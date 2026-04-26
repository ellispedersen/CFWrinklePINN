from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from training.cross_test_harness import REPORT_SCHEMA_VERSION


@dataclass(frozen=True)
class GateSpec:
    level: int
    run_dir: Path


@dataclass
class StepResult:
    name: str
    required: bool
    passed: bool
    criteria: str
    command: list[str] | None = None
    returncode: int | None = None
    duration_seconds: float = 0.0
    details: str = ""


@dataclass
class ReleaseSummary:
    passed: bool
    criteria: list[str]
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    steps: list[StepResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["steps"] = [asdict(step) for step in self.steps]
        return payload


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def parse_gate_specs(values: list[str]) -> list[GateSpec]:
    specs: list[GateSpec] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid gate spec {value!r}; expected LEVEL=RUN_DIR")
        level_str, run_dir_str = value.split("=", 1)
        try:
            level = int(level_str.strip())
        except ValueError as exc:
            raise ValueError(f"Invalid gate level {level_str!r}") from exc
        if level not in {1, 2, 3, 4}:
            raise ValueError(f"Unsupported gate level {level}; expected 1,2,3,4")
        run_dir = Path(run_dir_str.strip())
        if not run_dir_str.strip():
            raise ValueError(f"Gate run dir missing in spec {value!r}")
        specs.append(GateSpec(level=level, run_dir=run_dir))
    return specs


def parse_gate_specs_from_env() -> list[GateSpec]:
    raw = os.environ.get("RELEASE_CHECK_GATES", "").strip()
    if not raw:
        return []
    parts = [item.strip() for item in raw.split(";") if item.strip()]
    return parse_gate_specs(parts)


def validate_cross_test_report(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, f"Cross-test report not found: {path}"
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        return False, "Cross-test report must be a JSON object"
    if payload.get("schema_version") != REPORT_SCHEMA_VERSION:
        return False, (
            f"schema_version mismatch: got {payload.get('schema_version')!r}, "
            f"expected {REPORT_SCHEMA_VERSION!r}"
        )
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        return False, "Cross-test report must contain non-empty 'artifacts'"
    table = payload.get("comparison_table")
    if not isinstance(table, list) or not table:
        return False, "Cross-test report must contain non-empty 'comparison_table'"
    for idx, row in enumerate(table):
        if not isinstance(row, dict):
            return False, f"comparison_table[{idx}] must be an object"
        for key in ("rank", "artifact_name", "model_type", "checkpoint"):
            if key not in row:
                return False, f"comparison_table[{idx}] missing key {key!r}"
    return True, f"Validated cross-test report with {len(table)} ranked artifacts"


def evaluate_release(steps: list[StepResult]) -> bool:
    return all(step.passed for step in steps if step.required)


def _run_command(name: str, command: list[str], *, required: bool, criteria: str) -> StepResult:
    start = time.perf_counter()
    proc = subprocess.run(command, capture_output=True, text=True, check=False)
    duration = time.perf_counter() - start
    details_parts: list[str] = []
    if proc.stdout.strip():
        details_parts.append(f"stdout_tail={proc.stdout.strip().splitlines()[-1]}")
    if proc.stderr.strip():
        details_parts.append(f"stderr_tail={proc.stderr.strip().splitlines()[-1]}")
    details = "; ".join(details_parts)
    return StepResult(
        name=name,
        required=required,
        passed=proc.returncode == 0,
        criteria=criteria,
        command=command,
        returncode=proc.returncode,
        duration_seconds=round(duration, 3),
        details=details,
    )


def _build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Production release gate orchestrator")
    p.add_argument("--python-exe", default=os.environ.get("RELEASE_CHECK_PYTHON", sys.executable))
    p.add_argument("--contract-checkpoint", type=Path, default=os.environ.get("RELEASE_CHECK_CONTRACT_CHECKPOINT"))
    p.add_argument("--skip-contract-preflight", action="store_true")
    p.add_argument(
        "--label-suite-script",
        type=Path,
        default=Path(os.environ.get("RELEASE_CHECK_LABEL_SUITE_SCRIPT", "scripts/run_label_validation_suite.ps1")),
    )
    p.add_argument(
        "--include-label-integration",
        action="store_true",
        default=_env_flag("RELEASE_CHECK_INCLUDE_INTEGRATION_LABELS", default=False),
    )
    p.add_argument("--skip-label-validation", action="store_true")
    p.add_argument("--gate", action="append", default=[], help="Gate run spec: LEVEL=RUN_DIR")
    p.add_argument("--skip-gates", action="store_true")
    p.add_argument("--cross-test-report", type=Path, default=os.environ.get("RELEASE_CHECK_CROSS_TEST_REPORT"))
    p.add_argument(
        "--require-cross-test-report",
        action="store_true",
        default=_env_flag("RELEASE_CHECK_REQUIRE_CROSS_TEST", default=False),
    )
    p.add_argument(
        "--summary-path",
        type=Path,
        default=Path(os.environ.get("RELEASE_CHECK_SUMMARY_PATH", "reports/release_check_summary.json")),
    )
    return p.parse_args()


def main() -> None:
    args = _build_args()
    steps: list[StepResult] = []
    criteria = [
        "All required checks must pass (exit code 0 or validator pass).",
        "Required checks are contract preflight, label validation, and each provided gate check unless explicitly skipped.",
        "Cross-test report validation is required only when --require-cross-test-report (or RELEASE_CHECK_REQUIRE_CROSS_TEST=1) is set.",
    ]

    if not args.skip_contract_preflight:
        command = [args.python_exe, "-m", "training.validate_contracts"]
        if args.contract_checkpoint:
            command.extend(["--checkpoint", str(args.contract_checkpoint)])
        steps.append(
            _run_command(
                "contract_preflight",
                command,
                required=True,
                criteria="training.validate_contracts exits 0",
            )
        )

    if not args.skip_label_validation:
        command = [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(args.label_suite_script),
        ]
        if args.include_label_integration:
            command.append("-IncludeIntegration")
        steps.append(
            _run_command(
                "label_validation_suite",
                command,
                required=True,
                criteria="run_label_validation_suite.ps1 exits 0",
            )
        )

    if not args.skip_gates:
        gate_specs = parse_gate_specs(args.gate) if args.gate else parse_gate_specs_from_env()
        if not gate_specs:
            steps.append(
                StepResult(
                    name="gate_checks",
                    required=True,
                    passed=False,
                    criteria="At least one gate spec must be provided when gate checks are enabled",
                    details="Provide --gate LEVEL=RUN_DIR or set RELEASE_CHECK_GATES",
                )
            )
        for gate in gate_specs:
            command = [
                args.python_exe,
                "-m",
                "training.gate_check",
                "--level",
                str(gate.level),
                "--run-dir",
                str(gate.run_dir),
            ]
            steps.append(
                _run_command(
                    f"gate_level_{gate.level}",
                    command,
                    required=True,
                    criteria=f"training.gate_check --level {gate.level} exits 0",
                )
            )

    if args.cross_test_report is not None:
        passed, details = validate_cross_test_report(args.cross_test_report)
        steps.append(
            StepResult(
                name="cross_test_report_validation",
                required=args.require_cross_test_report,
                passed=passed,
                criteria="Cross-test report schema_version and ranking table are valid",
                details=details,
            )
        )
    elif args.require_cross_test_report:
        steps.append(
            StepResult(
                name="cross_test_report_validation",
                required=True,
                passed=False,
                criteria="Cross-test report path must be provided",
                details="Set --cross-test-report or RELEASE_CHECK_CROSS_TEST_REPORT",
            )
        )

    summary = ReleaseSummary(passed=evaluate_release(steps), criteria=criteria, steps=steps)
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary.to_dict(), f, indent=2)

    print(json.dumps(summary.to_dict(), indent=2))
    print(f"Release summary written to: {args.summary_path}")
    raise SystemExit(0 if summary.passed else 1)


if __name__ == "__main__":
    main()

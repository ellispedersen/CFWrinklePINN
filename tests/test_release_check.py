from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from training.cross_test_harness import REPORT_SCHEMA_VERSION
from training.release_check import StepResult, evaluate_release, parse_gate_specs, validate_cross_test_report


def test_evaluate_release_ignores_optional_failure() -> None:
    steps = [
        StepResult(name="contract", required=True, passed=True, criteria="ok"),
        StepResult(name="cross_test", required=False, passed=False, criteria="optional"),
    ]
    assert evaluate_release(steps) is True


def test_evaluate_release_fails_on_required_failure() -> None:
    steps = [
        StepResult(name="contract", required=True, passed=True, criteria="ok"),
        StepResult(name="gate", required=True, passed=False, criteria="must pass"),
    ]
    assert evaluate_release(steps) is False


def test_parse_gate_specs_accepts_windows_paths() -> None:
    specs = parse_gate_specs(["2=C:\\runs\\level2", "4=C:\\runs\\level4"])
    assert [(s.level, str(s.run_dir)) for s in specs] == [(2, "C:\\runs\\level2"), (4, "C:\\runs\\level4")]


def test_parse_gate_specs_rejects_invalid_level() -> None:
    with pytest.raises(ValueError, match="Unsupported gate level"):
        parse_gate_specs(["5=C:\\runs\\level5"])


def test_validate_cross_test_report_passes_for_minimal_valid_payload(tmp_path: Path) -> None:
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifacts": [{"artifact_name": "a"}],
        "comparison_table": [
            {"rank": 1, "artifact_name": "a", "model_type": "coarse", "checkpoint": "a.pt"},
        ],
    }
    path = tmp_path / "cross_report.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    passed, details = validate_cross_test_report(path)
    assert passed is True
    assert "Validated cross-test report" in details


def test_validate_cross_test_report_fails_when_required_fields_missing(tmp_path: Path) -> None:
    path = tmp_path / "cross_report_invalid.json"
    path.write_text(json.dumps({"schema_version": REPORT_SCHEMA_VERSION, "artifacts": []}), encoding="utf-8")

    passed, details = validate_cross_test_report(path)
    assert passed is False
    assert "artifacts" in details


def test_release_check_runs_level2_gate_with_legacy_history_schema(tmp_path: Path) -> None:
    run_dir = tmp_path / "level2_run"
    run_dir.mkdir()
    history = [
        {"loss/total": 2.0, "loss/severity": 0.2},
        {"loss/total": 0.9, "loss/severity": 0.01},
    ]
    (run_dir / "history.json").write_text(json.dumps(history), encoding="utf-8")

    summary_path = tmp_path / "release_summary.json"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "training.release_check",
            "--skip-contract-preflight",
            "--skip-label-validation",
            "--gate",
            f"2={run_dir}",
            "--summary-path",
            str(summary_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    gate_step = next(step for step in payload["steps"] if step["name"] == "gate_level_2")
    assert gate_step["passed"] is True

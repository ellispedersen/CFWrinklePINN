from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from training.gate_check import check_level_1, check_level_2, check_level_3, write_blocked_report


def _write_history(run_dir: Path, rows: list[dict]) -> None:
    run_dir.mkdir()
    with (run_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f)


def test_level2_track_a_history_skips_track_b_checks(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_history(
        run_dir,
        [
            {"loss/total": 2.0, "loss/severity": 0.2},
            {"loss/total": 0.9, "loss/severity": 0.01},
        ],
    )
    report = check_level_2(run_dir)
    check_names = {c.name for c in report.checks}
    assert "severity_loss_below_threshold" in check_names
    assert "total_loss_decreased_50pct" in check_names
    assert "fine_loss_components_finite" not in check_names


def test_level2_severity_prefers_loss_severity_when_available(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_history(
        run_dir,
        [
            {"loss/total": 2.0, "loss/severity": 0.2, "coarse/severity": 0.01, "loss/fine_stress_1": 0.3},
            {"loss/total": 0.9, "loss/severity": 0.01, "coarse/severity": 0.2, "loss/fine_stress_1": 0.2},
        ],
    )
    report = check_level_2(run_dir)
    check = next(c for c in report.checks if c.name == "severity_loss_below_threshold")
    assert check.passed is True
    assert "min(loss/severity)" in check.note
    assert "Using loss/severity" in check.note


def test_level2_severity_uses_cross_scale_fallback_when_loss_severity_missing(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_history(
        run_dir,
        [
            {"loss/total": 2.0, "coarse/severity": 0.2, "loss/fine_stress_1": 0.3},
            {"loss/total": 0.9, "coarse/severity": 0.01, "loss/fine_stress_1": 0.2},
        ],
    )
    report = check_level_2(run_dir)
    check = next(c for c in report.checks if c.name == "severity_loss_below_threshold")
    assert check.passed is True
    assert "min(coarse/severity)" in check.note
    assert "cross-scale history" in check.note


def test_level2_severity_fails_with_clear_note_when_no_valid_source(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_history(
        run_dir,
        [
            {"loss/total": 2.0, "loss/severity": "nan"},
            {"loss/total": 0.9, "loss/severity": None},
        ],
    )
    report = check_level_2(run_dir)
    check = next(c for c in report.checks if c.name == "severity_loss_below_threshold")
    assert check.passed is False
    assert "No valid severity metric source found" in check.note


def test_level2_track_b_accepts_legacy_fine_stress_key(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    row = {
        "loss/total": 1.0,
        "loss/severity": 0.01,
        "fine/stress_mae": 0.2,
        "fine/compressive_frac": 0.2,
        "loss/fine_stress": 0.1,
    }
    _write_history(run_dir, [row, row])
    report = check_level_2(run_dir)
    check = next(c for c in report.checks if c.name == "fine_loss_components_finite")
    assert check.passed is True


def test_level2_loss_reduction_prefers_val_loss_over_normalized_train_loss(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_history(
        run_dir,
        [
            {"loss/total": 7.5, "val_loss": 2.0, "loss/severity": 0.2},
            {"loss/total": 7.5, "val_loss": 0.8, "loss/severity": 0.01},
        ],
    )
    report = check_level_2(run_dir)
    check = next(c for c in report.checks if c.name == "total_loss_decreased_50pct")
    assert check.passed is True
    assert "val_loss" in check.note


def test_level2_loss_reduction_supports_loss_total_raw_fallback(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_history(
        run_dir,
        [
            {"loss/total": 7.5, "loss/total_raw": 1.8, "loss/severity": 0.2},
            {"loss/total": 7.5, "loss/total_raw": 0.7, "loss/severity": 0.01},
        ],
    )
    report = check_level_2(run_dir)
    check = next(c for c in report.checks if c.name == "total_loss_decreased_50pct")
    assert check.passed is True
    assert "loss/total_raw" in check.note


def test_write_blocked_report_emits_explicit_failure_payload(tmp_path: Path) -> None:
    run_dir = tmp_path / "missing_run"
    report_path = tmp_path / "reports" / "gate.json"
    report = write_blocked_report(
        level=2,
        run_dir=run_dir,
        report_path=report_path,
        reason="history_missing",
        note="history.json was not produced",
    )
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert report.passed is False
    assert payload["passed"] is False
    assert payload["checks"][0]["name"] == "blocked_precheck"
    assert "history_missing" in payload["checks"][0]["note"]


def test_level1_track_a_passes_with_loss_prefix(tmp_path: Path) -> None:
    """check_level_1 must pass when Track A emits loss/<name> keys."""
    run_dir = tmp_path / "run"
    _write_history(
        run_dir,
        [
            {
                "loss/total": 1.5,
                "loss/severity": 0.3,
                "loss/comp_frac": 0.1,
                "loss/oop": 0.05,
                "loss/thick_var": 0.02,
                "loss/physics": 0.0,
            }
        ],
    )
    report = check_level_1(run_dir)
    check = next(c for c in report.checks if c.name == "no_nan_in_components")
    assert check.passed is True
    assert "cross-scale" not in check.note


def test_level1_track_b_passes_with_coarse_prefix(tmp_path: Path) -> None:
    """check_level_1 must pass when Track B emits coarse/<name> keys instead of loss/<name>."""
    run_dir = tmp_path / "run"
    _write_history(
        run_dir,
        [
            {
                "loss/total": 1.5,
                # Track B: cross_scale_loss logs under coarse/ prefix, not loss/
                "coarse/severity": 0.3,
                "coarse/comp_frac": 0.1,
                "coarse/oop": 0.05,
                "coarse/thick_var": 0.02,
                "coarse/physics": 0.0,
                # Fine-mesh metrics also present
                "fine/stress_mae": 0.25,
                "fine/compressive_frac": 0.08,
            }
        ],
    )
    report = check_level_1(run_dir)
    check = next(c for c in report.checks if c.name == "no_nan_in_components")
    assert check.passed is True
    assert "cross-scale" in check.note


def test_level1_fails_when_components_are_nan(tmp_path: Path) -> None:
    """check_level_1 must fail when all component keys are absent (all NaN)."""
    run_dir = tmp_path / "run"
    _write_history(
        run_dir,
        [{"loss/total": 1.5}],  # no component keys at all
    )
    report = check_level_1(run_dir)
    check = next(c for c in report.checks if c.name == "no_nan_in_components")
    assert check.passed is False


def test_level3_track_b_mini_passes_with_calibrated_threshold(tmp_path: Path) -> None:
    """Level 3 Track B gate must pass when fine/stress_mae is below the calibrated 0.30 threshold.
    Anchored to the best observed mini-train result (0.2881, normon, 50 epochs).
    """
    run_dir = tmp_path / "run"
    rows = [
        {"val_loss": 0.30, "detection_rate": 0.8, "fine/stress_mae": 0.30, "fine/compressive_frac": 0.12},
        {"val_loss": 0.20, "detection_rate": 1.0, "fine/stress_mae": 0.29, "fine/compressive_frac": 0.12},
    ]
    _write_history(run_dir, rows)
    report = check_level_3(run_dir)
    check = next(c for c in report.checks if c.name == "fine_stress_mae_below_threshold")
    assert check.passed is True
    assert check.threshold == pytest.approx(0.30)


def test_level3_track_b_fails_when_fine_stress_mae_too_high(tmp_path: Path) -> None:
    """Level 3 Track B gate must fail when fine/stress_mae exceeds 0.30 threshold."""
    run_dir = tmp_path / "run"
    rows = [
        {"val_loss": 0.30, "detection_rate": 0.8, "fine/stress_mae": 0.35, "fine/compressive_frac": 0.12},
        {"val_loss": 0.20, "detection_rate": 1.0, "fine/stress_mae": 0.32, "fine/compressive_frac": 0.12},
    ]
    _write_history(run_dir, rows)
    report = check_level_3(run_dir)
    check = next(c for c in report.checks if c.name == "fine_stress_mae_below_threshold")
    assert check.passed is False


def test_level3_coarse_only_skips_fine_checks(tmp_path: Path) -> None:
    """Level 3 Track A (coarse-only) history must not trigger fine-mesh gate checks."""
    run_dir = tmp_path / "run"
    # 12 rows so baseline_idx=9 (epoch 10) is higher than the final row
    rows = [{"val_loss": 1.0 - i * 0.06, "detection_rate": min(0.5 + i * 0.05, 1.0)} for i in range(12)]
    _write_history(run_dir, rows)
    report = check_level_3(run_dir)
    check_names = {c.name for c in report.checks}
    assert "fine_stress_mae_below_threshold" not in check_names
    assert "fine_compressive_frac_above_threshold" not in check_names
    assert report.passed is True


def test_gate_check_cli_blocked_reason_writes_report_without_run_dir(tmp_path: Path) -> None:
    run_dir = tmp_path / "nonexistent_run"
    report_path = tmp_path / "reports" / "blocked.json"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "training.gate_check",
            "--level",
            "2",
            "--run-dir",
            str(run_dir),
            "--report-path",
            str(report_path),
            "--blocked-reason",
            "history_missing",
            "--blocked-note",
            "precheck failed",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert payload["checks"][0]["name"] == "blocked_precheck"

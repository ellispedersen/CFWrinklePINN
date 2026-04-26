from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from training import evaluate as eval_mod
from training.gate_check import check_level_2


def _coarse_tensor(severity: float, comp: float = 0.0) -> torch.Tensor:
    out = torch.zeros(1, 1, 4, dtype=torch.float32)
    out[..., 0] = severity
    out[..., 1] = comp
    return out


def test_compute_metrics_detection_uses_severity_channel(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(eval_mod, "_load_ground_truth_labels", lambda _sim_ids: {"sim_a": 1})
    predictions = [{"sim_id": "sim_a", "pred": _coarse_tensor(0.0, comp=1.0), "target": _coarse_tensor(0.0)}]
    metrics = eval_mod.compute_metrics(predictions, threshold=0.5)
    assert metrics["detection_rate"] == 0.0

    predictions[0]["pred"] = _coarse_tensor(0.8, comp=0.0)
    metrics = eval_mod.compute_metrics(predictions, threshold=0.5)
    assert metrics["detection_rate"] == 1.0
    assert metrics["mean_wrinkled_frac_pred"] == 1.0
    assert metrics["mean_wrinkled_frac_target"] == 0.0
    assert metrics["mean_wrinkled_frac_mae"] == 1.0


def test_compute_metrics_wrinkled_fraction_tracks_area(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(eval_mod, "_load_ground_truth_labels", lambda _sim_ids: {"sim_a": 1})
    pred = torch.zeros(2, 4, 4, dtype=torch.float32)
    tgt = torch.zeros(2, 4, 4, dtype=torch.float32)
    # Predicted wrinkled on 2/4 elements at least once across time.
    pred[0, :2, 0] = 0.8
    # Target wrinkled on 3/4 elements at least once across time.
    tgt[1, :3, 0] = 0.8
    metrics = eval_mod.compute_metrics([{"sim_id": "sim_a", "pred": pred, "target": tgt}], threshold=0.5)
    assert metrics["mean_wrinkled_frac_pred"] == pytest.approx(0.5)
    assert metrics["mean_wrinkled_frac_target"] == pytest.approx(0.75)
    assert metrics["mean_wrinkled_frac_mae"] == pytest.approx(0.25)


def test_compute_metrics_rejects_wrong_channel_count(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(eval_mod, "_load_ground_truth_labels", lambda _sim_ids: {"sim_a": 0})
    predictions = [{"sim_id": "sim_a", "pred": torch.zeros(1, 1, 3), "target": torch.zeros(1, 1, 3)}]
    with pytest.raises(ValueError, match="last dimension 4"):
        eval_mod.compute_metrics(predictions)


def test_compute_fine_metrics_channel_mapping_consistent() -> None:
    features = torch.tensor(
        [[[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]],
        dtype=torch.float32,
    )
    elems = torch.tensor([[0, 1, 2]], dtype=torch.int64)
    fine_tgt = features[:, elems, :].mean(dim=2)
    metrics = eval_mod.compute_fine_metrics([{"fine_pred": fine_tgt, "fine_features": features, "fine_elements": elems}])
    assert metrics["fine/stress_mae"] == 0.0
    assert metrics["fine/dz_mae"] == 0.0
    assert metrics["fine/thick_mae"] == 0.0
    assert metrics["fine/wrinkled_match_rate"] == 1.0
    assert metrics["fine/wrinkled_frac_pred"] == 0.0
    assert metrics["fine/wrinkled_frac_target"] == 0.0
    assert metrics["fine/wrinkled_frac_mae"] == 0.0


def test_compute_fine_metrics_wrinkled_match_rate() -> None:
    features = torch.zeros((1, 4, 4), dtype=torch.float32)
    elems = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.int64)
    # Target: only first element wrinkled (compressive).
    # Node 0 is exclusive to elem 0; set it to -0.6 so elem 0 mean = -0.2 < -0.05
    # but elem 1 (nodes 1,2,3 all zero) mean = 0.0 > -0.05 (not wrinkled).
    features[0, 0, 0] = -0.6
    fine_tgt = features[:, elems, :].mean(dim=2)
    # Prediction: both elements wrinkled -> one correct, one false positive.
    fine_pred = fine_tgt.clone()
    fine_pred[0, 1, 0] = -0.2
    metrics = eval_mod.compute_fine_metrics([{"fine_pred": fine_pred, "fine_features": features, "fine_elements": elems}])
    assert metrics["fine/wrinkled_frac_target"] == pytest.approx(0.5)
    assert metrics["fine/wrinkled_frac_pred"] == pytest.approx(1.0)
    assert metrics["fine/wrinkled_frac_mae"] == pytest.approx(0.5)
    assert metrics["fine/wrinkled_match_rate"] == pytest.approx(0.5)


def test_compute_fine_metrics_rejects_out_of_range_elements() -> None:
    item = {
        "fine_pred": torch.zeros(1, 1, 4),
        "fine_features": torch.zeros(1, 3, 4),
        "fine_elements": torch.tensor([[0, 1, 5]], dtype=torch.int64),
    }
    with pytest.raises(ValueError, match="out-of-range"):
        eval_mod.compute_fine_metrics([item])


def test_gate_level2_accepts_minimum_required_fine_loss_component_set(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    row = {
        "loss/total": 1.0,
        "loss/severity": 0.01,
        "fine/stress_mae": 0.2,
        "fine/compressive_frac": 0.2,
        "loss/fine_stress_1": 0.1,
        "loss/fine_dz": 0.1,
        "loss/fine_dz_mono": 0.1,
        "loss/fine_buckling": 0.1,
        "loss/fine_coupling": 0.1,
    }
    with (run_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump([row, row], f)

    report = check_level_2(run_dir)
    check = next(c for c in report.checks if c.name == "fine_loss_components_finite")
    assert check.passed is True

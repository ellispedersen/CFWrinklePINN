from __future__ import annotations

import math

from model.contracts import MODEL_TYPE_COARSE, MODEL_TYPE_CROSS_SCALE
from training.cross_test_harness import (
    FINE_METRIC_COLUMNS,
    build_comparison_table,
    standardize_metrics,
)


def test_standardize_metrics_sets_explicit_na_for_coarse_only_model() -> None:
    metrics, statuses = standardize_metrics(
        {"val_loss": 0.42, "f1": 0.9},
        model_type=MODEL_TYPE_COARSE,
    )
    for key in FINE_METRIC_COLUMNS:
        assert metrics[key] is None
        assert statuses[key] == "na:not_applicable"


def test_standardize_metrics_converts_non_finite_cross_scale_values_to_na() -> None:
    metrics, statuses = standardize_metrics(
        {"val_loss": 0.2, "fine/stress_mae": float("nan"), "fine/dz_mae": math.inf},
        model_type=MODEL_TYPE_CROSS_SCALE,
    )
    assert metrics["fine/stress_mae"] is None
    assert statuses["fine/stress_mae"] == "na:non_finite"
    assert metrics["fine/dz_mae"] is None
    assert statuses["fine/dz_mae"] == "na:non_finite"


def test_build_comparison_table_ranks_by_val_loss_then_f1() -> None:
    rows = [
        {
            "artifact_name": "model-b",
            "model_type": MODEL_TYPE_CROSS_SCALE,
            "checkpoint": "b.pt",
            "metrics": {"val_loss": 0.2, "f1": 0.7},
        },
        {
            "artifact_name": "model-a",
            "model_type": MODEL_TYPE_COARSE,
            "checkpoint": "a.pt",
            "metrics": {"val_loss": 0.2, "f1": 0.9},
        },
        {
            "artifact_name": "model-v1",
            "model_type": MODEL_TYPE_COARSE,
            "checkpoint": "v1.pt",
            "metrics": {"val_loss": 0.5, "f1": 0.99},
        },
    ]
    table = build_comparison_table(rows)
    assert [row["artifact_name"] for row in table] == ["model-a", "model-b", "model-v1"]
    assert [row["rank"] for row in table] == [1, 2, 3]

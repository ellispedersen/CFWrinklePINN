from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from model.loss import cross_scale_loss, wrinkle_loss
from training.evaluate import compute_fine_metrics, compute_metrics


def test_wrinkle_loss_rejects_wrong_channel_count() -> None:
    pred = torch.zeros(2, 3, 3)
    tgt = torch.zeros(2, 3, 3)
    with pytest.raises(ValueError, match="last dimension"):
        wrinkle_loss(pred, tgt)


def test_compute_metrics_rejects_shape_mismatch() -> None:
    pred_item = {
        "sim_id": "sim_1",
        "pred": torch.zeros(2, 3, 4),
        "target": torch.zeros(2, 4, 4),
    }
    with pytest.raises(ValueError, match="shape mismatch"):
        compute_metrics([pred_item])


def test_compute_fine_metrics_rejects_out_of_range_elements() -> None:
    fine_preds = [
        {
            "fine_pred": torch.zeros(2, 1, 4),
            "fine_features": torch.zeros(2, 2, 4),
            "fine_elements": torch.tensor([[0, 1, 2]], dtype=torch.int64),
        }
    ]
    with pytest.raises(ValueError, match="out-of-range"):
        compute_fine_metrics(fine_preds)


def test_cross_scale_loss_rejects_wrong_fine_channels() -> None:
    with pytest.raises(ValueError, match="last dimension"):
        cross_scale_loss(
            fine_pred=torch.zeros(2, 1, 3),
            fine_features=torch.zeros(2, 3, 4),
            fine_elements=torch.tensor([[0, 1, 2]], dtype=torch.int64),
            coarse_pred=torch.zeros(2, 1, 4),
            coarse_targets=torch.zeros(2, 1, 4),
        )

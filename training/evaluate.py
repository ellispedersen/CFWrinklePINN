from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch

from model.loss import wrinkle_loss

WP2_H5 = Path("data/cfwrinkle_dataset.h5")


def compute_metrics(
    predictions: list[dict],
    threshold: float = 0.15,
) -> dict[str, float]:
    gt_labels = _load_ground_truth_labels([p["sim_id"] for p in predictions])

    preds_binary: list[int] = []
    gt_binary: list[int] = []
    severity_maes: list[float] = []
    val_losses: list[float] = []

    for pred_item in predictions:
        sid = pred_item["sim_id"]
        pred = pred_item["pred"]
        target = pred_item["target"]
        pred_sev = pred[..., 0]
        tgt_sev = target[..., 0]

        max_pred = float(torch.nan_to_num(pred_sev, nan=0.0).max().detach().cpu())
        preds_binary.append(1 if max_pred >= threshold else 0)
        gt_binary.append(gt_labels.get(sid, 0))

        mask = ~torch.isnan(tgt_sev)
        if mask.sum() > 0:
            severity_maes.append(float((pred_sev[mask] - tgt_sev[mask]).abs().mean().detach().cpu()))

        loss, _ = wrinkle_loss(pred, target)
        val_losses.append(float(loss.detach().cpu()))

    tp = sum(p == 1 and g == 1 for p, g in zip(preds_binary, gt_binary))
    fp = sum(p == 1 and g == 0 for p, g in zip(preds_binary, gt_binary))
    fn = sum(p == 0 and g == 1 for p, g in zip(preds_binary, gt_binary))
    tn = sum(p == 0 and g == 0 for p, g in zip(preds_binary, gt_binary))

    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    far = fp / max(fp + tn, 1)

    return {
        "detection_rate": float(recall),
        "false_alarm_rate": float(far),
        "precision": float(precision),
        "f1": float(f1),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "mean_severity_mae": float(np.mean(severity_maes)) if severity_maes else 0.0,
        "val_loss": float(np.mean(val_losses)) if val_losses else 0.0,
    }


def _load_ground_truth_labels(sim_ids: list[str]) -> dict[str, int]:
    labels: dict[str, int] = {}
    with h5py.File(WP2_H5, "r") as f:
        for sid in sim_ids:
            path = f"simulations/{sid}"
            if path in f:
                labels[sid] = int(f[path].attrs.get("is_wrinkled", 0))
    return labels


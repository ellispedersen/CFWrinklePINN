from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch

from model.labels import COARSE_TARGET_INDEX, FINE_TARGET_INDEX, N_COARSE_TARGETS, N_FINE_TARGETS, require_last_dim
from model.loss import wrinkle_loss

import os as _os
WP2_H5 = Path(_os.environ.get("WP2_H5", "data/cfwrinkle_dataset.h5"))


def compute_metrics(
    predictions: list[dict],
    threshold: float = 0.15,
) -> dict[str, float]:
    sim_ids: list[str] = []
    for pred_item in predictions:
        sid = pred_item["sim_id"]
        pred = pred_item["pred"]
        target = pred_item["target"]
        if pred.shape != target.shape:
            raise ValueError(f"pred/target shape mismatch for {sid}: {tuple(pred.shape)} vs {tuple(target.shape)}")
        require_last_dim(pred, N_COARSE_TARGETS, f"pred[{sid}]")
        require_last_dim(target, N_COARSE_TARGETS, f"target[{sid}]")
        sim_ids.append(sid)

    gt_labels = _load_ground_truth_labels(sim_ids)

    preds_binary: list[int] = []
    gt_binary: list[int] = []
    pred_wrinkled_fracs: list[float] = []
    tgt_wrinkled_fracs: list[float] = []
    severity_maes: list[float] = []
    val_losses: list[float] = []

    for pred_item in predictions:
        sid = pred_item["sim_id"]
        pred = pred_item["pred"]
        target = pred_item["target"]
        pred_sev = pred[..., COARSE_TARGET_INDEX["severity"]]
        tgt_sev = target[..., COARSE_TARGET_INDEX["severity"]]

        max_pred = float(torch.nan_to_num(pred_sev, nan=0.0).max().detach().cpu())
        preds_binary.append(1 if max_pred >= threshold else 0)
        gt_binary.append(gt_labels.get(sid, 0))

        pred_wrinkled = torch.nan_to_num(pred_sev, nan=0.0).amax(dim=0) >= threshold
        tgt_wrinkled = torch.nan_to_num(tgt_sev, nan=0.0).amax(dim=0) >= threshold
        pred_wrinkled_fracs.append(float(pred_wrinkled.float().mean().detach().cpu()))
        tgt_wrinkled_fracs.append(float(tgt_wrinkled.float().mean().detach().cpu()))

        mask = ~torch.isnan(tgt_sev)
        if mask.sum() > 0:
            severity_maes.append(float((pred_sev[mask] - tgt_sev[mask]).abs().mean().detach().cpu()))

        _, log = wrinkle_loss(pred, target)
        val_losses.append(log["loss/total"])

    tp = sum(p == 1 and g == 1 for p, g in zip(preds_binary, gt_binary))
    fp = sum(p == 1 and g == 0 for p, g in zip(preds_binary, gt_binary))
    fn = sum(p == 0 and g == 1 for p, g in zip(preds_binary, gt_binary))
    tn = sum(p == 0 and g == 0 for p, g in zip(preds_binary, gt_binary))

    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    far = fp / max(fp + tn, 1)
    mean_pred_wrinkled_frac = float(np.mean(pred_wrinkled_fracs)) if pred_wrinkled_fracs else 0.0
    mean_tgt_wrinkled_frac = float(np.mean(tgt_wrinkled_fracs)) if tgt_wrinkled_fracs else 0.0

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
        "mean_wrinkled_frac_pred": mean_pred_wrinkled_frac,
        "mean_wrinkled_frac_target": mean_tgt_wrinkled_frac,
        "mean_wrinkled_frac_mae": abs(mean_pred_wrinkled_frac - mean_tgt_wrinkled_frac),
        "val_loss": float(np.mean(val_losses)) if val_losses else 0.0,
    }


def compute_fine_metrics(
    fine_preds: list[dict],
) -> dict[str, float]:
    """Compute fine-mesh evaluation metrics for CrossScaleNet.

    Each entry in fine_preds must have:
      "fine_pred"     (T, N_fine_elem, 4) — model output
      "fine_features" (T, N_fine_nodes, 4) — ground truth at node level
      "fine_elements" (N_fine_elem, 3) — node indices per element
    """
    agg = _init_fine_metric_agg()
    for item in fine_preds:
        _accumulate_fine_metrics(agg, item)
    return _finalize_fine_metric_agg(agg)


def _init_fine_metric_agg() -> dict[str, list[float]]:
    return {
        "stress_maes": [],
        "dz_maes": [],
        "thick_maes": [],
        "compressive_fracs": [],
        "wrinkled_match_rates": [],
        "wrinkled_pred_fracs": [],
        "wrinkled_tgt_fracs": [],
    }


def _accumulate_fine_metrics(agg: dict[str, list[float]], item: dict) -> None:
    pred = item["fine_pred"]          # (T, N_fine_elem, 4)
    features = item["fine_features"]  # (T, N_fine_nodes, 4)
    elems = item["fine_elements"]     # (N_fine_elem, 3)
    require_last_dim(pred, N_FINE_TARGETS, "fine_pred")
    require_last_dim(features, N_FINE_TARGETS, "fine_features")
    if pred.shape[0] != features.shape[0] or pred.shape[1] != elems.shape[0]:
        raise ValueError(
            f"fine_pred shape {tuple(pred.shape)} incompatible with fine_features {tuple(features.shape)} "
            f"and fine_elements {tuple(elems.shape)}"
        )
    if elems.numel() > 0:
        if int(elems.min()) < 0 or int(elems.max()) >= int(features.shape[1]):
            raise ValueError("fine_elements contains out-of-range node indices for fine_features")

    # Fine element targets: average corner node values
    fine_tgt = features[:, elems, :].mean(dim=2)  # (T, N_fine_elem, 4)

    # Fiber stress MAE (channels 0 and 1)
    mask_s = ~torch.isnan(fine_tgt[..., FINE_TARGET_INDEX["stress_1"]])
    if mask_s.any():
        s1_mae = (
            pred[..., FINE_TARGET_INDEX["stress_1"]][mask_s]
            - fine_tgt[..., FINE_TARGET_INDEX["stress_1"]][mask_s]
        ).abs().mean().item()
        s2_mae = (
            pred[..., FINE_TARGET_INDEX["stress_2"]][mask_s]
            - fine_tgt[..., FINE_TARGET_INDEX["stress_2"]][mask_s]
        ).abs().mean().item()
        agg["stress_maes"].append((s1_mae + s2_mae) / 2)

    # Displacement Z MAE (channel 2)
    mask_dz = ~torch.isnan(fine_tgt[..., FINE_TARGET_INDEX["dz"]])
    if mask_dz.any():
        agg["dz_maes"].append(
            (
                pred[..., FINE_TARGET_INDEX["dz"]][mask_dz]
                - fine_tgt[..., FINE_TARGET_INDEX["dz"]][mask_dz]
            ).abs().mean().item()
        )

    # Thickness MAE (channel 3)
    mask_th = ~torch.isnan(fine_tgt[..., FINE_TARGET_INDEX["thickness"]])
    if mask_th.any():
        agg["thick_maes"].append(
            (
                pred[..., FINE_TARGET_INDEX["thickness"]][mask_th]
                - fine_tgt[..., FINE_TARGET_INDEX["thickness"]][mask_th]
            ).abs().mean().item()
        )

    # Compressive fraction: fine elements where predicted fiber_stress_1 < -0.05 (normalised)
    compressive = (pred[..., FINE_TARGET_INDEX["stress_1"]] < -0.05).float().mean().item()
    agg["compressive_fracs"].append(compressive)

    # Wrinkle match metrics on fine elements (binary compressive mask agreement).
    pred_wrinkled = pred[..., FINE_TARGET_INDEX["stress_1"]] < -0.05
    tgt_wrinkled = fine_tgt[..., FINE_TARGET_INDEX["stress_1"]] < -0.05
    valid = ~torch.isnan(fine_tgt[..., FINE_TARGET_INDEX["stress_1"]])
    if valid.any():
        pred_valid = pred_wrinkled[valid]
        tgt_valid = tgt_wrinkled[valid]
        agg["wrinkled_match_rates"].append(float((pred_valid == tgt_valid).float().mean().item()))
        agg["wrinkled_pred_fracs"].append(float(pred_valid.float().mean().item()))
        agg["wrinkled_tgt_fracs"].append(float(tgt_valid.float().mean().item()))


def _finalize_fine_metric_agg(agg: dict[str, list[float]]) -> dict[str, float]:
    pred_frac = float(np.mean(agg["wrinkled_pred_fracs"])) if agg["wrinkled_pred_fracs"] else 0.0
    tgt_frac = float(np.mean(agg["wrinkled_tgt_fracs"])) if agg["wrinkled_tgt_fracs"] else 0.0
    return {
        "fine/stress_mae": float(np.mean(agg["stress_maes"])) if agg["stress_maes"] else 0.0,
        "fine/dz_mae": float(np.mean(agg["dz_maes"])) if agg["dz_maes"] else 0.0,
        "fine/thick_mae": float(np.mean(agg["thick_maes"])) if agg["thick_maes"] else 0.0,
        "fine/compressive_frac": float(np.mean(agg["compressive_fracs"])) if agg["compressive_fracs"] else 0.0,
        "fine/wrinkled_match_rate": float(np.mean(agg["wrinkled_match_rates"])) if agg["wrinkled_match_rates"] else 0.0,
        "fine/wrinkled_frac_pred": pred_frac,
        "fine/wrinkled_frac_target": tgt_frac,
        "fine/wrinkled_frac_mae": abs(pred_frac - tgt_frac),
    }


def _load_ground_truth_labels(sim_ids: list[str]) -> dict[str, int]:
    labels: dict[str, int] = {}
    with h5py.File(WP2_H5, "r") as f:
        for sid in sim_ids:
            path = f"simulations/{sid}"
            if path in f:
                labels[sid] = int(f[path].attrs.get("is_wrinkled", 0))
    return labels


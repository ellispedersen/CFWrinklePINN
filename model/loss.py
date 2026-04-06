from __future__ import annotations

import torch
import torch.nn.functional as F


LOSS_WEIGHTS = {
    "severity": 3.0,
    "comp_frac": 1.0,
    "oop": 1.0,
    "thick_var": 0.5,
    "physics": 2.0,
}


def _nan_huber(pred: torch.Tensor, target: torch.Tensor, delta: float = 0.1) -> torch.Tensor:
    mask = ~torch.isnan(target)
    if mask.sum() == 0:
        return pred.sum() * 0.0
    return F.huber_loss(pred[mask], target[mask], delta=delta, reduction="mean")


def _nan_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mask = ~torch.isnan(target)
    if mask.sum() == 0:
        return pred.sum() * 0.0
    return F.mse_loss(pred[mask], target[mask], reduction="mean")


def monotonicity_loss(severity_pred: torch.Tensor) -> torch.Tensor:
    delta = severity_pred[1:] - severity_pred[:-1]
    violations = torch.clamp(-delta, min=0.0)
    return violations.mean()


def wrinkle_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    weights: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    w = weights or LOSS_WEIGHTS

    pred_sev = predictions[..., 0]
    pred_comp = predictions[..., 1]
    pred_oop = predictions[..., 2]
    pred_tv = predictions[..., 3]

    tgt_sev = targets[..., 0]
    tgt_comp = targets[..., 1]
    tgt_oop = targets[..., 2]
    tgt_tv = targets[..., 3]

    l_sev = _nan_huber(pred_sev, tgt_sev)
    l_comp = _nan_mse(pred_comp, tgt_comp)
    l_oop = _nan_mse(pred_oop, tgt_oop)
    l_tv = _nan_mse(pred_tv, tgt_tv)
    l_phys = monotonicity_loss(pred_sev)

    def _safe_norm(loss: torch.Tensor) -> torch.Tensor:
        return loss / loss.detach().clamp(min=1e-8)

    total = (
        w["severity"] * _safe_norm(l_sev)
        + w["comp_frac"] * _safe_norm(l_comp)
        + w["oop"] * _safe_norm(l_oop)
        + w["thick_var"] * _safe_norm(l_tv)
        + w["physics"] * _safe_norm(l_phys)
    )

    log = {
        "loss/total": float(total.detach().cpu()),
        "loss/severity": float(l_sev.detach().cpu()),
        "loss/comp_frac": float(l_comp.detach().cpu()),
        "loss/oop": float(l_oop.detach().cpu()),
        "loss/thick_var": float(l_tv.detach().cpu()),
        "loss/physics": float(l_phys.detach().cpu()),
    }
    return total, log


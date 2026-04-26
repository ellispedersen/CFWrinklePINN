from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from .labels import (
    COARSE_TARGET_INDEX,
    FINE_TARGET_INDEX,
    N_COARSE_TARGETS,
    N_FINE_TARGETS,
    require_last_dim,
)


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


def _chunked_fine_supervision_losses(
    fine_pred: torch.Tensor,
    fine_features: torch.Tensor,
    fine_elements: torch.Tensor,
    chunk_elems: int,
    delta: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute fine-target losses without materializing full (T, N_elem, 3, 4) gathers.

    This bounds peak activation memory by chunking over fine elements.
    """
    chunk_elems = max(1, int(chunk_elems))
    dev = fine_pred.device
    total_mse_fs1 = torch.zeros((), device=dev, dtype=torch.float32)
    total_mse_fs2 = torch.zeros((), device=dev, dtype=torch.float32)
    total_huber_dz = torch.zeros((), device=dev, dtype=torch.float32)
    total_mse_thick = torch.zeros((), device=dev, dtype=torch.float32)
    total_buckling = torch.zeros((), device=dev, dtype=torch.float32)
    cnt_fs1 = torch.zeros((), device=dev, dtype=torch.float32)
    cnt_fs2 = torch.zeros((), device=dev, dtype=torch.float32)
    cnt_dz = torch.zeros((), device=dev, dtype=torch.float32)
    cnt_thick = torch.zeros((), device=dev, dtype=torch.float32)
    cnt_buckling = torch.zeros((), device=dev, dtype=torch.float32)

    for start in range(0, fine_elements.shape[0], chunk_elems):
        end = min(start + chunk_elems, fine_elements.shape[0])
        elem_idx = fine_elements[start:end]
        n0 = elem_idx[:, 0]
        n1 = elem_idx[:, 1]
        n2 = elem_idx[:, 2]
        tgt_chunk = (
            fine_features[:, n0, :]
            + fine_features[:, n1, :]
            + fine_features[:, n2, :]
        ) / 3.0
        pred_chunk = fine_pred[:, start:end, :]

        pred_fs1 = pred_chunk[..., FINE_TARGET_INDEX["stress_1"]]
        tgt_fs1 = tgt_chunk[..., FINE_TARGET_INDEX["stress_1"]]
        valid_fs1 = ~torch.isnan(tgt_fs1)
        if valid_fs1.any():
            diff = (pred_fs1[valid_fs1] - tgt_fs1[valid_fs1]).to(torch.float32)
            total_mse_fs1 = total_mse_fs1 + diff.pow(2).sum()
            cnt_fs1 = cnt_fs1 + valid_fs1.sum().to(torch.float32)

        pred_fs2 = pred_chunk[..., FINE_TARGET_INDEX["stress_2"]]
        tgt_fs2 = tgt_chunk[..., FINE_TARGET_INDEX["stress_2"]]
        valid_fs2 = ~torch.isnan(tgt_fs2)
        if valid_fs2.any():
            diff = (pred_fs2[valid_fs2] - tgt_fs2[valid_fs2]).to(torch.float32)
            total_mse_fs2 = total_mse_fs2 + diff.pow(2).sum()
            cnt_fs2 = cnt_fs2 + valid_fs2.sum().to(torch.float32)

        pred_dz = pred_chunk[..., FINE_TARGET_INDEX["dz"]]
        tgt_dz = tgt_chunk[..., FINE_TARGET_INDEX["dz"]]
        valid_dz = ~torch.isnan(tgt_dz)
        if valid_dz.any():
            abs_err = (pred_dz[valid_dz] - tgt_dz[valid_dz]).abs().to(torch.float32)
            quad = torch.minimum(abs_err, torch.tensor(delta, device=dev, dtype=torch.float32))
            lin = abs_err - quad
            total_huber_dz = total_huber_dz + (0.5 * quad.pow(2) + delta * lin).sum()
            cnt_dz = cnt_dz + valid_dz.sum().to(torch.float32)

        pred_thick = pred_chunk[..., FINE_TARGET_INDEX["thickness"]]
        tgt_thick = tgt_chunk[..., FINE_TARGET_INDEX["thickness"]]
        valid_thick = ~torch.isnan(tgt_thick)
        if valid_thick.any():
            diff = (pred_thick[valid_thick] - tgt_thick[valid_thick]).to(torch.float32)
            total_mse_thick = total_mse_thick + diff.pow(2).sum()
            cnt_thick = cnt_thick + valid_thick.sum().to(torch.float32)

        valid_buckling = valid_fs1
        if valid_buckling.any():
            tensile = (tgt_fs1 >= -0.05).to(torch.float32)
            pred_dz_sq = pred_dz.pow(2).to(torch.float32)
            valid_f = valid_buckling.to(torch.float32)
            total_buckling = total_buckling + (pred_dz_sq * tensile * valid_f).sum()
            cnt_buckling = cnt_buckling + valid_f.sum()

    zero = fine_pred.sum() * 0.0
    l_fs1 = total_mse_fs1 / cnt_fs1.clamp(min=1.0) if torch.gt(cnt_fs1, 0).item() else zero
    l_fs2 = total_mse_fs2 / cnt_fs2.clamp(min=1.0) if torch.gt(cnt_fs2, 0).item() else zero
    l_dz = total_huber_dz / cnt_dz.clamp(min=1.0) if torch.gt(cnt_dz, 0).item() else zero
    l_thick = total_mse_thick / cnt_thick.clamp(min=1.0) if torch.gt(cnt_thick, 0).item() else zero
    l_buckling = total_buckling / cnt_buckling.clamp(min=1.0) if torch.gt(cnt_buckling, 0).item() else zero
    return l_fs1, l_fs2, l_dz, l_thick, l_buckling


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def monotonicity_loss(severity_pred: torch.Tensor) -> torch.Tensor:
    delta = severity_pred[1:] - severity_pred[:-1]
    violations = torch.clamp(-delta, min=0.0)
    return violations.mean()


def wrinkle_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    weights: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    if predictions.shape != targets.shape:
        raise ValueError(
            f"predictions and targets must have identical shape for wrinkle_loss; "
            f"got {tuple(predictions.shape)} vs {tuple(targets.shape)}"
        )
    require_last_dim(predictions, N_COARSE_TARGETS, "predictions")
    require_last_dim(targets, N_COARSE_TARGETS, "targets")
    w = weights or LOSS_WEIGHTS

    pred_sev = predictions[..., COARSE_TARGET_INDEX["severity"]]
    pred_comp = predictions[..., COARSE_TARGET_INDEX["comp_frac"]]
    pred_oop = predictions[..., COARSE_TARGET_INDEX["oop"]]
    pred_tv = predictions[..., COARSE_TARGET_INDEX["thick_var"]]

    tgt_sev = targets[..., COARSE_TARGET_INDEX["severity"]]
    tgt_comp = targets[..., COARSE_TARGET_INDEX["comp_frac"]]
    tgt_oop = targets[..., COARSE_TARGET_INDEX["oop"]]
    tgt_tv = targets[..., COARSE_TARGET_INDEX["thick_var"]]

    l_sev = _nan_huber(pred_sev, tgt_sev)
    l_comp = _nan_mse(pred_comp, tgt_comp)
    l_oop = _nan_mse(pred_oop, tgt_oop)
    l_tv = _nan_mse(pred_tv, tgt_tv)
    l_phys = monotonicity_loss(pred_sev)

    # Weighted raw total — used for logging and early stopping
    raw_total = (
        w["severity"] * l_sev
        + w["comp_frac"] * l_comp
        + w["oop"] * l_oop
        + w["thick_var"] * l_tv
        + w["physics"] * l_phys
    )

    # Self-normalised total — each component contributes unit-scale gradients
    # scaled by its weight.  The forward value is always sum(weights) but the
    # gradient for each component is w_i / |l_i|, which balances learning across
    # targets of different magnitudes.
    def _safe_norm(loss: torch.Tensor) -> torch.Tensor:
        # Floor at 0.01 caps gradient amplification at ~100× per component.
        # Without this, near-zero losses (especially physics/monotonicity at T=256)
        # can amplify gradients by >10,000×, overwhelming clip_grad_norm.
        return loss / loss.detach().clamp(min=1e-2)

    grad_total = (
        w["severity"] * _safe_norm(l_sev)
        + w["comp_frac"] * _safe_norm(l_comp)
        + w["oop"] * _safe_norm(l_oop)
        + w["thick_var"] * _safe_norm(l_tv)
        + w["physics"] * _safe_norm(l_phys)
    )

    log = {
        "loss/total": float(raw_total.detach().cpu()),
        "loss/severity": float(l_sev.detach().cpu()),
        "loss/comp_frac": float(l_comp.detach().cpu()),
        "loss/oop": float(l_oop.detach().cpu()),
        "loss/thick_var": float(l_tv.detach().cpu()),
        "loss/physics": float(l_phys.detach().cpu()),
    }
    return grad_total, log


CROSS_SCALE_WEIGHTS = {
    "fine_stress": 2.0,    # fiber_stress_1 + fiber_stress_2 (primary wrinkle precursors)
    "fine_dz": 1.5,        # displacement_z (OOP motion)
    "fine_thick": 1.0,     # thickness (thinning indicator)
    "coarse": 1.0,         # auxiliary coarse loss (backward compat)
    "physics": 1.0,        # monotonicity on fine stress
    "dz_mono": 0.5,        # monotonicity on |dz|
    "buckling": 0.5,       # dz~0 in tensile zones
    "coupling": 0.5,       # thickness-|dz| anti-correlation
    "coherence": 0.3,      # optional fine-element spatial coherence
}

# Env-var overrides for per-run weight tuning (Track C and beyond).
# Set e.g. CFWRINKLE_FINE_DZ_WEIGHT=4.0 to override fine_dz without code changes.
_WEIGHT_ENV_VARS: dict[str, str] = {
    "CFWRINKLE_FINE_DZ_WEIGHT":     "fine_dz",
    "CFWRINKLE_FINE_STRESS_WEIGHT": "fine_stress",
    "CFWRINKLE_FINE_THICK_WEIGHT":  "fine_thick",
    "CFWRINKLE_BUCKLING_WEIGHT":    "buckling",
    "CFWRINKLE_COUPLING_WEIGHT":    "coupling",
    "CFWRINKLE_COHERENCE_WEIGHT":   "coherence",
    "CFWRINKLE_DZ_MONO_WEIGHT":     "dz_mono",
}


def _resolve_cross_scale_weights(overrides: dict[str, float] | None) -> dict[str, float]:
    """Return effective weights: defaults → caller overrides → env var overrides."""
    w = dict(CROSS_SCALE_WEIGHTS)
    if overrides:
        w.update(overrides)
    for env_key, weight_key in _WEIGHT_ENV_VARS.items():
        val = os.environ.get(env_key)
        if val is not None:
            w[weight_key] = float(val)
    return w


def buckling_onset_loss(
    fine_pred: torch.Tensor,
    fine_tgt: torch.Tensor,
) -> torch.Tensor:
    tgt_stress = fine_tgt[..., FINE_TARGET_INDEX["stress_1"]]
    pred_dz = fine_pred[..., FINE_TARGET_INDEX["dz"]]
    valid = ~torch.isnan(tgt_stress)
    tensile = (tgt_stress >= -0.05).to(pred_dz.dtype)
    if valid.sum() == 0:
        return pred_dz.sum() * 0.0
    valid_f = valid.to(pred_dz.dtype)
    return (pred_dz.pow(2) * tensile * valid_f).sum() / valid_f.sum().clamp(min=1.0)


def thickness_dz_coupling_loss(fine_pred: torch.Tensor) -> torch.Tensor:
    dz = fine_pred[..., FINE_TARGET_INDEX["dz"]]
    thick = fine_pred[..., FINE_TARGET_INDEX["thickness"]]
    dz_norm = dz - dz.mean(dim=-1, keepdim=True)
    thick_norm = thick - thick.mean(dim=-1, keepdim=True)
    cos_sim = F.cosine_similarity(dz_norm.abs(), thick_norm, dim=-1)
    return F.relu(cos_sim).mean()


def spatial_coherence_loss(
    fine_pred: torch.Tensor,
    fine_elem_edge_index: torch.Tensor,
) -> torch.Tensor:
    if fine_elem_edge_index.numel() == 0:
        return fine_pred.sum() * 0.0
    src, dst = fine_elem_edge_index[0], fine_elem_edge_index[1]
    diff = fine_pred[:, src, :] - fine_pred[:, dst, :]
    return diff.pow(2).mean()


def cross_scale_loss(
    fine_pred: torch.Tensor,
    fine_features: torch.Tensor,
    fine_elements: torch.Tensor,
    coarse_pred: torch.Tensor,
    coarse_targets: torch.Tensor,
    fine_elem_edge_index: torch.Tensor | None = None,
    weights: dict[str, float] | None = None,
    step_idx: int | None = None,
    physics_weight_scale: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Loss for CrossScaleNet.

    Args:
        fine_pred:      (T, N_fine_elem, 4) — model fine-element predictions
                        channels: [fiber_stress_1, fiber_stress_2, displacement_z, thickness]
        fine_features:  (T, N_fine_nodes, 4) — ground truth at fine node level
                        same channel order as fine_pred
        fine_elements:  (N_fine_elem, 3) int64 — node indices per fine element
        coarse_pred:    (T, M_coarse, 4) — coarse head output (backward compat)
        coarse_targets: (T, M_coarse, 4) — standard wrinkle_loss targets
        weights:        optional override of CROSS_SCALE_WEIGHTS
    """
    require_last_dim(fine_pred, N_FINE_TARGETS, "fine_pred")
    require_last_dim(fine_features, N_FINE_TARGETS, "fine_features")
    require_last_dim(coarse_pred, N_COARSE_TARGETS, "coarse_pred")
    require_last_dim(coarse_targets, N_COARSE_TARGETS, "coarse_targets")
    if fine_pred.shape[:2] != (fine_features.shape[0], fine_elements.shape[0]):
        raise ValueError(
            f"fine_pred must have shape (T, N_fine_elem, {N_FINE_TARGETS}); got {tuple(fine_pred.shape)} "
            f"for T={fine_features.shape[0]}, N_fine_elem={fine_elements.shape[0]}"
        )
    w = _resolve_cross_scale_weights(weights)
    physics_weight_scale = max(0.0, float(physics_weight_scale))
    fine_elem_regions = max(1, int(os.environ.get("CFWRINKLE_FINE_ELEM_REGIONS", "1")))

    fine_pred_active = fine_pred
    fine_elements_active = fine_elements
    fine_elem_edge_index_active = fine_elem_edge_index
    region_start = 0
    if fine_elem_regions > 1 and step_idx is not None:
        n_elem = fine_elements.shape[0]
        region = step_idx % fine_elem_regions
        region_start = (region * n_elem) // fine_elem_regions
        region_end = ((region + 1) * n_elem) // fine_elem_regions
        fine_pred_active = fine_pred[:, region_start:region_end, :]
        fine_elements_active = fine_elements[region_start:region_end]
        if fine_elem_edge_index is not None and fine_elem_edge_index.numel() > 0:
            src = fine_elem_edge_index[0]
            dst = fine_elem_edge_index[1]
            in_region = (
                (src >= region_start)
                & (src < region_end)
                & (dst >= region_start)
                & (dst < region_end)
            )
            if in_region.any():
                fine_elem_edge_index_active = torch.stack(
                    (src[in_region] - region_start, dst[in_region] - region_start),
                    dim=0,
                )
            else:
                fine_elem_edge_index_active = None
        else:
            fine_elem_edge_index_active = None

    fine_loss_chunk_elems = max(1, int(os.environ.get("CFWRINKLE_FINE_LOSS_CHUNK_ELEMS", "8192")))
    l_fs1, l_fs2, l_dz, l_thick, l_buckling = _chunked_fine_supervision_losses(
        fine_pred_active,
        fine_features,
        fine_elements_active,
        chunk_elems=fine_loss_chunk_elems,
    )
    l_fine_phys = monotonicity_loss(fine_pred_active[..., FINE_TARGET_INDEX["stress_1"]])
    zero = fine_pred.sum() * 0.0

    aux_interval = max(1, int(os.environ.get("CFWRINKLE_AUX_LOSS_INTERVAL", "1")))
    use_aux = step_idx is None or aux_interval <= 1 or (step_idx % aux_interval == 0)

    disable_dz_mono = _env_flag("CFWRINKLE_DISABLE_FINE_DZ_MONO", False)
    disable_buckling = _env_flag("CFWRINKLE_DISABLE_FINE_BUCKLING", False)
    disable_coupling = _env_flag("CFWRINKLE_DISABLE_FINE_COUPLING", False)
    disable_coherence = _env_flag("CFWRINKLE_DISABLE_FINE_COHERENCE", False)

    if use_aux and not disable_dz_mono:
        l_dz_mono = monotonicity_loss(fine_pred_active[..., FINE_TARGET_INDEX["dz"]].abs())
    else:
        l_dz_mono = zero
    if not use_aux or disable_buckling:
        l_buckling = zero
    if use_aux and not disable_coupling:
        l_coupling = thickness_dz_coupling_loss(fine_pred_active)
    else:
        l_coupling = zero
    if use_aux and not disable_coherence and fine_elem_edge_index_active is not None:
        l_coherence = spatial_coherence_loss(fine_pred_active, fine_elem_edge_index_active)
    else:
        l_coherence = zero

    l_coarse, coarse_log = wrinkle_loss(coarse_pred, coarse_targets)

    def _safe_norm(loss: torch.Tensor) -> torch.Tensor:
        return loss / loss.detach().clamp(min=1e-2)

    raw_fine = (
        w["fine_stress"] * (l_fs1 + l_fs2) / 2
        + w["fine_dz"] * l_dz
        + w["fine_thick"] * l_thick
        + physics_weight_scale
        * (
            w["physics"] * l_fine_phys
            + w["dz_mono"] * l_dz_mono
            + w["buckling"] * l_buckling
            + w["coupling"] * l_coupling
            + w["coherence"] * l_coherence
        )
    )

    grad_total = (
        w["fine_stress"] * (_safe_norm(l_fs1) + _safe_norm(l_fs2)) / 2
        + w["fine_dz"] * _safe_norm(l_dz)
        + w["fine_thick"] * _safe_norm(l_thick)
        + physics_weight_scale
        * (
            w["physics"] * _safe_norm(l_fine_phys)
            + w["dz_mono"] * _safe_norm(l_dz_mono)
            + w["buckling"] * _safe_norm(l_buckling)
            + w["coupling"] * _safe_norm(l_coupling)
            + w["coherence"] * _safe_norm(l_coherence)
        )
        + w["coarse"] * l_coarse  # l_coarse already self-normalised via wrinkle_loss
    )

    total_raw = float((raw_fine.detach().cpu())) + w["coarse"] * float(coarse_log["loss/total"])
    log = {
        "loss/total": total_raw,
        "loss/fine_stress_1": float(l_fs1.detach().cpu()),
        "loss/fine_stress_2": float(l_fs2.detach().cpu()),
        "loss/fine_dz": float(l_dz.detach().cpu()),
        "loss/fine_thick": float(l_thick.detach().cpu()),
        "loss/fine_physics": float(l_fine_phys.detach().cpu()),
        "loss/fine_dz_mono": float(l_dz_mono.detach().cpu()),
        "loss/fine_buckling": float(l_buckling.detach().cpu()),
        "loss/fine_coupling": float(l_coupling.detach().cpu()),
        "loss/fine_coherence": float(l_coherence.detach().cpu()),
        "loss/physics_weight_scale": physics_weight_scale,
        "loss/fine_elem_regions": float(fine_elem_regions),
        **{f"coarse/{k.split('/')[1]}": v for k, v in coarse_log.items()},
    }
    return grad_total, log


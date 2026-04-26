from __future__ import annotations

from typing import Any

import numpy as np


MATERIAL_FEATURES = [
    "n_plies",
    "ply_thickness_mm",
    "forming_speed_mm_s",
    "blank_initial_temp_C",
    "is_woven",
    "membrane_stiffness_ratio",
    "bending_stiffness_ratio",
    "coupling_indicator",
]
MATERIAL_FEATURE_INDEX = {name: i for i, name in enumerate(MATERIAL_FEATURES)}
if len(MATERIAL_FEATURE_INDEX) != len(MATERIAL_FEATURES):
    raise RuntimeError("MATERIAL_FEATURES must contain unique labels")


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _coalesce_value(sim_attrs: dict[str, Any], row: dict[str, Any], key: str) -> Any:
    if key in sim_attrs and sim_attrs[key] is not None:
        return sim_attrs[key]
    return row.get(key)


def _last_or_self(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        return value.reshape(-1)[-1].item()
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        return value[-1]
    return value


def compute_material_card(sim_attrs: dict[str, Any], registry_row: dict[str, Any] | None) -> np.ndarray:
    row = registry_row or {}
    if "n_plies" in row and row["n_plies"] is not None:
        row_n_plies = _safe_float(row["n_plies"], np.nan)
        if np.isfinite(row_n_plies) and int(round(row_n_plies)) != int(sim_attrs["n_plies"]):
            raise ValueError(
                f"n_plies mismatch between simulation attrs ({sim_attrs['n_plies']}) and registry ({row['n_plies']})"
            )
    if "batch" in row and row["batch"] is not None:
        if str(row["batch"]) != str(sim_attrs["batch"]):
            raise ValueError(
                f"batch mismatch between simulation attrs ({sim_attrs['batch']}) and registry ({row['batch']})"
            )

    n_plies = float(sim_attrs["n_plies"])
    if n_plies <= 0:
        raise ValueError(f"n_plies must be > 0, got {n_plies}")

    ply_thickness = _safe_float(_coalesce_value(sim_attrs, row, "ply_thickness_afi_mm"), 0.3)
    if ply_thickness <= 0:
        raise ValueError(f"ply_thickness_afi_mm must be > 0, got {ply_thickness}")

    stroke_mm = _safe_float(_coalesce_value(sim_attrs, row, "punch_stroke_loadset2_mm"), 75.0)
    solve_dt_val = _last_or_self(_coalesce_value(sim_attrs, row, "solve_dt"))
    solve_dt = _safe_float(solve_dt_val, 0.0333)
    if solve_dt <= 0:
        raise ValueError(f"solve_dt must be > 0, got {solve_dt}")
    forming_speed = stroke_mm / solve_dt

    blank_temp = _safe_float(_coalesce_value(sim_attrs, row, "blank_initial_temp_C"), 300.0)
    is_woven = 1.0 if str(sim_attrs["batch"]) == "B" else 0.0

    if is_woven:
        membrane_ratio = 1.0
        bending_ratio = 1.0
        coupling = 0.15
    else:
        membrane_ratio = 6.0
        bending_ratio = 9.0
        coupling = 0.05

    card = np.array(
        [n_plies, ply_thickness, forming_speed, blank_temp, is_woven, membrane_ratio, bending_ratio, coupling],
        dtype=np.float32,
    )
    if card.shape != (len(MATERIAL_FEATURES),):
        raise RuntimeError(
            f"Material card length mismatch: got {card.shape}, expected ({len(MATERIAL_FEATURES)},)"
        )
    if not np.isfinite(card).all():
        raise ValueError("Material card contains non-finite values")
    return card


def normalize_cards(cards: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = cards.mean(axis=0).astype(np.float32)
    std = cards.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-8, 1.0, std).astype(np.float32)
    norm = ((cards - mean) / std).astype(np.float32)
    return norm, mean, std


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


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def compute_material_card(sim_attrs: dict[str, Any], registry_row: dict[str, Any] | None) -> np.ndarray:
    row = registry_row or {}
    n_plies = float(sim_attrs["n_plies"])
    ply_thickness = _safe_float(row.get("ply_thickness_afi_mm"), 0.3)
    forming_speed = _safe_float(row.get("punch_stroke_loadset2_mm"), 75.0) / max(
        _safe_float((row.get("solve_dt") or [2.0, 2.0, 0.0333])[-1], 0.0333), 1e-6
    )
    blank_temp = _safe_float(row.get("blank_initial_temp_C"), 300.0)
    is_woven = 1.0 if str(sim_attrs["batch"]) == "B" else 0.0

    if is_woven:
        membrane_ratio = 1.0
        bending_ratio = 1.0
        coupling = 0.15
    else:
        membrane_ratio = 6.0
        bending_ratio = 9.0
        coupling = 0.05

    return np.array(
        [n_plies, ply_thickness, forming_speed, blank_temp, is_woven, membrane_ratio, bending_ratio, coupling],
        dtype=np.float32,
    )


def normalize_cards(cards: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = cards.mean(axis=0).astype(np.float32)
    std = cards.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-8, 1.0, std).astype(np.float32)
    norm = ((cards - mean) / std).astype(np.float32)
    return norm, mean, std


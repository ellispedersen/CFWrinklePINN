from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml

from wp2_build.schema import FIELD_ORDER as WP2_FIELD_ORDER

FIELD_ORDER = [
    "displacement",
    "temperature",
    "thickness",
    "eq_shear_rate",
    "fiber_dir_1",
    "fiber_dir_2",
    "shear_angle",
    "fiber_strain_1",
    "fiber_strain_2",
    "fiber_stress_1",
    "fiber_stress_2",
    "gl_strain",
    "stress",
]
_EXPECTED_WP3_FIELD_ORDER = tuple(name for name in WP2_FIELD_ORDER if name != "crystallinity")
if tuple(FIELD_ORDER) != _EXPECTED_WP3_FIELD_ORDER:
    raise RuntimeError(
        "WP3 FIELD_ORDER drifted from wp2_build.schema.FIELD_ORDER (excluding crystallinity). "
        f"expected={_EXPECTED_WP3_FIELD_ORDER}, got={tuple(FIELD_ORDER)}"
    )


def load_pipeline_config(root: Path) -> dict[str, Any]:
    cfg = root / "config" / "pipeline_config.yaml"
    return yaml.safe_load(cfg.read_text(encoding="utf-8"))


def get_wp2_h5_path(root: Path) -> Path:
    env_path = os.environ.get("WP2_H5")
    if env_path:
        p = Path(env_path)
        return p if p.is_absolute() else root / p
    cfg = load_pipeline_config(root)
    rel = str(cfg["dataset"]["hdf5_path"])
    p = Path(rel)
    return p if p.is_absolute() else root / p


def get_wp3_h5_path(root: Path) -> Path:
    env_path = os.environ.get("WP3_H5")
    if env_path:
        p = Path(env_path)
        return p if p.is_absolute() else root / p
    return root / "data" / "cfwrinkle_wp3_features.h5"


def strip_padding(arr: np.ndarray) -> np.ndarray:
    if arr.ndim < 2:
        return arr
    if arr.shape[-1] > 1 and np.allclose(arr[..., 0], 0.0):
        return arr[..., 1:]
    return arr


def ensure_2d(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 3 and arr.shape[-1] == 1:
        return arr[..., 0]
    return arr


def scalarize_vector_temperature(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 3 and arr.shape[-1] == 3:
        return arr.mean(axis=-1)
    return arr


def _attr_to_python(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def read_sim_attrs(wp2: h5py.File, sim_id: str) -> dict[str, Any]:
    g = wp2[f"simulations/{sim_id}"]
    out = {
        "batch": str(g.attrs["batch"]),
        "material": str(g.attrs["material"]),
        "n_plies": int(g.attrs["n_plies"]),
        "compound_severity": float(g.attrs["compound_severity"]),
        "is_wrinkled": bool(g.attrs["is_wrinkled"]),
        "ply_orientations_deg": [float(x) for x in g.attrs["ply_orientations_deg"]],
    }
    for key in ("ply_thickness_afi_mm", "punch_stroke_loadset2_mm", "blank_initial_temp_C", "solve_dt"):
        if key in g.attrs:
            out[key] = _attr_to_python(g.attrs[key])
    return out


def load_input_registry(root: Path) -> dict[str, dict[str, Any]]:
    p = root / "reports" / "input_param_registry.json"
    rows = json.loads(p.read_text(encoding="utf-8"))
    return {str(r["sim_id"]): r for r in rows if r.get("sim_id")}


def sim_to_registry_key(sim_id: str) -> str:
    return sim_id.replace("_pair1", "").replace("_pair2", "")


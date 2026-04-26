"""Sim-ID to AniForm material file mapping.

Maps simulation name patterns to .afl filenames in config/materials/.
Used by parse_material_file() when you need the full mechanical property
set; not required for the training pipeline (which uses compute_material_card).

Usage:
    from wp3_features.material_mapping import get_material_file, auto_detect_material

    path = get_material_file("geom_0_3_pair1")  # -> config/materials/UD ...afl
    kind = auto_detect_material("mold_set_005")  # -> "woven"
    card = get_physics_material_card("A", num_plies=2, ply_thickness_mm=0.15)  # pass actual sim metadata
"""
from __future__ import annotations

import fnmatch
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from wp3_features.material import MATERIAL_FEATURES

# Default materials directory (relative to repo root)
_DEFAULT_MATERIALS_DIR = "config/materials"

# Pattern -> filename mappings (evaluated in order; first match wins)
MATERIAL_MAPPING: Dict[str, str] = {
    # Batch B — woven Twintex (mold_set_* sim IDs)
    "*mold*":       "Twintex GF-PP-unconsolidated-2x2twill-1485gsm fromLiterature RT unitMPA 2025-04-14.afl",
    "*twintex*":    "Twintex GF-PP-unconsolidated-2x2twill-1485gsm fromLiterature RT unitMPA 2025-04-14.afl",
    "*woven*":      "Twintex GF-PP-unconsolidated-2x2twill-1485gsm fromLiterature RT unitMPA 2025-04-14.afl",
    "*2x2*":        "Twintex GF-PP-unconsolidated-2x2twill-1485gsm fromLiterature RT unitMPA 2025-04-14.afl",
    # Batch A — UD thermoplastic (geom_0_* sim IDs)
    "*geom*":       "UD reinforced thermoplastic unitMPa 2025-04-14.afl",
    "*ud*":         "UD reinforced thermoplastic unitMPa 2025-04-14.afl",
    "*unidirectional*": "UD reinforced thermoplastic unitMPa 2025-04-14.afl",
    "*tape*":       "UD reinforced thermoplastic unitMPa 2025-04-14.afl",
    # Fallback
    "*":            "UD reinforced thermoplastic unitMPa 2025-04-14.afl",
}


def get_material_file(
    simulation_name: str,
    materials_dir: str = _DEFAULT_MATERIALS_DIR,
) -> Optional[str]:
    """Return the absolute path to the .afl file for the given simulation name.

    Returns None if the directory or matched file does not exist.
    """
    mat_dir = Path(materials_dir)
    if not mat_dir.exists():
        print(f"Warning: materials directory not found: {mat_dir}")
        return None

    sim_lower = simulation_name.lower()
    for pattern, filename in MATERIAL_MAPPING.items():
        if fnmatch.fnmatch(sim_lower, pattern.lower()):
            path = mat_dir / filename
            if path.exists():
                return str(path)
            print(f"Warning: matched material file not found: {path}")
    print(f"Warning: no material mapping for simulation: {simulation_name}")
    return None


def auto_detect_material(simulation_name: str) -> str:
    """Return 'woven', 'ud', or 'unknown' from the simulation name alone."""
    s = simulation_name.lower()
    if any(k in s for k in ("mold", "twintex", "2x2", "twill", "woven")):
        return "woven"
    if any(k in s for k in ("geom", "ud", "unidirectional", "tape")):
        return "ud"
    return "unknown"


def load_material_mapping(
    config_file: str = "config/material_mapping.json",
) -> Dict[str, str]:
    """Load mapping overrides from a JSON file; falls back to MATERIAL_MAPPING."""
    path = Path(config_file)
    if not path.exists():
        return MATERIAL_MAPPING
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {config_file}: {e}. Using defaults.")
        return MATERIAL_MAPPING


def save_material_mapping(
    mapping: Dict[str, str],
    config_file: str = "config/material_mapping.json",
) -> None:
    path = Path(config_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"Saved material mapping to: {path}")


# ---------------------------------------------------------------------------
# Physics-based material cards — lazily cached, zero I/O after first call
# ---------------------------------------------------------------------------
# The .afl file defines per-ply properties. The actual number of plies and
# effective ply thickness in a given simulation come from the simulation
# metadata, NOT from the .afl file — they vary per sim (especially Batch B).
# Always pass the real n_plies from the simulation when calling
# get_physics_material_card(); do not rely on the batch-level defaults.

def _load_material_props(
    materials_dir: str = _DEFAULT_MATERIALS_DIR,
) -> Dict[str, "MaterialProperties"]:
    """Parse both .afl files once; return {batch: MaterialProperties}."""
    from wp3_features.material_parser import (
        parse_material_file, UD_THERMOPLASTIC, TWINTEX_GF_PP,
        AFL_UD, AFL_TWINTEX,
    )
    mat_dir = Path(materials_dir)
    props: Dict[str, "MaterialProperties"] = {}
    for batch, filename, fallback in (
        ("A", AFL_UD,      UD_THERMOPLASTIC),
        ("B", AFL_TWINTEX, TWINTEX_GF_PP),
    ):
        afl = mat_dir / filename
        props[batch] = parse_material_file(afl) if afl.exists() else fallback
    return props


# Parsed once at import — holds MaterialProperties (no n_plies baked in yet).
_MATERIAL_PROPS: Dict[str, "MaterialProperties"] = _load_material_props()

# Cache keyed by (batch, num_plies, ply_thickness_mm) — populated lazily.
_PHYSICS_CARD_CACHE: Dict[tuple, "np.ndarray"] = {}
_MATERIAL_CARD_DIM = len(MATERIAL_FEATURES)


def get_physics_material_card(
    batch: str,
    num_plies: int,
    ply_thickness_mm: float,
) -> "np.ndarray":
    """Return the physics card for batch 'A'/'B' using actual simulation metadata.

    Cached after the first call for each (batch, num_plies, ply_thickness_mm)
    combination — subsequent calls are a pure dict lookup with no computation
    or I/O.

    Args:
        batch: "A" (UD thermoplastic) or "B" (Twintex woven).
        num_plies: actual ply count from the simulation (NOT the .afl default).
        ply_thickness_mm: actual per-ply thickness from the simulation (NOT the
            .afl default — the two can differ, e.g. due to deconsolidation or
            different layup configurations).

    Returns:
        float32 array of shape (8,).
    """
    batch_norm = str(batch).upper()
    if batch_norm not in _MATERIAL_PROPS:
        raise ValueError(f"Unsupported batch '{batch}'. Expected one of {sorted(_MATERIAL_PROPS.keys())}")

    n_plies_i = int(num_plies)
    if n_plies_i <= 0:
        raise ValueError(f"num_plies must be > 0, got {num_plies}")

    t_mm = float(ply_thickness_mm)
    if not np.isfinite(t_mm) or t_mm <= 0:
        raise ValueError(f"ply_thickness_mm must be a finite value > 0, got {ply_thickness_mm}")

    key = (batch_norm, n_plies_i, t_mm)
    if key not in _PHYSICS_CARD_CACHE:
        card = _MATERIAL_PROPS[batch_norm].to_physics_features(
            num_plies=n_plies_i,
            ply_thickness_mm=t_mm,
        )
        if card.shape != (_MATERIAL_CARD_DIM,):
            raise RuntimeError(
                f"Physics material card length mismatch: got {card.shape}, expected ({_MATERIAL_CARD_DIM},)"
            )
        if not np.isfinite(card).all():
            raise ValueError("Physics material card contains non-finite values")
        _PHYSICS_CARD_CACHE[key] = card
    return _PHYSICS_CARD_CACHE[key]

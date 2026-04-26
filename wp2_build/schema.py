from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

H5_PATH = Path("data/cfwrinkle_dataset.h5")
SCHEMA_VERSION = "1.0"

# Field name -> (afr_id, sub_id, array_shape_suffix, groups_key)
# groups_key: "ply" for ScalarT/VectorT fields, "bending" for STensorPSST fields
FIELD_DEFS: dict[str, tuple[int, int, tuple[int, ...], str]] = {
    "displacement": (40, 1, (3,), "ply"),
    "temperature": (44, 1, (3,), "ply"),
    "thickness": (105, 1, (), "ply"),
    "eq_shear_rate": (106, 1, (), "ply"),
    "fiber_dir_1": (203, 1, (3,), "ply"),
    "fiber_dir_2": (203, 2, (3,), "ply"),
    "shear_angle": (204, 1, (), "ply"),
    "fiber_strain_1": (205, 1, (), "ply"),
    "fiber_strain_2": (205, 2, (), "ply"),
    "fiber_stress_1": (206, 1, (), "ply"),
    "fiber_stress_2": (206, 2, (), "ply"),
    "gl_strain": (102, 1, (3,), "bending"),
    "stress": (200, 1, (3,), "bending"),
    # Batch A only
    "crystallinity": (214, 1, (), "ply"),
}
FIELD_ORDER: tuple[str, ...] = tuple(FIELD_DEFS.keys())

CHUNK_TIME_DIM = 1
CHUNK_NODE_DIM = 4096
COMPRESSION = "lzf"

CV_N_FOLDS = 5
CV_RANDOM_SEED = 42


@dataclass(slots=True)
class SimRecord:
    sim_id: str
    batch: str
    fine_dir: Path
    coarse_dir: Path | None
    ply_groups: list[int]
    bending_groups: list[int]
    compound_severity: float
    is_wrinkled: bool
    onset_increment: int
    onset_stroke_frac: float
    uz_range_mm: list[float]
    max_rfz_N: float
    n_plies: int
    material: str
    ply_orientations_deg: list[float]


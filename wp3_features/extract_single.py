from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np


def extract_single_simulation(results_dir: str | Path, sim_id: str | None = None) -> dict[str, Any]:
    """
    Return a single simulation in model-ready format.

    Current implementation reads from the already-built WP3 HDF5 by `sim_id`.
    If the simulation cannot be found, an explicit error is raised because
    direct raw-results extraction has not yet been factored out from
    `wp3_features/build_features.py`.
    """
    sim_name = sim_id or Path(results_dir).stem.replace(".Results", "")
    wp3_h5 = Path("data/cfwrinkle_wp3_features.h5")
    with h5py.File(wp3_h5, "r") as f:
        path = f"simulations/{sim_name}"
        if path not in f:
            raise NotImplementedError(
                f"Simulation '{sim_name}' not found in {wp3_h5}. "
                "Raw-results extraction is not yet factored into extract_single.py."
            )
        grp = f[path]
        node_features = np.concatenate(
            [grp["coarse/resampled/fields"][:], grp["coarse/resampled/rates"][:]],
            axis=-1,
        ).astype(np.float32)
        data = {
            "sim_id": sim_name,
            "node_features": node_features,
            "edge_index": grp["graph/edge_index"][:].astype(np.int64),
            "edge_attr": grp["graph/edge_attr"][:].astype(np.float32),
            "material_card": grp["material_card"][:].astype(np.float32),
            "elements": grp["coarse/mesh_elements"][:].astype(np.int64),
            "targets": np.stack(
                [
                    grp["targets/wrinkle_severity"][:],
                    grp["targets/comp_frac_elem"][:],
                    grp["targets/oop_max_elem"][:],
                    grp["targets/thickness_variance_elem"][:],
                ],
                axis=-1,
            ).astype(np.float32),
            "h5_path": str(wp3_h5),
        }
    return data


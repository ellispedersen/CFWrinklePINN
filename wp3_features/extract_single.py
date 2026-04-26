from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .targets import TARGET_NAMES


def extract_single_simulation(
    results_dir: str | Path,
    sim_id: str | None = None,
    wp3_h5_path: str | Path | None = None,
    normalize: bool = True,
) -> dict[str, Any]:
    """
    Return a single simulation in model-ready format.

    Tries the pre-built WP3 HDF5 first (fast path).  If the simulation is not
    found in the HDF5, falls back to raw AniForm extraction via
    extract_single_from_raw().

    normalize=True (default) applies per-feature z-score normalization using
    statistics stored in the WP3 HDF5 (metadata/feature_stats/). Must match
    WrinkleDataset(normalize=True) used during training — disabling it produces
    incorrect model predictions.

    Returns the same dict schema in both cases, with these additional keys when
    the raw path is taken (useful for cross-scale inference and visualization):
        coarse_to_fine   (2, K) int64
        fine_elements    (M_f, 3) int64
        coarse_nodes     (N_c, 3) float32
        fine_nodes       (N_f, 3) float32
        stroke_fractions (T,) float32
    """
    sim_name = sim_id or Path(results_dir).stem.replace(".Results", "")
    wp3_h5 = Path(wp3_h5_path or os.environ.get("WP3_H5", "data/cfwrinkle_wp3_features.h5"))

    if wp3_h5.exists():
        with h5py.File(wp3_h5, "r") as f:
            path = f"simulations/{sim_name}"
            if path in f:
                return _extract_from_h5(f, sim_name, wp3_h5, normalize=normalize)

    # Fall through to raw AniForm extraction
    if not Path(results_dir).exists():
        raise FileNotFoundError(
            f"Simulation '{sim_name}' not found in WP3 HDF5 at {wp3_h5} "
            f"and results_dir does not exist: {results_dir}"
        )
    warnings.warn(
        f"Simulation '{sim_name}' not in WP3 HDF5 — extracting from raw AniForm results.",
        stacklevel=2,
    )
    return extract_single_from_raw(results_dir)


def _extract_from_h5(f: h5py.File, sim_name: str, wp3_h5: Path, normalize: bool = True) -> dict[str, Any]:
    """Read model-ready arrays from a pre-built WP3 HDF5 group."""
    grp = f[f"simulations/{sim_name}"]
    node_features = np.concatenate(
        [grp["coarse/resampled/fields"][:], grp["coarse/resampled/rates"][:]],
        axis=-1,
    ).astype(np.float32)
    if normalize:
        mu_path = "metadata/feature_stats/feat_mean"
        std_path = "metadata/feature_stats/feat_std"
        if mu_path in f and std_path in f:
            mu = f[mu_path][:].astype(np.float32)
            std = f[std_path][:].astype(np.float32)
            if mu.shape == (node_features.shape[-1],):
                node_features = (node_features - mu) / (std + 1e-8)
    data = {
        "sim_id": sim_name,
        "node_features": node_features,
        "edge_index": grp["graph/edge_index"][:].astype(np.int64),
        "edge_attr": grp["graph/edge_attr"][:].astype(np.float32),
        "material_card": grp["material_card"][:].astype(np.float32),
        "elements": grp["coarse/mesh_elements"][:].astype(np.int64),
        "h5_path": str(wp3_h5),
    }
    # Targets are derived from fine-mesh runs and stored at WP3 build time.
    # Only present for training-set sims — read conditionally so inference on
    # sims without a fine-mesh run does not fail.
    if all(f"targets/{name}" in grp for name in TARGET_NAMES):
        data["targets"] = np.stack(
            [grp[f"targets/{name}"][:] for name in TARGET_NAMES],
            axis=-1,
        ).astype(np.float32)
        data["target_names"] = list(TARGET_NAMES)
    # Include cross-scale keys if present
    if "graph/coarse_to_fine_index" in grp:
        data["coarse_to_fine"] = grp["graph/coarse_to_fine_index"][:].astype(np.int64)
    if "fine/mesh_elements" in grp:
        data["fine_elements"] = grp["fine/mesh_elements"][:].astype(np.int64)
    if "fine/mesh_nodes" in grp:
        data["fine_nodes"] = grp["fine/mesh_nodes"][:].astype(np.float32)
    for path, key in (
        ("mesh/coarse/nodes", "coarse_nodes"),
        ("coarse/mesh_nodes", "coarse_nodes"),
    ):
        if path in grp and "coarse_nodes" not in data:
            data["coarse_nodes"] = grp[path][:].astype(np.float32)
            break
    return data


def extract_single_from_raw(
    results_dir: str | Path,
    *,
    batch: str | None = None,
    n_resample_points: int = 256,
) -> dict[str, Any]:
    """
    Extract model-ready batch dict from a raw AniForm *.Results/ directory.

    This implements the raw extraction path previously marked NotImplementedError.
    Suitable for inference on new data not yet in the WP2/WP3 HDF5.

    Returns the same core dict schema as extract_single_simulation() (HDF5 path),
    plus cross-scale keys:
        coarse_to_fine   (2, K) int64
        fine_elements    (M_f, 3) int64
        coarse_nodes     (N_c, 3) float32
        fine_nodes       (N_f, 3) float32
        stroke_fractions (T,) float32

    Note: the material_card here is un-normalized (computed from process parameter
    defaults). For production use, prefer the pre-normalized card from the WP3 HDF5.
    """
    from validation.wrinkle_detector import find_coarse_and_fine
    from wp2_build.build_dataset import _collect_level_data, SimRecord  # type: ignore[attr-defined]
    from wp2_build.schema import FIELD_DEFS
    from wp3_features.graph import build_edge_attr, build_edge_index
    from wp3_features.correspondence import build_coarse_to_fine_map
    from wp3_features.material import compute_material_card
    from wp3_features.physics import FEATURE_NAMES, compute_feature_tensor
    from wp3_features.temporal import compute_rates, resample_to_uniform

    results_dir = Path(results_dir)
    sim_id = results_dir.stem.replace(".Results", "").replace(" ", "_")

    # Auto-detect batch
    if batch is None:
        sid = sim_id.lower()
        if "mold_set" in sid or "mold set" in sid:
            batch = "B"
        elif "geom_0" in sid or "geom 0" in sid:
            batch = "A"
        else:
            raise ValueError(
                f"Cannot auto-detect batch from sim_id '{sim_id}'. "
                "Pass batch='A' or batch='B' explicitly."
            )

    ply_groups = [6, 10] if batch == "A" else [4, 8, 12]
    bending_groups = [6, 7, 10, 11] if batch == "A" else [4, 5, 8, 9, 12, 13]
    n_plies = len(ply_groups)

    coarse_dir, fine_dir = find_coarse_and_fine(results_dir)
    if fine_dir is None or coarse_dir is None:
        raise FileNotFoundError(
            f"Could not find coarse+fine sub-runs under {results_dir}"
        )

    rec = SimRecord(
        sim_id=sim_id, batch=batch, fine_dir=results_dir, coarse_dir=results_dir,
        ply_groups=ply_groups, bending_groups=bending_groups,
        compound_severity=0.0, is_wrinkled=False, onset_increment=-1,
        onset_stroke_frac=float("nan"), uz_range_mm=[float("nan"), float("nan")],
        max_rfz_N=float("nan"), n_plies=n_plies, material=f"batch_{batch}",
        ply_orientations_deg=[],
    )

    coarse_data = _collect_level_data(rec, coarse_dir, batch)
    fine_data = _collect_level_data(rec, fine_dir, batch)

    if coarse_data is None or fine_data is None:
        raise RuntimeError("_collect_level_data returned None — check AFR files exist")

    coarse_nodes = coarse_data["mesh_nodes"].astype(np.float32)
    coarse_elements = coarse_data["mesh_elements"].astype(np.int64)
    fine_nodes = fine_data["mesh_nodes"].astype(np.float32)
    fine_elements = fine_data["mesh_elements"].astype(np.int64)

    _, coarse_feats = compute_feature_tensor(coarse_data["fields"], coarse_data["times_s"], batch)
    rates = compute_rates(coarse_data["times_s"], coarse_feats)
    u_stroke, res_fields = resample_to_uniform(coarse_data["stroke_frac"], coarse_feats, n_resample_points)
    _, res_rates = resample_to_uniform(coarse_data["stroke_frac"], rates, n_resample_points)

    node_features = np.concatenate(
        [res_fields.astype(np.float32), res_rates.astype(np.float32)], axis=-1
    )

    edge_index = build_edge_index(coarse_elements)
    edge_attr = build_edge_attr(edge_index, coarse_nodes)

    mapping = build_coarse_to_fine_map(coarse_nodes, coarse_elements, fine_nodes, fine_elements, radius_factor=1.5)
    coarse_to_fine = np.stack([mapping["coarse_index"], mapping["fine_index"]], axis=0).astype(np.int64)

    material_card = compute_material_card(
        {"batch": batch, "n_plies": n_plies,
         "ply_thickness_afi_mm": 0.30 if batch == "A" else 0.15},
        registry_row=None,
    )

    return {
        "sim_id": sim_id,
        "node_features": node_features,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "material_card": material_card,
        "elements": coarse_elements,
        "coarse_to_fine": coarse_to_fine,
        "fine_elements": fine_elements,
        "coarse_nodes": coarse_nodes,
        "fine_nodes": fine_nodes,
        "stroke_fractions": u_stroke.astype(np.float32),
        "target_names": list(TARGET_NAMES),
    }

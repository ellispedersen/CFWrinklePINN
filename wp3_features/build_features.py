from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .correspondence import build_coarse_to_fine_map
from .graph import build_edge_attr, build_edge_index, build_element_adjacency
from .h5_utils import (
    FIELD_ORDER,
    get_wp2_h5_path,
    get_wp3_h5_path,
    load_input_registry,
    read_sim_attrs,
    sim_to_registry_key,
)
from .material import MATERIAL_FEATURES, compute_material_card, normalize_cards
from .physics import FEATURE_NAMES, compute_feature_tensor
from .targets import TARGET_NAMES, compute_wrinkle_targets
from .temporal import compute_rates, resample_to_uniform

ROOT = Path(__file__).resolve().parents[1]


def _as_np(ds: h5py.Dataset) -> np.ndarray:
    return ds[:].astype(np.float32)


def _require_shape(arr: np.ndarray, shape_prefix: tuple[int, ...], name: str) -> None:
    if arr.ndim < len(shape_prefix):
        raise ValueError(f"{name} must have at least {len(shape_prefix)} dims, got {arr.shape}")
    for i, expected in enumerate(shape_prefix):
        if expected >= 0 and arr.shape[i] != expected:
            raise ValueError(f"{name} has invalid shape {arr.shape}, expected dim {i} == {expected}")


def _collect_fields(sim: h5py.Group, level: str) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    grp = sim[level]
    stroke = grp["stroke_frac"][:].astype(np.float32)
    times = grp["times_s"][:].astype(np.float32)
    fields: dict[str, np.ndarray] = {}
    for k in FIELD_ORDER:
        arr = _as_np(grp[k])
        fields[k] = arr
    return stroke, times, fields


def _resample_all(stroke: np.ndarray, times: np.ndarray, features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rates = compute_rates(times, features)
    u, res_fields = resample_to_uniform(stroke, features, n_points=256)
    _, res_rates = resample_to_uniform(stroke, rates, n_points=256)
    return u, res_fields, res_rates


def _build_sim(wp2: h5py.File, out: h5py.File, sim_id: str, reg: dict[str, dict[str, Any]], include_fine: bool = False) -> np.ndarray:
    sim = wp2[f"simulations/{sim_id}"]
    attrs = read_sim_attrs(wp2, sim_id)

    c_nodes = sim["mesh/coarse/nodes"][:].astype(np.float32)
    c_elem = sim["mesh/coarse/elements"][:].astype(np.int64)
    f_nodes = sim["mesh/fine/nodes"][:].astype(np.float32)
    f_elem = sim["mesh/fine/elements"][:].astype(np.int64)
    _require_shape(c_nodes, (-1, 3), "mesh/coarse/nodes")
    _require_shape(f_nodes, (-1, 3), "mesh/fine/nodes")
    _require_shape(c_elem, (-1, 3), "mesh/coarse/elements")
    _require_shape(f_elem, (-1, 3), "mesh/fine/elements")

    # Corruption guards: skip obviously invalid meshes before feature extraction.
    if c_elem.shape[0] < 1000:
        raise RuntimeError(f"coarse mesh too small: {c_elem.shape[0]} elements")
    if f_elem.shape[0] < 10000:
        raise RuntimeError(f"fine mesh too small: {f_elem.shape[0]} elements")

    edge_index = build_edge_index(c_elem)
    edge_attr = build_edge_attr(edge_index, c_nodes)
    if edge_attr.shape[0] != edge_index.shape[1] or edge_attr.shape[1] != 4:
        raise RuntimeError(
            f"edge_attr/edge_index shape mismatch: edge_index={edge_index.shape}, edge_attr={edge_attr.shape}"
        )
    mapping = build_coarse_to_fine_map(c_nodes, c_elem, f_nodes, f_elem, radius_factor=1.5)
    if mapping["coarse_index"].dtype.kind not in {"i", "u"} or mapping["fine_index"].dtype.kind not in {"i", "u"}:
        raise RuntimeError("coarse_to_fine indices must be integer arrays")
    mapping_idx = np.stack([mapping["coarse_index"], mapping["fine_index"]], axis=0).astype(np.int64, copy=False)
    # Keep only true outliers/corruption; Batch B naturally has lower pre-fallback
    # coverage due projection mismatch, so do not use a high cutoff.
    if float(mapping["pre_fallback_coverage"]) < 0.65:
        raise RuntimeError(f"pre_fallback_coverage outlier: {float(mapping['pre_fallback_coverage']):.3f}")
    if float(mapping["mapping_coverage"]) < 0.99:
        raise RuntimeError(f"mapping_coverage too low: {float(mapping['mapping_coverage']):.3f}")

    c_stroke, c_times, c_fields = _collect_fields(sim, "coarse")
    f_stroke, _, f_fields = _collect_fields(sim, "fine")
    feat_names, coarse_feats = compute_feature_tensor(c_fields, c_times, attrs["batch"])
    if feat_names != list(FEATURE_NAMES):
        raise RuntimeError(f"Feature names mismatch for {sim_id}: got={feat_names}, expected={list(FEATURE_NAMES)}")
    u, res_fields, res_rates = _resample_all(c_stroke, c_times, coarse_feats)
    res_fields = res_fields.astype(np.float32, copy=False)
    res_rates = res_rates.astype(np.float32, copy=False)
    _require_shape(res_fields, (256, c_nodes.shape[0]), "coarse/resampled/fields")
    _require_shape(res_rates, (256, c_nodes.shape[0]), "coarse/resampled/rates")

    fine_targets = compute_wrinkle_targets(
        mapping,
        f_fields["fiber_stress_1"],
        f_fields["displacement"],
        f_fields["thickness"],
        f_elem,
    )
    if tuple(fine_targets.keys()) != TARGET_NAMES:
        raise RuntimeError(f"Target names mismatch for {sim_id}: got={tuple(fine_targets.keys())}, expected={TARGET_NAMES}")
    resampled_targets: dict[str, np.ndarray] = {}
    for target_name in TARGET_NAMES:
        _, target_arr = resample_to_uniform(f_stroke, fine_targets[target_name], n_points=256)
        target_arr = target_arr.astype(np.float32, copy=False)
        _require_shape(target_arr, (256, c_elem.shape[0]), f"targets/{target_name}")
        resampled_targets[target_name] = target_arr

    sgrp = out.require_group("simulations").create_group(sim_id)
    sgrp.attrs["batch"] = attrs["batch"]
    sgrp.attrs["material"] = attrs["material"]
    sgrp.attrs["compound_severity"] = attrs["compound_severity"]
    sgrp.attrs["is_wrinkled"] = attrs["is_wrinkled"]

    g = sgrp.create_group("graph")
    g.create_dataset("edge_index", data=edge_index, compression="lzf")
    g.create_dataset("edge_attr", data=edge_attr, compression="lzf")
    g.create_dataset("coarse_to_fine_index", data=mapping_idx, compression="lzf")
    g.create_dataset("primary_hit_count", data=mapping["primary_hit_count"], compression="lzf")
    g.attrs["mapping_coverage"] = float(mapping["mapping_coverage"])
    g.attrs["pre_fallback_coverage"] = float(mapping["pre_fallback_coverage"])
    g.attrs["mean_refinement_ratio"] = float(mapping["mean_refinement_ratio"])
    g.attrs["boundary_elem_frac"] = float(mapping["boundary_elem_frac"])

    coarse = sgrp.create_group("coarse")
    coarse.create_dataset("mesh_elements", data=c_elem, compression="lzf")
    res = coarse.create_group("resampled")
    res.create_dataset("stroke_fracs", data=u.astype(np.float32, copy=False), compression="lzf")
    res.create_dataset("fields", data=res_fields, compression="lzf")
    res.create_dataset("rates", data=res_rates, compression="lzf")
    res.attrs["feature_names"] = json.dumps(feat_names)
    res.attrs["feature_name_to_channel"] = json.dumps({name: i for i, name in enumerate(feat_names)})
    rate_names = [f"d_dt:{name}" for name in feat_names]
    res.attrs["rate_feature_names"] = json.dumps(rate_names)
    res.attrs["rate_name_to_channel"] = json.dumps({name: i for i, name in enumerate(rate_names)})

    fine = sgrp.create_group("fine")
    fine.create_dataset("mesh_elements", data=f_elem, compression="lzf")

    if include_fine:
        fine.create_dataset("mesh_nodes", data=f_nodes, compression="lzf")

        f_edge_index = build_edge_index(f_elem)
        f_edge_attr = build_edge_attr(f_edge_index, f_nodes)
        f_elem_adj = build_element_adjacency(f_elem)
        fine.create_dataset("edge_index", data=f_edge_index, compression="lzf")
        fine.create_dataset("edge_attr", data=f_edge_attr, compression="lzf")
        fine.create_dataset("element_edge_index", data=f_elem_adj, compression="lzf")

        fine_res = fine.create_group("resampled")
        _, fs1 = resample_to_uniform(f_stroke, f_fields["fiber_stress_1"], n_points=256)
        _, fs2 = resample_to_uniform(f_stroke, f_fields["fiber_stress_2"], n_points=256)
        _, dz = resample_to_uniform(f_stroke, f_fields["displacement"][:, :, 2], n_points=256)
        _, thick = resample_to_uniform(f_stroke, f_fields["thickness"], n_points=256)
        fine_res.create_dataset("fiber_stress_1", data=fs1.astype(np.float32, copy=False), compression="lzf")
        fine_res.create_dataset("fiber_stress_2", data=fs2.astype(np.float32, copy=False), compression="lzf")
        fine_res.create_dataset("displacement_z", data=dz.astype(np.float32, copy=False), compression="lzf")
        fine_res.create_dataset("thickness", data=thick.astype(np.float32, copy=False), compression="lzf")

    tgt = sgrp.create_group("targets")
    for target_name in TARGET_NAMES:
        tgt.create_dataset(target_name, data=resampled_targets[target_name], compression="lzf")
    tgt.attrs["target_names"] = json.dumps(list(TARGET_NAMES))
    tgt.attrs["target_name_to_channel"] = json.dumps({name: i for i, name in enumerate(TARGET_NAMES)})

    row = reg.get(sim_to_registry_key(sim_id))
    card = compute_material_card(attrs, row)
    sgrp.create_dataset("material_card_raw", data=card, compression="lzf")
    return card


def main() -> None:
    p = argparse.ArgumentParser(description="WP3 feature builder")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--sim", default=None, type=str)
    p.add_argument("--include-fine-features", action="store_true",
                   help="Store fine-mesh node features for cross-scale training (adds ~15-20 GB)")
    args = p.parse_args()

    wp2_path = get_wp2_h5_path(ROOT)
    wp3_path = get_wp3_h5_path(ROOT)
    reg = load_input_registry(ROOT)

    with h5py.File(wp2_path, "r") as wp2:
        sim_ids = sorted(wp2["simulations"].keys())
        if args.sim:
            sim_ids = [s for s in sim_ids if s == args.sim]
        if args.dry_run:
            print(f"Dry-run: {len(sim_ids)} simulations available")
            for sid in sim_ids[:5]:
                s = wp2[f"simulations/{sid}"]
                print(
                    sid,
                    "coarse_elem",
                    s["mesh/coarse/elements"].shape,
                    "fine_elem",
                    s["mesh/fine/elements"].shape,
                )
            return

        wp3_path.parent.mkdir(parents=True, exist_ok=True)
        if wp3_path.exists():
            wp3_path.unlink()

        cards: list[np.ndarray] = []
        built_ids: list[str] = []
        skipped: list[dict[str, str]] = []
        with h5py.File(wp3_path, "w") as out:
            for i, sid in enumerate(sim_ids, 1):
                print(f"[{i}/{len(sim_ids)}] {sid}")
                try:
                    cards.append(_build_sim(wp2, out, sid, reg, include_fine=args.include_fine_features))
                    built_ids.append(sid)
                except Exception as exc:
                    skipped.append({"sim_id": sid, "error": str(exc)})
                    print(f"[SKIP] {sid}: {exc}")
                    if f"simulations/{sid}" in out:
                        del out[f"simulations/{sid}"]

            if not cards:
                raise RuntimeError("No simulations were successfully processed for WP3 features")

            cards_np = np.stack(cards, axis=0).astype(np.float32)
            norm, mean, std = normalize_cards(cards_np)
            for sid, c in zip(built_ids, norm):
                out[f"simulations/{sid}"].create_dataset("material_card", data=c, compression="lzf")

            meta = out.create_group("metadata")
            meta.attrs["n_input_simulations"] = len(sim_ids)
            meta.attrs["n_simulations"] = len(built_ids)
            meta.attrs["n_skipped"] = len(skipped)
            meta.attrs["material_feature_names"] = json.dumps(MATERIAL_FEATURES)
            meta.attrs["feature_names"] = json.dumps(list(FEATURE_NAMES))
            meta.attrs["target_names"] = json.dumps(list(TARGET_NAMES))
            meta.create_dataset("built_sim_ids", data=np.array(built_ids, dtype=h5py.string_dtype("utf-8")))
            meta.create_dataset("skipped_sim_ids", data=np.array([s["sim_id"] for s in skipped], dtype=h5py.string_dtype("utf-8")))
            meta.create_dataset("skipped_details_json", data=json.dumps(skipped))
            meta.create_dataset("material_card_mean", data=mean, compression="lzf")
            meta.create_dataset("material_card_std", data=std, compression="lzf")
    print(f"Built WP3 features: {wp3_path} (built={len(built_ids)}, skipped={len(skipped)})")


if __name__ == "__main__":
    main()


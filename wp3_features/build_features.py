from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .correspondence import build_coarse_to_fine_map
from .graph import build_edge_attr, build_edge_index
from .h5_utils import (
    FIELD_ORDER,
    get_wp2_h5_path,
    get_wp3_h5_path,
    load_input_registry,
    read_sim_attrs,
    sim_to_registry_key,
)
from .material import MATERIAL_FEATURES, compute_material_card, normalize_cards
from .physics import compute_feature_tensor
from .targets import compute_wrinkle_targets
from .temporal import compute_rates, resample_to_uniform

ROOT = Path(__file__).resolve().parents[1]


def _as_np(ds: h5py.Dataset) -> np.ndarray:
    return ds[:].astype(np.float32)


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


def _build_sim(wp2: h5py.File, out: h5py.File, sim_id: str, reg: dict[str, dict[str, Any]]) -> np.ndarray:
    sim = wp2[f"simulations/{sim_id}"]
    attrs = read_sim_attrs(wp2, sim_id)

    c_nodes = sim["mesh/coarse/nodes"][:].astype(np.float32)
    c_elem = sim["mesh/coarse/elements"][:].astype(np.int32)
    f_nodes = sim["mesh/fine/nodes"][:].astype(np.float32)
    f_elem = sim["mesh/fine/elements"][:].astype(np.int32)

    # Corruption guards: skip obviously invalid meshes before feature extraction.
    if c_elem.shape[0] < 1000:
        raise RuntimeError(f"coarse mesh too small: {c_elem.shape[0]} elements")
    if f_elem.shape[0] < 10000:
        raise RuntimeError(f"fine mesh too small: {f_elem.shape[0]} elements")

    edge_index = build_edge_index(c_elem)
    edge_attr = build_edge_attr(edge_index, c_nodes)
    mapping = build_coarse_to_fine_map(c_nodes, c_elem, f_nodes, f_elem, radius_factor=1.5)
    # Keep only true outliers/corruption; Batch B naturally has lower pre-fallback
    # coverage due projection mismatch, so do not use a high cutoff.
    if float(mapping["pre_fallback_coverage"]) < 0.65:
        raise RuntimeError(f"pre_fallback_coverage outlier: {float(mapping['pre_fallback_coverage']):.3f}")
    if float(mapping["mapping_coverage"]) < 0.99:
        raise RuntimeError(f"mapping_coverage too low: {float(mapping['mapping_coverage']):.3f}")

    c_stroke, c_times, c_fields = _collect_fields(sim, "coarse")
    f_stroke, _, f_fields = _collect_fields(sim, "fine")
    feat_names, coarse_feats = compute_feature_tensor(c_fields, c_times, attrs["batch"])
    u, res_fields, res_rates = _resample_all(c_stroke, c_times, coarse_feats)

    fine_targets = compute_wrinkle_targets(
        mapping,
        f_fields["fiber_stress_1"],
        f_fields["displacement"],
        f_fields["thickness"],
        f_elem,
    )
    # Per-timestep targets resampled to the same 256-grid
    _, wr_sev = resample_to_uniform(f_stroke, fine_targets["wrinkle_severity"], n_points=256)
    _, wr_comp = resample_to_uniform(f_stroke, fine_targets["comp_frac_elem"], n_points=256)
    _, wr_oop = resample_to_uniform(f_stroke, fine_targets["oop_max_elem"], n_points=256)
    _, wr_tv = resample_to_uniform(f_stroke, fine_targets["thickness_variance_elem"], n_points=256)

    sgrp = out.require_group("simulations").create_group(sim_id)
    sgrp.attrs["batch"] = attrs["batch"]
    sgrp.attrs["material"] = attrs["material"]
    sgrp.attrs["compound_severity"] = attrs["compound_severity"]
    sgrp.attrs["is_wrinkled"] = attrs["is_wrinkled"]

    g = sgrp.create_group("graph")
    g.create_dataset("edge_index", data=edge_index, compression="lzf")
    g.create_dataset("edge_attr", data=edge_attr, compression="lzf")
    g.create_dataset("coarse_to_fine_index", data=np.stack([mapping["coarse_index"], mapping["fine_index"]], axis=0), compression="lzf")
    g.create_dataset("primary_hit_count", data=mapping["primary_hit_count"], compression="lzf")
    g.attrs["mapping_coverage"] = float(mapping["mapping_coverage"])
    g.attrs["pre_fallback_coverage"] = float(mapping["pre_fallback_coverage"])
    g.attrs["mean_refinement_ratio"] = float(mapping["mean_refinement_ratio"])
    g.attrs["boundary_elem_frac"] = float(mapping["boundary_elem_frac"])

    coarse = sgrp.create_group("coarse")
    coarse.create_dataset("mesh_elements", data=c_elem, compression="lzf")
    res = coarse.create_group("resampled")
    res.create_dataset("stroke_fracs", data=u, compression="lzf")
    res.create_dataset("fields", data=res_fields, compression="lzf")
    res.create_dataset("rates", data=res_rates, compression="lzf")
    res.attrs["feature_names"] = json.dumps(feat_names)

    fine = sgrp.create_group("fine")
    fine.create_dataset("mesh_elements", data=f_elem, compression="lzf")

    tgt = sgrp.create_group("targets")
    tgt.create_dataset("wrinkle_severity", data=wr_sev, compression="lzf")
    tgt.create_dataset("comp_frac_elem", data=wr_comp, compression="lzf")
    tgt.create_dataset("oop_max_elem", data=wr_oop, compression="lzf")
    tgt.create_dataset("thickness_variance_elem", data=wr_tv, compression="lzf")

    row = reg.get(sim_to_registry_key(sim_id))
    card = compute_material_card(attrs, row)
    sgrp.create_dataset("material_card_raw", data=card, compression="lzf")
    return card


def main() -> None:
    p = argparse.ArgumentParser(description="WP3 feature builder")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--sim", default=None, type=str)
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
                    cards.append(_build_sim(wp2, out, sid, reg))
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
            meta.create_dataset("built_sim_ids", data=np.array(built_ids, dtype=h5py.string_dtype("utf-8")))
            meta.create_dataset("skipped_sim_ids", data=np.array([s["sim_id"] for s in skipped], dtype=h5py.string_dtype("utf-8")))
            meta.create_dataset("skipped_details_json", data=json.dumps(skipped))
            meta.create_dataset("material_card_mean", data=mean, compression="lzf")
            meta.create_dataset("material_card_std", data=std, compression="lzf")
    print(f"Built WP3 features: {wp3_path} (built={len(built_ids)}, skipped={len(skipped)})")


if __name__ == "__main__":
    main()


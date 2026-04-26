from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np

from .physics import FEATURE_NAMES
from .targets import TARGET_NAMES


def _ok(msg: str) -> None:
    print(f"PASS: {msg}")


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    raise SystemExit(1)


def _require(condition: bool, msg: str) -> None:
    if not condition:
        _fail(msg)
    _ok(msg)


def _resolve_path(path_str: str) -> Path:
    p = Path(path_str).expanduser()
    return p if p.is_absolute() else (Path.cwd() / p)


def _require_dataset(root: h5py.Group, rel: str, sim_id: str) -> h5py.Dataset:
    cur: h5py.Group | h5py.Dataset = root
    for part in rel.split("/"):
        if part not in cur:
            _fail(f"{sim_id} missing {rel}")
        cur = cur[part]
    if not isinstance(cur, h5py.Dataset):
        _fail(f"{sim_id} expected dataset at {rel}")
    return cur


def run_preflight(wp2_path: Path, wp3_path: Path, min_free_gb: float, require_fine: bool) -> None:
    _require(wp2_path.exists(), f"WP2 exists: {wp2_path}")
    wp3_path.parent.mkdir(parents=True, exist_ok=True)
    _require(wp3_path.parent.exists(), f"WP3 parent exists: {wp3_path.parent}")

    usage = shutil.disk_usage(wp3_path.parent)
    free_gb = usage.free / (1024**3)
    _require(free_gb >= min_free_gb, f"free disk {free_gb:.2f} GB >= {min_free_gb:.2f} GB")

    if wp3_path.exists():
        size_gb = wp3_path.stat().st_size / (1024**3)
        print(f"INFO: Existing WP3 file will be replaced: {wp3_path} ({size_gb:.2f} GB)")
    else:
        print(f"INFO: WP3 output will be created: {wp3_path}")

    with h5py.File(wp2_path, "r") as wp2:
        _require("simulations" in wp2, "WP2 has simulations group")
        sim_ids = sorted(wp2["simulations"].keys())
        _require(len(sim_ids) > 0, "WP2 has at least one simulation")
        sample_id = sim_ids[0]
        sim = wp2[f"simulations/{sample_id}"]
        _require("mesh/coarse/elements" in sim, f"{sample_id} has coarse elements")
        _require("mesh/fine/elements" in sim, f"{sample_id} has fine elements")
        if require_fine:
            _require("fine/displacement" in sim, f"{sample_id} has fine displacement")
            _require("fine/fiber_stress_1" in sim, f"{sample_id} has fine fiber_stress_1")
            _require("fine/fiber_stress_2" in sim, f"{sample_id} has fine fiber_stress_2")
            _require("fine/thickness" in sim, f"{sample_id} has fine thickness")

    print("Preflight checks passed.")


def run_postbuild(wp2_path: Path, wp3_path: Path, require_fine: bool) -> None:
    _require(wp3_path.exists(), f"WP3 exists: {wp3_path}")
    base_required = [
        "graph/edge_index",
        "graph/edge_attr",
        "graph/coarse_to_fine_index",
        "coarse/mesh_elements",
        "coarse/resampled/stroke_fracs",
        "coarse/resampled/fields",
        "coarse/resampled/rates",
        "fine/mesh_elements",
        "targets/wrinkle_severity",
        "material_card",
        "material_card_raw",
    ]
    fine_required = [
        "fine/mesh_nodes",
        "fine/edge_index",
        "fine/edge_attr",
        "fine/element_edge_index",
        "fine/resampled/fiber_stress_1",
        "fine/resampled/fiber_stress_2",
        "fine/resampled/displacement_z",
        "fine/resampled/thickness",
    ]
    with h5py.File(wp2_path, "r") as wp2, h5py.File(wp3_path, "r") as wp3:
        _require("metadata" in wp3, "WP3 has metadata group")
        _require("simulations" in wp3, "WP3 has simulations group")
        _require("built_sim_ids" in wp3["metadata"], "WP3 metadata has built_sim_ids")

        built_ids = wp3["metadata/built_sim_ids"].asstr()[:].tolist()
        sim_ids = sorted(wp3["simulations"].keys())
        _require(len(built_ids) > 0, "WP3 built_sim_ids is non-empty")
        _require(sorted(built_ids) == sim_ids, "WP3 built_sim_ids match simulations keys")

        expected_target_names = list(TARGET_NAMES)
        feature_count = len(FEATURE_NAMES)
        for sim_id in built_ids:
            s_wp3 = wp3[f"simulations/{sim_id}"]
            s_wp2 = wp2[f"simulations/{sim_id}"]
            for rel in base_required:
                _require_dataset(s_wp3, rel, sim_id)
            if require_fine:
                for rel in fine_required:
                    _require_dataset(s_wp3, rel, sim_id)

            stroke = _require_dataset(s_wp3, "coarse/resampled/stroke_fracs", sim_id)[:]
            fields = _require_dataset(s_wp3, "coarse/resampled/fields", sim_id)
            rates = _require_dataset(s_wp3, "coarse/resampled/rates", sim_id)
            wrinkle = _require_dataset(s_wp3, "targets/wrinkle_severity", sim_id)
            _require(stroke.shape == (256,), f"{sim_id} stroke_fracs shape is (256,)")
            _require(fields.shape[0] == 256 and fields.shape[-1] == feature_count, f"{sim_id} fields shape integrity")
            _require(rates.shape[0] == 256 and rates.shape[-1] == feature_count, f"{sim_id} rates shape integrity")
            _require(wrinkle.shape[0] == 256, f"{sim_id} wrinkle_severity has 256 timesteps")

            target_group = s_wp3["targets"]
            for tname in expected_target_names:
                _require(tname in target_group, f"{sim_id} has targets/{tname}")
                _require(target_group[tname].shape[0] == 256, f"{sim_id} targets/{tname} has 256 timesteps")

            wp2_coarse_elem = s_wp2["mesh/coarse/elements"][:].astype(np.int64, copy=False)
            wp3_coarse_elem = s_wp3["coarse/mesh_elements"][:].astype(np.int64, copy=False)
            _require(np.array_equal(wp2_coarse_elem, wp3_coarse_elem), f"{sim_id} coarse mesh integrity")

            wp2_fine_elem = s_wp2["mesh/fine/elements"][:].astype(np.int64, copy=False)
            wp3_fine_elem = s_wp3["fine/mesh_elements"][:].astype(np.int64, copy=False)
            _require(np.array_equal(wp2_fine_elem, wp3_fine_elem), f"{sim_id} fine mesh integrity")

            if require_fine:
                for rel in fine_required[4:]:
                    arr = _require_dataset(s_wp3, rel, sim_id)
                    _require(arr.shape[0] == 256, f"{sim_id} {rel} has 256 timesteps")

        print(f"Post-build checks passed for {len(built_ids)} simulations.")


def main() -> None:
    parser = argparse.ArgumentParser(description="WP3 include-fine rebuild pre/post checks")
    sub = parser.add_subparsers(dest="mode", required=True)

    p_pre = sub.add_parser("preflight", help="Storage + input preflight checks")
    p_pre.add_argument("--wp2-path", required=True, type=str)
    p_pre.add_argument("--wp3-path", required=True, type=str)
    p_pre.add_argument("--min-free-gb", default=140.0, type=float)
    p_pre.add_argument("--require-fine", action="store_true")

    p_post = sub.add_parser("postbuild", help="Post-build schema/shape/integrity checks")
    p_post.add_argument("--wp2-path", required=True, type=str)
    p_post.add_argument("--wp3-path", required=True, type=str)
    p_post.add_argument("--require-fine", action="store_true")

    args = parser.parse_args()
    wp2_path = _resolve_path(args.wp2_path)
    wp3_path = _resolve_path(args.wp3_path)
    if args.mode == "preflight":
        run_preflight(wp2_path, wp3_path, float(args.min_free_gb), bool(args.require_fine))
    elif args.mode == "postbuild":
        run_postbuild(wp2_path, wp3_path, bool(args.require_fine))
    else:
        _fail(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)

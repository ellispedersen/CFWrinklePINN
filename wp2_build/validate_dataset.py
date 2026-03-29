from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import h5py
import numpy as np

from .build_dataset import read_field_all_increments, read_increment_times
from .schema import CV_N_FOLDS, H5_PATH

REQUIRED_FINE_FIELDS = [
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
    "increments",
    "times_s",
    "stroke_frac",
    "derived/comp_frac_f1",
    "derived/dz_variance",
    "derived/severity",
]
BATCH_A_ONLY_FIELDS = ["crystallinity"]


def _pass(name: str) -> None:
    print(f"PASS: {name}")


def _fail(name: str, msg: str) -> None:
    print(f"FAIL: {name} - {msg}")


def _check_sim_count(h5: h5py.File) -> tuple[bool, str]:
    n = len(h5["simulations"].keys())
    return (n == 66, f"expected 66 got {n}")


def _check_required_fields(h5: h5py.File) -> tuple[bool, str]:
    for sim_id in h5["simulations"].keys():
        fine = h5[f"simulations/{sim_id}/fine"]
        for rel in REQUIRED_FINE_FIELDS:
            cur = fine
            for part in rel.split("/"):
                if part not in cur:
                    return False, f"{sim_id} missing fine/{rel}"
                cur = cur[part]
        batch = h5[f"simulations/{sim_id}"].attrs["batch"]
        if batch == "A":
            if "crystallinity" not in fine:
                return False, f"{sim_id} missing crystallinity"
    return True, "ok"


def _check_nan_inf(h5: h5py.File) -> tuple[bool, str]:
    for sim_id in h5["simulations"].keys():
        sim = h5[f"simulations/{sim_id}"]
        for field in ("displacement", "fiber_stress_1", "stress"):
            arr = sim[f"fine/{field}"][:]
            if np.isnan(arr).any() or np.isinf(arr).any():
                return False, f"{sim_id} has NaN/Inf in {field}"
        onset = sim.attrs.get("onset_stroke_frac", np.nan)
        if isinstance(onset, (float, np.floating)) and math.isnan(float(onset)):
            pass
    return True, "ok"


def _sample_sims_by_batch(h5: h5py.File) -> dict[str, list[str]]:
    by_batch = {"A": [], "B": []}
    for sim_id in h5["simulations"].keys():
        b = str(h5[f"simulations/{sim_id}"].attrs["batch"])
        if b in by_batch:
            by_batch[b].append(sim_id)
    rng = random.Random(42)
    return {k: sorted(rng.sample(v, min(3, len(v)))) for k, v in by_batch.items()}


def _check_node_counts(h5: h5py.File) -> tuple[bool, str]:
    sample = _sample_sims_by_batch(h5)
    for batch, sim_ids in sample.items():
        for sim_id in sim_ids:
            sim = h5[f"simulations/{sim_id}"]
            fine_run_dir = Path(sim.attrs["fine_run_dir"])
            ply_groups = [int(x) for x in sim.attrs["ply_groups"]]
            afr = fine_run_dir / "results" / "model_40_1.afr"
            data, _ = read_field_all_increments(afr, 40, 1, ply_groups)
            if data.shape[1] != int(sim.attrs["n_nodes_fine"]):
                return False, f"{sim_id} node mismatch h5={sim.attrs['n_nodes_fine']} afr={data.shape[1]}"
    return True, "ok"


def _check_increment_counts(h5: h5py.File) -> tuple[bool, str]:
    sample = _sample_sims_by_batch(h5)
    for _, sim_ids in sample.items():
        for sim_id in sim_ids:
            sim = h5[f"simulations/{sim_id}"]
            fine_run_dir = Path(sim.attrs["fine_run_dir"])
            afs_incr, _, _ = read_increment_times(fine_run_dir)
            h5_incr = sim["fine/increments"][:]
            if len(afs_incr) != len(h5_incr):
                return False, f"{sim_id} increment mismatch h5={len(h5_incr)} afs={len(afs_incr)}"
    return True, "ok"


def _check_severity(h5: h5py.File) -> tuple[bool, str]:
    reg_path = Path("reports/wrinkle_onset_registry.json")
    reg = json.loads(reg_path.read_text(encoding="utf-8"))
    sev_by_sim = {r["sim_id"]: float(r.get("max_compound_severity", 0.0) or 0.0) for r in reg if r.get("sim_id")}
    for sim_id in h5["simulations"].keys():
        sid = sim_id
        if sid not in sev_by_sim:
            sid = sim_id.replace("_pair2", "").replace("_pair1", "")
        if sid not in sev_by_sim:
            return False, f"{sim_id} missing from registry"
        got = float(h5[f"simulations/{sim_id}"].attrs["compound_severity"])
        exp = sev_by_sim[sid]
        if abs(got - exp) > 1e-9:
            return False, f"{sim_id} severity mismatch h5={got} reg={exp}"
    return True, "ok"


def _check_splits(h5: h5py.File) -> tuple[bool, str]:
    all_ids = set(h5["simulations"].keys())
    splits = h5["splits"]
    for k in range(CV_N_FOLDS):
        fold = splits[f"fold_{k}"]
        tr = set(fold["train"].asstr()[:].tolist())
        va = set(fold["val"].asstr()[:].tolist())
        te = set(fold["test"].asstr()[:].tolist())
        if tr & va or tr & te or va & te:
            return False, f"fold_{k} overlap detected"
        if (tr | va | te) != all_ids:
            return False, f"fold_{k} union mismatch"
    return True, "ok"


def _check_batch_b_crystallinity(h5: h5py.File) -> tuple[bool, str]:
    for sim_id in h5["simulations"].keys():
        sim = h5[f"simulations/{sim_id}"]
        if str(sim.attrs["batch"]) != "B":
            continue
        ds = sim["fine/crystallinity"]
        if ds.shape[1] != 0:
            return False, f"{sim_id} crystallinity shape[1]={ds.shape[1]}"
        if bool(ds.attrs.get("available", True)) is not False:
            return False, f"{sim_id} crystallinity available attr should be False"
    return True, "ok"


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate WP2 HDF5 dataset")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not H5_PATH.exists():
        print(f"FAIL: dataset missing: {H5_PATH}")
        sys.exit(1)

    checks = [
        ("sim-count", _check_sim_count),
        ("required-fields", _check_required_fields),
        ("nan-inf", _check_nan_inf),
        ("node-count-spotcheck", _check_node_counts),
        ("increment-count-spotcheck", _check_increment_counts),
        ("severity-match", _check_severity),
        ("split-integrity", _check_splits),
        ("batch-b-crystallinity", _check_batch_b_crystallinity),
    ]

    failed = 0
    with h5py.File(H5_PATH, "r") as h5:
        for name, fn in checks:
            ok, msg = fn(h5)
            if ok:
                if args.verbose:
                    _pass(name)
            else:
                failed += 1
                _fail(name, msg)

    if failed:
        print(f"Validation failed: {failed} checks")
        sys.exit(1)
    print("Validation passed")
    sys.exit(0)


if __name__ == "__main__":
    main()


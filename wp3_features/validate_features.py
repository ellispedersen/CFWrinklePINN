from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

from .h5_utils import get_wp3_h5_path
from .physics import FEATURE_NAMES
from .targets import TARGET_NAMES

ROOT = Path(__file__).resolve().parents[1]


def _pass(name: str) -> None:
    print(f"PASS: {name}")


def _fail(name: str, msg: str) -> None:
    print(f"FAIL: {name} - {msg}")


def _parse_json_attr(value: object) -> object:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    elif isinstance(value, np.bytes_):
        value = bytes(value).decode("utf-8")
    if not isinstance(value, str):
        raise TypeError(f"Expected JSON string attr, got {type(value)!r}")
    return json.loads(value)


def _check_graph_undirected(h5: h5py.File) -> tuple[bool, str]:
    for sid in h5["simulations"].keys():
        ei = h5[f"simulations/{sid}/graph/edge_index"][:]
        s = {tuple(x) for x in ei.T.tolist()}
        for a, b in list(s)[: min(1000, len(s))]:
            if (b, a) not in s:
                return False, f"{sid} missing reverse edge for {(a, b)}"
    return True, "ok"


def _check_stroke_grid(h5: h5py.File) -> tuple[bool, str]:
    target = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    for sid in h5["simulations"].keys():
        s = h5[f"simulations/{sid}/coarse/resampled/stroke_fracs"][:]
        if s.shape != (256,) or not np.allclose(s, target, atol=1e-6):
            return False, f"{sid} invalid stroke grid"
    return True, "ok"


def _check_rate_init(h5: h5py.File) -> tuple[bool, str]:
    for sid in h5["simulations"].keys():
        r0 = h5[f"simulations/{sid}/coarse/resampled/rates"][0]
        if np.nanmax(np.abs(r0)) > 1e-4:
            return False, f"{sid} non-zero initial rates"
    return True, "ok"


def _check_targets(h5: h5py.File) -> tuple[bool, str]:
    for sid in h5["simulations"].keys():
        t = h5[f"simulations/{sid}/targets/wrinkle_severity"][:]
        if np.nanmax(t[0]) > 0.2:
            return False, f"{sid} high initial wrinkle severity"
        if np.isnan(t).all():
            return False, f"{sid} all target values are NaN"
    return True, "ok"


def _check_label_schema(h5: h5py.File) -> tuple[bool, str]:
    expected_feature_names = list(FEATURE_NAMES)
    expected_rate_names = [f"d_dt:{name}" for name in expected_feature_names]
    expected_target_names = list(TARGET_NAMES)
    for sid in h5["simulations"].keys():
        res = h5[f"simulations/{sid}/coarse/resampled"]
        if "feature_names" not in res.attrs:
            return False, f"{sid} missing coarse/resampled feature_names attr"
        if "rate_feature_names" not in res.attrs:
            return False, f"{sid} missing coarse/resampled rate_feature_names attr"
        feat_names = _parse_json_attr(res.attrs["feature_names"])
        rate_names = _parse_json_attr(res.attrs["rate_feature_names"])
        if feat_names != expected_feature_names:
            return False, f"{sid} feature_names mismatch"
        if rate_names != expected_rate_names:
            return False, f"{sid} rate_feature_names mismatch"
        if int(res["fields"].shape[-1]) != len(expected_feature_names):
            return False, f"{sid} coarse/resampled/fields channel mismatch"
        if int(res["rates"].shape[-1]) != len(expected_feature_names):
            return False, f"{sid} coarse/resampled/rates channel mismatch"

        tgt = h5[f"simulations/{sid}/targets"]
        if "target_names" not in tgt.attrs:
            return False, f"{sid} missing targets target_names attr"
        target_names = _parse_json_attr(tgt.attrs["target_names"])
        if target_names != expected_target_names:
            return False, f"{sid} target_names mismatch"
        for target_name in target_names:
            if target_name not in tgt:
                return False, f"{sid} missing targets/{target_name}"
    return True, "ok"


def _check_material_norm(h5: h5py.File) -> tuple[bool, str]:
    cards = []
    raw_cards = []
    for sid in h5["simulations"].keys():
        if "material_card" not in h5[f"simulations/{sid}"] or "material_card_raw" not in h5[f"simulations/{sid}"]:
            return False, f"{sid} missing material card datasets"
        cards.append(h5[f"simulations/{sid}/material_card"][:].astype(np.float32))
        raw_cards.append(h5[f"simulations/{sid}/material_card_raw"][:].astype(np.float32))
    cards_np = np.stack(cards, axis=0)
    raw_np = np.stack(raw_cards, axis=0)
    m = np.abs(cards_np.mean(axis=0))
    s = np.abs(cards_np.std(axis=0) - 1.0)
    # Allow small floating-point residuals from float32 normalisation.
    if np.max(m) > 1e-3:
        return False, f"material card mean not zero: max={float(np.max(m))}"
    # Constant raw dimensions are expected to remain zero after normalisation.
    raw_std = raw_np.std(axis=0).astype(np.float32)
    mask = raw_std > 1e-8
    if np.any(mask) and np.max(s[mask]) > 1e-3:
        return False, f"material card std not one: max={float(np.max(s))}"
    return True, "ok"


def _check_required(h5: h5py.File) -> tuple[bool, str]:
    if "metadata" not in h5 or "built_sim_ids" not in h5["metadata"]:
        return False, "missing metadata/built_sim_ids"
    built_ids = h5["metadata/built_sim_ids"].asstr()[:].tolist()
    if not built_ids:
        return False, "no built simulations listed"

    req = [
        "graph/edge_index",
        "graph/edge_attr",
        "coarse/resampled/stroke_fracs",
        "coarse/resampled/fields",
        "coarse/resampled/rates",
        "targets/wrinkle_severity",
        "material_card",
    ]
    for sid in built_ids:
        if sid not in h5["simulations"]:
            return False, f"{sid} listed in built_sim_ids but missing in simulations/"
        for rel in req:
            cur = h5[f"simulations/{sid}"]
            for p in rel.split("/"):
                if p not in cur:
                    return False, f"{sid} missing {rel}"
                cur = cur[p]
    return True, "ok"


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate WP3 feature dataset")
    parser.add_argument("--path", default=None, type=str)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    h5_path = Path(args.path) if args.path else get_wp3_h5_path(ROOT)
    if not h5_path.exists():
        print(f"FAIL: missing {h5_path}")
        sys.exit(1)

    checks = [
        ("required", _check_required),
        ("graph-undirected", _check_graph_undirected),
        ("stroke-grid", _check_stroke_grid),
        ("rate-initial", _check_rate_init),
        ("targets", _check_targets),
        ("label-schema", _check_label_schema),
        ("material-normalisation", _check_material_norm),
    ]
    failed = 0
    with h5py.File(h5_path, "r") as h5:
        for name, fn in checks:
            ok, msg = fn(h5)
            if ok:
                if args.verbose:
                    _pass(name)
            else:
                failed += 1
                _fail(name, msg)
    if failed:
        print(f"Validation failed: {failed}")
        sys.exit(1)
    print("Validation passed")


if __name__ == "__main__":
    main()


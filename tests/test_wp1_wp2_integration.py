from __future__ import annotations

import json
from pathlib import Path, PureWindowsPath

import h5py
import pytest

from wp2_build.validate_dataset import (
    _check_batch_b_crystallinity,
    _check_increment_counts,
    _check_nan_inf,
    _check_node_counts,
    _check_required_fields,
    _check_severity,
    _check_sim_count,
    _check_splits,
)

pytestmark = pytest.mark.integration


def _candidate_run_paths(path_str: str) -> list[Path]:
    candidates = [Path(path_str)]
    if ":\\" in path_str:
        win = PureWindowsPath(path_str)
        tail = "/".join(win.parts[1:])
        candidates.append(Path(f"/mnt/{win.drive[0].lower()}/{tail}"))
    return candidates


def _raw_runs_accessible(h5: h5py.File) -> bool:
    first_sim = next(iter(h5["simulations"].keys()))
    run_dir_raw = str(h5[f"simulations/{first_sim}"].attrs["fine_run_dir"])
    for base in _candidate_run_paths(run_dir_raw):
        if (base / "results" / "model.afs").exists() or (base / "model.afs").exists():
            return True
    return False


def test_wp2_dataset_exists():
    h5_path = Path("data/cfwrinkle_dataset.h5")
    assert h5_path.exists(), "Expected WP2 dataset at data/cfwrinkle_dataset.h5"
    assert h5_path.stat().st_size > 500 * 1024 * 1024


def test_wp1_wp2_sim_count_alignment():
    wr = json.loads(Path("reports/wrinkle_onset_registry.json").read_text(encoding="utf-8"))
    wr_ids = {r["sim_id"] for r in wr if r.get("sim_id")} - {"geom_0_1_pair2"}
    with h5py.File("data/cfwrinkle_dataset.h5", "r") as h5:
        ds_ids = set(h5["simulations"].keys())
    assert ds_ids == wr_ids


def test_wp1_wp2_specific_severity_anchor_values():
    with h5py.File("data/cfwrinkle_dataset.h5", "r") as h5:
        mold = float(h5["simulations/mold_set_004"].attrs["compound_severity"])
        geom = float(h5["simulations/geom_0_8"].attrs["compound_severity"])
    assert abs(mold - 0.167) < 0.01
    assert abs(geom - 0.473) < 0.01


def test_wp2_validator_checks_pass_on_real_dataset():
    with h5py.File("data/cfwrinkle_dataset.h5", "r") as h5:
        if not _raw_runs_accessible(h5):
            pytest.skip("Raw AniForm run directories are not mounted in this environment")
        checks = [
            _check_sim_count,
            _check_required_fields,
            _check_nan_inf,
            _check_node_counts,
            _check_increment_counts,
            _check_severity,
            _check_splits,
            _check_batch_b_crystallinity,
        ]
        failed = []
        for fn in checks:
            ok, msg = fn(h5)
            if not ok:
                failed.append((fn.__name__, msg))
        assert not failed, f"Validator checks failed: {failed}"


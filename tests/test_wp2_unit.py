from __future__ import annotations

import numpy as np
import h5py
from pathlib import Path

from wp2_build.build_dataset import _severity_from_components, compute_derived, write_splits
from wp2_build.schema import SimRecord


def _mk_rec(sim_id: str, batch: str, sev: float) -> SimRecord:
    return SimRecord(
        sim_id=sim_id,
        batch=batch,
        fine_dir=Path("C:\\tmp"),
        coarse_dir=None,
        ply_groups=[6, 10] if batch == "A" else [4, 8, 12],
        bending_groups=[6, 7, 10, 11] if batch == "A" else [4, 5, 8, 9, 12, 13],
        compound_severity=sev,
        is_wrinkled=sev > 0.10,
        onset_increment=10,
        onset_stroke_frac=0.2,
        uz_range_mm=[0.0, 10.0],
        max_rfz_N=1000.0,
        n_plies=2 if batch == "A" else 3,
        material="UD_thermoplastic" if batch == "A" else "Twintex_2x2_twill",
        ply_orientations_deg=[0.0, 90.0] if batch == "A" else [0.0, 45.0, 90.0],
    )


def test_severity_components_clamped_and_weighted():
    comp = np.array([0.0, 0.25, 0.5, 1.0], dtype=np.float32)
    dz = np.array([0.0, 5.0, 10.0, 20.0], dtype=np.float32)
    shear = np.array([0.0, 45.0, 90.0, 120.0], dtype=np.float32)
    sev = _severity_from_components("A", comp, dz, shear)
    expected = np.array([0.0, 0.5, 1.0, 1.0], dtype=np.float32)
    assert np.allclose(sev, expected, atol=1e-6)


def test_compute_derived_shapes_and_expected_values():
    displacement = np.array(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 2.0], [0.0, 0.0, 4.0], [0.0, 0.0, 6.0]],
            [[0.0, 0.0, 1.0], [0.0, 0.0, 3.0], [0.0, 0.0, 5.0], [0.0, 0.0, 7.0]],
        ],
        dtype=np.float32,
    )
    fiber_stress_1 = np.array(
        [[-0.1, -0.2, 0.0, 0.1], [-0.06, 0.0, 0.2, -1.0]],
        dtype=np.float32,
    )
    shear_angle = np.array([[10.0, 20.0, 30.0, 5.0], [45.0, 44.0, 42.0, 40.0]], dtype=np.float32)

    out = compute_derived("B", displacement, fiber_stress_1, shear_angle)
    assert set(out.keys()) == {"comp_frac_f1", "dz_variance", "severity"}
    assert out["comp_frac_f1"].shape == (2,)
    assert out["dz_variance"].shape == (2,)
    assert out["severity"].shape == (2,)
    assert np.allclose(out["comp_frac_f1"], np.array([0.5, 0.5], dtype=np.float32))
    assert np.allclose(out["dz_variance"], np.array([5.0, 5.0], dtype=np.float32))


def test_write_splits_train_val_test_no_overlap_and_full_union(tmp_path):
    records = []
    # balanced labels per (batch, quartile) to make stratification feasible
    for q, sev in enumerate([0.05, 0.25, 0.5, 0.75]):
        for i in range(10):
            records.append(_mk_rec(f"A_q{q}_{i}", "A", sev))
            records.append(_mk_rec(f"B_q{q}_{i}", "B", sev))

    h5_path = tmp_path / "splits.h5"
    with h5py.File(h5_path, "w") as h5:
        write_splits(h5, records)
        all_ids = {r.sim_id for r in records}
        assert "splits" in h5
        assert int(h5["splits"].attrs["n_folds"]) == 5
        for k in range(5):
            fold = h5[f"splits/fold_{k}"]
            tr = set(fold["train"].asstr()[:].tolist())
            va = set(fold["val"].asstr()[:].tolist())
            te = set(fold["test"].asstr()[:].tolist())
            assert not (tr & va)
            assert not (tr & te)
            assert not (va & te)
            assert (tr | va | te) == all_ids


from __future__ import annotations

import json

import h5py
import numpy as np
import pytest
torch = pytest.importorskip("torch")

from model.dataset import WrinkleDataset
from wp3_features.physics import FEATURE_NAMES
from wp3_features.targets import TARGET_NAMES


def _write_minimal_wp3(
    path,
    *,
    edge_attr_rows: int = 3,
    feature_names: list[str] | None = None,
    target_names: list[str] | None = None,
) -> None:
    feature_names = feature_names or list(FEATURE_NAMES)
    target_names = target_names or list(TARGET_NAMES)
    with h5py.File(path, "w") as f:
        sgrp = f.create_group("simulations").create_group("sim_001")
        g = sgrp.create_group("graph")
        g.create_dataset("edge_index", data=np.array([[0, 1, 2], [1, 2, 0]], dtype=np.int32))
        g.create_dataset("edge_attr", data=np.zeros((edge_attr_rows, 4), dtype=np.float64))
        sgrp.create_dataset("material_card", data=np.ones((8,), dtype=np.float64))

        coarse = sgrp.create_group("coarse")
        coarse.create_dataset("mesh_elements", data=np.array([[0, 1, 2], [2, 1, 0]], dtype=np.int32))
        res = coarse.create_group("resampled")
        res.create_dataset("fields", data=np.zeros((4, 3, len(FEATURE_NAMES)), dtype=np.float64))
        res.create_dataset("rates", data=np.zeros((4, 3, len(FEATURE_NAMES)), dtype=np.float64))
        res.attrs["feature_names"] = json.dumps(feature_names)
        res.attrs["rate_feature_names"] = json.dumps([f"d_dt:{name}" for name in feature_names])

        tgt = sgrp.create_group("targets")
        tgt.attrs["target_names"] = json.dumps(target_names)
        for key in TARGET_NAMES:
            tgt.create_dataset(key, data=np.zeros((4, 2), dtype=np.float64))


def _write_minimal_wp3_with_fine(path) -> None:
    with h5py.File(path, "w") as f:
        sgrp = f.create_group("simulations").create_group("sim_001")
        g = sgrp.create_group("graph")
        g.create_dataset("edge_index", data=np.array([[0, 1, 2], [1, 2, 0]], dtype=np.int32))
        g.create_dataset("edge_attr", data=np.zeros((3, 4), dtype=np.float64))
        g.create_dataset("coarse_to_fine_index", data=np.array([[0], [0]], dtype=np.int64))
        sgrp.create_dataset("material_card", data=np.ones((8,), dtype=np.float64))

        coarse = sgrp.create_group("coarse")
        coarse.create_dataset("mesh_elements", data=np.array([[0, 1, 2], [2, 1, 0]], dtype=np.int32))
        res = coarse.create_group("resampled")
        res.create_dataset("fields", data=np.zeros((4, 3, len(FEATURE_NAMES)), dtype=np.float64))
        res.create_dataset("rates", data=np.zeros((4, 3, len(FEATURE_NAMES)), dtype=np.float64))
        res.attrs["feature_names"] = json.dumps(list(FEATURE_NAMES))
        res.attrs["rate_feature_names"] = json.dumps([f"d_dt:{name}" for name in FEATURE_NAMES])

        fine = sgrp.create_group("fine")
        fine.create_dataset("resampled/fiber_stress_1", data=np.full((4, 3), 10.0, dtype=np.float64))
        fine.create_dataset("resampled/fiber_stress_2", data=np.full((4, 3), 20.0, dtype=np.float64))
        fine.create_dataset("resampled/displacement_z", data=np.full((4, 3), 30.0, dtype=np.float64))
        fine.create_dataset("resampled/thickness", data=np.full((4, 3), 40.0, dtype=np.float64))
        fine.create_dataset("mesh_elements", data=np.array([[0, 1, 2]], dtype=np.int64))
        fine.create_dataset("mesh_nodes", data=np.zeros((3, 3), dtype=np.float64))
        fine.create_dataset("edge_index", data=np.array([[0], [1]], dtype=np.int64))
        fine.create_dataset("edge_attr", data=np.zeros((1, 4), dtype=np.float64))

        tgt = sgrp.create_group("targets")
        tgt.attrs["target_names"] = json.dumps(list(TARGET_NAMES))
        for key in TARGET_NAMES:
            tgt.create_dataset(key, data=np.zeros((4, 2), dtype=np.float64))


def test_dataset_enforces_output_dtypes(tmp_path) -> None:
    h5_path = tmp_path / "wp3.h5"
    _write_minimal_wp3(h5_path)

    ds = WrinkleDataset(h5_path=h5_path, sim_ids=["sim_001"], normalize=False)
    sample = ds[0]

    assert sample["node_features"].dtype == torch.float32
    assert sample["edge_index"].dtype == torch.int64
    assert sample["edge_attr"].dtype == torch.float32
    assert sample["elements"].dtype == torch.int64
    assert sample["targets"].dtype == torch.float32


def test_dataset_rejects_bad_graph_shapes(tmp_path) -> None:
    h5_path = tmp_path / "wp3_bad.h5"
    _write_minimal_wp3(h5_path, edge_attr_rows=2)

    ds = WrinkleDataset(h5_path=h5_path, sim_ids=["sim_001"], normalize=False)
    with pytest.raises(ValueError, match="edge_attr"):
        _ = ds[0]


def test_dataset_rejects_non_finite_node_features(tmp_path) -> None:
    h5_path = tmp_path / "wp3_nan_feat.h5"
    _write_minimal_wp3(h5_path)
    with h5py.File(h5_path, "r+") as f:
        f["simulations/sim_001/coarse/resampled/fields"][0, 0, 0] = np.nan

    ds = WrinkleDataset(h5_path=h5_path, sim_ids=["sim_001"], normalize=False)
    with pytest.raises(ValueError, match="node_features contains non-finite values"):
        _ = ds[0]


def test_dataset_rejects_non_finite_edge_attr(tmp_path) -> None:
    h5_path = tmp_path / "wp3_inf_edge_attr.h5"
    _write_minimal_wp3(h5_path)
    with h5py.File(h5_path, "r+") as f:
        f["simulations/sim_001/graph/edge_attr"][0, 0] = np.inf

    ds = WrinkleDataset(h5_path=h5_path, sim_ids=["sim_001"], normalize=False)
    with pytest.raises(ValueError, match="edge_attr contains non-finite values"):
        _ = ds[0]


def test_dataset_rejects_feature_label_mismatch(tmp_path) -> None:
    h5_path = tmp_path / "wp3_bad_feature_labels.h5"
    wrong_names = list(FEATURE_NAMES)
    wrong_names[0], wrong_names[1] = wrong_names[1], wrong_names[0]
    _write_minimal_wp3(h5_path, feature_names=wrong_names)

    ds = WrinkleDataset(h5_path=h5_path, sim_ids=["sim_001"], normalize=False)
    with pytest.raises(ValueError, match="Feature names mismatch"):
        _ = ds[0]


def test_dataset_rejects_target_label_mismatch(tmp_path) -> None:
    h5_path = tmp_path / "wp3_bad_target_labels.h5"
    wrong_names = list(TARGET_NAMES)
    wrong_names[0], wrong_names[1] = wrong_names[1], wrong_names[0]
    _write_minimal_wp3(h5_path, target_names=wrong_names)

    ds = WrinkleDataset(h5_path=h5_path, sim_ids=["sim_001"], normalize=False)
    with pytest.raises(ValueError, match="Target names mismatch"):
        _ = ds[0]


def test_fine_features_not_normalized_by_default(tmp_path) -> None:
    h5_path = tmp_path / "wp3_fine_default.h5"
    _write_minimal_wp3_with_fine(h5_path)
    ds = WrinkleDataset(h5_path=h5_path, sim_ids=["sim_001"], normalize=False, include_fine=True)
    sample = ds[0]
    fine = sample["fine_features"].numpy()
    assert np.allclose(fine[..., 0], 10.0)
    assert np.allclose(fine[..., 1], 20.0)
    assert np.allclose(fine[..., 2], 30.0)
    assert np.allclose(fine[..., 3], 40.0)


def test_fine_feature_normalization_uses_provided_stats(tmp_path) -> None:
    h5_path = tmp_path / "wp3_fine_norm.h5"
    _write_minimal_wp3_with_fine(h5_path)
    ds = WrinkleDataset(
        h5_path=h5_path,
        sim_ids=["sim_001"],
        normalize=False,
        include_fine=True,
        fine_normalize=True,
        fine_norm_stats={
            "fine_feat_mean": np.array([8.0, 18.0, 29.0, 39.0], dtype=np.float32),
            "fine_feat_std": np.array([2.0, 2.0, 1.0, 0.5], dtype=np.float32),
        },
    )
    fine = ds[0]["fine_features"].numpy()
    assert np.allclose(fine[..., 0], 1.0)
    assert np.allclose(fine[..., 1], 1.0)
    assert np.allclose(fine[..., 2], 1.0)
    assert np.allclose(fine[..., 3], 2.0)


def test_fine_feature_normalization_requires_include_fine(tmp_path) -> None:
    h5_path = tmp_path / "wp3_fine_guard.h5"
    _write_minimal_wp3(h5_path)
    with pytest.raises(ValueError, match="requires include_fine=True"):
        WrinkleDataset(h5_path=h5_path, sim_ids=["sim_001"], normalize=False, fine_normalize=True)

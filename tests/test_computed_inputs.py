from __future__ import annotations

import numpy as np

import pytest

from wp3_features.physics import FEATURE_NAMES, compute_feature_tensor


def _field_dict() -> dict[str, np.ndarray]:
    return {
        "displacement": np.array(
            [
                [[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]],
                [[0.0, 0.0, 0.0], [6.0, 8.0, 0.0]],
            ],
            dtype=np.float32,
        ),
        "temperature": np.array([[20.0, 25.0], [21.0, 26.0]], dtype=np.float32),
        "thickness": np.array([[2.0, 2.0], [1.0, 4.0]], dtype=np.float32),
        "eq_shear_rate": np.zeros((2, 2), dtype=np.float32),
        "fiber_dir_1": np.zeros((2, 2, 3), dtype=np.float32),
        "fiber_dir_2": np.zeros((2, 2, 3), dtype=np.float32),
        "shear_angle": np.array([[0.0, 0.0], [45.0, 10.0]], dtype=np.float32),
        "fiber_strain_1": np.zeros((2, 2), dtype=np.float32),
        "fiber_strain_2": np.zeros((2, 2), dtype=np.float32),
        "fiber_stress_1": np.array([[-0.1, 0.1], [-0.2, 0.2]], dtype=np.float32),
        "fiber_stress_2": np.zeros((2, 2), dtype=np.float32),
        "gl_strain": np.zeros((2, 2, 3), dtype=np.float32),
        "stress": np.zeros((2, 2, 3), dtype=np.float32),
    }


def test_compute_feature_tensor_derived_values_are_correct() -> None:
    names, feats = compute_feature_tensor(_field_dict(), np.array([0.0, 2.0], dtype=np.float32), batch="A")
    idx = {name: i for i, name in enumerate(names)}

    assert feats.shape == (2, 2, len(names))
    assert np.allclose(feats[1, :, idx["thickness_ratio"]], np.array([0.5, 2.0], dtype=np.float32), atol=1e-6)
    assert np.allclose(feats[1, :, idx["thickness_rate"]], np.array([-0.5, 1.0], dtype=np.float32), atol=1e-6)
    assert np.allclose(feats[1, :, idx["draw_in_distance"]], np.array([0.0, 10.0], dtype=np.float32), atol=1e-6)
    assert np.allclose(feats[1, :, idx["fiber_comp_indicator"]], np.array([1.0, 0.0], dtype=np.float32), atol=1e-6)


def test_compute_feature_tensor_locking_proximity_depends_on_batch() -> None:
    names_a, feats_a = compute_feature_tensor(_field_dict(), np.array([0.0, 1.0], dtype=np.float32), batch="A")
    names_b, feats_b = compute_feature_tensor(_field_dict(), np.array([0.0, 1.0], dtype=np.float32), batch="B")
    idx_a = {name: i for i, name in enumerate(names_a)}
    idx_b = {name: i for i, name in enumerate(names_b)}

    lock_a = feats_a[1, :, idx_a["locking_proximity"]]
    lock_b = feats_b[1, :, idx_b["locking_proximity"]]
    assert np.allclose(lock_a, np.array([0.5, 10.0 / 90.0], dtype=np.float32), atol=1e-6)
    assert np.allclose(lock_b, np.array([1.0, 10.0 / 45.0], dtype=np.float32), atol=1e-6)


def test_compute_feature_tensor_feature_name_contract() -> None:
    names, feats = compute_feature_tensor(_field_dict(), np.array([0.0, 1.0], dtype=np.float32), batch="A")
    assert tuple(names) == FEATURE_NAMES
    assert feats.shape[-1] == len(FEATURE_NAMES)


def test_compute_feature_tensor_rejects_invalid_batch() -> None:
    with pytest.raises(ValueError, match="batch must be 'A' or 'B'"):
        compute_feature_tensor(_field_dict(), np.array([0.0, 1.0], dtype=np.float32), batch="C")

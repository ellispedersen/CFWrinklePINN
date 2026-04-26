from __future__ import annotations

import numpy as np

from wp3_features.correspondence import build_coarse_to_fine_map
import pytest

from wp3_features.graph import build_edge_attr, build_edge_index, build_element_adjacency
from wp3_features.physics import FEATURE_NAMES, compute_feature_tensor
from wp3_features.targets import TARGET_NAMES, compute_wrinkle_targets
from wp3_features.temporal import compute_rates, resample_to_uniform


def test_graph_builders():
    nodes = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)
    elems = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    ei = build_edge_index(elems)
    ea = build_edge_attr(ei, nodes)
    assert ei.shape[0] == 2
    assert ei.dtype == np.int64
    assert ea.shape[0] == ei.shape[1]
    assert ea.shape[1] == 4
    assert ea.dtype == np.float32
    adj = build_element_adjacency(elems)
    assert adj.shape[0] == 2
    assert adj.dtype == np.int64


def test_graph_builders_validate_shapes():
    nodes = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
    bad_elems = np.array([[0, 1]], dtype=np.int64)
    with pytest.raises(ValueError, match="shape"):
        build_edge_index(bad_elems)
    with pytest.raises(ValueError, match="shape"):
        build_element_adjacency(bad_elems)
    with pytest.raises(ValueError, match="out-of-range"):
        build_edge_attr(np.array([[0, 2], [1, 0]], dtype=np.int64), nodes)


def test_temporal_rates_and_resample():
    t = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    x = np.array([[0.0], [1.0], [3.0]], dtype=np.float32)
    r = compute_rates(t, x)
    assert np.isclose(r[0, 0], 0.0)
    assert np.isclose(r[1, 0], 2.0)
    u, y = resample_to_uniform(t, x, n_points=5)
    assert u.shape == (5,)
    assert y.shape == (5, 1)


def test_correspondence_and_targets():
    coarse_nodes = np.array([[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0]], dtype=np.float32)
    coarse_elems = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    fine_nodes = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0], [1, 1, 0], [2, 1, 0]], dtype=np.float32)
    fine_elems = np.array([[0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4]], dtype=np.int32)
    m = build_coarse_to_fine_map(coarse_nodes, coarse_elems, fine_nodes, fine_elems)
    assert m["mapping_coverage"] >= 0.95
    f_fs = np.array([[-0.1, 0.0, 0.1, -0.2, 0.3, 0.2], [-0.2, -0.1, 0.1, -0.3, 0.3, 0.2]], dtype=np.float32)
    f_disp = np.zeros((2, 6, 3), dtype=np.float32)
    f_disp[1, :, 2] = np.array([0, 1, 0, 1, 0, 1], dtype=np.float32)
    f_th = np.ones((2, 6), dtype=np.float32)
    tgt = compute_wrinkle_targets(m, f_fs, f_disp, f_th, fine_elems)
    assert tuple(tgt.keys()) == TARGET_NAMES
    assert tgt["wrinkle_severity"].shape[1] == coarse_elems.shape[0]


def test_compute_feature_tensor_shape():
    n_t, n_n = 4, 5
    fields = {
        "displacement": np.zeros((n_t, n_n, 3), dtype=np.float32),
        "temperature": np.zeros((n_t, n_n), dtype=np.float32),
        "thickness": np.ones((n_t, n_n), dtype=np.float32),
        "eq_shear_rate": np.zeros((n_t, n_n), dtype=np.float32),
        "fiber_dir_1": np.zeros((n_t, n_n, 3), dtype=np.float32),
        "fiber_dir_2": np.zeros((n_t, n_n, 3), dtype=np.float32),
        "shear_angle": np.zeros((n_t, n_n), dtype=np.float32),
        "fiber_strain_1": np.zeros((n_t, n_n), dtype=np.float32),
        "fiber_strain_2": np.zeros((n_t, n_n), dtype=np.float32),
        "fiber_stress_1": np.zeros((n_t, n_n), dtype=np.float32),
        "fiber_stress_2": np.zeros((n_t, n_n), dtype=np.float32),
        "gl_strain": np.zeros((n_t, n_n, 3), dtype=np.float32),
        "stress": np.zeros((n_t, n_n, 3), dtype=np.float32),
    }
    names, feats = compute_feature_tensor(fields, np.linspace(0, 1, n_t, dtype=np.float32), "A")
    assert tuple(names) == FEATURE_NAMES
    assert feats.shape[:2] == (n_t, n_n)
    assert feats.shape[2] == len(names)


def test_targets_reject_out_of_range_fine_index() -> None:
    mapping = {
        "n_fine_per_coarse": np.array([1], dtype=np.int32),
        "coarse_index": np.array([0], dtype=np.int32),
        "fine_index": np.array([3], dtype=np.int32),
    }
    f_fs = np.zeros((2, 3), dtype=np.float32)
    f_disp = np.zeros((2, 3, 3), dtype=np.float32)
    f_th = np.ones((2, 3), dtype=np.float32)
    fine_elems = np.array([[0, 1, 2]], dtype=np.int32)
    with pytest.raises(ValueError, match="fine_index"):
        compute_wrinkle_targets(mapping, f_fs, f_disp, f_th, fine_elems)


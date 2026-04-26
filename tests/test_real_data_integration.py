"""Real-data integration tests.

These tests load an actual HDF5 simulation and run a forward pass with real data.
They are skipped automatically if the HDF5 files are not present (data transfer
may still be in progress).

Run with:
    pytest tests/test_real_data_integration.py -v -m integration
Or combined with the full suite excluding slow tests:
    pytest tests/ -m "integration and not slow" -v
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

# ---------------------------------------------------------------------------
# Determine HDF5 paths (mirror dataset.py / train.py env-var logic)
# ---------------------------------------------------------------------------
_WP3_H5 = Path(os.environ.get("WP3_H5", "data/cfwrinkle_wp3_features.h5"))
_WP2_H5 = Path(os.environ.get("WP2_H5", "data/cfwrinkle_dataset.h5"))

_DATA_AVAILABLE = _WP3_H5.exists() and _WP2_H5.exists()

pytestmark = pytest.mark.integration


def _skip_if_no_data() -> None:
    if not _DATA_AVAILABLE:
        pytest.skip(
            f"HDF5 files not found — skipping integration test. "
            f"Set WP3_H5 / WP2_H5 env vars or ensure files exist at "
            f"'{_WP3_H5}' and '{_WP2_H5}'."
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def first_sim_id() -> str:
    """Return the first simulation key found in the WP3 HDF5 file."""
    _skip_if_no_data()
    import h5py
    with h5py.File(_WP3_H5, "r") as f:
        sims = sorted(f["simulations"].keys())
    assert sims, "WP3 HDF5 has no simulations"
    return sims[0]


@pytest.fixture(scope="module")
def dataset_single(first_sim_id: str):
    """WrinkleDataset containing exactly one real simulation, T capped at 32."""
    _skip_if_no_data()
    from model.dataset import WrinkleDataset
    return WrinkleDataset(
        h5_path=str(_WP3_H5),
        sim_ids=[first_sim_id],
        max_timesteps=32,
        temporal_strategy="tail",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_load_single_sim_and_forward_pass(dataset_single, first_sim_id: str) -> None:
    """Load one real simulation and run a forward pass; verify output shapes."""
    _skip_if_no_data()
    from model.gnn import FormingGraphNet

    assert len(dataset_single) == 1, f"Expected 1 sample, got {len(dataset_single)}"
    batch = dataset_single[0]

    # Verify real mesh sizes are plausible (coarse mesh has ~3 700 nodes)
    n_nodes = batch["node_features"].shape[1]
    n_elems = batch["elements"].shape[0]
    t_steps = batch["node_features"].shape[0]
    assert n_nodes >= 500, f"Too few nodes: {n_nodes} — may have loaded wrong group"
    assert n_elems >= 500, f"Too few elements: {n_elems}"
    assert 1 <= t_steps <= 32, f"Unexpected timestep count: {t_steps}"

    model = FormingGraphNet(hidden_dim=32, attn_batch_nodes=512, decoder_chunk_t=32)
    model.eval()
    with torch.no_grad():
        pred = model(batch)

    assert pred.shape == (t_steps, n_elems, 4), (
        f"Unexpected output shape: {pred.shape} "
        f"(expected ({t_steps}, {n_elems}, 4))"
    )
    assert not torch.isnan(pred).any(), "NaN in forward pass output on real data"
    assert torch.isfinite(pred).all(), "Non-finite values in forward pass output on real data"


def test_loss_computable_on_real_data(dataset_single) -> None:
    """wrinkle_loss on real data returns finite positive values for all components."""
    _skip_if_no_data()
    from model.gnn import FormingGraphNet
    from model.loss import wrinkle_loss

    batch = dataset_single[0]
    model = FormingGraphNet(hidden_dim=32, attn_batch_nodes=512, decoder_chunk_t=32)
    model.eval()
    with torch.no_grad():
        pred = model(batch)

    grad_total, log = wrinkle_loss(pred, batch["targets"])

    assert torch.isfinite(grad_total), f"grad_total is not finite: {grad_total}"
    assert float(grad_total) > 0, f"grad_total is not positive: {grad_total}"

    expected_keys = [
        "loss/total", "loss/severity", "loss/comp_frac",
        "loss/oop", "loss/thick_var", "loss/physics",
    ]
    for key in expected_keys:
        assert key in log, f"Missing log key: {key}"
        val = log[key]
        assert torch.isfinite(torch.tensor(float(val))), f"{key} is not finite: {val}"
        assert float(val) >= 0, f"{key} is negative: {val}"


def test_extract_single_simulation_applies_normalization(first_sim_id: str) -> None:
    """extract_single_simulation must apply the same z-score normalization as
    WrinkleDataset(normalize=True).  Omitting normalization produces features
    far outside [−10, 10] and breaks model predictions (train/serve skew).

    Regression test for the normalization fix in extract_single.py.
    """
    _skip_if_no_data()
    import numpy as np
    from wp3_features.extract_single import extract_single_simulation

    results_dir = f"/tmp/{first_sim_id}"  # HDF5 fast-path — dir need not exist
    normed = extract_single_simulation(results_dir, sim_id=first_sim_id,
                                       wp3_h5_path=str(_WP3_H5), normalize=True)
    raw = extract_single_simulation(results_dir, sim_id=first_sim_id,
                                    wp3_h5_path=str(_WP3_H5), normalize=False)

    nf_normed = normed["node_features"]
    nf_raw = raw["node_features"]

    # Normalized features must have smaller magnitude than raw
    assert nf_normed.std() < nf_raw.std(), (
        "Normalized features have larger std than raw — normalization may not be applied"
    )
    # Normalized mean should be close to 0 (dataset-wide z-score)
    assert abs(float(nf_normed.mean())) < 1.0, (
        f"Normalized feature mean too large: {nf_normed.mean():.4f} — expected ~0"
    )
    # Raw features must differ meaningfully from normalized
    assert not np.allclose(nf_normed, nf_raw, atol=1e-3), (
        "Normalized and raw features are identical — normalization is not being applied"
    )

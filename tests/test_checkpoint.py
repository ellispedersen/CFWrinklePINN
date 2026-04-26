"""Checkpoint save/load round-trip tests.

Verifies that saving and loading a model checkpoint produces identical predictions
and that the checkpoint format matches what train.py writes.
"""
from __future__ import annotations

import pytest
torch = pytest.importorskip("torch")
optim = pytest.importorskip("torch.optim")

from model.contracts import (
    CHECKPOINT_FORMAT_VERSION,
    MODEL_CONTRACT_VERSION,
    MODEL_TYPE_COARSE,
    COARSE_TARGET_DATASET_NAMES,
    validate_checkpoint_compatibility,
)
from model.gnn import FormingGraphNet
from model.labels import COARSE_TARGET_CHANNELS, FINE_TARGET_CHANNELS
from wp3_features.physics import FEATURE_NAMES


def _fake_batch() -> dict:
    torch.manual_seed(99)
    n, t, m, e = 100, 16, 80, 200
    elements = torch.randint(0, n, (m, 3), dtype=torch.int64)
    return {
        "node_features": torch.randn(t, n, 74),
        "edge_index": torch.randint(0, n, (2, e), dtype=torch.int64),
        "edge_attr": torch.randn(e, 4),
        "material_card": torch.randn(8),
        "elements": elements,
        "targets": torch.rand(t, m, 4),
    }


def _make_checkpoint(model: FormingGraphNet, model_cfg: dict, val_loss: float = 0.5) -> dict:
    """Build a checkpoint dict matching the format written by training/train.py line 161-171."""
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    return {
        "epoch": 5,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "val_metrics": {"detection_rate": 0.6, "f1": 0.5},
        "model_config": model_cfg,
        "model_type": MODEL_TYPE_COARSE,
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "model_version": MODEL_CONTRACT_VERSION,
        "contract": {
            "model_family": "cfwrinkle-pinn",
            "model_contract_version": MODEL_CONTRACT_VERSION,
            "model_type": MODEL_TYPE_COARSE,
            "coarse_target_channels": list(COARSE_TARGET_CHANNELS),
            "fine_target_channels": list(FINE_TARGET_CHANNELS),
            "dataset_target_names": list(COARSE_TARGET_DATASET_NAMES),
            "feature_names": list(FEATURE_NAMES),
        },
    }


def test_checkpoint_save_load_roundtrip(tmp_path: pytest.TempPathFactory) -> None:
    """Save model, load into new instance, predictions must match."""
    torch.manual_seed(42)
    model_cfg = {"hidden_dim": 32}
    model1 = FormingGraphNet(**model_cfg)
    model1.eval()
    batch = _fake_batch()

    with torch.no_grad():
        pred_before = model1(batch).clone()

    ckpt = _make_checkpoint(model1, model_cfg)
    ckpt_path = tmp_path / "test.pt"
    torch.save(ckpt, ckpt_path)

    loaded = torch.load(ckpt_path, weights_only=False)
    model2 = FormingGraphNet(**loaded["model_config"])
    model2.load_state_dict(loaded["model_state_dict"])
    model2.eval()

    with torch.no_grad():
        pred_after = model2(batch)

    assert torch.allclose(pred_before, pred_after, atol=1e-6), (
        "Predictions changed after checkpoint save/load. "
        f"Max diff: {(pred_before - pred_after).abs().max():.2e}"
    )


def test_checkpoint_contains_required_keys(tmp_path: pytest.TempPathFactory) -> None:
    """Checkpoint must contain all keys expected by inference and resumption code."""
    torch.manual_seed(43)
    model_cfg = {"hidden_dim": 32}
    model = FormingGraphNet(**model_cfg)
    ckpt = _make_checkpoint(model, model_cfg)
    ckpt_path = tmp_path / "keys_test.pt"
    torch.save(ckpt, ckpt_path)

    loaded = torch.load(ckpt_path, weights_only=False)
    required = {"epoch", "model_state_dict", "optimizer_state_dict",
                "val_loss", "val_metrics", "model_config", "model_type",
                "checkpoint_format_version", "model_version", "contract"}
    missing = required - set(loaded.keys())
    assert not missing, f"Checkpoint missing required keys: {missing}"

    assert loaded["model_config"] == model_cfg
    assert isinstance(loaded["epoch"], int)
    assert isinstance(loaded["val_loss"], float)
    validate_checkpoint_compatibility(loaded, expected_model_type=MODEL_TYPE_COARSE, require_contract_fields=True)


def test_checkpoint_mismatched_config_raises(tmp_path: pytest.TempPathFactory) -> None:
    """Loading a hidden_dim=32 checkpoint into a hidden_dim=64 model must fail
    with a clear size mismatch error."""
    torch.manual_seed(44)
    model32 = FormingGraphNet(hidden_dim=32)
    ckpt = _make_checkpoint(model32, {"hidden_dim": 32})
    ckpt_path = tmp_path / "mismatch.pt"
    torch.save(ckpt, ckpt_path)

    loaded = torch.load(ckpt_path, weights_only=False)
    model64 = FormingGraphNet(hidden_dim=64)
    with pytest.raises(RuntimeError):
        model64.load_state_dict(loaded["model_state_dict"])

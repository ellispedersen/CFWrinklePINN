from __future__ import annotations

import os

import torch

from model.gnn import FormingGraphNet
from model.loss import wrinkle_loss

T, N, M, E = 32, 200, 150, 500  # T=32 matches default --max-timesteps to avoid OOM in CI


def _synthetic_batch() -> dict:
    elements = torch.randint(0, N, (M, 3), dtype=torch.int64)
    return {
        "sim_id": "synthetic_test",
        "node_features": torch.randn(T, N, 74),
        "edge_index": torch.randint(0, N, (2, E), dtype=torch.int64),
        "edge_attr": torch.randn(E, 4),
        "material_card": torch.randn(8),
        "elements": elements,
        "targets": torch.rand(T, M, 4),
    }


def test_full_training_step() -> None:
    use_gpu = os.environ.get("WP7_SMOKE_USE_GPU", "0") == "1"
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    model = FormingGraphNet(hidden_dim=32).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in _synthetic_batch().items()}

    model.train()
    optimizer.zero_grad()
    pred = model(batch)
    assert pred.shape == (T, M, 4), f"Expected ({T}, {M}, 4), got {pred.shape}"

    loss, _ = wrinkle_loss(pred, batch["targets"])
    assert not torch.isnan(loss)
    assert torch.isfinite(loss)

    loss.backward()
    optimizer.step()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No grad: {name}"


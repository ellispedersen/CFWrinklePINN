"""Verify the model actually learns — loss must decrease over multiple gradient steps.

These tests use synthetic data (no HDF5 dependency) and run on CPU in seconds.
They catch the most critical class of failure: code runs but nothing is learned.
"""
from __future__ import annotations

import torch
import torch.optim as optim

from model.gnn import FormingGraphNet
from model.loss import wrinkle_loss


def _fixed_batch(seed: int = 42) -> dict:
    """Synthetic batch with fixed seed for reproducible learning curves."""
    g = torch.Generator()
    g.manual_seed(seed)
    n, t, m, e = 100, 16, 80, 200
    elements = torch.randint(0, n, (m, 3), dtype=torch.int64, generator=g)
    return {
        "node_features": torch.randn(t, n, 74, generator=g),
        "edge_index": torch.randint(0, n, (2, e), dtype=torch.int64, generator=g),
        "edge_attr": torch.randn(e, 4, generator=g),
        "material_card": torch.randn(8, generator=g),
        "elements": elements,
        "targets": torch.rand(t, m, 4, generator=g),
    }


def _run_steps(model: FormingGraphNet, batch: dict, n_steps: int = 20) -> list[dict]:
    """Run n_steps of forward-backward-optimizer and return log dicts."""
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    logs: list[dict] = []
    for _ in range(n_steps):
        optimizer.zero_grad()
        pred = model(batch)
        grad_total, log = wrinkle_loss(pred, batch["targets"])
        grad_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        logs.append(log)
    return logs


def test_loss_decreases_over_20_steps() -> None:
    """Total loss must drop by at least 50% when overfitting a single synthetic sample."""
    torch.manual_seed(0)
    model = FormingGraphNet(hidden_dim=32)
    model.train()
    batch = _fixed_batch()

    logs = _run_steps(model, batch, n_steps=20)

    loss_initial = logs[0]["loss/total"]
    loss_final = logs[-1]["loss/total"]

    assert torch.isfinite(torch.tensor(loss_initial)), "Initial loss is not finite"
    assert torch.isfinite(torch.tensor(loss_final)), "Final loss is not finite"
    assert loss_final < loss_initial * 0.5, (
        f"Loss did not decrease enough: initial={loss_initial:.4f}, "
        f"final={loss_final:.4f} (expected < {loss_initial * 0.5:.4f})"
    )


def test_severity_channel_improves() -> None:
    """Severity loss must decrease — catches gradient balancing failures where
    the total drops but the primary wrinkle target doesn't learn."""
    torch.manual_seed(1)
    model = FormingGraphNet(hidden_dim=32)
    model.train()
    batch = _fixed_batch(seed=1)

    logs = _run_steps(model, batch, n_steps=20)

    sev_initial = logs[0]["loss/severity"]
    sev_final = logs[-1]["loss/severity"]

    assert sev_final < sev_initial, (
        f"Severity loss did not decrease: initial={sev_initial:.4f}, "
        f"final={sev_final:.4f}"
    )


def test_loss_components_all_finite() -> None:
    """All loss components must be finite and non-negative after one step."""
    torch.manual_seed(2)
    model = FormingGraphNet(hidden_dim=32)
    model.train()
    batch = _fixed_batch(seed=2)

    pred = model(batch)
    _, log = wrinkle_loss(pred, batch["targets"])

    expected_keys = ["loss/total", "loss/severity", "loss/comp_frac",
                     "loss/oop", "loss/thick_var", "loss/physics"]
    for key in expected_keys:
        assert key in log, f"Missing log key: {key}"
        val = log[key]
        assert torch.isfinite(torch.tensor(val)), f"{key} is not finite: {val}"
        assert val >= 0, f"{key} is negative: {val}"


def test_grad_total_differs_from_raw_total() -> None:
    """grad_total (for backprop) must differ from raw_total (for logging).
    If they are equal, the self-normalisation is broken and early stopping
    will see a constant val_loss of sum(weights)=7.5."""
    torch.manual_seed(3)
    model = FormingGraphNet(hidden_dim=32)
    batch = _fixed_batch(seed=3)

    pred = model(batch)
    grad_total, log = wrinkle_loss(pred, batch["targets"])

    raw_total = log["loss/total"]
    grad_val = float(grad_total.detach())

    # grad_total is bounded by sum(weights)=7.5 when _safe_norm floors tiny losses.
    # raw_total should remain different (actual weighted sum of raw losses).
    assert 0.0 < grad_val <= 7.5 + 1e-6, (
        f"grad_total should be positive and <=7.5, got {grad_val:.4f}. "
        "Self-normalisation may be broken."
    )
    assert abs(raw_total - 7.5) > 0.1, (
        f"raw_total is also ~7.5 ({raw_total:.4f}). The log is using the "
        "normalised value — early stopping will not work."
    )

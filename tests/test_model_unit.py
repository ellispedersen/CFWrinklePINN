from __future__ import annotations

import torch

from model.gnn import FormingGraphNet
from model.layers import MaterialEncoder, MessagePassingLayer, TemporalAggregator
from model.loss import wrinkle_loss

N, T, M = 100, 16, 80
E = 200
HIDDEN = 32


def _fake_batch(n: int = N, t: int = T, m: int = M, e: int = E) -> dict:
    elements = torch.randint(0, n, (m, 3), dtype=torch.int64)
    return {
        "node_features": torch.randn(t, n, 74),
        "edge_index": torch.randint(0, n, (2, e), dtype=torch.int64),
        "edge_attr": torch.randn(e, 4),
        "material_card": torch.randn(8),
        "elements": elements,
        "targets": torch.rand(t, m, 4),
    }


def test_material_encoder_shape() -> None:
    enc = MaterialEncoder(8, 32)
    out = enc(torch.randn(8))
    assert out.shape == (32,)


def test_message_passing_shape() -> None:
    layer = MessagePassingLayer(HIDDEN, edge_dim=4, material_dim=16)
    h = torch.randn(N, HIDDEN)
    ei = torch.randint(0, N, (2, E), dtype=torch.int64)
    ea = torch.randn(E, 4)
    mat = torch.randn(16)
    out = layer(h, ei, ea, mat)
    assert out.shape == (N, HIDDEN)


def test_temporal_aggregator_shape() -> None:
    agg = TemporalAggregator(HIDDEN, n_heads=2, dropout=0.0)
    x = torch.randn(T, N, HIDDEN)
    out = agg(x)
    assert out.shape == (N, HIDDEN)


def test_forming_graph_net_output_shape() -> None:
    model = FormingGraphNet(hidden_dim=HIDDEN)
    batch = _fake_batch()
    pred = model(batch)
    assert pred.shape == (T, M, 4)


def test_loss_no_nan_with_clean_targets() -> None:
    pred = torch.rand(T, M, 4)
    tgt = torch.rand(T, M, 4)
    loss, _ = wrinkle_loss(pred, tgt)
    assert not torch.isnan(loss)
    assert loss > 0


def test_loss_with_nan_targets() -> None:
    pred = torch.rand(T, M, 4)
    tgt = torch.rand(T, M, 4)
    tgt[:, :5, :] = float("nan")
    loss, _ = wrinkle_loss(pred, tgt)
    assert not torch.isnan(loss)


def test_gradient_flows() -> None:
    model = FormingGraphNet(hidden_dim=HIDDEN)
    batch = _fake_batch()
    pred = model(batch)
    tgt = torch.rand_like(pred)
    loss, _ = wrinkle_loss(pred, tgt)
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"


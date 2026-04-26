from __future__ import annotations

import pytest
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


def test_forming_graph_net_rejects_nonstandard_target_count() -> None:
    with pytest.raises(ValueError, match="n_targets=4"):
        FormingGraphNet(hidden_dim=HIDDEN, n_targets=3)


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


# ---------------------------------------------------------------------------
# Realistic-scale tests: exercise chunking code paths
# ---------------------------------------------------------------------------

def test_attention_batching_triggered() -> None:
    """N=600 > attn_batch_nodes=512 forces the chunked attention path
    in TemporalAggregator.forward() (layers.py:84-90)."""
    model = FormingGraphNet(hidden_dim=HIDDEN, attn_batch_nodes=512)
    batch = _fake_batch(n=600, t=16, m=400, e=1200)
    pred = model(batch)
    assert pred.shape == (16, 400, 4)
    assert not torch.isnan(pred).any(), "NaN in output with attention batching"


def test_decoder_chunking_triggered() -> None:
    """T=48 > decoder_chunk_t=32 forces the GRU decoder to run in 2 chunks
    with hidden state threaded between them (gnn.py:67-74)."""
    model = FormingGraphNet(hidden_dim=HIDDEN, decoder_chunk_t=32)
    batch = _fake_batch(n=100, t=48, m=80, e=200)
    pred = model(batch)
    assert pred.shape == (48, 80, 4)
    assert not torch.isnan(pred).any(), "NaN in output with decoder chunking"


def test_gradient_checkpointing_output_matches() -> None:
    """use_checkpoint=True must produce identical outputs to use_checkpoint=False.
    Uses eval() to disable dropout so results are deterministic and comparable."""
    torch.manual_seed(42)
    batch = _fake_batch(n=600, t=48, m=400, e=1200)  # triggers both chunking paths

    torch.manual_seed(0)
    model_ckpt = FormingGraphNet(hidden_dim=HIDDEN, attn_batch_nodes=512,
                                 decoder_chunk_t=32, use_checkpoint=True)
    model_no_ckpt = FormingGraphNet(hidden_dim=HIDDEN, attn_batch_nodes=512,
                                    decoder_chunk_t=32, use_checkpoint=False)
    model_no_ckpt.load_state_dict(model_ckpt.state_dict())

    model_ckpt.eval()
    model_no_ckpt.eval()
    with torch.no_grad():
        out_ckpt = model_ckpt(batch)
        out_no_ckpt = model_no_ckpt(batch)

    assert torch.allclose(out_ckpt, out_no_ckpt, atol=1e-5), (
        f"Max diff: {(out_ckpt - out_no_ckpt).abs().max():.2e}"
    )


def test_gradient_checkpointing_gradients_flow() -> None:
    """Gradients must flow through checkpointed paths without NaN."""
    model = FormingGraphNet(hidden_dim=HIDDEN, attn_batch_nodes=512,
                            decoder_chunk_t=32, use_checkpoint=True)
    batch = _fake_batch(n=600, t=48, m=400, e=1200)
    pred = model(batch)
    loss, _ = wrinkle_loss(pred, batch["targets"])
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"


@pytest.mark.slow
def test_production_scale_forward() -> None:
    """Near-real mesh sizes: N=3700, E=20000, T=64, M=3000.
    Forward pass only (backward on CPU would be very slow).
    Verifies no shape errors, no NaN, and that chunking paths scale correctly."""
    model = FormingGraphNet(hidden_dim=64, attn_batch_nodes=512, decoder_chunk_t=32)
    batch = _fake_batch(n=3700, t=64, m=3000, e=20000)
    with torch.no_grad():
        pred = model(batch)
    assert pred.shape == (64, 3000, 4), f"Unexpected shape: {pred.shape}"
    assert not torch.isnan(pred).any(), "NaN in production-scale forward pass"
    assert torch.isfinite(pred).all(), "Non-finite values in production-scale output"


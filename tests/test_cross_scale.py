from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from model.cross_scale import CrossScaleNet
from model.loss import buckling_onset_loss, cross_scale_loss, thickness_dz_coupling_loss

N, T, M, E = 80, 12, 40, 160
HIDDEN = 32
N_FINE_NODES, N_FINE_ELEM = 320, 200
N_MAPPING = 160  # must be <= N_FINE_ELEM


def _fake_batch(
    n: int = N,
    t: int = T,
    m: int = M,
    e: int = E,
    n_fine_nodes: int = N_FINE_NODES,
    n_fine_elem: int = N_FINE_ELEM,
    n_mapping: int = N_MAPPING,
) -> dict:
    torch.manual_seed(42)
    return {
        "node_features": torch.randn(t, n, 74),
        "edge_index": torch.randint(0, n, (2, e), dtype=torch.int64),
        "edge_attr": torch.randn(e, 4),
        "material_card": torch.randn(8),
        "elements": torch.randint(0, n, (m, 3), dtype=torch.int64),
        # coarse_to_fine: row 0 = coarse elem idx, row 1 = fine elem idx (unique)
        "coarse_to_fine": torch.stack([
            torch.randint(0, m, (n_mapping,), dtype=torch.int64),
            torch.arange(n_mapping, dtype=torch.int64),
        ]),
        "fine_elements": torch.randint(0, n_fine_nodes, (n_fine_elem, 3), dtype=torch.int64),
    }


def _with_fine_mp_fields(batch: dict) -> dict:
    n_fine_nodes = int(batch["fine_elements"].max().item()) + 1
    n_fine_nodes = max(n_fine_nodes, N_FINE_NODES)
    n_fine_edges = max(10, n_fine_nodes * 2)
    batch["fine_node_coarse_map"] = torch.randint(-1, M, (n_fine_nodes,), dtype=torch.int64)
    batch["fine_edge_index"] = torch.randint(0, n_fine_nodes, (2, n_fine_edges), dtype=torch.int64)
    batch["fine_edge_attr"] = torch.randn(n_fine_edges, 4)
    return batch


def test_cross_scale_net_output_shapes() -> None:
    model = CrossScaleNet(hidden_dim=HIDDEN)
    batch = _fake_batch()
    out = model(batch)
    assert "coarse" in out and "fine" in out
    assert out["coarse"].shape == (T, M, 4)
    assert out["fine"].shape == (T, N_FINE_ELEM, 4)


def test_cross_scale_net_rejects_nonstandard_target_count() -> None:
    with pytest.raises(ValueError, match="n_targets=4"):
        CrossScaleNet(hidden_dim=HIDDEN, n_targets=5)


def test_no_nan_in_output() -> None:
    model = CrossScaleNet(hidden_dim=HIDDEN)
    batch = _fake_batch()
    out = model(batch)
    assert not torch.isnan(out["coarse"]).any(), "NaN in coarse output"
    assert not torch.isnan(out["fine"]).any(), "NaN in fine output"


def test_cross_scale_loss_no_nan() -> None:
    model = CrossScaleNet(hidden_dim=HIDDEN)
    batch = _fake_batch()
    out = model(batch)
    fine_features = torch.randn(T, N_FINE_NODES, 4)
    coarse_targets = torch.rand(T, M, 4)
    loss, log = cross_scale_loss(
        out["fine"], fine_features, batch["fine_elements"], out["coarse"], coarse_targets
    )
    assert not torch.isnan(loss), "NaN cross_scale_loss"
    assert loss > 0
    assert "loss/fine_stress_1" in log


def test_cross_scale_loss_with_nan_targets() -> None:
    model = CrossScaleNet(hidden_dim=HIDDEN)
    batch = _fake_batch()
    out = model(batch)
    fine_features = torch.randn(T, N_FINE_NODES, 4)
    fine_features[:, :10, :] = float("nan")
    coarse_targets = torch.rand(T, M, 4)
    loss, _ = cross_scale_loss(
        out["fine"], fine_features, batch["fine_elements"], out["coarse"], coarse_targets
    )
    assert not torch.isnan(loss)


def test_gradient_flows() -> None:
    model = CrossScaleNet(hidden_dim=HIDDEN)
    batch = _fake_batch()
    out = model(batch)
    fine_features = torch.randn(T, N_FINE_NODES, 4)
    coarse_targets = torch.rand(T, M, 4)
    loss, _ = cross_scale_loss(
        out["fine"], fine_features, batch["fine_elements"], out["coarse"], coarse_targets
    )
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"


def test_coarse_output_compatible_with_wrinkle_loss() -> None:
    """CrossScaleNet coarse head must be drop-in compatible with wrinkle_loss."""
    from model.loss import wrinkle_loss
    model = CrossScaleNet(hidden_dim=HIDDEN)
    batch = _fake_batch()
    out = model(batch)
    targets = torch.rand(T, M, 4)
    loss, log = wrinkle_loss(out["coarse"], targets)
    assert not torch.isnan(loss)
    assert "loss/severity" in log


def test_cross_scale_scatter_coverage() -> None:
    """Fine elements not covered by the mapping get zero embeddings — output must still be finite."""
    model = CrossScaleNet(hidden_dim=HIDDEN)
    # Only map half the fine elements
    partial_mapping = N_FINE_ELEM // 2
    batch = _fake_batch(n_mapping=partial_mapping)
    out = model(batch)
    assert torch.isfinite(out["fine"]).all(), "Non-finite values for unmapped fine elements"


def test_cross_scale_scatter_autocast_dtype_compatibility() -> None:
    model = CrossScaleNet(hidden_dim=HIDDEN).eval()
    batch = _fake_batch(t=4, n=32, m=16, e=64, n_fine_nodes=96, n_fine_elem=64, n_mapping=48)
    with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = model(batch)
    assert out["coarse"].shape == (4, 16, 4)
    assert out["fine"].shape == (4, 64, 4)
    assert torch.isfinite(out["fine"].float()).all()


def test_cross_scale_handles_duplicate_fine_mapping_indices() -> None:
    model = CrossScaleNet(hidden_dim=HIDDEN)
    batch = _fake_batch(n_mapping=8)
    batch["coarse_to_fine"][1] = torch.tensor([0, 1, 2, 3, 3, 4, 5, 6], dtype=torch.int64)
    out = model(batch)
    assert out["fine"].shape == (T, N_FINE_ELEM, 4)
    assert torch.isfinite(out["fine"]).all()


def test_cross_scale_duplicate_fine_mapping_uses_mean_aggregation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = CrossScaleNet(hidden_dim=4, n_mp_steps=0, use_checkpoint=False)
    model.fine_head = nn.Identity()
    batch = _fake_batch(t=1, n=8, m=3, e=8, n_fine_nodes=24, n_fine_elem=6, n_mapping=3)
    batch["coarse_to_fine"] = torch.tensor(
        [
            [0, 1, 2],
            [0, 0, 1],
        ],
        dtype=torch.int64,
    )

    def fake_decode_chunk(
        chunk_in: torch.Tensor,
        gru_hidden: torch.Tensor,
        coarse_node_idx: torch.Tensor,
        m_coarse: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del coarse_node_idx
        chunk_t = int(chunk_in.shape[1])
        coarse_embed = torch.tensor(
            [
                [2.0, 4.0, 6.0, 8.0],
                [10.0, 12.0, 14.0, 16.0],
                [1.0, 3.0, 5.0, 7.0],
            ],
            dtype=chunk_in.dtype,
            device=chunk_in.device,
        )
        return coarse_embed.unsqueeze(0).expand(chunk_t, m_coarse, -1).clone(), gru_hidden

    monkeypatch.setattr(model, "_decode_chunk", fake_decode_chunk)

    out = model(batch)
    assert torch.allclose(out["fine"][0, 0], torch.tensor([6.0, 8.0, 10.0, 12.0]), atol=1e-6)
    assert torch.allclose(out["fine"][0, 1], torch.tensor([1.0, 3.0, 5.0, 7.0]), atol=1e-6)
    assert torch.allclose(out["fine"][0, 2:], torch.zeros(4, 4), atol=1e-6)


def test_attention_batching_triggered() -> None:
    """n=600 > attn_batch_nodes=512 forces chunked attention in CrossScaleNet."""
    model = CrossScaleNet(hidden_dim=HIDDEN, attn_batch_nodes=512)
    batch = _fake_batch(n=600, t=12, m=300, e=1200, n_fine_nodes=1200, n_fine_elem=800, n_mapping=600)
    out = model(batch)
    assert out["coarse"].shape == (12, 300, 4)
    assert not torch.isnan(out["coarse"]).any()
    assert not torch.isnan(out["fine"]).any()


def test_decoder_chunking_triggered() -> None:
    """T=48 > decoder_chunk_t=32 forces multi-chunk GRU decode."""
    model = CrossScaleNet(hidden_dim=HIDDEN, decoder_chunk_t=32)
    batch = _fake_batch(t=48)
    out = model(batch)
    assert out["coarse"].shape == (48, M, 4)
    assert out["fine"].shape == (48, N_FINE_ELEM, 4)
    assert not torch.isnan(out["fine"]).any()


def test_gradient_checkpointing_output_matches() -> None:
    """use_checkpoint=True must produce identical outputs to use_checkpoint=False (in eval mode)."""
    torch.manual_seed(0)
    batch = _fake_batch(n=600, t=48, n_fine_nodes=1200, n_fine_elem=800, n_mapping=600)

    model_ckpt = CrossScaleNet(hidden_dim=HIDDEN, attn_batch_nodes=512, decoder_chunk_t=32,
                                use_checkpoint=True)
    model_no_ckpt = CrossScaleNet(hidden_dim=HIDDEN, attn_batch_nodes=512, decoder_chunk_t=32,
                                   use_checkpoint=False)
    model_no_ckpt.load_state_dict(model_ckpt.state_dict())

    model_ckpt.eval()
    model_no_ckpt.eval()
    with torch.no_grad():
        out_ckpt = model_ckpt(batch)
        out_no_ckpt = model_no_ckpt(batch)

    for key in ("coarse", "fine"):
        max_diff = (out_ckpt[key] - out_no_ckpt[key]).abs().max()
        assert torch.allclose(out_ckpt[key], out_no_ckpt[key], atol=1e-5), (
            f"Checkpoint mismatch in {key}: max_diff={max_diff:.2e}"
        )


def test_gradient_checkpointing_gradients_flow() -> None:
    """Gradients must flow through all checkpointed paths."""
    model = CrossScaleNet(hidden_dim=HIDDEN, attn_batch_nodes=512, decoder_chunk_t=32,
                          use_checkpoint=True)
    batch = _fake_batch(n=600, t=48, n_fine_nodes=1200, n_fine_elem=800, n_mapping=600)
    out = model(batch)
    fine_features = torch.randn(48, 1200, 4)
    coarse_targets = torch.rand(48, M, 4)
    loss, _ = cross_scale_loss(out["fine"], fine_features, batch["fine_elements"],
                                out["coarse"], coarse_targets)
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"


def test_buckling_onset_loss_no_nan() -> None:
    t, m = 16, 100
    fine_pred = torch.randn(t, m, 4)
    fine_tgt = torch.randn(t, m, 4)
    loss = buckling_onset_loss(fine_pred, fine_tgt)
    assert not torch.isnan(loss)
    assert loss >= 0


def test_thickness_dz_coupling_loss_no_nan() -> None:
    loss = thickness_dz_coupling_loss(torch.randn(16, 100, 4))
    assert not torch.isnan(loss)
    assert loss >= 0


def test_cross_scale_loss_with_physics_losses() -> None:
    t, m, nf, ne = 16, 40, 160, 200
    fine_pred = torch.randn(t, ne, 4)
    fine_features = torch.randn(t, nf, 4)
    fine_elements = torch.randint(0, nf, (ne, 3))
    coarse_pred = torch.rand(t, m, 4)
    coarse_targets = torch.rand(t, m, 4)
    loss, log = cross_scale_loss(fine_pred, fine_features, fine_elements, coarse_pred, coarse_targets)
    assert not torch.isnan(loss)
    assert "loss/fine_buckling" in log
    assert "loss/fine_dz_mono" in log
    assert "loss/fine_coupling" in log


def test_fine_mp_output_shapes() -> None:
    model = CrossScaleNet(hidden_dim=HIDDEN, use_fine_mp=True, fine_mp_steps=2)
    batch = _with_fine_mp_fields(_fake_batch())
    out = model(batch)
    assert out["fine"].shape == (T, N_FINE_ELEM, 4)
    assert not torch.isnan(out["fine"]).any()


def test_fine_mp_gradients_flow() -> None:
    model = CrossScaleNet(hidden_dim=HIDDEN, use_fine_mp=True)
    batch = _with_fine_mp_fields(_fake_batch())
    out = model(batch)
    fine_features = torch.randn(T, N_FINE_NODES, 4)
    coarse_targets = torch.rand(T, M, 4)
    loss, _ = cross_scale_loss(
        out["fine"],
        fine_features,
        batch["fine_elements"],
        out["coarse"],
        coarse_targets,
    )
    loss.backward()
    for name, p in model.named_parameters():
        if "fine_mp" in name:
            assert p.grad is not None, f"No gradient for fine MP param {name}"


def test_phase1_phase2_output_consistent_without_fine_mp() -> None:
    torch.manual_seed(0)
    batch = _with_fine_mp_fields(_fake_batch())
    m1 = CrossScaleNet(hidden_dim=HIDDEN, use_fine_mp=False)
    m2 = CrossScaleNet(hidden_dim=HIDDEN, use_fine_mp=True, fine_mp_steps=0)
    m2.load_state_dict(m1.state_dict())
    m1.eval()
    m2.eval()
    with torch.no_grad():
        o1, o2 = m1(batch), m2(batch)
    assert torch.allclose(o1["coarse"], o2["coarse"], atol=1e-6)

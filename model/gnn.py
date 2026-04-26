from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as _ckpt

from .layers import MaterialEncoder, MessagePassingLayer, TemporalAggregator
from .labels import COARSE_TARGET_INDEX, N_COARSE_TARGETS


class FormingGraphNet(nn.Module):
    def __init__(
        self,
        in_dim: int = 74,
        hidden_dim: int = 64,
        material_dim: int = 32,
        n_mp_steps: int = 3,
        n_heads: int = 2,
        dropout: float = 0.3,
        attn_batch_nodes: int = 512,
        n_targets: int = 4,
        decoder_chunk_t: int = 32,
        use_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        if int(n_targets) != N_COARSE_TARGETS:
            raise ValueError(
                f"FormingGraphNet requires n_targets={N_COARSE_TARGETS} for coarse label mapping, got {n_targets}"
            )
        self.node_encoder = nn.Linear(in_dim, hidden_dim)
        self.material_encoder = MaterialEncoder(in_dim=8, out_dim=material_dim)
        self.temporal = TemporalAggregator(
            hidden_dim,
            n_heads=n_heads,
            dropout=dropout,
            attn_batch_nodes=attn_batch_nodes,
            use_checkpoint=use_checkpoint,
        )
        self.mp_layers = nn.ModuleList(
            [MessagePassingLayer(hidden_dim, edge_dim=4, material_dim=material_dim) for _ in range(n_mp_steps)]
        )
        self.timestep_decoder = nn.GRU(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.decoder_chunk_t = max(1, int(decoder_chunk_t))
        self.use_checkpoint = use_checkpoint
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, n_targets),
        )

    def _decode_chunk(
        self,
        chunk_in: torch.Tensor,
        gru_hidden: torch.Tensor,
        node_idx: torch.Tensor,
        m: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One GRU decoder chunk + element aggregation + output head. Extracted for gradient checkpointing."""
        chunk_dec, new_hidden = self.timestep_decoder(chunk_in, gru_hidden)
        chunk_dec = chunk_dec.permute(1, 0, 2)  # (chunk_t, N, D)
        h_corner = chunk_dec[:, node_idx, :].reshape(chunk_dec.shape[0], m, 3, -1).mean(dim=2)
        return self.output_head(h_corner), new_hidden

    def forward(self, batch: dict) -> torch.Tensor:
        x = batch["node_features"]
        edge_index = batch["edge_index"]
        edge_attr = batch["edge_attr"]
        material = batch["material_card"]
        elements = batch["elements"]

        t, n, _ = x.shape
        m = int(elements.shape[0])

        mat_embed = self.material_encoder(material)
        x_enc = self.node_encoder(x)
        h = self.temporal(x_enc)
        for mp in self.mp_layers:
            h = mp(h, edge_index, edge_attr, mat_embed)

        # Temporal decoder: add MP context to each timestep then decode with GRU.
        # Chunked over T to bound peak activation memory; GRU hidden state is
        # threaded between chunks so temporal continuity is preserved.
        h_exp = h.unsqueeze(0).expand(t, -1, -1)  # (T, N, D) — broadcast, no copy
        h_seq = x_enc + h_exp                      # (T, N, D)

        node_idx = elements.reshape(-1)  # (M*3,)
        # Initialise as zeros so it is always a tensor (required for gradient checkpointing).
        gru_hidden = torch.zeros(1, n, self.timestep_decoder.hidden_size,
                                 device=x.device, dtype=x.dtype)

        # Disable checkpointing during inference (no_grad → requires_grad=False).
        use_ckpt = self.use_checkpoint and h_seq.requires_grad
        out_chunks: list[torch.Tensor] = []
        for t_start in range(0, t, self.decoder_chunk_t):
            t_end = min(t_start + self.decoder_chunk_t, t)
            chunk_in = h_seq[t_start:t_end].permute(1, 0, 2)   # (N, chunk, D)
            if use_ckpt:
                out_chunk, gru_hidden = _ckpt(
                    self._decode_chunk, chunk_in, gru_hidden, node_idx, m,
                    use_reentrant=False,
                )
            else:
                out_chunk, gru_hidden = self._decode_chunk(chunk_in, gru_hidden, node_idx, m)
            out_chunks.append(out_chunk)
        return torch.cat(out_chunks, dim=0)  # (T, M, n_targets)

    def predict_severity(self, batch: dict) -> torch.Tensor:
        return self.forward(batch)[..., COARSE_TARGET_INDEX["severity"]]


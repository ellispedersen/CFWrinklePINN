from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as _ckpt


class MaterialEncoder(nn.Module):
    def __init__(self, in_dim: int = 8, out_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.SiLU(),
            nn.Linear(64, out_dim),
        )

    def forward(self, material_card: torch.Tensor) -> torch.Tensor:
        return self.net(material_card)


class MessagePassingLayer(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int = 4, material_dim: int = 32) -> None:
        super().__init__()
        msg_in = hidden_dim * 2 + edge_dim + material_dim
        self.message_fn = nn.Sequential(
            nn.Linear(msg_in, hidden_dim),
            nn.SiLU(),
        )
        self.update_fn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    @torch.compiler.disable
    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        material_embed: torch.Tensor,
    ) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        n_nodes = h.shape[0]
        mat = material_embed.unsqueeze(0).expand(src.shape[0], -1)
        msg_input = torch.cat([h[src], h[dst], edge_attr, mat], dim=-1)
        messages = self.message_fn(msg_input)
        if messages.dtype != h.dtype:
            messages = messages.to(dtype=h.dtype)

        agg = torch.zeros(n_nodes, messages.shape[-1], device=h.device, dtype=h.dtype)
        idx = dst.unsqueeze(1).expand_as(messages)
        agg.scatter_add_(0, idx, messages)

        h_new = self.update_fn(torch.cat([h, agg], dim=-1))
        return self.norm(h + h_new)


class TemporalAggregator(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        n_heads: int = 2,
        dropout: float = 0.3,
        attn_batch_nodes: int = 512,
        use_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attn_batch_nodes = max(1, int(attn_batch_nodes))
        self.use_checkpoint = use_checkpoint
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def _forward_chunk(self, x_chunk: torch.Tensor) -> torch.Tensor:
        """GRU + self-attention for one node chunk. Extracted for gradient checkpointing."""
        gru_chunk, _ = self.gru(x_chunk)
        attn_chunk, _ = self.attn(gru_chunk, gru_chunk, gru_chunk)
        return self.dropout(attn_chunk)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t, n, d = x.shape
        x_flat = x.permute(1, 0, 2)
        x_flat = x_flat + self._make_sinusoidal(t, d, x.device, x.dtype).unsqueeze(0).expand(n, -1, -1)

        # Disable checkpointing during inference (no_grad context → requires_grad=False).
        use_ckpt = self.use_checkpoint and x_flat.requires_grad

        if n <= self.attn_batch_nodes:
            if use_ckpt:
                return _ckpt(self._forward_chunk, x_flat, use_reentrant=False).mean(dim=1)
            return self._forward_chunk(x_flat).mean(dim=1)

        chunks: list[torch.Tensor] = []
        for start in range(0, n, self.attn_batch_nodes):
            end = min(start + self.attn_batch_nodes, n)
            x_chunk = x_flat[start:end]
            if use_ckpt:
                chunks.append(_ckpt(self._forward_chunk, x_chunk, use_reentrant=False))
            else:
                chunks.append(self._forward_chunk(x_chunk))
        return torch.cat(chunks, dim=0).mean(dim=1)

    @staticmethod
    def _make_sinusoidal(length: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        pos = torch.linspace(0.0, 1.0, length, device=device, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / dim))
        enc = torch.zeros(length, dim, device=device, dtype=torch.float32)
        enc[:, 0::2] = torch.sin(pos * div)
        enc[:, 1::2] = torch.cos(pos * div[: enc[:, 1::2].shape[1]])
        return enc.to(dtype=dtype)


from __future__ import annotations

import math

import torch
import torch.nn as nn


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

        agg = torch.zeros(n_nodes, messages.shape[-1], device=h.device, dtype=h.dtype)
        idx = dst.unsqueeze(1).expand_as(messages)
        agg.scatter_add_(0, idx, messages)

        h_new = self.update_fn(torch.cat([h, agg], dim=-1))
        return self.norm(h + h_new)


class TemporalAggregator(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int = 2, dropout: float = 0.3) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t, n, d = x.shape
        x_flat = x.permute(1, 0, 2)
        x_flat = x_flat + self._make_sinusoidal(t, d, x.device, x.dtype).unsqueeze(0).expand(n, -1, -1)

        gru_out, _ = self.gru(x_flat)
        attn_out, _ = self.attn(gru_out, gru_out, gru_out)
        attn_out = self.dropout(attn_out)
        return attn_out.mean(dim=1)

    @staticmethod
    def _make_sinusoidal(length: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        pos = torch.linspace(0.0, 1.0, length, device=device, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / dim))
        enc = torch.zeros(length, dim, device=device, dtype=torch.float32)
        enc[:, 0::2] = torch.sin(pos * div)
        enc[:, 1::2] = torch.cos(pos * div[: enc[:, 1::2].shape[1]])
        return enc.to(dtype=dtype)


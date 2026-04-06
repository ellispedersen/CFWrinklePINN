from __future__ import annotations

import torch
import torch.nn as nn

from .layers import MaterialEncoder, MessagePassingLayer, TemporalAggregator


class FormingGraphNet(nn.Module):
    def __init__(
        self,
        in_dim: int = 74,
        hidden_dim: int = 64,
        material_dim: int = 32,
        n_mp_steps: int = 3,
        n_heads: int = 2,
        dropout: float = 0.3,
        n_targets: int = 4,
    ) -> None:
        super().__init__()
        self.node_encoder = nn.Linear(in_dim, hidden_dim)
        self.material_encoder = MaterialEncoder(in_dim=8, out_dim=material_dim)
        self.temporal = TemporalAggregator(hidden_dim, n_heads=n_heads, dropout=dropout)
        self.mp_layers = nn.ModuleList(
            [MessagePassingLayer(hidden_dim, edge_dim=4, material_dim=material_dim) for _ in range(n_mp_steps)]
        )
        self.timestep_decoder = nn.GRU(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, n_targets),
        )

    def forward(self, batch: dict) -> torch.Tensor:
        x = batch["node_features"]
        edge_index = batch["edge_index"]
        edge_attr = batch["edge_attr"]
        material = batch["material_card"]
        elements = batch["elements"]

        t, _, _ = x.shape
        m = int(elements.shape[0])

        mat_embed = self.material_encoder(material)
        x_enc = self.node_encoder(x)
        h = self.temporal(x_enc)
        for mp in self.mp_layers:
            h = mp(h, edge_index, edge_attr, mat_embed)

        h_exp = h.unsqueeze(0).expand(t, -1, -1)
        h_seq = x_enc + h_exp
        h_dec, _ = self.timestep_decoder(h_seq.permute(1, 0, 2))
        h_dec = h_dec.permute(1, 0, 2)

        node_idx = elements.reshape(-1)
        h_corner = h_dec[:, node_idx, :].reshape(t, m, 3, -1).mean(dim=2)
        return self.output_head(h_corner)

    def predict_severity(self, batch: dict) -> torch.Tensor:
        return self.forward(batch)[..., 0]


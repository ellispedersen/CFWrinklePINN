from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as _ckpt

from .layers import MaterialEncoder, MessagePassingLayer, TemporalAggregator
from .labels import N_COARSE_TARGETS, N_FINE_TARGETS


class CrossScaleNet(nn.Module):
    """Dual-mesh physics-informed GNN.

    Runs a coarse-mesh GNN encoder (identical hyperparameters to FormingGraphNet),
    then scatters coarse-element embeddings to fine-element space via the
    coarse_to_fine_index mapping stored in WP3.  Two output heads are produced:

      "coarse": (T, M_coarse, 4) — same shape as FormingGraphNet output; backward-
                                   compatible with existing gate checks / wrinkle_loss.
      "fine":   (T, N_fine_elem, 4) — per-fine-element predictions against fine-mesh
                                       ground truth (cross_scale_loss).

    Required batch keys (in addition to the standard Model A keys):
      coarse_to_fine  (2, N_mapping) int64 — [0]=coarse_elem_idx, [1]=fine_elem_idx
      fine_elements   (N_fine_elem, 3) int64
    """

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
        fine_mp_steps: int = 2,
        use_fine_mp: bool = False,
    ) -> None:
        super().__init__()
        if int(n_targets) != N_COARSE_TARGETS or int(n_targets) != N_FINE_TARGETS:
            raise ValueError(
                f"CrossScaleNet requires n_targets={N_COARSE_TARGETS} for coarse/fine label mapping, got {n_targets}"
            )
        # Coarse encoder — same architecture as FormingGraphNet.
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
            [MessagePassingLayer(hidden_dim, edge_dim=4, material_dim=material_dim)
             for _ in range(n_mp_steps)]
        )
        self.timestep_decoder = nn.GRU(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.decoder_chunk_t = max(1, int(decoder_chunk_t))
        self.use_checkpoint = use_checkpoint
        self.use_fine_mp = use_fine_mp

        # Coarse output head — backward compatible with wrinkle_loss / gate_check.
        self.coarse_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, n_targets),
        )

        # Fine output head — takes scattered coarse element embedding → fine prediction.
        self.fine_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_targets),
        )
        self.fine_mp_layers = nn.ModuleList(
            [MessagePassingLayer(hidden_dim, edge_dim=4, material_dim=material_dim)
             for _ in range(max(0, int(fine_mp_steps)) if use_fine_mp else 0)]
        )

    def _decode_chunk(
        self,
        chunk_in: torch.Tensor,
        gru_hidden: torch.Tensor,
        coarse_node_idx: torch.Tensor,
        m_coarse: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """GRU decode one chunk → coarse element embeddings. Extracted for gradient checkpointing."""
        chunk_dec, new_hidden = self.timestep_decoder(chunk_in, gru_hidden)
        chunk_dec = chunk_dec.permute(1, 0, 2)  # (chunk_t, N_coarse, D)
        # Average the 3 corner nodes of each coarse element.
        h_coarse_elem = (
            chunk_dec[:, coarse_node_idx, :]
            .reshape(chunk_dec.shape[0], m_coarse, 3, -1)
            .mean(dim=2)
        )
        return h_coarse_elem, new_hidden  # (chunk_t, M_coarse, D), (1, N_coarse, D)

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        x = batch["node_features"]           # (T, N_coarse, in_dim)
        edge_index = batch["edge_index"]     # (2, E_coarse)
        edge_attr = batch["edge_attr"]       # (E_coarse, 4)
        material = batch["material_card"]    # (8,)
        coarse_elements = batch["elements"]  # (M_coarse, 3)
        coarse_to_fine = batch["coarse_to_fine"]   # (2, N_mapping)
        fine_elements = batch["fine_elements"]      # (N_fine_elem, 3)
        node_coarse_map = batch.get("fine_node_coarse_map")
        fine_edge_index = batch.get("fine_edge_index")
        fine_edge_attr = batch.get("fine_edge_attr")

        t, n, _ = x.shape
        m_coarse = int(coarse_elements.shape[0])
        n_fine_elem = int(fine_elements.shape[0])
        if coarse_to_fine.ndim != 2 or int(coarse_to_fine.shape[0]) != 2:
            raise ValueError(f"coarse_to_fine must have shape (2, K), got {tuple(coarse_to_fine.shape)}")
        c_idx = coarse_to_fine[0]  # (N_mapping,) — which coarse element
        f_idx = coarse_to_fine[1]  # (N_mapping,) — which fine element
        # Bounds validation is done once at dataset load time (_validate_fine_shapes).
        # Removed int(tensor.min/max) calls here — they cause a graph break in
        # torch.compile (data-dependent Python extraction), doubling kernel count.

        # --- Coarse encoder ---
        mat_embed = self.material_encoder(material)
        x_enc = self.node_encoder(x)    # (T, N_coarse, D)
        h = self.temporal(x_enc)        # (N_coarse, D)
        for mp in self.mp_layers:
            h = mp(h, edge_index, edge_attr, mat_embed)

        h_seq = x_enc + h.unsqueeze(0).expand(t, -1, -1)  # (T, N_coarse, D)
        coarse_node_idx = coarse_elements.reshape(-1)       # (M_coarse*3,)
        gru_hidden = torch.zeros(
            1, n, self.timestep_decoder.hidden_size,
            device=x.device, dtype=x.dtype,
        )
        use_ckpt = self.use_checkpoint and h_seq.requires_grad

        coarse_chunks: list[torch.Tensor] = []
        fine_chunks: list[torch.Tensor] = []

        for t_start in range(0, t, self.decoder_chunk_t):
            t_end = min(t_start + self.decoder_chunk_t, t)
            chunk_in = h_seq[t_start:t_end].permute(1, 0, 2)  # (N_coarse, chunk_t, D)

            if use_ckpt:
                h_coarse_elem, gru_hidden = _ckpt(
                    self._decode_chunk, chunk_in, gru_hidden, coarse_node_idx, m_coarse,
                    use_reentrant=False,
                )
            else:
                h_coarse_elem, gru_hidden = self._decode_chunk(
                    chunk_in, gru_hidden, coarse_node_idx, m_coarse,
                )
            # h_coarse_elem: (chunk_t, M_coarse, D)
            chunk_t = h_coarse_elem.shape[0]
            d = h_coarse_elem.shape[2]

            # Coarse prediction
            coarse_chunks.append(self.coarse_head(h_coarse_elem))  # (chunk_t, M_coarse, 4)

            if self.use_fine_mp and len(self.fine_mp_layers) > 0:
                if node_coarse_map is None or fine_edge_index is None or fine_edge_attr is None:
                    raise KeyError(
                        "CrossScaleNet(use_fine_mp=True) requires fine_node_coarse_map, "
                        "fine_edge_index, and fine_edge_attr in batch."
                    )
                mapped_nodes = node_coarse_map.clamp(min=0)
                valid_nodes = (node_coarse_map >= 0).unsqueeze(-1).to(dtype=x.dtype)

                fine_t_list: list[torch.Tensor] = []
                # Gradient-checkpoint each timestep to avoid retaining N_fine_nodes
                # activations across all T steps (fine mesh is ~25x coarse mesh size).
                def _fine_mp_step(
                    h_c: torch.Tensor,
                    _mapped: torch.Tensor,
                    _valid: torch.Tensor,
                    _fe: torch.Tensor,
                    _ei: torch.Tensor,
                    _ea: torch.Tensor,
                    _me: torch.Tensor,
                ) -> torch.Tensor:
                    # Disable autocast for indexing ops — torch.compile + inductor +
                    # autocast has a known regression in 2.11 where IndexBackward
                    # fails with "Unexpected floating ScalarType in autocast::prioritize".
                    with torch.autocast("cuda", enabled=False):
                        h = h_c.float()[_mapped] * _valid.float()
                    h = h.to(dtype=h_c.dtype)
                    for mp_layer in self.fine_mp_layers:
                        h = mp_layer(h, _ei, _ea, _me)
                    with torch.autocast("cuda", enabled=False):
                        out = self.fine_head(h[_fe].float()).mean(dim=1)
                    return out.to(dtype=h_c.dtype)

                for t_local in range(chunk_t):
                    h_c = h_coarse_elem[t_local]  # (M_coarse, D)
                    if use_ckpt:
                        step_out = _ckpt(
                            _fine_mp_step,
                            h_c, mapped_nodes, valid_nodes,
                            fine_elements, fine_edge_index, fine_edge_attr, mat_embed,
                            use_reentrant=False,
                        )
                    else:
                        step_out = _fine_mp_step(
                            h_c, mapped_nodes, valid_nodes,
                            fine_elements, fine_edge_index, fine_edge_attr, mat_embed,
                        )
                    fine_t_list.append(step_out)
                fine_chunks.append(torch.stack(fine_t_list, dim=0))
            else:
                # Phase-1 path: scatter coarse element embeddings directly to fine elements.
                h_mapped = h_coarse_elem[:, c_idx, :]  # (chunk_t, N_mapping, D)
                h_fine_elem = torch.zeros(
                    chunk_t, n_fine_elem, d, device=x.device, dtype=h_mapped.dtype
                )
                idx_exp = f_idx.unsqueeze(0).unsqueeze(-1).expand(chunk_t, -1, d)
                h_fine_elem.scatter_add_(1, idx_exp, h_mapped)
                map_counts = torch.zeros(n_fine_elem, device=x.device, dtype=h_mapped.dtype)
                map_counts.scatter_add_(
                    0,
                    f_idx,
                    torch.ones_like(f_idx, dtype=h_mapped.dtype),
                )
                h_fine_elem = h_fine_elem / map_counts.clamp_min(1).view(1, -1, 1)
                fine_chunks.append(self.fine_head(h_fine_elem))  # (chunk_t, N_fine_elem, 4)

        return {
            "coarse": torch.cat(coarse_chunks, dim=0),  # (T, M_coarse, 4)
            "fine": torch.cat(fine_chunks, dim=0),      # (T, N_fine_elem, 4)
        }

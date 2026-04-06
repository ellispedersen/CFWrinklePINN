# WP6 — Model Architecture
## PyTorch Dataset Interface, GNN Design, Loss Functions

**Depends on:** WP1–5 complete, `data/cfwrinkle_wp3_features.h5` validated
**Gate before WP7:** unit tests pass + overfit-on-1-sim check passes
**Status:** Ready to implement

---

## Objective

Build the PyTorch model and dataset interface that reads `data/cfwrinkle_wp3_features.h5` and produces per-element wrinkle severity predictions. The model processes a full coarse-mesh simulation (256 timesteps, variable-size graph) and must be usable at inference time with only a single set of coarse-mesh AniForm results.

---

## Inputs (from WP3 HDF5)

Per simulation `simulations/<sim_id>/` in `data/cfwrinkle_wp3_features.h5`:

| Dataset | Shape | dtype | Notes |
|---|---|---|---|
| `coarse_fields_resampled` | (256, N_nodes, 37) | float32 | Physics features |
| `coarse_rates_resampled` | (256, N_nodes, 37) | float32 | Temporal rates |
| `graph/edge_index` | (2, N_edges) | int32 | Undirected triangle edges |
| `graph/edge_attr` | (N_edges, 4) | float32 | [dx, dy, dz, dist] |
| `material_card` | (8,) | float32 | Global material params |
| `targets/wrinkle_severity` | (256, N_elem) | float32 | Primary target [0,1] |
| `targets/comp_frac_elem` | (256, N_elem) | float32 | Auxiliary target |
| `targets/oop_max_elem` | (256, N_elem) | float32 | Auxiliary target |
| `targets/thickness_variance_elem` | (256, N_elem) | float32 | Auxiliary target |
| `mesh/coarse_nodes` | (N_nodes, 3) | float32 | Reference XYZ positions |
| `mesh/coarse_elements` | (N_elem, 3) | int32 | Triangle node indices |

**Derived input to model:** concat features + rates → (256, N_nodes, 74)

---

## File Structure to Create

```
model/
  __init__.py           — exports FormingGraphNet, wrinkle_loss
  dataset.py            — WrinkleDataset (PyTorch Dataset)
  gnn.py                — FormingGraphNet
  layers.py             — MessagePassingLayer, TemporalAggregator, MaterialEncoder
  loss.py               — composite physics loss

tests/
  test_model_unit.py    — unit tests for all model components
```

---

## 6.1 Dataset Interface

**File:** `model/dataset.py`

```python
"""
WP6 — WrinkleDataset
Reads cfwrinkle_wp3_features.h5 and yields one simulation per item.
All computation is done at parse time — no on-the-fly field lookups.
"""
from __future__ import annotations
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class WrinkleDataset(Dataset):
    """
    One item = one simulation. Variable-size graphs — do NOT use default_collate.
    Use DataLoader with batch_size=1 and collate_fn=lambda x: x[0].

    Args:
        h5_path: path to cfwrinkle_wp3_features.h5
        sim_ids: list of sim_id strings to include (e.g. from fold['train'])
        normalize: if True, apply stored global feature stats (mean/std)
        device: torch device for returned tensors

    Item dict keys:
        sim_id:         str
        node_features:  (256, N_nodes, 74)  float32 — features + rates concatenated
        edge_index:     (2, N_edges)        int64
        edge_attr:      (N_edges, 4)        float32
        material_card:  (8,)                float32
        elements:       (N_elem, 3)         int64   — for node→element aggregation
        targets:        (256, N_elem, 4)    float32 — [severity, comp_frac, oop, thick_var]
        sim_meta:       dict with batch, n_nodes, n_elem, n_edges
    """

    TARGET_KEYS = [
        "targets/wrinkle_severity",
        "targets/comp_frac_elem",
        "targets/oop_max_elem",
        "targets/thickness_variance_elem",
    ]

    def __init__(
        self,
        h5_path: str | Path,
        sim_ids: list[str],
        normalize: bool = True,
        device: str | torch.device = "cpu",
    ) -> None:
        self.h5_path = Path(h5_path)
        self.sim_ids = sim_ids
        self.normalize = normalize
        self.device = torch.device(device)
        self._stats: dict | None = None
        if normalize:
            self._stats = self._load_or_compute_stats()

    def __len__(self) -> int:
        return len(self.sim_ids)

    def __getitem__(self, idx: int) -> dict:
        sim_id = self.sim_ids[idx]
        with h5py.File(self.h5_path, "r") as f:
            grp = f[f"simulations/{sim_id}"]
            feats = grp["coarse_fields_resampled"][:]   # (256, N, 37)
            rates = grp["coarse_rates_resampled"][:]    # (256, N, 37)
            edge_index = grp["graph/edge_index"][:]     # (2, E)
            edge_attr = grp["graph/edge_attr"][:]       # (E, 4)
            material = grp["material_card"][:]          # (8,)
            elements = grp["mesh/coarse_elements"][:]   # (M, 3)
            targets = np.stack(
                [grp[k][:] for k in self.TARGET_KEYS], axis=-1
            )  # (256, M, 4)
            batch = str(grp.attrs.get("batch", "?"))
            n_nodes = feats.shape[1]
            n_elem = elements.shape[0]

        node_features = np.concatenate([feats, rates], axis=-1)  # (256, N, 74)

        if self.normalize and self._stats:
            mu = self._stats["feat_mean"]   # (74,)
            std = self._stats["feat_std"]   # (74,)
            node_features = (node_features - mu) / (std + 1e-8)

        def _t(arr, dtype=torch.float32) -> torch.Tensor:
            return torch.as_tensor(arr, dtype=dtype, device=self.device)

        return {
            "sim_id": sim_id,
            "node_features": _t(node_features),
            "edge_index": _t(edge_index, dtype=torch.int64),
            "edge_attr": _t(edge_attr),
            "material_card": _t(material),
            "elements": _t(elements, dtype=torch.int64),
            "targets": _t(targets),
            "sim_meta": {"batch": batch, "n_nodes": n_nodes, "n_elem": n_elem, "n_edges": edge_index.shape[1]},
        }

    def _load_or_compute_stats(self) -> dict:
        """
        Compute global feature mean/std across ALL sims in the h5 file.
        This includes held-out sims — normalisation statistics must be
        computed on the full dataset before any split is applied, then
        frozen. Never recompute per-fold.
        """
        # Try to read cached stats from h5 metadata
        with h5py.File(self.h5_path, "r") as f:
            if "metadata/feature_stats/feat_mean" in f:
                return {
                    "feat_mean": f["metadata/feature_stats/feat_mean"][:],
                    "feat_std":  f["metadata/feature_stats/feat_std"][:],
                }
        # Compute on the fly (only needed once — see compute_and_store_stats())
        return self._compute_stats_online()

    def _compute_stats_online(self) -> dict:
        """Welford online mean/var across all sims × timesteps × nodes."""
        print("Computing feature normalisation statistics (one-time)...")
        count = 0
        mean = None
        M2 = None
        with h5py.File(self.h5_path, "r") as f:
            all_ids = list(f["simulations"].keys())
            for sid in all_ids:
                grp = f[f"simulations/{sid}"]
                feats = grp["coarse_fields_resampled"][:]   # (256, N, 37)
                rates = grp["coarse_rates_resampled"][:]    # (256, N, 37)
                x = np.concatenate([feats, rates], axis=-1).reshape(-1, 74)
                for row in x:
                    count += 1
                    if mean is None:
                        mean = row.copy()
                        M2 = np.zeros_like(row)
                    else:
                        delta = row - mean
                        mean += delta / count
                        M2 += delta * (row - mean)
        std = np.sqrt(M2 / max(count - 1, 1)).astype(np.float32)
        mean = mean.astype(np.float32)
        # Store back into h5
        with h5py.File(self.h5_path, "a") as f:
            f.require_group("metadata/feature_stats")
            f["metadata/feature_stats/feat_mean"] = mean
            f["metadata/feature_stats/feat_std"] = std
        print(f"Stats computed from {count:,} node-timestep samples.")
        return {"feat_mean": mean, "feat_std": std}


def load_fold_sim_ids(h5_path: str | Path, fold: int) -> tuple[list[str], list[str]]:
    """
    Read train/val sim_id lists from the CV split stored in cfwrinkle_dataset.h5.
    Returns (train_ids, val_ids).
    NOTE: splits are stored in the WP2 HDF5, not the WP3 HDF5.
    """
    wp2_h5 = Path(str(h5_path).replace("cfwrinkle_wp3_features.h5", "cfwrinkle_dataset.h5"))
    with h5py.File(wp2_h5, "r") as f:
        grp = f[f"splits/fold_{fold}"]
        train_ids = [s.decode() if isinstance(s, bytes) else s for s in grp["train"][:]]
        val_ids   = [s.decode() if isinstance(s, bytes) else s for s in grp["val"][:]]
    return train_ids, val_ids
```

---

## 6.2 Model Layers

**File:** `model/layers.py`

### MaterialEncoder
```python
class MaterialEncoder(nn.Module):
    """
    MLP: (8,) → (material_dim,) material embedding.
    Injected at every message passing step.
    """
    def __init__(self, in_dim: int = 8, out_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.SiLU(),
            nn.Linear(64, out_dim),
        )

    def forward(self, material_card: torch.Tensor) -> torch.Tensor:
        # material_card: (8,) or (B, 8)
        return self.net(material_card)
```

### MessagePassingLayer
```python
class MessagePassingLayer(nn.Module):
    """
    One step of custom scatter_add message passing.
    No PyTorch Geometric — uses torch.scatter_add_.

    Message: concat(h_src, h_dst, edge_attr, material_embed) → message
    Update:  h_dst + aggregate_messages (residual)
    """
    def __init__(self, hidden_dim: int, edge_dim: int = 4, material_dim: int = 32) -> None:
        super().__init__()
        msg_in = hidden_dim * 2 + edge_dim + material_dim
        self.message_fn = nn.Sequential(
            nn.Linear(msg_in, hidden_dim), nn.SiLU(),
        )
        self.update_fn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.SiLU(),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h: torch.Tensor,           # (N_nodes, hidden_dim)
        edge_index: torch.Tensor,  # (2, N_edges) int64
        edge_attr: torch.Tensor,   # (N_edges, 4)
        material_embed: torch.Tensor,  # (material_dim,)
    ) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]  # (N_edges,)
        n_nodes = h.shape[0]

        # Broadcast material embed to each edge
        mat = material_embed.unsqueeze(0).expand(src.shape[0], -1)  # (E, material_dim)

        msg_input = torch.cat([h[src], h[dst], edge_attr, mat], dim=-1)  # (E, msg_in)
        messages = self.message_fn(msg_input)  # (E, hidden_dim)

        # Aggregate to destination nodes
        agg = torch.zeros(n_nodes, messages.shape[-1], device=h.device, dtype=h.dtype)
        idx = dst.unsqueeze(1).expand_as(messages)
        agg.scatter_add_(0, idx, messages)

        # Update with residual
        h_new = self.update_fn(torch.cat([h, agg], dim=-1))
        return self.norm(h + h_new)  # residual connection + LayerNorm
```

### TemporalAggregator
```python
class TemporalAggregator(nn.Module):
    """
    Per-node GRU over 256 timesteps, then attention-weighted pooling.
    Produces a single hidden state per node summarising the temporal sequence.

    Input:  (T, N_nodes, hidden_dim)
    Output: (N_nodes, hidden_dim)
    """
    def __init__(self, hidden_dim: int, n_heads: int = 2, dropout: float = 0.3) -> None:
        super().__init__()
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        # Sinusoidal positional encoding (over stroke fraction 0→1)
        self.register_buffer("pos_enc", self._make_sinusoidal(256, hidden_dim))

    @staticmethod
    def _make_sinusoidal(T: int, d: int) -> torch.Tensor:
        pos = torch.linspace(0, 1, T).unsqueeze(1)  # (T, 1)
        div = torch.exp(torch.arange(0, d, 2).float() * (-np.log(10000.0) / d))
        enc = torch.zeros(T, d)
        enc[:, 0::2] = torch.sin(pos * div)
        enc[:, 1::2] = torch.cos(pos * div[:d // 2])
        return enc  # (T, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, N_nodes, hidden_dim)
        T, N, D = x.shape
        x_flat = x.permute(1, 0, 2)  # (N_nodes, T, D) — batch_first for GRU

        # Add positional encoding
        pos = self.pos_enc.unsqueeze(0).expand(N, -1, -1)  # (N, T, D)
        x_flat = x_flat + pos

        # GRU — processes each node's temporal sequence independently
        gru_out, _ = self.gru(x_flat)  # (N, T, D)

        # Self-attention over timesteps per node
        attn_out, _ = self.attn(gru_out, gru_out, gru_out)  # (N, T, D)
        attn_out = self.dropout(attn_out)

        # Mean pool over timesteps
        return attn_out.mean(dim=1)  # (N, D)
```

---

## 6.3 Main Model

**File:** `model/gnn.py`

```python
"""
WP6 — FormingGraphNet
Physics-informed spatiotemporal GNN for wrinkle prediction.

Input:  one simulation dict from WrinkleDataset
Output: (256, N_elem, 4) predicted targets — same shape as ground truth
"""
import numpy as np
import torch
import torch.nn as nn
from .layers import MaterialEncoder, MessagePassingLayer, TemporalAggregator


class FormingGraphNet(nn.Module):
    """
    Architecture:
      1. Node feature encoder: (74,) → hidden_dim per node per timestep
      2. Material encoder: (8,) → material_embed_dim
      3. Temporal aggregator (GRU + attention): collapse 256 timesteps → 1 state per node
      4. Message passing (n_mp_steps): propagate spatial information
      5. Node→element aggregation: mean of 3 corner nodes per triangle
      6. Output head: hidden_dim → 4 target predictions per element per timestep

    Capacity rationale:
      hidden_dim=64 for 65 training simulations.
      Grow to 128 only with evidence of underfitting (val loss plateauing
      while train loss continues decreasing after epoch 30+).
    """

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
        self.hidden_dim = hidden_dim
        self.n_mp_steps = n_mp_steps

        # Encoders
        self.node_encoder = nn.Linear(in_dim, hidden_dim)  # no activation — preserve linear relations
        self.material_encoder = MaterialEncoder(in_dim=8, out_dim=material_dim)

        # Temporal aggregator
        self.temporal = TemporalAggregator(hidden_dim, n_heads=n_heads, dropout=dropout)

        # Message passing layers
        self.mp_layers = nn.ModuleList([
            MessagePassingLayer(hidden_dim, edge_dim=4, material_dim=material_dim)
            for _ in range(n_mp_steps)
        ])

        # Output head: node embeddings → per-element predictions
        # Applied per timestep after decoding
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, n_targets),
        )

        # Per-timestep decoder (after temporal aggregation, refine per timestep)
        self.timestep_decoder = nn.GRU(
            hidden_dim, hidden_dim, num_layers=1, batch_first=True
        )

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Args:
            batch: dict from WrinkleDataset.__getitem__
                node_features: (T, N, 74)
                edge_index:    (2, E)   int64
                edge_attr:     (E, 4)
                material_card: (8,)
                elements:      (M, 3)  int64

        Returns:
            predictions: (T, M, 4) — per timestep, per element, 4 targets
        """
        x = batch["node_features"]          # (T, N, 74)
        edge_index = batch["edge_index"]    # (2, E)
        edge_attr = batch["edge_attr"]      # (E, 4)
        material = batch["material_card"]   # (8,)
        elements = batch["elements"]        # (M, 3)

        T, N, _ = x.shape
        M = elements.shape[0]

        # Encode material globally
        mat_embed = self.material_encoder(material)  # (material_dim,)

        # Encode node features at every timestep
        x_enc = self.node_encoder(x)  # (T, N, hidden_dim)

        # Temporal aggregation → single state per node
        h_temporal = self.temporal(x_enc)  # (N, hidden_dim)

        # Message passing on temporally-aggregated graph
        h = h_temporal
        for mp in self.mp_layers:
            h = mp(h, edge_index, edge_attr, mat_embed)  # (N, hidden_dim)

        # Decode per-timestep predictions by conditioning on GMP output
        # Expand spatial embedding to all timesteps, then run per-node GRU
        h_exp = h.unsqueeze(0).expand(T, -1, -1)  # (T, N, hidden_dim)
        h_enc_seq = x_enc + h_exp                  # residual: temporal features + graph context

        # Per-node temporal decode
        h_dec, _ = self.timestep_decoder(
            h_enc_seq.permute(1, 0, 2)  # (N, T, hidden_dim)
        )  # (N, T, hidden_dim)
        h_dec = h_dec.permute(1, 0, 2)  # (T, N, hidden_dim)

        # Node → element aggregation (mean of 3 corner nodes)
        # elements: (M, 3) → gather 3 node embeddings per element → mean
        corners = elements.unsqueeze(0).expand(T, -1, -1)  # (T, M, 3)
        # Gather: for each element, get hidden states of its 3 nodes
        h_corners = h_dec[:, corners.reshape(T, -1), :]     # (T, M*3, hidden_dim)
        h_corners = h_corners.reshape(T, M, 3, -1).mean(dim=2)  # (T, M, hidden_dim)

        # Output head
        predictions = self.output_head(h_corners)  # (T, M, 4)
        return predictions

    def predict_severity(self, batch: dict) -> torch.Tensor:
        """Convenience: return only wrinkle_severity (channel 0)."""
        return self.forward(batch)[..., 0]  # (T, M)
```

---

## 6.4 Loss Function

**File:** `model/loss.py`

```python
"""
WP6 — Composite physics loss for wrinkle prediction.

Components:
  L_severity:   Huber loss on wrinkle_severity (primary target, highest weight)
  L_comp_frac:  MSE on comp_frac_elem
  L_oop:        MSE on oop_max_elem
  L_thick_var:  MSE on thickness_variance_elem
  L_physics:    Monotonicity: severity should be non-decreasing for wrinkled sims

All components are NaN-masked — NaN in targets = boundary element, excluded.
Components are scale-normalised before weighting to prevent any one from dominating.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


LOSS_WEIGHTS = {
    "severity":   3.0,   # primary — spatially sparse, needs emphasis
    "comp_frac":  1.0,
    "oop":        1.0,
    "thick_var":  0.5,
    "physics":    2.0,
}


def _nan_huber(pred: torch.Tensor, target: torch.Tensor, delta: float = 0.1) -> torch.Tensor:
    """Huber loss with NaN masking. Returns scalar mean over valid elements."""
    mask = ~torch.isnan(target)
    if mask.sum() == 0:
        return pred.sum() * 0.0  # differentiable zero
    return F.huber_loss(pred[mask], target[mask], delta=delta, reduction="mean")


def _nan_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mask = ~torch.isnan(target)
    if mask.sum() == 0:
        return pred.sum() * 0.0
    return F.mse_loss(pred[mask], target[mask], reduction="mean")


def monotonicity_loss(severity_pred: torch.Tensor) -> torch.Tensor:
    """
    Physics prior: wrinkle severity should not decrease once initiated.
    Penalise decreases in predicted severity over time.
    severity_pred: (T, M) predictions for one sim.
    """
    delta = severity_pred[1:] - severity_pred[:-1]   # (T-1, M)
    violations = torch.clamp(-delta, min=0.0)         # penalise decreases only
    return violations.mean()


def wrinkle_loss(
    predictions: torch.Tensor,  # (T, M, 4)
    targets: torch.Tensor,       # (T, M, 4)
    weights: dict | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute composite loss.

    Returns:
        total_loss: scalar tensor (differentiable)
        component_log: dict of float values for logging
    """
    w = weights or LOSS_WEIGHTS
    pred_sev  = predictions[..., 0]  # (T, M)
    pred_comp = predictions[..., 1]
    pred_oop  = predictions[..., 2]
    pred_tv   = predictions[..., 3]

    tgt_sev  = targets[..., 0]
    tgt_comp = targets[..., 1]
    tgt_oop  = targets[..., 2]
    tgt_tv   = targets[..., 3]

    l_sev  = _nan_huber(pred_sev,  tgt_sev)
    l_comp = _nan_mse(pred_comp, tgt_comp)
    l_oop  = _nan_mse(pred_oop,  tgt_oop)
    l_tv   = _nan_mse(pred_tv,   tgt_tv)
    l_phys = monotonicity_loss(pred_sev)

    # Scale-normalise: divide each by its own detached magnitude (running estimate)
    # This keeps relative weights meaningful when component magnitudes differ by orders of magnitude.
    def _safe_norm(loss: torch.Tensor) -> torch.Tensor:
        mag = loss.detach().clamp(min=1e-8)
        return loss / mag

    total = (
        w["severity"]  * _safe_norm(l_sev)  +
        w["comp_frac"] * _safe_norm(l_comp) +
        w["oop"]       * _safe_norm(l_oop)  +
        w["thick_var"] * _safe_norm(l_tv)   +
        w["physics"]   * _safe_norm(l_phys)
    )

    log = {
        "loss/total":     float(total),
        "loss/severity":  float(l_sev),
        "loss/comp_frac": float(l_comp),
        "loss/oop":       float(l_oop),
        "loss/thick_var": float(l_tv),
        "loss/physics":   float(l_phys),
    }
    return total, log
```

---

## 6.5 Unit Tests

**File:** `tests/test_model_unit.py`

```python
"""
WP6 unit tests — all use synthetic data, no HDF5 required.
Run: pytest tests/test_model_unit.py -v
"""
import torch
import pytest
from model.layers import MaterialEncoder, MessagePassingLayer, TemporalAggregator
from model.gnn import FormingGraphNet
from model.loss import wrinkle_loss


N, T, M = 100, 16, 80   # small sizes for fast tests
E = 200
HIDDEN = 32


def _fake_batch(N=N, T=T, M=M, E=E, device="cpu"):
    """Synthetic batch mimicking WrinkleDataset output."""
    elements = torch.randint(0, N, (M, 3))
    return {
        "node_features":  torch.randn(T, N, 74),
        "edge_index":     torch.randint(0, N, (2, E), dtype=torch.int64),
        "edge_attr":      torch.randn(E, 4),
        "material_card":  torch.randn(8),
        "elements":       elements,
        "targets":        torch.rand(T, M, 4),
    }


def test_material_encoder_shape():
    enc = MaterialEncoder(8, 32)
    out = enc(torch.randn(8))
    assert out.shape == (32,)


def test_message_passing_shape():
    layer = MessagePassingLayer(HIDDEN, edge_dim=4, material_dim=16)
    h = torch.randn(N, HIDDEN)
    ei = torch.randint(0, N, (2, E), dtype=torch.int64)
    ea = torch.randn(E, 4)
    mat = torch.randn(16)
    out = layer(h, ei, ea, mat)
    assert out.shape == (N, HIDDEN), f"expected ({N}, {HIDDEN}), got {out.shape}"


def test_temporal_aggregator_shape():
    agg = TemporalAggregator(HIDDEN, n_heads=2, dropout=0.0)
    x = torch.randn(T, N, HIDDEN)
    out = agg(x)
    assert out.shape == (N, HIDDEN)


def test_forming_graph_net_output_shape():
    model = FormingGraphNet(hidden_dim=HIDDEN)
    batch = _fake_batch()
    pred = model(batch)
    assert pred.shape == (T, M, 4), f"expected ({T}, {M}, 4), got {pred.shape}"


def test_loss_no_nan_with_clean_targets():
    pred = torch.rand(T, M, 4)
    tgt  = torch.rand(T, M, 4)
    loss, log = wrinkle_loss(pred, tgt)
    assert not torch.isnan(loss), "loss is NaN with clean targets"
    assert loss > 0


def test_loss_with_nan_targets():
    pred = torch.rand(T, M, 4)
    tgt  = torch.rand(T, M, 4)
    tgt[:, :5, :] = float("nan")  # boundary elements
    loss, log = wrinkle_loss(pred, tgt)
    assert not torch.isnan(loss), "loss is NaN with partial NaN targets"


def test_gradient_flows():
    model = FormingGraphNet(hidden_dim=HIDDEN)
    batch = _fake_batch()
    pred = model(batch)
    tgt = torch.rand_like(pred)
    loss, _ = wrinkle_loss(pred, tgt)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"
        assert not torch.isnan(p.grad).any(), f"NaN gradient in {name}"
```

---

## WP6 Gate Checklist

```
Dataset Interface
  [ ] WrinkleDataset.__getitem__ returns correct shapes for 3+ real sims
  [ ] node_features shape: (256, N_nodes, 74) — verify N_nodes varies across sims
  [ ] targets shape: (256, N_elem, 4)
  [ ] Feature normalisation stats computed and stored in HDF5 metadata
  [ ] load_fold_sim_ids() correctly reads from WP2 HDF5 splits

Model Architecture (ROCm)
  [ ] All ops verified ROCm-compatible (no custom CUDA, no Flash Attention)
  [ ] Model instantiates and runs forward pass on GPU without OOM
  [ ] Output shape matches targets shape: (256, N_elem, 4)

Loss Function
  [ ] NaN masking verified — boundary elements excluded from all loss components
  [ ] Log each loss component magnitude for first training batch — confirm none near 0 or dominating
  [ ] Gradient flows to all parameters (test_gradient_flows passes)

Unit Tests
  [ ] pytest tests/test_model_unit.py -v → all pass

Sanity: Overfit on 1 Simulation
  [ ] Train on 1 sim for 50 epochs — severity loss should decrease to < 0.01
  [ ] If loss stagnates after 20 epochs, check LR and loss scale-normalisation
```

---

## Handoff to WP7

WP7 receives:
- `model/dataset.py` — WrinkleDataset + load_fold_sim_ids
- `model/gnn.py` — FormingGraphNet
- `model/loss.py` — wrinkle_loss
- `data/cfwrinkle_wp3_features.h5` — normalisation stats now stored in `metadata/feature_stats/`
- All unit tests passing

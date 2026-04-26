# WP10 — Fine-Mesh Message Passing + Spatial Coherence
## Track B Phase 2 — Local Spatial Reasoning on Fine Mesh

**Depends on:** WP8 Level 3 gate passing, WP9 physics losses validated
**Gate:** Level 2 re-run with fine MP — no OOM, fine/stress_mae not regressing vs WP9
**Status:** NOT STARTED — implement after WP8+WP9 baseline confirmed

---

## Context

CrossScaleNet (WP8) scatters coarse element embeddings to fine elements. All fine elements under the same coarse parent receive **identical embeddings** — the model has no mechanism to differentiate them spatially. This is the fundamental limitation of Phase 1: the fine head output is piecewise-constant at coarse-element resolution.

WP10 adds fine-mesh message passing (fine MP) that propagates information between neighboring fine elements, enabling:
- Spatial gradients within each coarse element patch
- Learning of wrinkle wavelength-scale patterns
- Edge effects and boundary conditions at coarse-element boundaries

---

## Architecture Addition to CrossScaleNet

### New batch keys required (added to `_read_fine()` in dataset.py)

```python
batch["fine_node_coarse_map"]   # (N_fine_nodes,) int64 — which coarse elem owns each fine node
batch["fine_elem_edge_index"]   # (2, E_adj) int64 — element adjacency graph (from WP3)
```

### Fine MP flow (inserted between scatter and fine_head)

```
h_coarse_elem (chunk_t, M_coarse, D)
  ↓ for each t_local in chunk_t:   [process one timestep at a time — memory-safe]
h_coarse_one (M_coarse, D)
  ↓ scatter via node_coarse_map
h_fine_nodes (N_fine_nodes, D)    [36 MB for Batch A at D=64 FP32]
  ↓ fine_mp_layers × 2 (reuses MessagePassingLayer on fine node graph)
h_fine_nodes_refined (N_fine_nodes, D)
  ↓ node-to-element aggregation (mean of 3 corner nodes)
h_fine_elem_t (N_fine_elem, D)
  ↓ fine_head
fine_pred_t (N_fine_elem, 4)
  ↓ stack over chunk_t timesteps
fine_chunk (chunk_t, N_fine_elem, 4)
```

---

## Step 10.1 — Fine Element Adjacency Graph (WP3 addition)

Two fine triangular elements are **adjacent** if they share an edge (2 common nodes). This must be built during WP3 rebuild.

**File**: `wp3_features/graph.py` — add function:

```python
def build_element_adjacency(elements: np.ndarray) -> np.ndarray:
    """
    Build element-level adjacency from triangle connectivity.
    Two elements are adjacent if they share an edge (2 common nodes).
    Returns (2, E_adj) int32 — directed adjacency (both directions).

    For a triangular mesh with N_elem elements:
      ~N_elem interior edges → ~2×N_elem directed adjacency entries
    Batch A (~299K elements): ~900K directed entries ≈ 7 MB uncompressed
    Batch B (~240K elements): ~720K directed entries ≈ 6 MB uncompressed
    """
    from collections import defaultdict
    edge_to_elems: defaultdict = defaultdict(list)
    for ei, tri in enumerate(elements):
        for i in range(3):
            a, b = int(tri[i]), int(tri[(i + 1) % 3])
            edge_to_elems[(min(a, b), max(a, b))].append(ei)
    src, dst = [], []
    for elems_sharing_edge in edge_to_elems.values():
        if len(elems_sharing_edge) == 2:
            e0, e1 = elems_sharing_edge
            src += [e0, e1]
            dst += [e1, e0]
    if not src:
        return np.zeros((2, 0), dtype=np.int32)
    return np.array([src, dst], dtype=np.int32)
```

**Wire into `_build_sim()`** in `wp3_features/build_features.py` (inside `if include_fine:` block):

```python
if include_fine:
    # ... existing fine data storage ...
    f_elem_adj = build_element_adjacency(f_elem)
    fine.create_dataset("element_edge_index", data=f_elem_adj, compression="lzf")
```

**Storage**: ~7 MB/sim × 65 sims ≈ 455 MB total — negligible. Added to WP3 during next rebuild.

---

## Step 10.2 — Node-to-Coarse Mapping (dataset.py)

For the scatter from coarse elements to fine nodes, we need `node_coarse_map[n]` = which coarse element owns fine node n.

**Add to `_read_fine()`** in `model/dataset.py` after loading `coarse_to_fine` and `fine_elements`:

```python
# Build node_coarse_map: (N_fine_nodes,) int64
# Maps each fine node to the coarse element that contains its parent fine element.
n_fine_nodes = int(nodes.shape[0])
node_coarse_map = np.full(n_fine_nodes, -1, dtype=np.int64)
c_to_f = coarse_to_fine   # (2, N_mapping) int64
c_idx_arr = c_to_f[0]     # coarse element indices
f_idx_arr = c_to_f[1]     # fine element indices
for i in range(c_idx_arr.shape[0]):
    ci = int(c_idx_arr[i])
    fi = int(f_idx_arr[i])
    for ni in fine_elements[fi]:   # 3 corner nodes of this fine element
        if node_coarse_map[ni] < 0:
            node_coarse_map[ni] = ci
# Fine nodes not covered: leave as -1 (will be zeroed in scatter)
```

**Runtime cost**: ~900K iterations for Batch A (3 nodes × 299K mapping entries) — milliseconds.

Add `"node_coarse_map": node_coarse_map` to the returned fine_data dict, and expose as `batch["fine_node_coarse_map"]` in `WrinkleDataset.__getitem__`.

---

## Step 10.3 — CrossScaleNet Constructor Changes

**File**: `model/cross_scale.py`

```python
def __init__(
    self, ...,
    fine_mp_steps: int = 2,
    use_fine_mp: bool = False,  # OFF by default — Phase 1 unchanged
) -> None:
    ...
    self.use_fine_mp = use_fine_mp
    # Reuses existing MessagePassingLayer — no new class needed
    self.fine_mp_layers = nn.ModuleList(
        [MessagePassingLayer(hidden_dim, edge_dim=4, material_dim=material_dim)
         for _ in range(fine_mp_steps if use_fine_mp else 0)]
    )
```

**When `use_fine_mp=False`**: `fine_mp_layers` is an empty ModuleList. The decoder chunk loop falls through to existing behaviour. Zero overhead.

---

## Step 10.4 — Per-Timestep Fine MP Loop

**File**: `model/cross_scale.py` — replace the fine scatter block inside the decoder chunk loop.

**Before (Phase 1)**:
```python
h_mapped = h_coarse_elem[:, c_idx, :]
h_fine_elem = torch.zeros(chunk_t, n_fine_elem, d, ...)
h_fine_elem.scatter_(1, idx_exp, h_mapped)
fine_chunks.append(self.fine_head(h_fine_elem))
```

**After (Phase 2, when `use_fine_mp=True`)**:
```python
if self.use_fine_mp and self.fine_mp_layers:
    node_coarse_map = batch["fine_node_coarse_map"]   # (N_fine_nodes,) int64
    fine_edge_index = batch["fine_edge_index"]        # (2, E_fine) — node graph
    fine_edge_attr  = batch["fine_edge_attr"]         # (E_fine, 4)
    fine_elements   = batch["fine_elements"]          # (N_fine_elem, 3)
    n_fine_nodes = node_coarse_map.shape[0]

    fine_t_list = []
    for t_local in range(chunk_t):
        h_c = h_coarse_elem[t_local]  # (M_coarse, D)
        # Scatter coarse element embeddings to fine nodes
        h_fn = h_c[node_coarse_map.clamp(min=0)]   # (N_fine_nodes, D)
        # Zero out unmapped nodes (-1 entries)
        h_fn = h_fn * (node_coarse_map >= 0).unsqueeze(-1).to(h_fn.dtype)
        # Fine node message passing
        for fine_mp in self.fine_mp_layers:
            h_fn = fine_mp(h_fn, fine_edge_index, fine_edge_attr, mat_embed)
        # Node → element aggregation (mean of 3 corners)
        h_fe = h_fn[fine_elements].mean(dim=1)    # (N_fine_elem, D)
        fine_t_list.append(self.fine_head(h_fe))  # (N_fine_elem, 4)
    fine_chunks.append(torch.stack(fine_t_list, dim=0))  # (chunk_t, N_fine_elem, 4)
else:
    # Phase 1 path — unchanged
    h_mapped = h_coarse_elem[:, c_idx, :]
    h_fine_elem = torch.zeros(chunk_t, n_fine_elem, d, ...)
    h_fine_elem.scatter_(1, idx_exp, h_mapped)
    fine_chunks.append(self.fine_head(h_fine_elem))
```

---

## Step 10.5 — CLI Flag in `training/train.py`

Add `--use-fine-mp` flag to `_parse_args()`:
```python
p.add_argument("--use-fine-mp", action="store_true",
               help="Enable fine-mesh message passing in CrossScaleNet (Phase 2 architecture)")
```

Wire into model config:
```python
"model": {
    "hidden_dim": args.hidden_dim,
    "attn_batch_nodes": args.attn_batch_nodes,
    "decoder_chunk_t": args.decoder_chunk_t,
    "use_fine_mp": getattr(args, "use_fine_mp", False),  # new
},
```

Training command with fine MP enabled:
```bash
python -u -m training.train \
  --model-type cross-scale --use-fine-mp \
  --fold 3 --max-train-sims 13 --max-val-sims 2 \
  --epochs 50 --hidden-dim 64 --attn-batch-nodes 512 \
  --max-timesteps 128 --temporal-strategy tail --amp --device cuda \
  --output /home/ellis/cfwrinkle/checkpoints/progressive/cross_scale_fine_mp_level3
```

---

## Memory Analysis

Per-timestep fine MP memory (peak within the timestep loop):

| Tensor | Batch A (FP16) | Batch B (FP16) |
|---|---|---|
| `h_fn` (N_fine_nodes × D) | 150K×64×2 = **18 MB** | 121K×64×2 = **15 MB** |
| Fine node edge messages (E_fine × msg_in) | 900K×164×2 = **296 MB** | 360K×164×2 = **118 MB** |
| Aggregated messages (N_fine_nodes × D) | **18 MB** | **15 MB** |
| `h_fe` (N_fine_elem × D) | 299K×64×2 = **37 MB** | 240K×64×2 = **29 MB** |
| **Peak per timestep** | **~370 MB** | **~180 MB** |

This is processed sequentially per timestep, so the maximum simultaneous memory footprint is one timestep's worth. Very safe within 21.5 GB.

The fine MP adds `chunk_t` inner iterations per decoder chunk. At chunk_t=32, T=128: 4 outer chunks × 32 inner iterations = 128 fine MP passes. Each pass is fast (just MessagePassingLayer on 150K nodes).

---

## Step 10.6 — Spatial Coherence Loss (using element_edge_index)

Once `batch["fine_elem_edge_index"]` is available (from WP3 rebuild in this WP), enable the deferred loss from WP9:

**No code change needed in loss.py** — it's already written with `fine_elem_edge_index=None` default.

In dataset `_read_fine()`, add:
```python
if "element_edge_index" in fine_grp:
    result["elem_edge_index"] = fine_grp["element_edge_index"][:].astype(np.int64)
```

Expose as `batch["fine_elem_edge_index"]` in `__getitem__`. `cross_scale_loss()` will automatically pick it up via the keyword argument.

---

## Step 10.7 — WP3 Rebuild (Second Time)

WP10 adds `element_edge_index` to the WP3 fine group. This requires a second WP3 rebuild with the updated `build_features.py`.

**Only needed if WP8 rebuild did NOT include element adjacency.** The rebuild procedure is identical to WP8 Step 8.1.

WP3 size after WP10 rebuild: ~79 GB (element adjacency adds ~0.5 GB — negligible).

---

## Tests to Add

```python
def test_fine_mp_output_shapes():
    model = CrossScaleNet(hidden_dim=HIDDEN, use_fine_mp=True, fine_mp_steps=2)
    batch = _fake_batch()
    # Add node_coarse_map to batch
    batch["fine_node_coarse_map"] = torch.randint(0, M, (N_FINE_NODES,))
    out = model(batch)
    assert out["fine"].shape == (T, N_FINE_ELEM, 4)
    assert not torch.isnan(out["fine"]).any()

def test_fine_mp_gradients_flow():
    model = CrossScaleNet(hidden_dim=HIDDEN, use_fine_mp=True)
    batch = _fake_batch()
    batch["fine_node_coarse_map"] = torch.randint(0, M, (N_FINE_NODES,))
    out = model(batch)
    fine_features = torch.randn(T, N_FINE_NODES, 4)
    coarse_targets = torch.rand(T, M, 4)
    loss, _ = cross_scale_loss(out["fine"], fine_features, batch["fine_elements"],
                                out["coarse"], coarse_targets)
    loss.backward()
    for name, p in model.named_parameters():
        if "fine_mp" in name:
            assert p.grad is not None, f"No gradient for fine MP param {name}"

def test_phase1_phase2_output_consistent_without_fine_mp():
    """use_fine_mp=False and use_fine_mp=True with no fine_mp_layers produce same coarse output."""
    torch.manual_seed(0)
    batch = _fake_batch()
    batch["fine_node_coarse_map"] = torch.randint(0, M, (N_FINE_NODES,))
    m1 = CrossScaleNet(hidden_dim=HIDDEN, use_fine_mp=False)
    m2 = CrossScaleNet(hidden_dim=HIDDEN, use_fine_mp=True, fine_mp_steps=0)
    m2.load_state_dict(m1.state_dict())
    m1.eval(); m2.eval()
    with torch.no_grad():
        o1, o2 = m1(batch), m2(batch)
    assert torch.allclose(o1["coarse"], o2["coarse"], atol=1e-6)
```

---

## WP10 Gate Checklist

```
WP3 changes:
  [ ] build_element_adjacency() in wp3_features/graph.py
  [ ] element_edge_index stored in WP3 under fine/element_edge_index
  [ ] WP3 rebuilt (or fine group patched for existing sims)
  [ ] dataset._read_fine() loads elem_edge_index when present

CrossScaleNet:
  [ ] fine_mp_layers (empty ModuleList when use_fine_mp=False)
  [ ] Per-timestep fine MP loop implemented (when use_fine_mp=True)
  [ ] node_coarse_map scatter working (zero for unmapped nodes)
  [ ] --use-fine-mp flag in train.py

Level 2 overfit test (use_fine_mp=True):
  [ ] No OOM on Batch A sims (~370 MB peak per timestep)
  [ ] Gradient flows to fine_mp_layers parameters
  [ ] fine/stress_mae not regressing vs Phase 1 (WP9) baseline
  [ ] Spatial coherence loss enabled (fine_elem_edge_index in batch)

Tests:
  [ ] test_fine_mp_output_shapes
  [ ] test_fine_mp_gradients_flow
  [ ] test_phase1_phase2_output_consistent_without_fine_mp
```

---

## Files Modified

| File | Change |
|---|---|
| `wp3_features/graph.py` | Add `build_element_adjacency()` |
| `wp3_features/build_features.py` | Store `element_edge_index` in `--include-fine-features` build |
| `model/dataset.py` | Add `fine_node_coarse_map` + `fine_elem_edge_index` to `_read_fine()` |
| `model/cross_scale.py` | Add `use_fine_mp`, `fine_mp_layers`, per-timestep MP loop |
| `training/train.py` | Add `--use-fine-mp` flag, wire into model config |
| `tests/test_cross_scale.py` | Add 3 new tests |

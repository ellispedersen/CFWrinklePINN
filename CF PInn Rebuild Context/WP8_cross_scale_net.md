# WP8 — CrossScaleNet: Dual-Mesh Physics-Informed Architecture
## Track B Baseline — Cross-Scale Feature Transfer

**Depends on:** WP1–WP7 complete, WP3 rebuilt with `--include-fine-features`
**Gate before WP9:** Level 2 overfit passes (fine/stress_mae decreasing, no NaN)
**Status:** Architecture COMPLETE — awaiting WP3 rebuild to begin training

---

## Context

Track A (FormingGraphNet) correlates coarse mesh features to aggregated fine-mesh targets. It works but is limited: the wrinkle signal lives at fine-mesh resolution. Two geometrically similar sims can have nearly identical coarse features yet different wrinkle outcomes — the discriminating signal (compressive fiber stress at nucleation sites) only resolves at fine scale.

Track B (CrossScaleNet) trains the same coarse encoder but adds a direct fine-mesh supervision signal: it predicts fine-mesh physics fields per element and is supervised by ground-truth fine-mesh data. The coarse-fine mesh divergence IS the wrinkle signal.

**Both tracks coexist in the same repo. Switching is `--model-type coarse` vs `--model-type cross-scale`.**

---

## Architecture (already implemented in `model/cross_scale.py`)

```
Coarse encoder (shared with FormingGraphNet):
  node_encoder: Linear(74, D)
  material_encoder: MaterialEncoder(8, 32)
  temporal: TemporalAggregator(D) — GRU + attention per node, mean-pooled
  mp_layers: MessagePassingLayer × 3
  timestep_decoder: GRU(D, D), chunked over decoder_chunk_t=32
  → h_coarse_elem: (T, M_coarse, D) — per-element, per-timestep

Cross-scale scatter:
  c_idx = coarse_to_fine[0]  # (N_mapping,) — coarse element
  f_idx = coarse_to_fine[1]  # (N_mapping,) — fine element
  h_fine_elem = zeros(T, N_fine_elem, D)
  h_fine_elem[:, f_idx, :] = h_coarse_elem[:, c_idx, :]
  (fine elements without a mapping entry receive zero embedding)

Output heads:
  coarse_head: Linear(D, D//2) + SiLU + Linear(D//2, 4)
    → (T, M_coarse, 4) — backward-compatible with wrinkle_loss / Track A gates
  fine_head: Linear(D, D) + SiLU + Linear(D, 4)
    → (T, N_fine_elem, 4) — [fiber_stress_1, fiber_stress_2, displacement_z, thickness]
```

---

## Data Requirements

### WP3 Fine Features (added by `--include-fine-features` rebuild)

Per sim under `simulations/{sim_id}/fine/`:

| Path | Shape | Notes |
|---|---|---|
| `mesh_nodes` | (N_fine, 3) | Fine node XYZ positions |
| `mesh_elements` | (N_fine_elem, 3) | Fine triangle connectivity |
| `edge_index` | (2, E_fine) | Fine node-level graph |
| `edge_attr` | (E_fine, 4) | Fine edge [dx, dy, dz, dist] |
| `resampled/fiber_stress_1` | (256, N_fine) | Resampled fine-node field |
| `resampled/fiber_stress_2` | (256, N_fine) | |
| `resampled/displacement_z` | (256, N_fine) | dz component only |
| `resampled/thickness` | (256, N_fine) | |

And under `simulations/{sim_id}/graph/`:
- `coarse_to_fine_index` (2, N_mapping) — already in WP3, row 0=coarse_elem, row 1=fine_elem

### Dataset Batch Keys (when `include_fine=True`)

```python
batch["fine_features"]      # (T, N_fine_nodes, 4) — [fs1, fs2, dz, thick] at node level
batch["fine_elements"]      # (N_fine_elem, 3) int64 — node connectivity
batch["fine_edge_index"]    # (2, E_fine) int64 — node-level graph
batch["fine_edge_attr"]     # (E_fine, 4)
batch["fine_nodes"]         # (N_fine_nodes, 3)
batch["coarse_to_fine"]     # (2, N_mapping) int64
```

---

## Step 8.1 — Rebuild WP3 with Fine Features

**Run once** before any Track B training. Deletes and rebuilds the 59 GB WP3 file:

```bash
# New file will be ~79 GB (+20 GB fine data)
# 769 GB free on WSL — no storage risk
cd '/mnt/c/Users/ellis/Documents/VS Code/CFWrinklePINN'
source /home/ellis/venvs/cfwrinkle/bin/activate
export LD_PRELOAD=/opt/rocm-7.2.0/lib/libamdhip64.so
export PYTHONUNBUFFERED=1
python -u -m wp3_features.build_features \
  --include-fine-features \
  2>&1 | tee /home/ellis/cfwrinkle/logs/wp3_rebuild_fine.log
```

Expected duration: 2–4 hours.

**Validate after rebuild:**
```bash
python -c "
import h5py
f = h5py.File('/home/ellis/cfwrinkle/data/cfwrinkle_wp3_features.h5', 'r')
sim = sorted(f['simulations'].keys())[0]
g = f[f'simulations/{sim}/fine']
assert 'mesh_nodes' in g, 'mesh_nodes missing'
assert g['resampled/fiber_stress_1'].shape == (256, g['mesh_nodes'].shape[0]), 'shape mismatch'
# Confirm coarse data intact
assert f[f'simulations/{sim}/coarse/resampled/fields'].shape[2] == 37, 'coarse broken'
f.close()
print('WP3 fine features validated OK')
"
```

---

## Step 8.2 — Training Scripts

These are parallel to Track A scripts. Train in separate tmux sessions; checkpoints written to separate directories.

### `run_cross_scale_level2.sh` (Level 2 overfit, 3 sims)

```bash
#!/usr/bin/env bash
set -e
source /home/ellis/venvs/cfwrinkle/bin/activate
export LD_PRELOAD=/opt/rocm-7.2.0/lib/libamdhip64.so
export PYTHONUNBUFFERED=1
export PYTORCH_HIP_ALLOC_CONF="garbage_collection_threshold:0.8,max_split_size_mb:512"

python -u -m training.train \
  --model-type cross-scale \
  --sim-ids "geom_0_0_pair1,geom_0_0_pair2,geom_0_10" \
  --epochs 50 --hidden-dim 64 --attn-batch-nodes 512 \
  --max-timesteps 128 --temporal-strategy tail --amp --device cuda \
  --output /home/ellis/cfwrinkle/checkpoints/progressive/cross_scale_level2
```

### `run_cross_scale_level3.sh` (Level 3, 13 sims)

```bash
#!/usr/bin/env bash
set -e
source /home/ellis/venvs/cfwrinkle/bin/activate
export LD_PRELOAD=/opt/rocm-7.2.0/lib/libamdhip64.so
export PYTHONUNBUFFERED=1
export PYTORCH_HIP_ALLOC_CONF="garbage_collection_threshold:0.8,max_split_size_mb:512"

python -u -m training.train \
  --model-type cross-scale \
  --fold 3 --max-train-sims 13 --max-val-sims 2 \
  --epochs 50 --hidden-dim 64 --attn-batch-nodes 512 \
  --max-timesteps 128 --temporal-strategy tail --amp --device cuda \
  --output /home/ellis/cfwrinkle/checkpoints/progressive/cross_scale_level3
```

### `run_cross_scale_level4.sh` (Level 4, full 5-fold CV)

```bash
#!/usr/bin/env bash
set -e
source /home/ellis/venvs/cfwrinkle/bin/activate
export LD_PRELOAD=/opt/rocm-7.2.0/lib/libamdhip64.so
export PYTHONUNBUFFERED=1
export PYTORCH_HIP_ALLOC_CONF="garbage_collection_threshold:0.8,max_split_size_mb:512"

python -u -m training.train \
  --model-type cross-scale \
  --all-folds \
  --epochs 50 --hidden-dim 64 --attn-batch-nodes 512 \
  --max-timesteps 128 --temporal-strategy tail --amp --device cuda \
  --output /home/ellis/cfwrinkle/checkpoints/progressive/cross_scale_level4_cv
```

---

## Step 8.3 — VRAM Budget

Fine feature memory per sim at T=128, AMP FP16:

| Tensor | Batch A | Batch B |
|---|---|---|
| `fine_features` (128 × N_fine × 4) | **154 MB** | 124 MB |
| `h_fine_elem` per decoder chunk (32 × N_fine_elem × 64) | **1.22 GB** | 0.98 GB |
| `h_fine_elem` stored with grad checkpointing | recomputed | recomputed |
| Estimated peak forward | **6–8 GB** | 4–6 GB |
| Estimated backward (~2×) | **12–16 GB** | 8–12 GB |

Run dry-run before first Level 3 to confirm:
```bash
python -m training.train \
  --model-type cross-scale --dry-run \
  --sim-ids "geom_0_0_pair1" \
  --hidden-dim 64 --attn-batch-nodes 512 \
  --max-timesteps 128 --amp --device cuda
```

If Batch A sims OOM: reduce `--decoder-chunk-t 16` (halves h_fine_elem peak at cost of more GRU chunks).

---

## Step 8.4 — Loss Function (existing `cross_scale_loss`)

**File**: `model/loss.py` lines 97–167 (already implemented)

```
Fine targets computed on-the-fly: fine_features[:, fine_elements, :].mean(dim=2)
→ (T, N_fine_elem, 4) — corner-node average per fine element

Components:
  l_fs1 (MSE on fiber_stress_1)       weight 2.0
  l_fs2 (MSE on fiber_stress_2)       weight 2.0 (shared with fs1)
  l_dz  (Huber on displacement_z)     weight 1.5
  l_thick (MSE on thickness)          weight 1.0
  l_fine_phys (monotonicity on fs1)   weight 1.0
  l_coarse (wrinkle_loss on coarse)   weight 1.0 — backward-compat Track A

All components self-normalized via _safe_norm(floor=1e-2).
```

Additional physics losses are added in WP9.

---

## Step 8.5 — Fine-Mesh Evaluation Metrics

**File**: `training/evaluate.py:compute_fine_metrics()` (already implemented)

Added to `evaluate_split()` return dict when model is CrossScaleNet:

| Metric | Meaning |
|---|---|
| `fine/stress_mae` | Average of fiber_stress_1 and fs2 MAE vs element-averaged targets |
| `fine/dz_mae` | Displacement_z MAE |
| `fine/thick_mae` | Thickness MAE |
| `fine/compressive_frac` | Fraction of fine elements with predicted fs1 < −0.05 |

Gate check (`training/gate_check.py`) reads these from `history.json` for Track B gates.

---

## WP8 Gate Checklist

```
Rebuild:
  [ ] WP3 rebuilt with --include-fine-features
  [ ] fine/mesh_nodes shape confirmed (150,712 for Batch A)
  [ ] fine/resampled/fiber_stress_1 shape (256, 150,712) for Batch A
  [ ] coarse data intact after rebuild

Level 2 (overfit on 3 sims):
  [ ] Loss components: no NaN in loss/fine_stress_1 or loss/fine_dz
  [ ] fine/stress_mae decreasing over epochs
  [ ] fine/compressive_frac > 0 (model predicting compressive zones)
  [ ] coarse head detection_rate tracking (backward-compat)

Level 3 (mini-train):
  [ ] val_loss (combined) decreasing
  [ ] fine/stress_mae < 0.15 on val set
  [ ] No OOM on Batch B sims (N=30,603 coarse nodes)

Level 4 (full CV):
  [ ] All 5 folds complete
  [ ] Coarse gate: mean detection_rate > 0.5, best > 0.7, mean F1 > 0.4
  [ ] Fine gate: mean fine/stress_mae < 0.1
  [ ] Comparison table: Model A vs Model B per fold
```

---

## Files

| File | Status | Change |
|---|---|---|
| `model/cross_scale.py` | ✅ Done | CrossScaleNet architecture |
| `model/loss.py` | ✅ Done | cross_scale_loss (expanded in WP9) |
| `model/dataset.py` | ✅ Done | include_fine=True loading |
| `training/train.py` | ✅ Done | --model-type cross-scale wiring |
| `training/evaluate.py` | ✅ Done | compute_fine_metrics() |
| `wp3_features/build_features.py` | ✅ Done | --include-fine-features flag |
| `tests/test_cross_scale.py` | ✅ Done | 11 unit tests passing |
| `run_cross_scale_level2.sh` | 🔲 Create | Step 8.2 |
| `run_cross_scale_level3.sh` | 🔲 Create | Step 8.2 |
| `run_cross_scale_level4.sh` | 🔲 Create | Step 8.2 |

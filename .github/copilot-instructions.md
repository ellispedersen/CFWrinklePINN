# Copilot Instructions — CFWrinklePINN
## Physics-Informed GNN for Composite Forming Wrinkle Prediction

**Updated:** 2026-04-06
**Current status:** WP1–5 complete. Implement WP6 (model) then WP7 (training).

---

## Work Package Status

| WP | Name | Status | Key Deliverable |
|---|---|---|---|
| WP1 | Reverse Engineering | ✅ Done | `config/field_registry.yaml` locked |
| WP2 | Data Structures | ✅ Done | `data/cfwrinkle_dataset.h5` (55 GB) |
| WP3 | Extraction & Ingestion | ✅ Done | `data/cfwrinkle_wp3_features.h5` (58 GB) |
| WP4 | Feature Engineering | ✅ Done | `wp3_features/physics.py` (37 features) + `wp3_features/targets.py` |
| WP5 | Stratification & CV | ✅ Done | Stratified 5-fold splits in WP2 HDF5 under `splits/` |
| WP6 | Model Architecture | 🔲 **Next** | `model/` — GNN, dataset, loss |
| WP7 | Training & Evaluation | 🔲 After WP6 | `training/` — train loop, eval, viz, inference |

**Do not begin WP7 until WP6 gate checklist passes.**

---

## WP Plan Documents

Full Codex-ready implementation instructions are in:

```
C:\Users\ellis\Documents\VS Code\CFWrinklePINN\CF PInn Rebuild Context
  INDEX.md                    ← Master index, dataset facts, HDF5 schema
  WP6_model_architecture.md   ← GNN design, dataset interface, loss — implement first
  WP7_training_evaluation.md  ← Training loop, progressive tests, visualisation, inference
```

**Read INDEX.md first.** It contains the WP3 HDF5 dataset structure (shapes, paths, field names) that all WP6/7 code depends on.

---

## Project Goal (Overall Directive)

Build a physics-informed spatiotemporal GNN that:
1. Takes a **single set of coarse-mesh AniForm results** as input (no fine mesh at inference)
2. Predicts **where and when wrinkles will form** on the composite sheet
3. Provides a **screening tool** to flag high-risk geometry/process combinations

Deliver:
- Progressive test runs (smoke → overfit → mini-train → full CV) with human-verifiable checkpoints at each stage
- Dynamic visualisations: training curves, mesh-based severity maps, temporal animations, cross-fold summaries
- Single-simulation inference pipeline (raw AniForm dir → wrinkle predictions + visualisation)

---

## Hard Constraints

- **Inference uses only coarse mesh** — no fine mesh topology, no refinement ratio as a feature
- **No PyTorch Geometric** — custom `scatter_add` message passing only
- **ROCm compatibility** — no custom CUDA extensions. Safe ops: `nn.GRU`, `scatter_add_`, `nn.LayerNorm`, `nn.MultiheadAttention` (standard), `torch.linalg.eigvalsh`
- **Batch size = 1** (variable-size graphs — no default batching across sims)
- **65 training simulations** — keep hidden_dim=64, regularise aggressively
- Raw data in `CFWrinklePredict2/` is **read-only** — never copy or modify

---

## Environment

```powershell
cd "C:\Users\ellis\Documents\VS Code\CFWrinklePINN"
.\.venv\Scripts\Activate.ps1   # Python 3.12, --system-site-packages (inherits PyTorch+ROCm)
```

```bash
# Verify GPU
python -c "import torch; print(torch.version.hip); print(torch.cuda.get_device_name(0))"

# Run existing tests (must stay green)
pytest tests/ -q
```

**GPU:** AMD RX 7900 XT, 21.5 GB VRAM, ROCm 7.2, PyTorch 2.9.1+rocmsdk

---

## Critical Files to Read Before Writing WP6/7 Code

| File | Why |
|---|---|
| `CLAUDE.md` | AniForm reader APIs, field identity table, data conventions |
| `config/field_registry.yaml` | All 20 confirmed AFR fields (locked) |
| `config/pipeline_config.yaml` | All paths (use these, no hardcoding) |
| `wp3_features/physics.py` | 37 physics feature names and computation |
| `wp3_features/targets.py` | 4 wrinkle target field definitions |
| `wp2_build/schema.py` | WP2 HDF5 field definitions |
| `C:\...\CF PInn Rebuild Context\INDEX.md` | WP3 HDF5 structure reference |

---

## WP3 HDF5 Quick Reference

**File:** `data/cfwrinkle_wp3_features.h5`
**Per-sim path:** `simulations/<sim_id>/`

| Path | Shape | Notes |
|---|---|---|
| `coarse_fields_resampled` | (256, N_nodes, 37) | Physics features |
| `coarse_rates_resampled` | (256, N_nodes, 37) | Temporal rates |
| `graph/edge_index` | (2, N_edges) | int32, undirected |
| `graph/edge_attr` | (N_edges, 4) | [dx, dy, dz, dist] |
| `material_card` | (8,) | Global conditioning |
| `targets/wrinkle_severity` | (256, N_elem) | PRIMARY target [0,1] |
| `targets/comp_frac_elem` | (256, N_elem) | Auxiliary |
| `targets/oop_max_elem` | (256, N_elem) | Auxiliary |
| `targets/thickness_variance_elem` | (256, N_elem) | Auxiliary |
| `mesh/coarse_nodes` | (N_nodes, 3) | Reference XYZ |
| `mesh/coarse_elements` | (N_elem, 3) | Triangle connectivity |

**CV splits** are in the WP2 HDF5: `data/cfwrinkle_dataset.h5` under `splits/fold_*`.

---

## Commands

```bash
# Run all tests
pytest tests/ -q

# Lint
ruff check .

# Level 1 smoke test (after WP6 implemented)
python -m training.train --sim-ids geom_0_2_pair1 --epochs 1 --output checkpoints/smoke/

# Interactive data verification
python -c "
from training.visualize import quick_verify_sample
quick_verify_sample('geom_0_2_pair1', 'data/cfwrinkle_wp3_features.h5')
"

# Single-sim inference (after WP7 implemented)
python -m training.infer --results-dir 'path/to/geom_X.Results' --checkpoint checkpoints/fold_0/best.pt --visualize
```

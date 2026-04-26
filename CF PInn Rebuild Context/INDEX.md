# Aniform Composite Forming — Physics-Informed GNN
## Master Index

---

## Project Goal

A proof-of-concept physics-informed spatiotemporal GNN that takes coarse-mesh Aniform forming simulations as input and predicts wrinkle onset — where and when the fine-mesh solution diverges from the coarse. The deliverable is a screening tool that flags high-risk geometry/process combinations before committing to multi-day fine-mesh runs.

**This is a wrinkle risk classifier trained on coarse-to-fine residuals. It is not a surrogate model.**

**At inference: only a single set of coarse-mesh AniForm results is needed — no fine mesh.**

---

## Dataset Facts

| Property | Value |
|---|---|
| Total simulations | 65 valid (66 input — 1 excluded: `geom_0_6` corrupted coarse mesh) |
| Unique geometries | Batch A: 21 UD thermoplastic 2-ply, Batch B: 44 Twintex 2×2 twill 3-ply |
| Raw output | ~55 GB WP2 HDF5, ~79 GB WP3 feature HDF5 (fine features included) |
| Timestep grid | 256 uniform stroke-fraction points per simulation (resampled) |
| Hardware | AMD 7900 XT (21.5 GB VRAM, ROCm 7.2) |
| PyTorch | 2.9.1+rocmsdk, Python 3.12 venv |
| CV splits | Stratified 5-fold (material × severity quartile), written into WP2 HDF5 |

---

## Hard Constraints (never violate)

- At inference: **only the coarse mesh exists** — no fine mesh topology, no refinement ratio as a feature
- Coarse and fine meshes **share no nodes** — correspondence built at WP3 parse time
- **65 training simulations** — every architecture and regularisation decision must respect this
- **ROCm compatibility** — no custom CUDA extensions; GRU, scatter_add, LayerNorm, standard attention are safe
- **No PyTorch Geometric** — use custom scatter_add message passing
- Batch size is 1 (variable-size graphs — cannot naively batch across sims)

---

## Work Package Sequence

```
TRACK A — Geometry-Based Baseline (FormingGraphNet)
────────────────────────────────────────────────────────────

WP1  Reverse Engineering          ✅ COMPLETE — gate passed 2026-03-29
  └─ field_registry.yaml locked, all 20 AFR fields identified

WP2  Data Structures & Schema     ✅ COMPLETE — gate passed
  └─ data/cfwrinkle_dataset.h5 (55 GB), stratified 5-fold CV splits written

WP3  Extraction & Ingestion       ✅ COMPLETE — gate passed
  └─ data/cfwrinkle_wp3_features.h5 (59 GB, coarse only), 65/66 sims built

WP4  Feature Engineering          ✅ COMPLETE (delivered as part of WP3)
  └─ wp3_features/physics.py: 37-feature tensor per node per timestep
  └─ wp3_features/targets.py: 4 wrinkle target fields per element per timestep

WP5  Stratification & CV Setup    ✅ COMPLETE (delivered as part of WP2)
  └─ Stratified 5-fold CV splits in WP2 HDF5 under splits/fold_*

WP6  Model Architecture           ✅ COMPLETE — gate passed
  └─ model/gnn.py (FormingGraphNet), model/layers.py, model/loss.py, model/dataset.py

WP7  Training & Evaluation        ✅ COMPLETE — Level 4 gate PASSED
  └─ Gate artifact: reports/wp7_gate_level4.json
  └─ training/train.py, training/gate_check.py, wsl_progressive_suite.sh


TRACK B — Physics-Informed Cross-Scale Model (CrossScaleNet)
────────────────────────────────────────────────────────────
Both tracks coexist. Switch with --model-type coarse|cross-scale.

WP8  CrossScaleNet Architecture   ✅ COMPLETE
  └─ model/cross_scale.py integrated with train/eval/gate flow
  └─ See WP8_cross_scale_net.md

WP9  Physics-Informed Losses      ✅ COMPLETE
  └─ buckling_onset_loss, thickness_dz_coupling_loss, dz_mono, spatial_coherence implemented
  └─ See WP9_pinn_losses.md

WP10 Fine-Mesh Message Passing    ✅ COMPLETE
  └─ Fine element adjacency graph, node_coarse_map scatter, per-timestep MP loop implemented
  └─ See WP10_fine_mesh_mp.md

WP11 Track B Training & Gate      ✅ COMPLETE — Level 4 gate PASSED 2026-04-25
  └─ Full report: reports/CROSSSCALE_LEVEL4_TRAINING_REPORT_2026-04-25.md
  └─ WP doc: WP11_track_b_training.md (issues, ROCm constraints, Track C forward work)
  └─ Gate artifact: reports/wp7_gate_level4_cross_scale.json
  └─ 5-fold CV: mean val_loss=0.0854; Track A vs B: reports/track_ab_comparison.json
  └─ Inference pipeline verified; normalization bug fixed in extract_single.py
  └─ Test suite: 170 passed, 1 skipped (2026-04-25)
  └─ 4 open issues → Track C (Track B checkpoints preserved as reference baseline):
     1. dz R²<-8 in folds 0/2/4 — increase fine_dz loss weight (see WP11)
     2. Stress spatial R²≈0.09 — enable use_fine_mp=True (see WP10/WP11)
     3. Batch B R² unstable — MAE-only evaluation for near-flat fields
     4. mold_set_004 low-signal sim inflating summaries


TRACK C — CrossScaleNet Improved (fine_mp ON, dz loss tuned)
────────────────────────────────────────────────────────────
Track B preserved as reference. Track C checkpoints to separate output directory.
Switch with --model-type cross-scale (same flag; checkpoint config carries use_fine_mp).

WP12 Track C Training              🔲 NOT STARTED
  └─ Changes from Track B:
     - use_fine_mp=True (enable WP10 fine-mesh message passing)
     - fine_dz loss weight: 1.5 → 3.0–5.0 (fix dz fold-lottery)
     - Batch B R² guard in evaluate.py (min-variance threshold)
  └─ Output dir: /home/ellis/cfwrinkle/checkpoints/progressive/cross_scale_level4_cv_trackc/
  └─ Track B reference dir: /home/ellis/cfwrinkle/checkpoints/progressive/cross_scale_level4_cv/
  └─ See WP11_track_b_training.md for full issue specs
```

---

## Work Package Files

| File | Content | Track | Status |
|---|---|---|---|
| `WP1_reverse_engineering.md` | Input archaeology, field identification, validation gate | A | ✅ Done |
| `WP2_data_structures.md` | HDF5 schema, field registry, provenance contract | A | ✅ Done |
| `WP3_extraction_ingestion.md` | Aniform readers, mesh correspondence, temporal resampling | A | ✅ Done |
| `WP4_feature_engineering.md` | Physics features, targets — reference doc (implemented in WP3) | A | ✅ Done |
| `WP5_stratification.md` | Label scheme, fold construction — reference doc (implemented in WP2) | A | ✅ Done |
| `WP6_model_architecture.md` | FormingGraphNet design, dataset interface, loss functions | A | ✅ Done |
| `WP7_training_evaluation.md` | Training loop, progressive test runs, visualisation, inference | A | ✅ Done |
| `WP8_cross_scale_net.md` | CrossScaleNet architecture, WP3 rebuild, Phase 1 training | B | ✅ Done |
| `WP9_pinn_losses.md` | Physics-informed losses: buckling onset, coupling, coherence | B | ✅ Done |
| `WP10_fine_mesh_mp.md` | Fine element adjacency, node_coarse_map, per-timestep fine MP | B | ✅ Done |
| `WP11_track_b_training.md` | Level 4 CV outcomes, 4 open quality issues, ROCm constraints, Track C forward work | B | ✅ Done — 4 open items |
| `WP12_track_c_training.md` | Track C training plan: use_fine_mp=True, dz loss tuning, Batch B eval fix | C | 🔲 Not started |

---

## WP3 Feature HDF5 Structure (input to WP6/7)

Each simulation in `data/cfwrinkle_wp3_features.h5` under `simulations/<sim_id>/`:

```
graph/
  edge_index              (2, N_edges)          int32  — undirected triangle mesh edges
  edge_attr               (N_edges, 4)          float32 — [dx, dy, dz, dist] relative geometry
coarse_fields_resampled   (256, N_nodes, 37)    float32 — physics feature tensor
coarse_rates_resampled    (256, N_nodes, 37)    float32 — temporal rates of features
targets/
  wrinkle_severity        (256, N_coarse_elem)  float32 — PRIMARY target [0,1]
  comp_frac_elem          (256, N_coarse_elem)  float32 — compressive fiber fraction
  oop_max_elem            (256, N_coarse_elem)  float32 — max out-of-plane displacement
  thickness_variance_elem (256, N_coarse_elem)  float32 — spatial thickness variance
material_card             (8,)                  float32 — normalised material params
coarse_to_fine/
  coarse_index            (N_mappings,)         int32   — coarse element index
  fine_index              (N_mappings,)         int32   — fine element index
  n_fine_per_coarse       (N_coarse_elem,)      int32
mesh/
  coarse_nodes            (N_nodes, 3)          float32 — XYZ reference positions
  coarse_elements         (N_coarse_elem, 3)    int32   — triangle connectivity
```

Feature names (37): dx, dy, dz, temperature, eq_shear_rate, fiber_dir_1_x/y/z, fiber_dir_2_x/y/z, E11, E22, E12, eps_1, eps_2, s11, s22, s12, sigma_1, sigma_2, sigma_comp, fiber_stress_1, fiber_stress_2, fiber_strain_1, fiber_strain_2, thickness, thickness_ratio, shear_angle, locking_proximity, area_change_ratio, draw_in_distance, fiber_comp_indicator, bending_energy_proxy, thickness_rate, shear_rate, fiber_stress_rate

---

## Cross-Cutting Conventions (apply everywhere)

- **All paths** live in `config/pipeline_config.yaml` — no hardcoded paths in module code
- **field_registry.yaml** is locked — do not modify
- **Raw solver outputs are immutable** — reference in place from `CFWrinklePredict2/`
- Precompute everything the dataloader needs at parse time — no on-the-fly spatial queries
- `batch_first=True` for all GRU/attention ops
- Windows environment, bash shell, forward slashes in Python paths

---

## Environment Setup

```bash
cd "C:\Users\ellis\Documents\VS Code\CFWrinklePINN"
.\.venv\Scripts\Activate.ps1

# Verify ROCm PyTorch
python -c "import torch; print(torch.version.hip); print(torch.cuda.is_available())"

# Verify key ops
python -c "
import torch
x = torch.randn(10, 64, device='cuda')
gru = torch.nn.GRU(64, 64, batch_first=True).cuda()
idx = torch.zeros(10, dtype=torch.long, device='cuda')
out = torch.zeros(5, 64, device='cuda').scatter_add_(0, idx.unsqueeze(1).expand(-1,64), x)
print('GRU + scatter_add OK')
"
```

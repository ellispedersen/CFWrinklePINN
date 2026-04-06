# WP6 & WP7 — Model Architecture & Training
## Outline Only — Do Not Implement Until WP1–5 Gates Pass

**Depends on:** WP5 gate passed, fold_composition_report.txt reviewed
**Status:** This document is an implementation outline. Open it; do not act on it until WP5 is complete.

---

## WP6 — Model Architecture

### Context from WP5 (read before any architecture decisions)

- **Effective training set:** ~32 pairs per fold
- **Validation set:** ~8 pairs per fold, stratified across material × geometry × outcome
- **Node count per graph:** varies by geometry — check actual range from dataset.h5 before fixing hidden_dim
- **Feature tensor shape:** (256 timesteps, N_nodes, N_physics_features) — N_physics_features determined by WP4

Every capacity decision must be defended against 32 training simulations. Default to smaller; grow only with evidence from validation curves.

---

### Architecture

**Physics-Informed Spatiotemporal GNN**

```
Input per pair:
  material_card:   (N_mat_params,)                — scalar conditioning
  node_features:   (256, N_nodes, N_physics_feat) — temporal feature sequence
  edge_index:      (2, N_edges)                   — coarse mesh topology
  edge_attr:       (N_edges, 4)                   — [dx, dy, dz, dist]

─────────────────────────────────────────────────

Material Encoder
  MLP: N_mat_params → 64 → material_embed_dim (32)
  Activation: SiLU
  Injected at every message passing step, not only at input

Node Feature Encoder
  Linear: N_physics_feat → hidden_dim
  No activation — preserve linear feature relationships

Hybrid Temporal Aggregator
  GRU:
    input_size  = hidden_dim
    hidden_size = hidden_dim
    num_layers  = 1
    batch_first = True
    Applied per-node over the 256-step sequence
  Attention-weighted pooling over GRU states:
    n_heads      = 2 (maximum — more heads = more overfit risk)
    dropout      = 0.3
    Positional encoding: sinusoidal over stroke_fraction (0→1)
    NOT over sequence index — physical time, not position in array

Message Passing (3 steps — start with 2, add 1 if val loss warrants)
  Per step:
    message_fn: Linear(hidden_dim*2 + edge_feat_dim + material_embed_dim → hidden_dim)
    update_fn:  Linear(hidden_dim*2 → hidden_dim) with residual
    norm:       LayerNorm(hidden_dim)
    material_embed injected at every step

Output Head
  MLP: hidden_dim → hidden_dim//2 → 1
  Activation: SiLU → identity
  Per-node wrinkle severity scalar
  NaN-masked elements excluded from loss (boundary elements from WP4)

─────────────────────────────────────────────────

Recommended starting hidden_dim: 64
Grow to 128 only if val curves show underfitting after 3+ folds
```

### Heterogeneous Edge Types

The graph has two qualitatively different edge types that carry different information:
- **Mesh topology edges** — structural connectivity, load transfer paths
- **Geometric proximity edges** — spatial neighbourhood, relevant for diffuse stress fields

Consider using separate `message_fn` weights per edge type if initial results suggest the model conflates them. This is optional complexity — start with a single message function.

---

### Composite Physics Loss

```
L_total = w₁·L_thickness    + w₂·L_wrinkle    + w₃·L_strain_rate
        + w₄·L_principal     + w₅·L_physics

Component          Weight    Type         Notes
──────────────────────────────────────────────────────────────────
L_thickness        1.0       Huber δ=0.1  Base reconstruction — all elements
L_wrinkle          3.0       Huber        Weighted by target wrinkle_severity
                                          spatially sparse — this weight matters
L_strain_rate      1.5       MSE          Weighted by stroke phase
                                          (higher weight near wrinkle onset timing)
L_principal        1.5       MSE          Compressive principal stress only;
                                          signed — zero loss for tensile regions
L_physics          2.0       custom       CLT consistency, volume conservation,
                                          shear locking constraint (2×2 only),
                                          Huber wrinkling criterion consistency
```

**Implementation notes:**
- Normalise each component by its running mean magnitude before applying relative weights — prevents any single component from dominating due to scale differences
- Material-conditioned weights: UD emphasises L_principal; 2×2 twill emphasises L_strain_rate and shear locking term in L_physics
- NaN elements (boundary, no fine-mesh coverage) masked in all loss components — use `torch.nan_to_num` or explicit mask

---

### Regularisation

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=20, T_mult=2
)
```

---

### ROCm Compatibility Checklist

All ops used must be in the tested PyTorch ROCm path:

| Op | Safe? |
|---|---|
| `nn.GRU` | ✓ |
| `scatter_add_` | ✓ |
| `nn.LayerNorm` | ✓ |
| `nn.MultiheadAttention` (standard) | ✓ |
| `torch.linalg.eigvalsh` | ✓ |
| Custom CUDA extensions | ✗ — do not use |
| Flash Attention | Verify before use — not always in ROCm path |

---

## WP7 — Training & Evaluation

### Dataloader

```python
# io/hdf5_reader.py — load_pair_for_training() is the only entry point
# No spatial queries, no field lookups, no on-the-fly computation
# Reads: resampled features, edge_index, edge_attr, material_card, events, targets
# Raises if feature_version doesn't match config

# Batch size: 1 simulation (graphs are variable-size — no batching across sims)
# Shuffle training set each epoch
```

### Training Loop (per fold)

```python
for fold in folds:
    model = FormingGraphNet(config)
    for epoch in range(n_epochs):
        model.train()
        for pair_id in fold['train']:
            data = hdf5_reader.load_pair_for_training(h5_path, pair_id, feature_version)
            pred = model(data)
            loss = physics_loss(pred, data['targets'], data, stroke_fractions)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_metrics = evaluate(model, fold['val'], h5_path, config)

        log_epoch(fold['fold'], epoch, loss, val_metrics)
```

### Primary Evaluation Metric

**Simulation-level wrinkle detection rate** — not node-level MSE.

For each validation simulation:
1. Aggregate per-node predictions → simulation-level max severity
2. Apply threshold → binary wrinkle prediction
3. Compare against `wrinkle_outcome` from fold labels

Report per fold and across folds:

| Metric | Definition | Priority |
|---|---|---|
| Detection rate (sensitivity) | TP / (TP + FN) | Highest — missed wrinkles are expensive |
| False alarm rate | FP / (FP + TN) | Acceptable — just triggers a fine mesh run |
| F1 score | Harmonic mean of precision and recall | Summary metric |
| Mean node-level severity MAE | Secondary — diagnostic only | Low |

### Commands

```bash
# Train on one fold (development)
python -m training.train \
    --fold 0 \
    --dataset data/dataset.h5 \
    --assignments config/fold_assignments.yaml \
    --config config/pipeline_config.yaml \
    --output checkpoints/fold_0/

# Train all 5 folds
python -m training.train \
    --all-folds \
    --dataset data/dataset.h5 \
    --assignments config/fold_assignments.yaml \
    --config config/pipeline_config.yaml \
    --output checkpoints/

# Evaluate a trained fold
python -m training.evaluate \
    --fold 0 \
    --checkpoint checkpoints/fold_0/best.pt \
    --dataset data/dataset.h5 \
    --assignments config/fold_assignments.yaml \
    --report reports/eval_fold_0.json

# Print aggregated metrics across all folds
python -m training.evaluate \
    --all-folds \
    --checkpoint-dir checkpoints/ \
    --report reports/eval_all_folds.json
```

### WP6/7 Gate

```
Architecture (WP6)
  [ ] fold_composition_report.txt read and acknowledged
  [ ] hidden_dim justified against 32 training pairs
  [ ] All ops verified ROCm-compatible
  [ ] Loss component scale check: log each component magnitude for first training batch
  [ ] NaN masking verified for boundary elements

Training (WP7)
  [ ] Training runs without OOM on 7900 XT for largest pair in dataset
  [ ] Val loss decreasing (not flat or increasing) after first 10 epochs on fold 0
  [ ] Detection rate > 0.5 on fold 0 val set before running all 5 folds
  [ ] All 5 folds complete without crash
  [ ] Final F1 reported across folds
```

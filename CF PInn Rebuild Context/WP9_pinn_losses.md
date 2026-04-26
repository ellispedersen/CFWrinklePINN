# WP9 — Physics-Informed Losses for CrossScaleNet
## Track B — Beyond Monotonicity

**Depends on:** WP8 Level 2 gate passing (cross_scale_loss working, no NaN)
**Gate before WP10:** Level 2 re-run with physics losses — each new component finite and contributing
**Status:** NOT STARTED — implement after WP8 Level 2 baseline confirmed

---

## Context

The current `cross_scale_loss` (WP8) uses only one physics constraint: `monotonicity_loss` on predicted fiber_stress_1. This penalizes temporal decreases in stress but encodes no spatial physics.

This WP adds three additional physics-derived loss components, each grounded in composite forming mechanics:

1. **Buckling onset coupling** — wrinkle amplitude only appears in compressive zones
2. **Thickness–displacement coupling** — out-of-plane buckling reduces in-plane thickness
3. **Spatial coherence** — wrinkle patterns have characteristic wavelength (Phase 3, deferred)

These are **soft constraints** (low weights) that guide training without overriding the data-fit signal.

---

## Physics Background

### Wrinkle formation mechanism
1. Compressive fiber stress builds as sheet conforms to mold geometry
2. When σ_fiber_1 < σ_critical (material-dependent), local buckling initiates
3. Buckled region deforms out-of-plane (dz increases) → thickness decreases (material conservation)
4. Wrinkle patterns have wavelength λ = 2π√(EI/σ_comp) — determined by bending stiffness and stress

### What the existing coarse model misses
- The critical buckling stress σ_critical is sub-element-scale physics
- Spatial correlation between compressive zone extent and wrinkle location only resolves at fine mesh
- Thickness–displacement coupling is a conservation law, not a statistical correlation

---

## Step 9.1 — Buckling Onset Loss

**File**: `model/loss.py` — add after existing `monotonicity_loss()`

```python
def buckling_onset_loss(
    fine_pred: torch.Tensor,  # (T, N_fine_elem, 4): [fs1, fs2, dz, thick]
    fine_tgt: torch.Tensor,   # (T, N_fine_elem, 4): element-averaged targets
) -> torch.Tensor:
    """
    Physics: Out-of-plane displacement (dz) should be ~0 in tensile zones.
    Penalises predicted dz where ground-truth fiber_stress_1 is not compressive.

    Uses target stress as the physical mask (available at train time — not a data leak
    because the constraint is a physics prior, not a shortcut to the target label).
    """
    tgt_stress = fine_tgt[..., 0]           # (T, N_fine_elem)
    pred_dz = fine_pred[..., 2]
    valid = ~torch.isnan(tgt_stress)
    tensile = (tgt_stress >= -0.05).float()  # 1 where tensile, 0 where compressive
    if valid.sum() == 0:
        return pred_dz.sum() * 0.0
    return (pred_dz.pow(2) * tensile * valid).sum() / valid.sum().clamp(min=1)
```

**Why -0.05**: Same threshold used in `comp_frac_elem` target computation (`wp3_features/targets.py` line 41). Consistent with the wrinkle severity definition.

**Weight: 0.5** (soft; does not suppress dz in valid compressive zones)

---

## Step 9.2 — Thickness–Displacement Coupling Loss

**File**: `model/loss.py`

```python
def thickness_dz_coupling_loss(fine_pred: torch.Tensor) -> torch.Tensor:
    """
    Physics: Wrinkling = material buckles out-of-plane → local thickness decreases.
    The spatial anti-correlation between |dz| and thickness should hold at each timestep.

    Formulation: cosine similarity between |dz| and thickness (spatially zero-meaned)
    should be negative. Penalise only positive correlation (relu gate).
    """
    dz = fine_pred[..., 2]       # (T, N_fine_elem)
    thick = fine_pred[..., 3]
    # Zero-mean spatially per timestep
    dz_norm = dz - dz.mean(dim=-1, keepdim=True)
    thick_norm = thick - thick.mean(dim=-1, keepdim=True)
    # cosine similarity: positive → dz and thickness co-vary (wrong)
    cos_sim = F.cosine_similarity(dz_norm.abs(), thick_norm, dim=-1)  # (T,)
    return F.relu(cos_sim).mean()  # penalise wrong sign only
```

**Weight: 0.5**

---

## Step 9.3 — Extended Monotonicity (dz)

Already in `cross_scale_loss()`: `l_fine_phys = monotonicity_loss(fine_pred[..., 0])` (fiber_stress_1).

Extend to also enforce non-decreasing wrinkle amplitude:

```python
# In cross_scale_loss():
l_fine_phys = monotonicity_loss(fine_pred[..., 0])        # fiber_stress_1 (existing)
l_dz_mono   = monotonicity_loss(fine_pred[..., 2].abs())  # NEW: |dz| non-decreasing
```

**Weight: 0.5**

---

## Step 9.4 — Updated `cross_scale_loss()` Signature

**File**: `model/loss.py` lines 97–167 — modify in place

```python
CROSS_SCALE_WEIGHTS = {
    "fine_stress": 2.0,   # fs1 + fs2 MSE
    "fine_dz":     1.5,   # dz Huber
    "fine_thick":  1.0,   # thickness MSE
    "coarse":      1.0,   # auxiliary wrinkle_loss (Track A compat)
    "physics":     1.0,   # fs1 monotonicity (existing)
    "dz_mono":     0.5,   # NEW: |dz| monotonicity
    "buckling":    0.5,   # NEW: dz~0 in tensile zones
    "coupling":    0.5,   # NEW: thick-dz anti-correlation
    "coherence":   0.3,   # deferred (WP10): graph Laplacian
}

def cross_scale_loss(
    fine_pred: torch.Tensor,
    fine_features: torch.Tensor,
    fine_elements: torch.Tensor,
    coarse_pred: torch.Tensor,
    coarse_targets: torch.Tensor,
    fine_elem_edge_index: torch.Tensor | None = None,  # for coherence (WP10)
    weights: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
```

**Updated log keys** (added to history.json):
```json
"loss/fine_buckling": 0.012,
"loss/fine_dz_mono": 0.008,
"loss/fine_coupling": 0.015
```

---

## Step 9.5 — Wire into `_compute_loss()` in `training/train.py`

```python
def _compute_loss(model_out, batch: dict) -> tuple[torch.Tensor, dict]:
    if isinstance(model_out, dict):
        return cross_scale_loss(
            model_out["fine"],
            batch["fine_features"],
            batch["fine_elements"],
            model_out["coarse"],
            batch["targets"],
            fine_elem_edge_index=batch.get("fine_elem_edge_index"),  # None until WP10
        )
    return wrinkle_loss(model_out, batch["targets"])
```

No other changes to `training/train.py`. Track A loss path unchanged.

---

## Step 9.6 — Spatial Coherence Loss (Deferred — WP10)

Requires fine element adjacency graph (not yet in WP3). Documented here for completeness.

```python
def spatial_coherence_loss(
    fine_pred: torch.Tensor,           # (T, N_fine_elem, 4)
    fine_elem_edge_index: torch.Tensor, # (2, E_adj) — element adjacency
) -> torch.Tensor:
    """
    Graph Laplacian on fine element predictions.
    Wrinkles span ~10-50 fine elements — suppresses sub-element noise
    without suppressing real wrinkle patterns.
    """
    src, dst = fine_elem_edge_index[0], fine_elem_edge_index[1]
    diff = fine_pred[:, src, :] - fine_pred[:, dst, :]  # (T, E_adj, 4)
    return diff.pow(2).mean()
```

Weight: **0.3** (very soft).

Enabled automatically when `batch["fine_elem_edge_index"]` is present (added in WP10 WP3 rebuild).

---

## Tests to Add (`tests/test_cross_scale.py`)

```python
def test_buckling_onset_loss_no_nan():
    T, M = 16, 100
    fine_pred = torch.randn(T, M, 4)
    fine_tgt = torch.randn(T, M, 4)
    loss = buckling_onset_loss(fine_pred, fine_tgt)
    assert not torch.isnan(loss)
    assert loss >= 0

def test_thickness_dz_coupling_loss_no_nan():
    loss = thickness_dz_coupling_loss(torch.randn(16, 100, 4))
    assert not torch.isnan(loss)
    assert loss >= 0

def test_cross_scale_loss_with_physics_losses():
    T, M, Nf, Ne = 16, 40, 160, 200
    fine_pred = torch.randn(T, Ne, 4)
    fine_features = torch.randn(T, Nf, 4)
    fine_elements = torch.randint(0, Nf, (Ne, 3))
    coarse_pred = torch.rand(T, M, 4)
    coarse_targets = torch.rand(T, M, 4)
    loss, log = cross_scale_loss(fine_pred, fine_features, fine_elements,
                                  coarse_pred, coarse_targets)
    assert not torch.isnan(loss)
    assert "loss/fine_buckling" in log
    assert "loss/fine_dz_mono" in log
    assert "loss/fine_coupling" in log
```

---

## WP9 Gate Checklist

```
Implementation:
  [ ] buckling_onset_loss() in loss.py — no NaN, non-negative
  [ ] thickness_dz_coupling_loss() in loss.py — no NaN, non-negative
  [ ] l_dz_mono added to cross_scale_loss()
  [ ] cross_scale_loss() signature updated (fine_elem_edge_index=None default)
  [ ] CROSS_SCALE_WEIGHTS updated
  [ ] All new loss keys in log dict

Level 2 re-run with physics losses:
  [ ] All 3 new loss components finite and positive in log
  [ ] loss/fine_buckling < 0.1 (reasonable magnitude)
  [ ] No regression in fine/stress_mae vs WP8 baseline
  [ ] Gradient flows to all model parameters

Tests:
  [ ] pytest tests/test_cross_scale.py::test_buckling_onset_loss_no_nan
  [ ] pytest tests/test_cross_scale.py::test_thickness_dz_coupling_loss_no_nan
  [ ] pytest tests/test_cross_scale.py::test_cross_scale_loss_with_physics_losses
```

---

## Files Modified

| File | Change |
|---|---|
| `model/loss.py` | Add `buckling_onset_loss()`, `thickness_dz_coupling_loss()`, extend `cross_scale_loss()`, update `CROSS_SCALE_WEIGHTS` |
| `training/train.py` | Wire `fine_elem_edge_index=None` into `_compute_loss()` |
| `tests/test_cross_scale.py` | Add 3 new tests |

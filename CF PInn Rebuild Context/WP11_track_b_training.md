# WP11 — Track B Training, Evaluation & Inference
## CrossScaleNet Level 4 Full CV — Outcomes, Issues, and Forward Work

**Depends on:** WP8 (architecture), WP9 (physics losses), WP10 (fine mesh MP)
**Gate artifact:** `reports/wp7_gate_level4_cross_scale.json` — PASS (2026-04-25)
**Full report:** `reports/CROSSSCALE_LEVEL4_TRAINING_REPORT_2026-04-25.md`
**Status:** COMPLETE (gate passed) — 4 open quality issues addressed in Track C (WP12)

> **Track B checkpoints are the reference baseline.** Do not overwrite or retrain in-place.
> Improvements are implemented as **Track C** with a separate output directory so A/B/C comparisons remain possible.
> Track C output: `/home/ellis/cfwrinkle/checkpoints/progressive/cross_scale_level4_cv_trackc/`
> Track B reference: `/home/ellis/cfwrinkle/checkpoints/progressive/cross_scale_level4_cv/`

---

## Hardware & ROCm Constraints (7900 XT Optimization Package)

Source: `reports/ROCM_OPTIMIZATION_HANDOFF_2026-04-18.md`

These constraints are **non-negotiable** for any further Track B training on this hardware:

| Constraint | Value | Reason |
|---|---|---|
| `MAX_TIMESTEPS` | ≥ 96 | Temporal fidelity floor; below this physics losses degrade |
| `--normalize-fine-features` | ON | HIP allocator instability without normalization; also required for fine metric quality |
| AMP dtype | **bfloat16 only** | float16 causes HIP segfault on RX 7900 XT (exit 139 confirmed 2026-04-24) |
| `ATTN_BATCH_NODES` | ≤ 64 for stable runs | 512 caused HIP segfault; 64 is proven stable |
| `CFWRINKLE_FINE_LOSS_CHUNK_ELEMS` | 2048 | Prevents HIP OOM on fine loss chunking |
| `CFWRINKLE_AUX_LOSS_INTERVAL` | 1 | Keep enabled; disabling causes auxiliary loss drift |
| `LD_PRELOAD` | `/opt/rocm-7.2.0/lib/libamdhip64.so` | Required for HIP visibility in WSL2 Ubuntu-24.04 |
| VRAM available | 19.94 GB | Current training uses only 0.76–1.42 GB (3.8–7.1%) — large headroom |
| Batch size | 1 | Variable-size graphs; cannot batch across sims |

Resume script: `scripts/resume_cross_scale_level4.sh` — all constraints baked in as defaults.

---

## Training Outcome Summary

| Fold | val_loss | detect_rate | F1 | Buckling loss | dz quality |
|---|---|---|---|---|---|
| fold_0 | 0.0899 | 1.000 | 1.000 | Stable ~0.18 | **FAILS** (R²≈−10) |
| fold_1 | 0.0741 | 1.000 | 1.000 | Rising → 0.471 | **WORKS** (R²≈+0.95) |
| fold_2 | 0.0842 | 1.000 | 1.000 | Stable ~0.20 | **FAILS** (R²≈−11) |
| fold_3 | 0.0889 | 1.000 | 1.000 | Rising → 0.560 | **WORKS** (R²≈+0.96) |
| fold_4 | 0.0898 | 1.000 | 1.000 | Stable ~0.21 | **FAILS** (R²≈−9) |

`detect_rate=1.000` is trivially achieved — all 65 sims in the dataset are wrinkled. Not a useful discrimination metric.

---

## 4 Open Issues

---

### Issue 1 — Displacement_z fails in folds 0, 2, 4
**Priority: HIGH**

**What happens:** Batch A (geom_0_*) validation sims show R²=−8 to −14 for fine/displacement_z in folds 0/2/4. The same sims evaluated by fold_1/fold_3 checkpoints yield R²>0.94.

**Evidence:**
- `geom_0_2` in fold_3 val: R²=+0.955 (fold_3 checkpoint)
- `geom_0_2` in fold_0 val: R²=−9.687 (fold_0 checkpoint)
- All 4 Batch A sims in fold_1 and fold_3 val: R²>0.69, mean>0.94

**Root cause:** `loss/fine_buckling` plateaued at ~0.18–0.22 early in folds 0/2/4. In folds 1/3 it rose continuously to 0.47–0.56 throughout training. The buckling onset loss (`buckling_onset_loss` in `model/loss.py`) penalises predicted dz in tensile zones. When this pressure is sustained, the model learns the spatial dz distribution. When it plateaued early, the model converged on predicting dz near the dataset mean (7.15 mm) regardless of input — catastrophic for Batch A sims where actual dz is 22–55 mm.

**Immediate workaround:** Use fold_1 or fold_3 checkpoint for Batch A production inference. All folds correctly predict dz for Batch B (near-zero dz, mae≈0.045).

**Planned fix:**
```python
# In model/loss.py cross_scale_loss — increase fine_dz weight
# Current: weight 1.5
# Proposed: weight 3.0–5.0
# Also: consider not plateauing the buckling loss weight (keep warmup ramp active longer)
```
Increase `fine_dz` loss weight so all folds maintain dz learning pressure throughout training, independent of buckling loss dynamics.

---

### Issue 2 — Fiber stress spatial R²≈0.09 everywhere
**Priority: MEDIUM**

**What happens:** `fiber_stress_1` R²=0.09–0.14 and `fiber_stress_2` R²≈0.01–0.06 uniformly across all folds and all sims. The model predicts the right mean stress level but not spatial patterns.

**Root cause:** The coarse-to-fine scatter in `model/cross_scale.py`:
```python
h_fine_elem[:, f_idx, :] = h_coarse_elem[:, c_idx, :]
```
All fine elements sharing a coarse parent receive identical embeddings. The fine head is piecewise-constant at coarse-element resolution. There is no mechanism for within-patch spatial differentiation, so spatial R² is structurally bounded regardless of training.

**Checkpoint config shows `use_fine_mp=False`** — fine-mesh message passing (WP10) was never enabled in any Level 4 run.

**Planned fix:** Enable `use_fine_mp=True` in CrossScaleNet. Architecture fully implemented in `model/cross_scale.py` (WP10). With 2 layers of fine-element message passing, fine elements within a patch can differentiate based on their local neighbourhood. Expected to substantially improve stress spatial R².

**VRAM budget:** Current peak 1.42 GB / 19.94 GB. Fine MP adds ~2× overhead over the fine head but remains well within budget.

**To enable:**
```bash
# In run_cross_scale_level4.sh or a new run_cross_scale_level5.sh
python -m training.train \
  --model-type cross-scale \
  --all-folds \
  --epochs 50 \
  --use-fine-mp \          # ← NEW
  --hidden-dim 64 \
  ...
```

---

### Issue 3 — R² metric undefined for near-flat fine fields in Batch B
**Priority: LOW**

**What happens:** Thickness and dz R² values for Batch B (mold_set_*) sims range from −105,206 to +0.6. Summary statistics are dominated by extreme outliers. The metric appears to indicate catastrophic failure but the predictions are numerically reasonable.

**Root cause:** Batch B (3-ply Twintex 2×2 twill, uniform stacking) produces near-constant fine fields:
- `fine/thickness std ≈ 0.000013 mm` (thickness = 0.8500 mm ± 0.0001 mm everywhere)
- `fine/displacement_z` range ≈ 5 mm with std < 1 mm

R² = 1 − ss_res/ss_tot collapses to −∞ when ss_tot → 0, even for accurate predictions.

**Fix:** Adopt batch-conditional evaluation:
- Batch A sims (`geom_0_*`): report both MAE and R²
- Batch B sims (`mold_set_*`): report MAE only; exclude R² or add minimum-variance guard:
```python
r2 = 1 - ss_res/ss_tot if ss_tot > MIN_VARIANCE_THRESHOLD else float("nan")
MIN_VARIANCE_THRESHOLD = 1e-4  # skip R² for near-flat fields
```
Add to `training/evaluate.py::compute_fine_metrics`.

---

### Issue 4 — mold_set_004 inflates apparent error rates
**Priority: LOW**

**What happens:** `mold_set_004` (fold_1 validation) shows dz_r2=−33.4, s1_r2=−7.9, thick_r2=−105,206 — the worst metrics of any sim in the dataset by a large margin.

**Root cause:** Same as Issue 3 but extreme:
```
fine/thickness:      std = 0.000013 mm  (constant 0.8500 mm)
fine/fiber_stress_1: std = 0.016        (vs dataset std ~0.69)
fine/displacement_z: range [−5.0, 0.0] mm
```
This is a low-amplitude, uniform Batch B sim. The model predictions are numerically reasonable in absolute terms. All catastrophic R² values are denominator-collapse artifacts.

**Fix:** Same as Issue 3 — minimum-variance guard in R² computation. Additionally, flag this sim as a known low-signal validation case in evaluation reports.

---

## Inference Pipeline

| Component | Status |
|---|---|
| `training/infer.py` | ✅ Fully implemented, batch and single-sim modes |
| `wp3_features/extract_single.py` | ✅ HDF5 fast path + raw AniForm fallback |
| Feature normalization in inference | ✅ Fixed 2026-04-25 (was returning raw un-normalized features) |
| Targets read conditionally | ✅ Fixed 2026-04-25 (no longer fails if fine mesh run absent) |
| Contract validation | ✅ `model/contracts.py`, tested |
| End-to-end inference test | ✅ Verified on real WP3 HDF5 (geom_0_1_pair1 via fold_2 best.pt) |

**Production guidance:** For Batch A sims requiring fine dz prediction, use `fold_1/best.pt` or `fold_3/best.pt`. For Batch B sims or coarse-only prediction, any fold checkpoint is equivalent.

---

## Track C — Next Training Stage (WP12)

Track B is preserved as a reference baseline. Track C implements the following changes in a separate output directory. See `WP12_track_c_training.md` for full spec.

In order of impact on fine prediction quality:

1. **Enable `use_fine_mp=True`** — largest architectural improvement available; expected to lift stress spatial R² from 0.09 to 0.4+. Architecture fully implemented in `model/cross_scale.py`.
2. **Increase `fine_dz` loss weight** (1.5 → 3.0–5.0) — eliminate the dz fold-lottery problem; all folds should converge to R²>0.9 on Batch A. Tune in conjunction with buckling loss weight to maintain pressure throughout training.
3. **Fix Batch B R² metric** — add minimum-variance guard to `evaluate.py::compute_fine_metrics` before Track C runs so reporting is clean from the start.
4. **Increase `hidden_dim`** (64 → 128) — optional; 137k → ~540k parameters; still <10% VRAM; revisit gate thresholds if done.

**Track C checkpoint dir**: `/home/ellis/cfwrinkle/checkpoints/progressive/cross_scale_level4_cv_trackc/`
**Track B reference dir** (do not modify): `/home/ellis/cfwrinkle/checkpoints/progressive/cross_scale_level4_cv/`

---

## Files

| File | Status |
|---|---|
| `model/cross_scale.py` | ✅ Architecture complete including `use_fine_mp` |
| `model/loss.py` | ✅ All physics losses implemented; `fine_dz` weight needs tuning |
| `training/train.py` | ✅ Full CV harness |
| `training/evaluate.py` | ⚠️ R² metric needs minimum-variance guard (Issue 3/4) |
| `training/infer.py` | ✅ Complete |
| `training/gate_check.py` | ✅ Thresholds calibrated to hidden_dim=64 baseline |
| `wp3_features/extract_single.py` | ✅ Normalization fix applied |
| `scripts/resume_cross_scale_level4.sh` | ✅ All ROCm constraints baked in |
| `reports/wp7_gate_level4_cross_scale.json` | ✅ PASS (2026-04-25) |
| `reports/track_ab_comparison.json` | ✅ Track A vs B per-fold comparison |
| `reports/CROSSSCALE_LEVEL4_TRAINING_REPORT_2026-04-25.md` | ✅ Full training report |
| `reports/ROCM_OPTIMIZATION_HANDOFF_2026-04-18.md` | ✅ Hardware constraints reference |

# WP12 — Track C Training
## CrossScaleNet Level 4 — fine_mp ON, dz weight tuned

**Depends on:** WP11 (Track B analysis, 4 open issues)
**Track B reference:** `/home/ellis/cfwrinkle/checkpoints/progressive/cross_scale_level4_cv/` — do not modify
**Track C output:** `/home/ellis/cfwrinkle/checkpoints/progressive/cross_scale_level4_cv_trackc/`
**Gate report:** `reports/wp7_gate_level4_cross_scale_trackc.json`
**Status:** 🔲 NOT STARTED

---

## Changes from Track B

| Parameter | Track B | Track C | Reason |
|---|---|---|---|
| `use_fine_mp` | False | **True** | Fix Issue 2: stress spatial R²≈0.09 (no within-patch variation) |
| `CFWRINKLE_FINE_DZ_WEIGHT` | 1.5 | **4.0** | Fix Issue 1: dz fails in folds 0/2/4 (buckling pressure plateau) |
| `hidden_dim` | 64 | **96** | Perf gains from all 8 optimizations provide budget; Track C starts fresh (no ckpt compat concern) |
| `EPOCHS` | 50 | **100** | ~60-150s/epoch after optimizations; 100 epochs ≈ same wall-clock as Track B 50 epochs |
| `MAX_TIMESTEPS` | 96 | **96** | T=128 preload OOM-killed at 25.8 GB on WSL2 27 GB ceiling (confirmed 2026-04-25); T=96 preload ~19.5 GB — stays within budget |
| `ATTN_BATCH_NODES` | 64 | **512** | All 8 optimizations applied; 512 fills RDNA3 kernels (8× less Python loop overhead) |
| `AMP_DTYPE` | bfloat16 | **float16** | RDNA3 gfx1100 native fp16; 10–30% faster than bfloat16 on this arch |
| `FINE_FEATURE_NORMALIZE` | 1 | 1 | Hard constraint — required for stable training |
| Output dir | `cross_scale_level4_cv` | `cross_scale_level4_cv_trackc` | Separate to preserve Track B |

---

## ROCm Hard Constraints (7900 XT — never change)

See `reports/ROCM_OPTIMIZATION_HANDOFF_2026-04-18.md` for full context.

- `AMP_DTYPE=float16` — RDNA3 gfx1100 native fp16 (bfloat16 was needed only for Track B checkpoint resume; Track C starts fresh)
- `MAX_TIMESTEPS >= 96` — below this, temporal physics resolution degrades; Track C uses 128
- `ATTN_BATCH_NODES=512` — HIP segfault was from bfloat16/float16 checkpoint dtype mismatch during Track B resume; Track C is segfault-free with float16 + 512
- `LD_PRELOAD=/opt/rocm-7.2.0/lib/libamdhip64.so` — required for HIP visibility in WSL2
- `CFWRINKLE_FINE_LOSS_CHUNK_ELEMS=2048` — prevents HIP OOM on fine loss chunking
- `FINE_FEATURE_NORMALIZE=1` — must remain ON

---

## How to Run

```bash
# From a WSL terminal
cd "/mnt/c/Users/ellis/Documents/VS Code/CFWrinklePINN"
bash scripts/resume_cross_scale_level4_trackc.sh

# Attach to monitor
tmux attach -t level4_trackc

# Log
tail -f /home/ellis/cfwrinkle/logs/level4_trackc.log
```

Track C launches in tmux session `level4_trackc` (separate from Track B's `level4_finalexec`).

---

## Expected Outcomes

### Issue 1 — dz prediction (CFWRINKLE_FINE_DZ_WEIGHT: 1.5 → 4.0)

Track B: dz R²=+0.95 in folds 1/3, R²<-8 in folds 0/2/4.
Target: R²>0.7 for Batch A (geom_0_*) in all 5 folds.

The 4.0 weight increases the gradient signal from dz loss relative to stress by 2.7×. This should maintain dz learning pressure in folds where buckling loss plateaued early. If folds 0/2/4 still fail at 4.0, escalate to 6.0–8.0 in a follow-up run.

### Issue 2 — Fiber stress spatial (use_fine_mp=True)

Track B: fiber_stress R²≈0.09–0.13 everywhere (piecewise-constant at coarse element resolution).
Target: R²>0.3 for Batch A. Fine MP propagates information between adjacent fine elements within a patch, enabling within-patch spatial gradients.

`fine_mp_steps=2` (default in CrossScaleNet). Two MP layers should be sufficient for 1-hop neighbourhood differentiation. VRAM impact remains low: smoke test at hidden_dim=64 showed 1.4 GB / 19.94 GB; at hidden_dim=96 expect ~2–3 GB (gradient checkpointing per timestep keeps fine MP from accumulating across T=96).

### Gate

Gate thresholds remain unchanged (calibrated to Track B baseline):
- `level_4_mean_fine_stress_mae_max`: 0.55
- `level_4_mean_fine_dz_mae_max`: 0.40

If Track C's fine_mp significantly improves stress, the stress threshold could be tightened. Revisit after results.

---

## Evaluation Plan (post-training)

After all 5 folds complete, run the same per-sim fine prediction comparison used for Track B analysis:
1. Load each fold's `best.pt`
2. Evaluate against fine ground truth from WP3 HDF5
3. Report per-field R² and MAE for Batch A and Batch B separately
4. Compare directly against Track B per-sim results in `reports/CROSSSCALE_LEVEL4_TRAINING_REPORT_2026-04-25.md`

Key comparison points:
- dz R² per Batch A sim per fold (did folds 0/2/4 improve?)
- stress_1 R² (did fine_mp lift it above 0.09?)
- thickness R² Batch A (should remain ~0.85+)
- val_loss per fold (should remain comparable to Track B ~0.085)

---

## Files

| File | Status | Notes |
|---|---|---|
| `run_cross_scale_level4_trackc.sh` | ✅ Created | Main run script |
| `scripts/resume_cross_scale_level4_trackc.sh` | ✅ Created | Detached tmux resume |
| `model/loss.py` | ✅ Updated | `CFWRINKLE_FINE_DZ_WEIGHT` + all weight env vars via `_resolve_cross_scale_weights` |
| `model/cross_scale.py` | ✅ Updated | fine MP per-timestep gradient checkpointing (prevents OOM on 150k-node fine mesh) |
| `model/dataset.py` | ✅ Updated | HDF5 slice-at-read-time (Issue 1) + `preload_fold()` (Issue 4) |
| `training/train.py` | ✅ Updated | `--torch-compile` flag, `preload_fold()` call, `torch.compile` wiring |
| `training/gate_check.py` | ✅ Ready | Same Level 4 gate; Track C report path is separate |

## All Optimizations Applied (from PERF_OPTIMIZATION_7900XT_2026-04-24.md)

| # | Optimization | Status | Expected gain |
|---|---|---|---|
| 1 | HDF5 slice at read time (`dataset.py`) | ✅ Applied | Eliminates 2.67× RAM spike per sample |
| 2 | `ATTN_BATCH_NODES` 64 → 512 | ✅ Applied | ~8× kernel utilisation improvement |
| 3 | AMP dtype bfloat16 → float16 | ✅ Applied | 10–30% GPU compute speedup on RDNA3 |
| 4 | Per-fold preload (`preload_fold()`) | ❌ Disabled | Batch B coarse=30k nodes × 40 sims × T=96 = ~31 GB; exceeds WSL2 27 GB RAM + usable swap. Fine stats persisted to HDF5 (2026-04-25) eliminating 200s startup cost. |
| 5 | scatter_add int32 | ❌ Skipped | No speedup on this ROCm build (1.0× measured) |
| 6 | HIP allocator (`expandable_segments:True`) | ✅ Applied | Reduces RDNA3 fragmentation stalls |
| 7 | CK flash attention (`TORCH_ROCM_FA_PREFER_CK=1`) | ✅ Applied | 20–30% attention speedup on RDNA3 |
| 8 | `torch.compile` (inductor, dynamic, ROCm-safe) | ✅ Applied | 1.3–2.5× GPU compute speedup |

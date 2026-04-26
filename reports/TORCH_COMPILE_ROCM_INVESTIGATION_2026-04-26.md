# torch.compile + ROCm 7.1 Investigation Report
**Date:** 2026-04-26  
**GPU:** AMD RX 7900 XT (gfx1100/RDNA3), 19.94 GB VRAM  
**Stack:** PyTorch 2.11.0+rocm7.1, ROCm 7.1.52802, WSL2 Ubuntu 24.04  
**Model:** CrossScaleNet, hidden_dim=64, use_fine_mp=True, T=96, ~175k params

---

## Problem Statement

`torch.compile` is required for ~40% epoch speedup (778s compiled vs ~1112s uncompiled). However, the combination of torch.compile + fine_mp training consistently causes GPU deadlocks mid-training on this hardware. Multiple approaches were investigated; all encountered one of two failure modes.

---

## Failure Mode A — HSA Code-Object Pool Exhaustion (hidden_dim=96 only)

**Symptom:** Infinite `HSA exception: MemoryRegion::BlockAllocator::alloc failed.` loop, no epoch completion, no crash, process alive but computing nothing.

**Root cause:** Triton compiles backward-pass kernels for each unique tensor shape. At hidden_dim=96 (376k params), the total compiled kernel binary size exceeds the ROCm HSA code-object memory pool. The pool is a fixed-size region; there is no env var to resize it in ROCm 7.1.

**Not a factor at hidden_dim=64:** The 175k-param model's kernel binaries fit. Epochs 1 and 2 completed successfully (777.9s, 1415.5s) with transient HSA exceptions that resolved via retry.

**Mitigations attempted:**
- `TORCHINDUCTOR_MAX_AUTOTUNE=0` — no effect (pool exhaustion from kernel loading, not autotuning)
- `TRITON_BUILD_NUMWORKERS=1` — undocumented, no measurable effect
- Clearing `~/.triton/cache/` — forces recompile but same pool limit
- Removing graph break at `cross_scale.py:123` (`int(c_idx.min())`) — reduced VRAM usage (1.40→0.75 GB after first sample) but HSA pool still filled
- `max_autotune_gemm_backends="ATEN"` — uses pre-compiled rocBLAS for GEMM; eliminates Triton GEMM kernels (confirmed: `num_triton_choices: 0` in logs); this was effective at keeping hidden_dim=64 compilation working

**Conclusion for hidden_dim=96:** Not viable with `torch.compile+inductor` on ROCm 7.1/gfx1100. hidden_dim=64 is the maximum that compiles stably.

---

## Failure Mode B — PyTorch VRAM Pool Bloat + HSA Deadlock (hidden_dim=64, mid-epoch)

**Symptom:** Training runs successfully for 2 complete epochs, then GPU utilization drops to 0%, system RAM drops to baseline, VRAM stays loaded, process alive indefinitely — a silent GPU deadlock.

**Observed pattern:**
```
Epoch 1: 777.9s  | VRAM reserved=12.07 GB at epoch 2 start
Epoch 2: 1415.5s | VRAM reserved=19.02 GB at epoch 3 start  ← WARNING
Epoch 3: hangs   | GPU 0%, RAM baseline, VRAM occupied
```

**Root cause (confirmed):** PyTorch's caching allocator retains all freed GPU tensors in a pool that only grows, never shrinks. Over 40 training samples per epoch — each including a gradient-checkpointed fine MP backward pass (96 re-materialization steps on 121–150k-node fine meshes) — the pool grows from ~12 GB to ~19 GB. At 95%+ reserved:

With `HSA_DISABLE_FRAGMENT_ALLOCATOR=1` active: when a new tensor allocation needs a contiguous block that doesn't fit in the fragmented pool, ROCm cannot fragment existing blocks to serve it. The `hipMalloc` call retries forever → GPU deadlock.

Without `HSA_DISABLE_FRAGMENT_ALLOCATOR`: the retry uses fragmentation and eventually succeeds (observed as transient `HSA exception` log lines in epochs 1–2). However, this re-enables the Failure Mode A risk during compilation, and the mid-epoch deadlock shifts: the pool still grows to ~19 GB and some subsequent operation fails differently.

**The core tension:**
- `HSA_DISABLE_FRAGMENT_ALLOCATOR=1` is needed during Triton compilation (prevents code-object pool fragmentation)
- `HSA_DISABLE_FRAGMENT_ALLOCATOR=0` (default) is needed during training (allows tensor allocation retry via fragmentation)
- These cannot be toggled mid-process (env var is read by the HSA runtime at init)
- The PyTorch `torch.cuda.empty_cache()` at epoch boundaries (implemented) resets the pool to ~6 GB, but mid-epoch growth still triggers the deadlock before the next boundary

---

## What Was Successfully Implemented

All changes below are in the codebase and can be activated:

| Component | Change | Status |
|---|---|---|
| `model/cross_scale.py` | Removed `int(c_idx.min/max)` graph breaks from `forward()` | ✅ Done |
| `model/dataset.py` | Fine feature stats persisted to HDF5 (eliminates 200s silent startup) | ✅ Done |
| `training/train.py` | `torch.cuda.empty_cache()` at epoch start (prevents 12→19 GB bloat at boundaries) | ✅ Done |
| `training/train.py` | Bidirectional `_orig_mod.` key normalization on checkpoint load (compile↔no-compile resume) | ✅ Done |
| `training/train.py` | VRAM logging shows `alloc=X GB reserved=Y GB` (replaces misleading alloc-only metric) | ✅ Done |
| `training/train.py` | `--no-preload-fold` flag (preload exceeds 27 GB WSL2 RAM for Batch B) | ✅ Done |
| `run_cross_scale_level4_trackc.sh` | `max_autotune_gemm_backends="ATEN"` (zero Triton GEMM kernels) | ✅ Done |
| `run_cross_scale_level4_trackc.sh` | `TORCH_ROCM_FA_PREFER_CK=0` (CK targets Wave64, gfx1100 is Wave32) | ✅ Done |
| `run_cross_scale_level4_trackc.sh` | `DECODER_CHUNK_T=6` (halved from 12, reduces GRU backward peak) | ✅ Done |
| `run_cross_scale_level4_trackc.sh` | `CFWRINKLE_FINE_LOSS_CHUNK_ELEMS=512` (reduced from 2048) | ✅ Done |

---

## Approaches to Investigate Next

### Option 1 — Two-Phase Training (Highest confidence)

Run the process in two phases with different ROCm configs:

**Phase 1 (compilation):** `HSA_DISABLE_FRAGMENT_ALLOCATOR=1`, `torch.compile ON`, run until end of epoch 1. Save checkpoint. Exit.

**Phase 2 (training):** `HSA_DISABLE_FRAGMENT_ALLOCATOR=0`, `torch.compile ON`, resume from epoch 2. The Triton cache from Phase 1 is warm; all kernel shapes were compiled. No new `hipMalloc` for code objects during training. Fragment allocator handles any mid-epoch pool pressure.

Implementation: add a `--compile-only-epoch-1` flag to the training script that exits after epoch 1 with a specific exit code, then a separate resume invocation without `HSA_DISABLE_FRAGMENT_ALLOCATOR`.

### Option 2 — Per-Sample empty_cache with HSA_DISABLE_FRAGMENT_ALLOCATOR=0

Without `HSA_DISABLE_FRAGMENT_ALLOCATOR`, the per-epoch `empty_cache()` might be sufficient to prevent pool growth — the question is whether the pool grows to a problematic level within one epoch's 40 samples before the next boundary call. With `--cuda-empty-cache-per-sample` (already implemented as a flag), this becomes per-sample, essentially keeping the pool near-zero throughout. Cost: ~1–10 ms overhead per sample (negligible at 40 samples × ~30s/sample).

Test: run with `HSA_DISABLE_FRAGMENT_ALLOCATOR` unset + `--cuda-empty-cache-per-sample`.

### Option 3 — Torch.compile with backend="aot_eager"

Uses AOT Autograd tracing without Triton codegen. Eliminates ALL HSA code-object allocations. Expected speedup: ~1.1× (vs 1.4–1.8× from inductor). No Triton compilation, no HSA pool issues.

Test: change `compile_kwargs["backend"]` from `"inductor"` to `"aot_eager"` in `training/train.py`.

### Option 4 — Accept no torch.compile

Without compile: epoch times ~900–1400s (source doc estimate 700–900s with HDF5 slice fix). At 50 epochs × 5 folds × 1100s average ≈ 76 hours ≈ 3.2 days.  
With compile (when working): ~700–800s/epoch ≈ 2.4 days. The performance gap is ~25%.

If Options 1–3 are too complex to debug, this is the reliable fallback. The model improvements (fine_mp, dz weight) are what matter scientifically.

---

## Hardware Constraints Confirmed (do not change)

| Constraint | Value | Reason |
|---|---|---|
| `MAX_TIMESTEPS` | 96 | Physics fidelity floor |
| `hidden_dim` | 64 | Max for torch.compile on ROCm 7.1/gfx1100 |
| `FINE_FEATURE_NORMALIZE` | 1 | Required for stable training |
| `AMP_DTYPE` | float16 | RDNA3 native fp16; bfloat16 is slower |
| `ATTN_BATCH_NODES` | 512 | Fills RDNA3 kernels; 64 underutilises |
| `use_checkpoint` | True | Without it: fine_mp allocates ~18.78 GB VRAM → OOM |
| `preload_fold` | OFF | Batch B coarse 30k nodes × 40 sims × T=96 = ~31 GB → OOM |
| `TORCH_ROCM_FA_PREFER_CK` | 0 | CK targets Wave64; gfx1100 is Wave32 → extra Triton kernels |

---

## Current Checkpoint State

Fold 0 checkpoints survive at:  
`/home/ellis/cfwrinkle/checkpoints/progressive/cross_scale_level4_cv_trackc/fold_0/`
- `best.pt`: epoch 2, `best_val_loss=0.2021` (no `_orig_mod.` prefix — safe to resume with or without compile)
- `latest.pt`: same

All key normalization (compile↔no-compile resume) is handled automatically by `training/train.py`.

## Recommended Next Step

Try **Option 2** first — it requires the smallest code change (just removing `HSA_DISABLE_FRAGMENT_ALLOCATOR` from the scripts and enabling `--cuda-empty-cache-per-sample`). If the per-sample pool reset keeps VRAM below 85% throughout training, the deadlock never occurs regardless of the fragment allocator setting.

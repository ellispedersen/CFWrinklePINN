# RX 7900 XT Performance Optimization — Revised Plan
**Date:** 2026-04-24  
**GPU:** AMD RX 7900 XT (gfx1100/RDNA3), 19.94 GB VRAM, ROCm 7.2, PyTorch 2.9.1+rocm  
**System RAM:** 32 GB | **Dataset on disk:** ~58 GB (T=256 full resolution, 66 sims)  
**Design intent:** MAX_TIMESTEPS tuned so the per-fold working set fills ~32 GB RAM  
**Current epoch time:** 1200–1400s | **Current VRAM use:** ~7%

---

## The Core Problem

The training loop processes one simulation at a time:
**read full T=256 from HDF5 → slice to MAX_TIMESTEPS in Python → move to GPU → forward/backward → repeat.**

GPU compute takes under a second per sample. Everything else is dead time.
There are three compounding issues that need to be addressed in order.

---

## Issue 1 — Transient Memory Spike on Every Sample Load (Critical)

**This is the most important thing to fix before the fold preload.**

`_read_features`, `_read_rates`, `_read_targets`, and `_read_fine_feature_stack` all use
`arr[:]` — they read the full T=256 array into RAM, *then* slice to `MAX_TIMESTEPS` in
Python (lines 308, 314, 342, 382–385). The sliced result is kept; the T=256 array is
discarded. But **both exist in RAM simultaneously** during that window.

If your MAX_TIMESTEPS is tuned so the final working set fills ~32 GB, the peak memory
during each `__getitem__` call is:

```
peak = final_size × (256 / MAX_TIMESTEPS)
```

At MAX_TIMESTEPS=96: peak = 2.67× final size.  
At MAX_TIMESTEPS=220: peak = 1.16× final size — more forgiving, but still spikes.

With RAM already near capacity, these transient spikes hit the OS swap path repeatedly
throughout every epoch. This is likely contributing substantially to the 1200–1400s time.

**Fix — slice at HDF5 read time:**

HDF5 supports fancy indexing natively. Reading `arr[t_idx]` only fetches those rows from
disk; the T=256 array is never allocated.

```python
# model/dataset.py — compute t_idx before any HDF5 read

def __getitem__(self, idx: int) -> dict:
    sim_id = self.sim_ids[idx]
    f = self._get_h5_file()
    grp = f[f"simulations/{sim_id}"]
    static_data = self._read_static_graph_data(grp, sim_id)

    # ── Compute t_idx FIRST, before any feature read ──────────────────────
    # Read one array header only to get n_t, without loading data
    raw_grp = grp["coarse/resampled"] if "coarse" in grp else grp
    feat_key = "coarse_fields_resampled" if "coarse_fields_resampled" in grp \
               else "coarse/resampled/fields"
    n_t = grp[feat_key].shape[0]          # reads shape metadata only — no data
    t_idx = None
    if self.max_timesteps is not None and self.max_timesteps < n_t:
        t_idx = self._temporal_indices(n_t, self.max_timesteps, self.temporal_strategy)

    # ── Read features with early slice ────────────────────────────────────
    feats = self._read_features_sliced(grp, t_idx)    # reads only t_idx rows
    rates = self._read_rates_sliced(grp, t_idx)
    targets = self._read_targets_sliced(grp, t_idx)
    # ... rest of __getitem__ unchanged, remove the post-read slice block ...
```

```python
@staticmethod
def _read_features_sliced(grp: h5py.Group, t_idx: np.ndarray | None) -> np.ndarray:
    key = "coarse_fields_resampled" if "coarse_fields_resampled" in grp \
          else "coarse/resampled/fields"
    ds = grp[key]
    if t_idx is None:
        return ds[:].astype(np.float32)
    return ds[t_idx].astype(np.float32)   # HDF5 fetches only these rows from disk

@staticmethod
def _read_rates_sliced(grp: h5py.Group, t_idx: np.ndarray | None) -> np.ndarray:
    key = "coarse_rates_resampled" if "coarse_rates_resampled" in grp \
          else "coarse/resampled/rates"
    ds = grp[key]
    if t_idx is None:
        return ds[:].astype(np.float32)
    return ds[t_idx].astype(np.float32)

@staticmethod
def _read_fine_feature_stack_sliced(grp: h5py.Group, t_idx: np.ndarray | None) -> np.ndarray:
    fine = grp["fine/resampled"]
    def _read(key):
        ds = fine[key]
        return (ds[t_idx] if t_idx is not None else ds[:]).astype(np.float32)
    fs1 = _read("fiber_stress_1")
    fs2 = _read("fiber_stress_2")
    dz  = _read("displacement_z")
    thick = _read("thickness")
    return np.stack([fs1, fs2, dz, thick], axis=-1)
```

**HDF5 fancy-index caveat:** `h5py` fancy indexing (`ds[t_idx]`) requires the index to be
monotonically increasing. The "tail" strategy (`np.arange(n_t - max_t, n_t)`) is already
monotonic. "stride" (`np.arange(0, n_t, step)[:max_t]`) is also monotonic. "linspace"
produces a sorted int array — also fine. All three strategies are safe.

**Peak memory impact:** Eliminated. Peak is now exactly the final working set size,
not 2.67× it. This directly reduces swap pressure and should remove a large fraction of
the per-sample overhead.

**Physics safety:** Reads the exact same rows, just without allocating the full T=256
intermediate. No numerical change.

---

## Issue 2 — Per-Fold Preload Strategy

Once Issue 1 is fixed (reads are cheap and don't spike memory), it's feasible to preload
the entire training fold's feature data into RAM at the start of each fold.

**Why per-fold, not full dataset:**
- Full dataset: 66 sims × (T=256 rate) ≈ 58 GB — doesn't fit
- Per fold: ~42 train sims × final-timestep size — sized to fit ~32 GB by design
- After Issue 1's fix, reading a fold's data costs exactly what it uses: no overhead

**Implementation — add `preload_fold` method to `WrinkleDataset`:**

```python
def preload_fold(self) -> None:
    """Pre-read all feature tensors for this dataset's sim_ids into RAM.

    Call once at the start of each fold, before the epoch loop.
    After this, __getitem__ serves from _feature_cache with no HDF5 access.
    """
    self._feature_cache: dict[str, dict] = {}
    f = self._get_h5_file()
    for sim_id in self.sim_ids:
        grp = f[f"simulations/{sim_id}"]
        # Reuse the sliced read path from Issue 1's fix
        n_t = self._get_n_t(grp)
        t_idx = None
        if self.max_timesteps is not None and self.max_timesteps < n_t:
            t_idx = self._temporal_indices(n_t, self.max_timesteps,
                                           self.temporal_strategy)
        feats   = self._read_features_sliced(grp, t_idx)
        rates   = self._read_rates_sliced(grp, t_idx)
        targets = self._read_targets_sliced(grp, t_idx)
        node_features = np.concatenate([feats, rates], axis=-1)
        if self.normalize and self._stats is not None:
            node_features = (node_features - self._stats["feat_mean"]) \
                            / (self._stats["feat_std"] + 1e-8)
        entry = {"node_features": node_features, "targets": targets}
        if self.include_fine:
            static_data = self._read_static_graph_data(grp, sim_id)
            fine_feats = self._read_fine_feature_stack_sliced(grp, t_idx)
            if self.fine_normalize and self._fine_stats is not None:
                fine_feats = (fine_feats - self._fine_stats["fine_feat_mean"]) \
                             / (self._fine_stats["fine_feat_std"] + 1e-8)
            entry["fine_features"] = fine_feats
        self._feature_cache[sim_id] = entry

def __getitem__(self, idx: int) -> dict:
    sim_id = self.sim_ids[idx]
    if hasattr(self, "_feature_cache") and sim_id in self._feature_cache:
        cached = self._feature_cache[sim_id]
        # Static graph still comes from _static_graph_cache (already cached)
        # ... assemble result dict from cached tensors ...
        return self._assemble_from_cache(sim_id, cached)
    # ... existing HDF5 path unchanged ...
```

**In `training/train.py`, call at fold start:**
```python
# After creating train_ds and val_ds for the fold:
if config.get("preload_fold", True):
    logger.info("Preloading fold feature data into RAM...")
    train_ds.preload_fold()
    # val_ds.preload_fold() — optional, val set is small enough to be fast anyway
    logger.info("Preload complete.")
```

**Expected impact:** After the one-time fold preload (which itself benefits from Issue 1's
fix — no 2.67× spike during the preload), every epoch within that fold has zero HDF5 I/O.
The training loop's per-sample cost drops to a pure dict lookup and tensor transfer.

**Memory note:** The preload uses exactly the same RAM as was being used transiently during
training anyway — the difference is it's allocated once at fold start rather than
repeatedly allocated and freed. GC pressure is eliminated.

---

## Issue 3 & 4 — Env-Var-Only Fixes (No Code Changes, Apply Now)

These two changes are single-line edits in `scripts/resume_cross_scale_level4.sh` and
have no physics or accuracy risk at all.

### ATTN_BATCH_NODES: 64 → 512

Line 15 of the resume script:
```bash
ATTN_BATCH_NODES="${ATTN_BATCH_NODES:-512}"   # was 64
```

With ~3,700 coarse nodes and chunks of 64, the TemporalAggregator launches ~58 separate
GRU + attention kernel pairs per forward pass. Each is tiny — far too small to saturate
the GPU. This is documented in CLAUDE.md as the Level 2 tuned value; the resume script
regressed it back to 64.

Increasing to 512 reduces Python-loop overhead in the attention path by ~8×. At 512 nodes/
chunk, each kernel call is large enough to actually occupy the GPU. VRAM cost is trivial
(~6 MB per chunk at fp16 vs 19.94 GB available). Output is numerically identical regardless
of chunk size — this is pure batching granularity.

### AMP dtype: bfloat16 → float16

Line 17 of the resume script:
```bash
AMP_DTYPE="${AMP_DTYPE:-float16}"   # was bfloat16
```

RDNA3/gfx1100 has native hardware fp16 instructions (rocWMMA). bfloat16 requires extra
lane-swap operations and has a confirmed kernel maturity issue on gfx1100 (PyTorch issue
#165141). Switching to float16 gives 10–30% GPU compute speedup on GRU, attention, and
linear layers with no change to training logic — GradScaler is already active for float16
in `train.py` line 263.

Monitor the `grad_scale` value in training logs for the first fold after switching. If it
collapses and stays very small, the loss path has an fp16 range issue. In practice this
is unlikely given the `_safe_norm` 1e-2 floor in the physics losses, but worth watching.

---

## Issue 5 — scatter_add_ Index Dtype (Conditional on Test)

All edge indices are loaded as `torch.int64` (dataset.py line 129). `scatter_add_` with
int64 indices is known to be significantly slower than int32 on GPU. Node counts here
(max ~75k) are well within int32 range.

**Run this test first in WSL before changing anything:**

```python
# scripts/test_scatter_int32.py
import torch

torch.manual_seed(0)
N, E, D = 75000, 500000, 64  # fine mesh scale

h = torch.randn(N, D, device="cuda")
src_64 = torch.randint(0, N, (E,), dtype=torch.int64, device="cuda")
messages = torch.randn(E, D, device="cuda")

# int64 path
agg_64 = torch.zeros(N, D, device="cuda")
idx_64 = src_64.unsqueeze(1).expand_as(messages)
agg_64.scatter_add_(0, idx_64, messages)

# int32 path
src_32 = src_64.to(torch.int32)
agg_32 = torch.zeros(N, D, device="cuda")
idx_32 = src_32.unsqueeze(1).expand_as(messages).to(torch.int32)
try:
    agg_32.scatter_add_(0, idx_32, messages)
    diff = (agg_64 - agg_32).abs().max().item()
    # Also test that h[src_32] works for advanced indexing
    g64 = h[src_64]
    g32 = h[src_32]
    idx_diff = (g64 - g32).abs().max().item()
    print(f"scatter_add diff: {diff}  ({'OK' if diff == 0 else 'MISMATCH'})")
    print(f"advanced index diff: {idx_diff}  ({'OK' if idx_diff == 0 else 'MISMATCH'})")
    import time
    def bench(fn, n=50):
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(n): fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t) / n * 1000
    t64 = bench(lambda: agg_64.scatter_add_(0, idx_64, messages))
    t32 = bench(lambda: agg_32.scatter_add_(0, idx_32, messages))
    print(f"int64: {t64:.2f}ms  int32: {t32:.2f}ms  speedup: {t64/t32:.1f}x")
except RuntimeError as e:
    print(f"int32 scatter not supported on this build: {e}")
```

If the test shows identical values and a meaningful speedup, apply the cast only at the
scatter site in `model/layers.py` line 53 — keep storage as int64 everywhere else:

```python
# layers.py line 53 — localized cast only:
idx = dst.unsqueeze(1).expand_as(messages).to(torch.int32)
agg.scatter_add_(0, idx, messages)
```

If the test fails or shows no speedup on this ROCm build, skip this change entirely.
Issues 1–4 are sufficient for a major improvement.

---

## Supplementary (apply after validating 1–5)

**HIP allocator — fixes intermittent step-time spikes on RDNA3:**
```bash
# resume_cross_scale_level4.sh line 27:
PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.8}"
```
Addresses a known RDNA3 memory pool fragmentation bug (ROCm issue #6007). Test
`expandable_segments:True` in WSL2 first — remove it and keep the other two if it
causes startup errors.

**CK flash attention — 20–30% attention speedup on RDNA3:**
```bash
export TORCH_ROCM_FA_PREFER_CK=1
```
Switches SDPA in MultiheadAttention from AOTriton to AMD Composable Kernel. Stable in
PyTorch 2.9.1+ROCm 7.2 but benchmark with a 5-epoch run before committing.

**torch.compile — 1.3–2.5× GPU compute speedup:**
```python
# training/train.py, after model construction:
if config.get("torch_compile", False):
    compile_kwargs = dict(backend="inductor", fullgraph=False, dynamic=True)
    if hasattr(torch.version, "hip"):
        compile_kwargs["mode"] = "default"  # NEVER use reduce-overhead on ROCm
    model = torch.compile(model, **compile_kwargs)
```
First epoch is slow (Triton compilation, 2–10 min). Apply last.

---

## Application Order

| # | Fix | Where | Risk |
|---|---|---|---|
| 1 | Slice at HDF5 read time | `model/dataset.py` | None — same data, no intermediate |
| 2 | ATTN_BATCH_NODES 64 → 512 | Env var in resume script | None |
| 3 | AMP dtype bfloat16 → float16 | Env var in resume script | Very low |
| 4 | Per-fold preload | `model/dataset.py` + `training/train.py` | None |
| 5 | scatter_add int32 (if test passes) | `model/layers.py` | Low |
| 6 | HIP allocator config | Env var in resume script | Very low |
| 7 | CK flash attention | Env var | Low |
| 8 | torch.compile | `training/train.py` | Medium |

**Do immediately (no code, no risk):** Apply fixes 2 and 3 right now in the resume script.
**Fix 1 before Fix 4:** The HDF5 slice-at-read-time fix makes the per-fold preload
memory-safe. Without it, `preload_fold()` itself would spike to 2.67× RAM during loading.

---

## Expected Epoch Times

| State | Epoch time |
|---|---|
| Baseline | 1200–1400s |
| After fixes 2 + 3 (env vars only) | ~900–1100s |
| After fix 1 (HDF5 slice at read) | ~700–900s (swap eliminated) |
| After fix 4 (per-fold preload) | ~150–300s (zero disk I/O per epoch) |
| After fixes 5–8 | ~60–150s |

---

## Hard Constraints — Do Not Change

- `MAX_TIMESTEPS >= 96` (the current value is the physics data density floor)
- `FINE_FEATURE_NORMALIZE=1` (normalization state baked into current checkpoints)
- `hidden_dim=64` (validated model identity at Level 3)
- `AUTO_RESUME=1` pointing to existing checkpoint dir (fold_4 at epoch 11)
- All `CFWRINKLE_DISABLE_FINE_*` remain 0

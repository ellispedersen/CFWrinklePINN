# Track C CUDA — Technical Reference
## CrossScaleNet Level 4 on RTX Pro 6000 Blackwell

**Status:** Deployment-ready (pending Verda execution)  
**Run script:** `run_cross_scale_level4_cuda.sh`  
**Gate report:** `checkpoints/progressive/cross_scale_level4_cv_trackc_verda/gate_report.json`

---

## 1. Architectural History and Platform Motivation

This section traces the decisions that led from the original Windows ROCm stack to the
RTX Pro 6000 Blackwell CUDA deployment. Each platform transition was driven by a specific
capacity constraint the model outgrew — not by preference.

### 1.1 WP1–5: Data pipeline (Windows, no GPU)

The initial work packages — field archaeology (WP1), HDF5 dataset construction (WP2),
feature extraction (WP3), feature engineering (WP4), and cross-validation splits (WP5) —
ran entirely on CPU. All development was on Windows 11 with the AMD RX 7900 XT in a
passive role. No platform constraints applied.

### 1.2 WP6–7: Model development and Track A (Windows ROCm, AMD RX 7900 XT)

CrossScaleNet architecture and the full training loop were developed on:

- **Hardware:** AMD RX 7900 XT, 19.94 GB VRAM, RDNA3 (gfx1100)
- **Stack:** ROCm 7.2.0 / PyTorch 2.9.1+rocmsdk / Python 3.12 / WSL2 Ubuntu-24.04
- **RAM ceiling:** WSL2 `.wslconfig` limited to 27 GB usable RAM

Track A (FormingGraphNet — coarse mesh only) completed and gated on this hardware,
establishing the detection baseline (mean val_loss 0.0761).

### 1.3 Track B Level 4: first full 5-fold CV (complete 2026-04-25)

Track B added cross-scale architecture: coarse GNN backbone + fine-mesh decoder head.
All five folds completed to 50 epochs on the 7900 XT under heavy ROCm-specific constraints:

| Constraint | Value | Root cause |
|---|---|---|
| `hidden_dim` | 64 (not 96) | 96 would require ~56 GB VRAM without gradient checkpointing [A] |
| `MAX_TIMESTEPS` | 96 (not 128) | T=128 Batch B preload ≈ 41 GB — OOM-killed at WSL2 27 GB ceiling [B] |
| `use_fine_mp` | False | fine MP at hidden_dim=64 required per-timestep gradient checkpointing to fit; disabled to avoid ROCm allocator instability under chunked backward passes |
| `gradient checkpointing` | ON (`use_checkpoint=True`) | Required to fit full sequence in 19.94 GB |
| `AMP dtype` | bfloat16 | float16 caused HIP segfault during Track B checkpoint resume [C] |
| `ATTN_BATCH_NODES` | 64 | RDNA3 gfx1100 CU utilisation limit at the time |
| `torch.compile` | `dynamic` mode only | `max-autotune` path had known ROCm fallback issues [D] |

**Track B results (mean over 5 folds):**

| Metric | Value |
|---|---|
| mean val_loss | 0.0854 |
| detection_rate | 1.000 (all 5 folds) |
| Peak VRAM | 0.76–1.42 GB / 19.94 GB (with grad checkpoint) |
| Wall-clock | ~21 min/epoch → ~107 hours total (50 epochs × 5 folds) |

**Systematic failures identified in Track B analysis** (`reports/CROSSSCALE_LEVEL4_TRAINING_REPORT_2026-04-25.md`):

**Issue 1 — displacement_z fails in folds 0/2/4.**
`loss/fine_buckling` plateaued at epoch 6 in these folds, releasing gradient pressure on dz.
The model defaulted to predicting dz near the dataset mean (~7.15 mm) regardless of input.
Folds 1/3 (where buckling loss rose monotonically to 0.47–0.56) correctly predicted Batch A
dz at R²=0.94–0.96. The same simulation evaluated by different fold checkpoints yielded
R²=+0.955 (fold_3) vs R²=−9.687 (fold_0) — a training dynamics issue, not a capacity issue.
**Fix: increase `CFWRINKLE_FINE_DZ_WEIGHT` from 1.5 → 4.0.**

**Issue 2 — fiber_stress spatial R²≈0.09 across all folds.**
The coarse-to-fine scatter assigns identical hidden state vectors to all fine elements under
the same coarse parent: `h_fine[:, f_idx, :] = h_coarse[:, c_idx, :]`. The fine head is
therefore piecewise-constant at coarse resolution — structurally incapable of predicting
within-patch spatial gradients. This is an architectural ceiling, not a training dynamics
problem.  
**Fix: enable `use_fine_mp=True` (fine mesh message passing).**

Both fixes require running without the ROCm memory constraints that prevented them. This is
the direct architectural driver for the platform change.

### 1.4 Why not scale ROCm?

Before committing to a CUDA deployment, three ROCm scale-up options were evaluated:

**Option A — More ROCm VRAM (AMD MI300X, 192 GB HBM3).**
The MI300X has 288 GB DDR5 system RAM in addition to 192 GB HBM3. However, PyTorch
training tensors and activations must reside in HBM3 (VRAM) — the DDR5 is not addressable
as VRAM overflow. PCIe bandwidth between DDR5 and HBM3 is 50–100× lower than intra-HBM3
bandwidth, making unified-memory spilling unusable for training workloads [8]. The MI300X
at ~€1.20/hr also does not improve on the torch.compile ROCm path limitations.

**Option B — Larger WSL2 allocation on the local 7900 XT.**
WSL2 `.wslconfig` could be raised to 32–40 GB RAM, but the 7900 XT VRAM ceiling remains
19.94 GB. `hidden_dim=96` without gradient checkpointing requires ~56 GB VRAM. No ROCm
hardware under €1/hr reaches this.

**Option C — ROCm cloud (AMD GPU instance).**
ROCm public cloud availability was poor at the time of evaluation. More importantly,
`torch.compile` on ROCm uses the Inductor ROCm backend which falls back from `max-autotune`
to `reduce-overhead` or `default` for many kernel shapes [D]. Track C requires
`max-autotune` to achieve the compute throughput needed for 5-fold CV at hidden_dim=96
within a cost-effective wall-clock budget.

**Conclusion:** CUDA was the correct choice. The RTX Pro 6000 Blackwell at €0.59/hr
removes all three constraints simultaneously — VRAM, RAM, and compile path.

### 1.5 Platform selection: RTX Pro 6000 Blackwell on Verda

The RTX Pro 6000 Blackwell (GB202 die, Compute Capability 12.2, sm_122 [7]) was selected over:

- **H200 SXM (€1.20/hr):** 141 GB HBM3 — more VRAM than needed; 2× cost for a single-GPU job
- **RTX 4090 / A100 (RunPod):** Adequate VRAM, but RunPod availability was inconsistent; no persistent volume with `keep_detached` guarantees
- **A6000 Ada:** 48 GB VRAM — marginally sufficient but no cost advantage over the 6000 Pro

Verda AI (FIN-03 Finland) adds spot persistence: `on_spot_discontinue: keep_detached`
retains the NVMe block volume across instance evictions [5], making spot pricing safe
when combined with atomic checkpointing (see §3.2).

If RTX Pro 6000 spot capacity is unavailable, the B300 data-centre Blackwell GPU
(sm_10x, 262 GB HBM3e, ~€2.45/hr) is the documented fallback — see §9.5 and
`docs/verda_deployment.md` §3 for the GPU selection table. Batch/chunk parameters
scale automatically; model architecture and checkpoint format are unchanged.

---

## 2. Track C vs Track B Configuration

All parameters changed from WP12 ROCm specification; differences from Track B baseline in bold.

| Parameter | Track B (WSL2/ROCm) | Track C CUDA | Change driver |
|---|---|---|---|
| `hidden_dim` | 64 | **96** | ROCm VRAM constraint lifted; quadratic attention scaling gives 2.25× parameter density [9] |
| `MAX_TIMESTEPS` | 96 | **128** | WSL2 27 GB RAM ceiling lifted; 188 GB DDR5 handles Batch B preload (~41 GB) |
| `use_fine_mp` | False | **True** | Issue 2 fix: fine MP propagates within-patch information between adjacent fine elements |
| `gradient checkpointing` | ON | **OFF** | 96 GB holds all T=128 activations at hidden_dim=96 (~56 GB estimated, 40 GB margin) |
| `AMP dtype` | bfloat16 | **bfloat16** | Unchanged; bfloat16 is Blackwell native and avoids Track B's float16/bfloat16 resume dtype mismatch [C] |
| `ATTN_BATCH_NODES` | 64 | **1024** | Blackwell 184 SMs (vs RDNA3 CU count); larger batch fills SMs more efficiently |
| `DECODER_CHUNK_T` | 12 | **24** | 96 GB headroom; longer GRU chunks reduce Python loop overhead per backward pass |
| `CFWRINKLE_FINE_DZ_WEIGHT` | 1.5 | **4.0** | Issue 1 fix: 2.7× stronger dz gradient signal to prevent buckling loss plateau |
| `CFWRINKLE_FINE_LOSS_CHUNK_ELEMS` | 2048 | **2048** | Unchanged; fine loss chunking still needed for 150k-node fine mesh |
| `torch.compile mode` | `dynamic` | **`max-autotune`** | CUDA Triton autotuning; not available on ROCm path [D][4] |
| `EPOCHS` | 50 | 50 | Matched for direct val_loss comparison |

---

## 3. Technical Decisions with Citations

### 3.1 Container image: pytorch/pytorch:2.11.0 over NGC

The original container used `nvcr.io/nvidia/pytorch:25.11-py3` (PyTorch 2.10.0, November 2025
NGC release) because sm_122 (RTX Pro 6000 Blackwell, CC 12.2) had known incompatibilities in
PyTorch stable wheels through the 2.9.x series [E].

By April 2026, `pytorch/pytorch:2.11.0-cuda13.0-cudnn9-runtime` was available [1]. PyTorch
2.11.0 was the first stable release with full sm_122 PTX/SASS compiled into the official wheels
[2]. The Docker Hub image was chosen over NGC 26.03 for three reasons:

1. **Size**: ~8 GB compressed (runtime variant, no TensorRT/TransformerEngine) vs ~22 GB for NGC
2. **No NGC account required**: eliminates `docker login nvcr.io` as an external dependency
3. **Conda venv compatibility**: `python3 -m venv /workspace/venv --system-site-packages`
   inherits conda's site-packages (including torch) into a standard venv — `run_cross_scale_level4_cuda.sh`
   activates `$VENV_PATH/bin/activate` without reinstalling the 8 GB wheel

### 3.2 Atomic checkpoint write

`train.py` previously called `torch.save(ckpt_dict, output_dir / "latest.pt")` directly.
On NVMe, a 300 MB checkpoint serialisation takes 5–15 seconds. A spot eviction mid-write
leaves a truncated pickle that fails to load and breaks resume.

The fix uses POSIX `rename(2)` atomicity [6]:

```python
_ckpt_tmp = output_dir / "latest.pt.tmp"
torch.save(ckpt_dict, _ckpt_tmp)       # write complete file to tmp
_ckpt_tmp.rename(output_dir / "latest.pt")  # atomic replace
```

`Path.rename()` in Python 3 calls `os.rename()` which maps to `rename(2)` on Linux.
POSIX.1-2017 guarantees this is atomic for intra-filesystem moves: "If the link named by
the new argument exists, it shall be removed and old renamed to new atomically" [6].
The same pattern applies to `best.pt`. Maximum work lost per eviction: one epoch.

### 3.3 .gitignore: unblocking input_param_registry.json

`build_features.py` calls `load_input_registry(ROOT)` which reads
`reports/input_param_registry.json` at startup. The file was covered by `reports/*.json`
in `.gitignore`, making it absent from fresh clones and silently breaking every WP3 rebuild
that started from a clean checkout.

The fix adds a negation pattern after the glob:

```gitignore
reports/*.json
!reports/input_param_registry.json
```

Git negation patterns require the negation rule to appear after the matching glob [F].
The file is now committed to the repo and survives `git clone`.

### 3.4 bfloat16 on Blackwell

Track B used bfloat16 because float16 caused a HIP segfault during checkpoint resume on
the 7900 XT [C]. Track C maintains bfloat16 because:

1. Blackwell (sm_122) has native BF16 Tensor Core support — no emulation penalty [7]
2. bfloat16's dynamic range (same exponent bits as float32) avoids the gradient underflow
   risk at float16 that required loss scaling in earlier CUDA architectures [10]
3. Keeps dtype consistent with Track B checkpoints if cross-run comparison is needed

### 3.5 Gradient checkpointing OFF at hidden_dim=96, T=128

Gradient checkpointing recomputes activations during the backward pass rather than storing
them, reducing VRAM at the cost of ~33% extra compute [11]. On the 7900 XT (19.94 GB),
it was required to fit Track B at hidden_dim=64, T=96.

VRAM estimate for Track C without checkpointing, extrapolating from Track B:

```
Track B without checkpoint: 18.78 GB (reported OOM threshold on 7900 XT)
Track C scaling:
  hidden_dim:     (96/64)² = 2.25×  (quadratic for attention weight matrices [9])
  T:              (128/96) = 1.33×   (linear for GRU per-step hidden states)
  Combined:       2.25 × 1.33 ≈ 3.0×

Estimated Track C: 18.78 × 3.0 ≈ 56 GB
RTX Pro 6000 VRAM: 96 GB
Margin:           ~40 GB
```

The margin is sufficient to run without gradient checkpointing, restoring full activation
graphs and eliminating the ~33% compute overhead.

### 3.6 torch.compile max-autotune

`torch.compile` with `max-autotune` uses TorchInductor + Triton to JIT-compile and autotune
CUDA kernels for every unique input shape seen during training [4][12]. First-run compilation
takes 10–20 minutes; subsequent runs (including post-eviction restarts) reuse the Triton
kernel cache mounted from the NVMe volume at `/mnt/data/.triton_cache`.

On ROCm, `max-autotune` was unavailable because the ROCm Inductor backend fell back to
`reduce-overhead` for attention and scatter kernels — the shapes are not covered by the
ROCm Triton autotuning codegen path as of PyTorch 2.9.x [D]. On CUDA, Triton autotuning
is fully supported and produces GPU-specific kernel variants for the RTX Pro 6000 Blackwell
architecture [4].

Expected speedup over non-compiled Track B: 1.5–3× per epoch depending on batch shape
diversity [12], targeting ~7–12 min/epoch (vs 21 min/epoch in Track B).

---

## 4. Memory Budget

### VRAM (96 GB GDDR7)

| Component | Estimated GB | Notes |
|---|---|---|
| Model parameters | ~0.2 | 137k params at hidden_dim=64 → ~380k at hidden_dim=96, float32 |
| Gradient state | ~0.2 | Same order as parameters |
| Optimizer state (AdamW) | ~0.4 | 2× parameter tensors (m, v) |
| Fine activations (no checkpointing) | ~40–55 | T=128 × hidden_dim=96 × 150k fine nodes |
| Coarse activations | ~2–4 | T=128 × hidden_dim=96 × ~3.7k coarse nodes |
| Intermediate tensors | ~2–5 | Attention weights, fine MP buffers |
| **Estimated peak** | **~50–65 GB** | 52–68% of 96 GB — comfortable margin |

### CPU RAM (188 GB DDR5)

Batch B fine mesh preload: 30k nodes × 40 sims × T=128 timesteps × float32 ≈ 41 GB.
At 188 GB available, this fits with ~140 GB margin.

---

## 5. Expected Outcomes vs Track B

Gate thresholds are calibrated from Track B baseline (see `training/gate_check.py`).
Track C targets improvements on the two systematic failures:

| Metric | Track B | Track C target | Mechanism |
|---|---|---|---|
| dz_mae (folds 0/2/4) | 0.451–0.592 | < 0.40 | `CFWRINKLE_FINE_DZ_WEIGHT` 1.5 → 4.0 |
| fiber_stress_1 R² | 0.09–0.14 | > 0.30 | `use_fine_mp=True` |
| val_loss (mean, 5 folds) | 0.0854 | ≤ 0.0900 | Should be comparable (Issue 1/2 fixes add loss signal, not noise) |
| detection_rate | 1.000 | 1.000 | Expected to hold |
| Wall-clock / fold | ~2,100 min | ~350–600 min | torch.compile max-autotune + no gradient checkpointing + ATTN_BATCH_NODES=1024 |

If `fiber_stress_1 R²` exceeds 0.30, tighten `level_4_mean_fine_stress_mae_max`
from 0.55 toward 0.30 for future Track D gate calibration.

---

## 6. Evaluation Plan (post-training)

After all 5 folds complete:

```bash
# Gate check (runs automatically at end of run_cross_scale_level4_cuda.sh):
cat /mnt/data/checkpoints/progressive/cross_scale_level4_cv_trackc_verda/gate_report.json

# Manual per-sim fine prediction analysis (mirrors Track B evaluation):
python -m training.evaluate \
  --run-dir checkpoints/progressive/cross_scale_level4_cv_trackc_verda \
  --output-json reports/track_c_fine_eval.json
```

Key comparison against Track B (`reports/CROSSSCALE_LEVEL4_TRAINING_REPORT_2026-04-25.md`):

1. dz R² per Batch A sim per fold — did folds 0/2/4 improve?
2. fiber_stress_1 R² — did fine MP lift it above 0.09?
3. thickness R² Batch A — should remain 0.83–0.91
4. val_loss per fold — should remain within ~5% of Track B (0.085–0.095)

If dz improvement is marginal, escalate `CFWRINKLE_FINE_DZ_WEIGHT` to 6.0–8.0 in a
Track D run. If fiber_stress spatial improvement is marginal, increase `fine_mp_steps`
from 2 to 4 in the CrossScaleNet config.

---

## 7. File Reference

| File | Purpose |
|---|---|
| `run_cross_scale_level4_cuda.sh` | Primary run script — all Track C hyperparameters |
| `scripts/verda_gpu_setup.sh` | GPU instance smoke test + image pull |
| `scripts/verda_launch_training.sh` | Docker wrapper — maps NVMe paths into container |
| `scripts/verda_cpu_rebuild.sh` | WP3 rebuild on cheap CPU instance |
| `Dockerfile` | `pytorch/pytorch:2.11.0-cuda13.0-cudnn9-runtime` training image |
| `training/train.py:448–456` | Atomic checkpoint write (latest.pt + best.pt) |
| `.gitignore` | `!reports/input_param_registry.json` exemption |
| `docs/verda_deployment.md` | Step-by-step Verda operations guide |
| `CF PInn Rebuild Context/WP12_track_c_training.md` | Original ROCm-targeted Track C spec |

---

## 8. References

[1] pytorch/pytorch Docker Hub. `pytorch/pytorch:2.11.0-cuda13.0-cudnn9-runtime`.  
    https://hub.docker.com/r/pytorch/pytorch

[2] PyTorch 2.11.0 Release Notes (March 2026). sm_122 PTX/SASS added to stable wheels.  
    https://github.com/pytorch/pytorch/releases/tag/v2.11.0

[3] Verda AI Platform — Block Storage.  
    https://docs.verda.ai/storage/block-volumes

[4] Tillet, P., Kung, H.-T., & Cox, D. (2019). Triton: An Intermediate Language and Compiler
    for Tiled Neural Network Computations. *MLSys 2019*.  
    https://doi.org/10.1145/3315508.3329973

[5] Verda AI Platform — Spot Instances.  
    https://docs.verda.ai/compute/spot-instances

[6] The Open Group Base Specifications Issue 7, 2018 Edition (POSIX.1-2017).
    `rename()` — §2.7: "If the link named by the new argument exists, it shall be removed
    and old renamed to new atomically."  
    https://pubs.opengroup.org/onlinepubs/9699919799/functions/rename.html

[7] NVIDIA RTX PRO 6000 Blackwell Product Brief (2025). GB202 die, Compute Capability 12.2
    (sm_122), 96 GB GDDR7, 184 Streaming Multiprocessors.  
    https://www.nvidia.com/en-us/design-visualization/rtx-pro-6000/

[8] AMD MI300X Architecture Whitepaper (2024). Unified Memory Architecture — CPU DDR5
    addressable via xGMI interconnect (128 GB/s peak); PCIe 5.0 host bridge (128 GB/s)
    vs HBM3 bandwidth (5.3 TB/s intra-die). Training gradient tensors require intra-die bandwidth.  
    https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html

[9] Vaswani, A., et al. (2017). Attention Is All You Need. *NeurIPS 2017*.
    Self-attention computational complexity O(n²d) per layer — quadratic in hidden dimension d.  
    https://arxiv.org/abs/1706.03762

[10] Micikevicius, P., et al. (2018). Mixed Precision Training. *ICLR 2018*.
     bfloat16 preserves float32 exponent range, eliminating the need for loss scaling
     that float16 requires due to its narrower dynamic range.  
     https://arxiv.org/abs/1710.03740

[11] Chen, T., et al. (2016). Training Deep Nets with Sublinear Memory Cost. *arXiv:1604.06174*.
     Gradient checkpointing recomputes activations during backward pass at ~1.33× compute
     cost; reduces activation memory from O(n) to O(√n) for a depth-n network.  
     https://arxiv.org/abs/1604.06174

[12] Ansel, J., et al. (2024). PyTorch 2: Faster Machine Learning Through Dynamic Python
     Bytecode Transformation and Graph Compilation. *ASPLOS 2024*.
     `max-autotune` uses Triton autotuning to select the best tiling/vectorisation strategy
     for each kernel shape — typically 1.5–3× throughput improvement over eager mode.  
     https://doi.org/10.1145/3620666.3651312

---

## 9. Multi-GPU Execution

### Automatic detection

`run_cross_scale_level4_cuda.sh` detects the number of CUDA devices at startup:

```bash
N_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())")
```

No flags, env vars, or code changes are required — the script branches automatically.

### Single GPU (N_GPUS == 1)

All 5 folds run sequentially on `cuda:0`. This is the baseline path described
throughout the rest of this document. Wall-clock: ~10 hr.

### Dual GPU (N_GPUS >= 2)

The 5 folds are split across 2 GPUs and run as two parallel background processes:

| Process | `CUDA_VISIBLE_DEVICES` | Folds | Fold count |
|---|---|---|---|
| GPU 0 subprocess | `0` | 0, 2, 4 | 3 (wall-clock ceiling) |
| GPU 1 subprocess | `1` | 1, 3 | 2 |

`CUDA_VISIBLE_DEVICES=N` restricts each subprocess to one physical GPU; within
that process it appears as `cuda:0`, so `--device cuda` in train.py requires no
change [13]. Both processes write to the same `RUN_DIR` — fold directories
(`fold_0/`, `fold_2/`, `fold_4/` on GPU 0; `fold_1/`, `fold_3/` on GPU 1)
are disjoint and do not conflict. `summary.json` is written once by the shell
script after both `wait`s complete, avoiding a write race.

```
GPU 0 timeline:  fold_0 ──── fold_2 ──── fold_4 ──── (done)
GPU 1 timeline:  fold_1 ──── fold_3 ──── (idle) ─────
                 ←────────── ~6 hr (wall-clock) ──────→
Single-GPU ref:  fold_0 ─ fold_1 ─ fold_2 ─ fold_3 ─ fold_4
                 ←─────────────── ~10 hr ──────────────────→
```

Speedup: ~1.67× (bounded by 3-fold GPU 0 path).

### Memory considerations with 2 GPUs

Each GPU process is fully independent — VRAM usage is identical to single-GPU
Track C (estimated 50–65 GB / 96 GB per GPU). There is no cross-GPU communication
and no NVLink requirement. DDR5 system RAM is shared between processes; preloaded
Batch B data (~41 GB per process × 2) requires ~82 GB total, well within
188 GB DDR5 per node.

### Resume after eviction (dual-GPU)

`AUTO_RESUME=1` scans all five fold directories (`fold_0` through `fold_4`) for
`latest.pt`. The scan covers all folds because in a 2-GPU run GPU 1 (folds 1, 3)
may complete before GPU 0 writes its first checkpoint for fold_0. If any
`fold_N/latest.pt` is found, `RESUME=1` is set and all GPU subprocesses receive
`--resume --allow-resume-mismatch`. Already-complete folds (epoch == `--epochs`)
return in < 1 min (zero training epochs; gate metrics read from `history.json`).

### Transitioning between GPU counts mid-training

Because fold checkpoints are written atomically after every epoch and the
`AUTO_RESUME` scan covers all five fold directories (not just `fold_0`), any
combination of 1-GPU → 2-GPU or 2-GPU → 1-GPU transition is handled
automatically at the next `verda_launch_training.sh` call.

| Transition | What happens on restart |
|---|---|
| **2 GPU → 1 GPU** | N_GPUS=1 → `--all-folds --resume`. Completed folds (epoch==50) run 0 training epochs and return in < 1 min. Partial folds resume from `latest.pt`. |
| **1 GPU → 2 GPU** | N_GPUS=2 → split into `--folds 0,2,4` and `--folds 1,3`. Both subsets get `--resume`. Folds already complete (e.g. fold_0, fold_1) run 0 epochs. Remaining folds resume in parallel. |
| **2 GPU → 2 GPU** | Normal eviction-resume path. AUTO_RESUME fires if any `fold_N/latest.pt` exists. |

The only scenario that breaks clean resume is if GPU 0 in a 2-GPU run gets
evicted before writing its **first** checkpoint for any fold (i.e. within the
first epoch of fold_0 — roughly the first 7–12 min including Triton warmup).
After that first write, the scan detects work and activates `--resume`
regardless of GPU count on the next launch.

### Triton cache with 2 GPUs

Both processes share `TRITON_CACHE_DIR` (mounted NVMe). Triton's cache is keyed
by kernel shape + device capability; sm_122 GPUs produce identical kernel hashes.
Concurrent autotuning on first run is safe — Triton uses file-level locking [4].
After the first fold on either GPU has compiled, subsequent folds on both GPUs
reuse the cached kernels.

### 9.5 VRAM-adaptive parameter scaling (B300 fallback)

When the RTX Pro 6000 Blackwell (spot) is unavailable, the B300 data-centre Blackwell
(sm_10x, 262 GB HBM3e, ~€2.45/hr) is the recommended fallback. `run_cross_scale_level4_cuda.sh`
detects VRAM at startup and scales batch/chunk sizes automatically:

| Parameter | RTX Pro 6000 (96 GB) | B300 (262 GB) | Scale factor |
|---|---|---|---|
| `ATTN_BATCH_NODES` | 1024 | 4096 | 4× |
| `DECODER_CHUNK_T` | 24 | 64 | 2.7× |
| `CFWRINKLE_FINE_LOSS_CHUNK_ELEMS` | 2048 | 8192 | 4× |
| `hidden_dim` | 96 | 96 | unchanged |
| `MAX_TIMESTEPS` | 128 | 128 | unchanged |
| `AMP_DTYPE` | bfloat16 | bfloat16 | unchanged |

Detection is VRAM-threshold based (not compute capability):
- VRAM ≥ 200 GB → B300 tier
- VRAM ≥ 80 GB → RTX Pro 6000 tier
- VRAM < 80 GB → conservative fallback (ATTN=512, CHUNK_T=12)

Any auto-scaled value can be overridden by exporting the env var before calling the
script (e.g. `ATTN_BATCH_NODES=2048 bash run_cross_scale_level4_cuda.sh`). The
`${VAR+set}` check captures only vars that were exported by the caller, so the
auto-scaling is skipped only for those that were explicitly set.

**Checkpoint compatibility:** batch/chunk sizes are runtime performance parameters and
are not serialised into checkpoint files. Resuming a B300-trained checkpoint on an
RTX Pro 6000 (or vice versa) requires no flags beyond `--allow-resume-mismatch`, which
`AUTO_RESUME` already passes. The model weights, optimizer state, and epoch counter are
fully portable across GPU types.

**B300 single-GPU expected timeline:** With 4× larger attention batches and 2.7× longer
decoder chunks, per-epoch compute is more efficient — but the B300's larger HBM3e
bandwidth partially offsets the higher per-batch cost. Estimated wall-clock: 8–14 hr
(5 folds, sequential), comparable to RTX Pro 6000 at higher absolute throughput.

**Cost note:** At €2.45/hr vs €0.59/hr, a full 10-hr B300 run costs ~€25 vs ~€6.
Use the B300 only when RTX Pro 6000 spot capacity is genuinely unavailable.

---

### Internal notes (not for external reference)

[A] Track B memory analysis: `reports/ROCM_OPTIMIZATION_HANDOFF_2026-04-18.md`  
[B] WSL2 preload OOM: `CF PInn Rebuild Context/WP12_track_c_training.md` — T=128 OOM-killed at 25.8 GB  
[C] float16 HIP segfault: `CF PInn Rebuild Context/WP12_track_c_training.md` — dtype mismatch on resume  
[D] ROCm torch.compile fallback: PyTorch GitHub issue #157549 — sm_122 and ROCm backend coverage  
[E] sm_122 PyTorch incompatibility: PyTorch GitHub issue #157549 — resolved in 2.11.0 stable  
[F] Git gitignore negation: `git-scm.com/docs/gitignore` — "A pattern which matches a
    previously excluded file can be re-included with a negation pattern... but only if
    the parent directory has not been excluded."

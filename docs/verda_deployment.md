# Verda Deployment Guide — Track C CrossScaleNet

**Target:** RTX Pro 6000 Blackwell spot instance on Verda AI (FIN-03)  
**Training config:** Track C — `hidden_dim=96`, `T=128`, `bfloat16`, `torch.compile max-autotune`  
**Estimated total cost:** ~€8–12 (CPU rebuild + NVMe volume + GPU training, 5-fold CV)

---

## Quick Reference

| Phase | Instance | Duration | Cost |
|---|---|---|---|
| 0 — Infrastructure | Verda console only | 5 min | €0 |
| 1 — WP3 CPU rebuild | CPU instance (any), FIN-03 | 3–4 hr | < €0.50 |
| 2 — Build + push image | Local Docker | 20–30 min | €0 |
| 3 — GPU training | RTX Pro 6000 spot (or B200/B300 fallback), FIN-03 | ~10 hr | ~€6 (RTX) / ~€18 (B200) / ~€25 (B300) |
| NVMe volume (2 weeks) | — | — | ~€5 |

Scripts involved:

```
scripts/verda_cpu_rebuild.sh     Phase 1 — WP3 rebuild on CPU instance
scripts/verda_gpu_setup.sh       Phase 3 — GPU instance setup + smoke test
scripts/verda_launch_training.sh Phase 3 — Docker training launcher
run_cross_scale_level4_cuda.sh   Called inside container by verda_launch_training.sh
```

---

## Prerequisites (one-time local setup)

Before creating any Verda instance:

### 1. Commit all local changes

```bash
cd "C:/Users/ellis/Documents/VS Code/CFWrinklePINN"
git add training/train.py .gitignore Dockerfile scripts/verda_*.sh
git commit -m "feat: Verda deployment — Track C CUDA config, atomic checkpoints, pt2110 image"
git push
```

This ensures the CPU instance clones the correct code including `reports/input_param_registry.json` (now unblocked from `.gitignore`).

### 2. Build and push the Docker image

```bash
# Build locally with a local tag (requires ~20 GB free disk)
docker build -t cfwrinkle-train:pt2110 .

# Tag for Verda container registry
docker tag cfwrinkle-train:pt2110 \
  vccr.io/20175b95-1ac5-4808-89b0-b08dc612c71e/cfwrinkle-train:pt2110

# Log in to Verda container registry
docker login -u vcr-20175b95-1ac5-4808-89b0-b08dc612c71e+erjp-cred-1 \
  -p fTSvVgcig4eA8T7S7TEFSc8nP3d3RnGn \
  vccr.io/20175b95-1ac5-4808-89b0-b08dc612c71e

# Push (~8 GB compressed; within 100 GB free tier)
docker push vccr.io/20175b95-1ac5-4808-89b0-b08dc612c71e/cfwrinkle-train:pt2110
```

Base image: `pytorch/pytorch:2.11.0-cuda13.0-cudnn9-runtime` — no NGC account required [1][2].

---

## Phase 0 — Infrastructure (Verda Console)

### Create the NVMe block volume

In the Verda console (console.verda.ai):

1. Navigate to **Storage → Block Volumes → Create**
2. Settings:
   - **Name:** `cfwrinkle-data`
   - **Type:** NVMe *(HDD volumes deprecated April 2026)*
   - **Size:** 200 GB
   - **Location:** FIN-03 (same datacenter as the GPU instance)
3. Leave the volume unattached — it exists independently of any instance.

The volume persists across instance terminations. It costs ~€0.20/GB/month (retained for ~2 weeks ≈ €5 total) [3].

> **Critical:** When creating the GPU spot instance in Phase 3, set
> `on_spot_discontinue: keep_detached` so the volume survives eviction without
> data loss.

---

## Phase 1 — WP3 Rebuild (CPU Instance)

### Launch a CPU instance

- **OS image:** Ubuntu 24.04 (any CPU variant, FIN-03)
- **GPU:** Not required
- **Storage:** Attach `cfwrinkle-data` volume
- **Startup script:** Paste the contents of `scripts/verda_cpu_rebuild.sh`, or SSH in and run it manually

### What the script does

```
[1/7] Mount NVMe volume
      Detects /dev/vdb (or /dev/sdb); mkfs.ext4 on first use only (blkid check).
      Mounts to /mnt/data.

[2/7] Install Python deps
      apt: python3-venv, git, curl
      pip: h5py>=3.8, numpy>=1.24, scipy>=1.10, tqdm>=4.65
      No torch — WP3 rebuild is pure NumPy/SciPy/h5py.

[3/7] Clone repo
      git clone $GITHUB_REPO /mnt/data/repo
      Verifies reports/input_param_registry.json exists (required by build_features.py).

[4/7] Download WP2 from SharePoint
      If /mnt/data/cfwrinkle_dataset.h5 already present: skip.
      Otherwise: prints rclone and curl instructions and exits.
      Re-run the script after downloading.

[5/7] Sanity-check WP2
      Verifies ≥60 simulations present and mesh/fine/nodes exists.

[6/7] WP3 preflight
      python -m wp3_features.rebuild_checks preflight --require-fine --min-free-gb 110

[7/7] Rebuild + validate
      python -m wp3_features.build_features --include-fine-features   (3–4 hr)
      Postbuild schema check, 7 semantic checks, simulation count assert ≥60.
```

### Download WP2 from SharePoint

**Option A — rclone (recommended; works with university tenant restrictions):**

The Verda instance has no browser, so use rclone's two-machine auth flow:

```bash
# ── Step 1: on your LOCAL Windows machine ──────────────────────────────────
rclone authorize "onedrive"
# Sign in with your University of Twente account in the browser that opens.
# rclone prints a JSON token block — copy the entire thing.

# ── Step 2: on the CPU instance (SSH) ─────────────────────────────────────
curl https://rclone.org/install.sh | bash
rclone config
# → New remote → name: onedrive
# → Storage type: Microsoft OneDrive
# → client_id / client_secret: (leave blank, press Enter)
# → Edit advanced config: No
# → Use auto config? → No   ← Verda has no browser
# → Paste the JSON token from Step 1 when prompted
# → Drive type: OneDrive (business or personal)
# → Confirm

# ── Step 3: download ───────────────────────────────────────────────────────
rclone copy "onedrive:CFWrinkle/cfwrinkle_dataset.h5" /mnt/data/ --progress
```

**Option B — SharePoint sharing link (requires "Anyone with link" permission):**

```bash
# Only works if your tenant allows anonymous sharing links.
# University of Twente restricts this — use Option A (rclone) instead.
curl -L -o /mnt/data/cfwrinkle_dataset.h5 'https://univ.sharepoint.com/:u:/s/...'
```

Expect ~55 GB. Verda FIN-03 → Microsoft Azure backbone is fast (~500 MB/s sustained).

### After rebuild

```bash
# Optional: back up WP3 to SharePoint before terminating
rclone copy /mnt/data/cfwrinkle_wp3_features.h5 "onedrive:CFWrinkle/" --progress
```

Then: **detach `cfwrinkle-data` volume** in the Verda console, terminate the CPU instance.

---

## Phase 2 — Build and Push Docker Image (Local)

See Prerequisites §3 above. The image must be pushed before the GPU instance can pull it.

Image contents: `pytorch/pytorch:2.11.0-cuda13.0-cudnn9-runtime` base + `h5py`, `numpy`, `scipy`, `tqdm` in a venv at `/workspace/venv`. Repo COPY'd to `/workspace/repo`. Compressed size ~8 GB [1].

---

## Phase 3 — GPU Training (RTX Pro 6000 Spot or B300 Fallback)

`run_cross_scale_level4_cuda.sh` detects GPU count and VRAM at startup and scales
hyperparameters automatically — no flags or config changes needed when switching GPU types.

| GPU option | VRAM | Cost | Auto-configured params |
|---|---|---|---|
| RTX Pro 6000 Blackwell (primary) | 96 GB GDDR7 | ~€0.59/hr spot | ATTN=1024, CHUNK_T=24, FINE_LOSS=2048 |
| RTX Pro 6000 × 2 (primary) | 2 × 96 GB | ~€1.18/hr spot | Dual-GPU fold split, ~1.67× speedup |
| B200 data-centre Blackwell (fallback) | 192 GB HBM3e | ~€1.80/hr | ATTN=4096, CHUNK_T=64, FINE_LOSS=8192; **parallel fold mode auto-enabled** |
| B200 × 2 (fallback) | 2 × 192 GB | ~€3.60/hr | Dual-GPU fold split + large params; ~1.67× speedup |
| B300 data-centre Blackwell (fallback) | 262 GB HBM3e | ~€2.45/hr | ATTN=4096, CHUNK_T=64, FINE_LOSS=8192; **parallel fold mode auto-enabled** |

> **B200/B300 note:** When VRAM ≥ 160 GB is detected (B200: ≥160 GB, B300: ≥200 GB), the
> script (a) upgrades batch and chunk sizes to fill the extra headroom, and (b) automatically
> runs two parallel fold processes on the same device (folds 0,2,4 ∥ 1,3), giving the same
> ~1.67× speedup as 2× RTX Pro 6000. `hidden_dim`, `T`, and model architecture are unchanged.
> Set `B300_PARALLEL=0` to force sequential if debugging or if VRAM is unexpectedly tight.
> Override any auto-scaled value via env vars (e.g. `ATTN_BATCH_NODES=2048`).

### Launch a GPU spot instance

- **OS image:** `ubuntu-24.04-cuda-13.0-open-docker`
  *(Ubuntu 24.04, CUDA 13.0, open kernel modules for sm_12x / sm_10x, Docker pre-installed)*
- **GPU:** RTX Pro 6000 Blackwell (spot), FIN-03 — 1 or 2 units; or 1–2× B200 / 1× B300 as fallback
- **`on_spot_discontinue`:** `keep_detached` ← **set this at creation**
- **Storage:** Attach `cfwrinkle-data` volume
- **Startup script:** Paste `scripts/verda_gpu_setup.sh`

### Setup script (verda_gpu_setup.sh)

```
[1/4] Mount volume        — no mkfs (already formatted in Phase 1)
[2/4] Pull Docker image   — from vccr.io/<PROJECT>/cfwrinkle-train:pt2110
[3/4] GPU smoke test      — sm_122 check, SDPA Flash, BF16 matmul < 5 ms
[4/4] HDF5 readability    — verifies both .h5 files inside container
```

Expected smoke test output:

```
GPU:        NVIDIA RTX 6000 Pro Ada Generation [or similar]
VRAM:       96.0 GB
Capability: sm_122
SDPA Flash: available ✓
BF16 matmul 4096×4096: 2.x ms/call ✓
SMOKE TEST PASSED
```

> If `sm_maj` is not 12, the script prints a warning but does not abort —
> verify you have the correct instance type before proceeding.

### Launch training

```bash
bash /mnt/data/repo/scripts/verda_launch_training.sh
```

This runs `docker run` with the following volume mounts:

| Host path (NVMe) | Container path | Purpose |
|---|---|---|
| `/mnt/data` | `/workspace/data_vol` | WP2 + WP3 HDF5 |
| `/mnt/data/checkpoints` | `/workspace/checkpoints` | Fold checkpoints |
| `/mnt/data/logs` | `/workspace/logs` | Run log |
| `/mnt/data/.triton_cache` | `/workspace/.triton_cache` | torch.compile kernel cache [4] |

The script passes env vars so `run_cross_scale_level4_cuda.sh` inside the container needs no edits.

### Training progress

```bash
# From the host (outside Docker):
tail -f /mnt/data/logs/level4_trackc_verda.log

# Fold progress:
ls -la /mnt/data/checkpoints/progressive/cross_scale_level4_cv_trackc_verda/

# Gate report (written after all folds):
cat /mnt/data/checkpoints/progressive/cross_scale_level4_cv_trackc_verda/gate_report.json
```

Expected timeline:
- First run: 10–20 min torch.compile autotune + kernel warmup, then ~1.5–2 hr/fold × 5 folds ≈ 10 hr total
- Resume after eviction: autotune cache warm (~2 min), training resumes from last epoch

---

## Resume After Spot Eviction

1. Volume retained automatically (`keep_detached`).
2. In Verda console: create a new instance, attach `cfwrinkle-data`. Use any available GPU type — RTX Pro 6000 (1 or 2), B200 (1 or 2), or B300 (1).
3. Run `scripts/verda_gpu_setup.sh` (pulls image, smoke test reports GPU count).
4. Run `scripts/verda_launch_training.sh` — `AUTO_RESUME=1` scans `fold_0/` through `fold_4/` for any `latest.pt` and activates `--resume --allow-resume-mismatch`.
5. Maximum work lost per eviction: 1 epoch (~5–8 min), because `latest.pt` is written atomically after every epoch [5].

### GPU count changes across evictions

The script detects GPU count at every launch and adjusts automatically — no
flags or config changes needed:

| Previous run | New instance | Behaviour |
|---|---|---|
| 2 GPUs | 1 GPU available | `--all-folds --resume`. Completed folds finish in < 1 min. |
| 1 GPU | 2 GPUs available | Dual-GPU split; already-complete folds resume in < 1 min each. |
| 2 GPUs | 2 GPUs available | Dual-GPU split as before; resumes each fold subset from checkpoint. |
| RTX Pro 6000 (any) | B200 (1×) | B200 parallel mode (2 procs, same device). ATTN/chunk auto-scaled up. Model weights compatible. |
| RTX Pro 6000 (any) | B200 (2×) | Dual-GPU split + large params. ATTN/chunk auto-scaled up. Model weights compatible. |
| RTX Pro 6000 (any) | B300 | B300 parallel mode (2 procs, same device). ATTN/chunk auto-scaled up. Model weights compatible. |
| B200 / B300 | RTX Pro 6000 | `--all-folds --resume` (1 GPU sequential) or dual-GPU split if 2× available. ATTN/chunk auto-scaled down to 96 GB defaults. |
| B200 (1×) | B300 | B300 parallel mode. ATTN/chunk unchanged (both use same large defaults). AUTO_RESUME picks up all fold checkpoints. |
| B300 | B200 (1×) | B200 parallel mode. ATTN/chunk unchanged. AUTO_RESUME picks up all fold checkpoints. |
| B200 (1×) | B200 (2×) | Switches to dual-GPU split. AUTO_RESUME resumes all fold checkpoints. |

The resume scan covers all five fold directories (`fold_0` through `fold_4`), so
partial progress from either GPU in a prior 2-GPU run is detected even if `fold_0`
has no checkpoint yet (i.e. GPU 1 completed fold_1 before GPU 0 finished epoch 1).

> **Why atomicity matters:** `torch.save` writes a single serialised pickle in a
> sequential write that can take 5–15 s on NVMe for a ~300 MB checkpoint. An
> eviction mid-write leaves a truncated file that fails to unpickle and breaks
> resume. The atomic write pattern (`latest.pt.tmp` → `rename` → `latest.pt`)
> guarantees readers see either the complete previous checkpoint or the complete
> new one — never a partial write. POSIX `rename(2)` is guaranteed atomic for
> intra-filesystem moves [6].

---

## Downloading Results

```bash
# From local machine (SSH to Verda instance first):
rsync -avz --progress \
  root@<VERDA_IP>:/mnt/data/checkpoints/progressive/cross_scale_level4_cv_trackc_verda/ \
  ./checkpoints/track_c_verda/

# Gate report:
scp root@<VERDA_IP>:/mnt/data/checkpoints/progressive/cross_scale_level4_cv_trackc_verda/gate_report.json \
    reports/wp7_gate_level4_cross_scale_trackc_verda.json
```

After downloading, terminate the GPU instance. Detach or delete the volume depending on whether another run is planned.

---

## Environment Variable Reference

All variables have defaults in `run_cross_scale_level4_cuda.sh`. Override at launch:

```bash
IMAGE=vccr.io/<PROJECT>/cfwrinkle-train:pt2110 \
VOLUME_ROOT=/mnt/data \
bash /mnt/data/repo/scripts/verda_launch_training.sh
```

Or override training variables by adding `-e VAR=VALUE` to the `docker run` call in `verda_launch_training.sh`:

| Variable | Default | Notes |
|---|---|---|
| `HIDDEN_DIM` | `96` | Track C target; increase to 128 only after memory validation |
| `MAX_TIMESTEPS` | `128` | Full temporal sequence; 96 for memory-constrained debugging |
| `EPOCHS` | `50` | Matches Track B for direct comparison |
| `ATTN_BATCH_NODES` | `1024` | Sized for 184 SM Blackwell; reduce to 512 if VRAM unexpectedly tight |
| `DECODER_CHUNK_T` | `24` | Longer chunk → fewer backward passes; reduce to 8 if OOM |
| `AMP_DTYPE` | `bfloat16` | Blackwell native; do NOT change to float16 (see Track C reference) |
| `TORCH_COMPILE_MODE` | `max-autotune` | Change to `reduce-overhead` for short test runs |
| `TRITON_CACHE_DIR` | `/workspace/.triton_cache` | Mounted from NVMe; survives eviction |
| `AUTO_RESUME` | `1` | Detects `fold_*/latest.pt` automatically |

---

## Troubleshooting

**`docker pull` fails with auth error**
→ Re-run `docker login vccr.io/<PROJECT_ID>` with the correct registry secret from the Verda console.

**Smoke test: sm_maj unexpected**
→ Verify you launched the RTX Pro 6000 Blackwell instance type. sm_122 is the expected capability (GB202 die, CC 12.2) [7].

**`WP3_H5 not found` error at training launch**
→ Phase 1 did not complete. Re-attach the NVMe volume to a CPU instance and re-run `verda_cpu_rebuild.sh` from step [6/7].

**torch.compile hangs at first epoch (> 20 min)**
→ This is normal on first run — Triton autotuning compiles kernels for all shapes. Subsequent runs reuse `/mnt/data/.triton_cache`. Do not interrupt [4].

**OOM during training**
→ Reduce `ATTN_BATCH_NODES` to `512`, `DECODER_CHUNK_T` to `12`, or add `--checkpoint` flag to `run_cross_scale_level4_cuda.sh`. See Track C CUDA Reference for memory budget.

**Resume fails: `best_val_loss` mismatch**
→ `--allow-resume-mismatch` is already set by `AUTO_RESUME`. If the error persists, load `latest.pt` manually and inspect `ckpt['model_config']`.

**MPI `not enough slots`**
→ Not applicable to this training stack (single-GPU, no MPI). If seen in logs, it is from a stray process — kill and restart.

---

## References

[1] pytorch/pytorch Docker Hub. `pytorch/pytorch:2.11.0-cuda13.0-cudnn9-runtime`.  
    https://hub.docker.com/r/pytorch/pytorch

[2] PyTorch 2.11.0 Release Notes, March 2026. sm_122 support landed in stable wheels.  
    https://github.com/pytorch/pytorch/releases/tag/v2.11.0

[3] Verda AI Platform — Block Storage documentation.  
    https://docs.verda.ai/storage/block-volumes

[4] Tillet, P., Kung, H.-T., & Cox, D. (2019). Triton: An Intermediate Language and Compiler
    for Tiled Neural Network Computations. *MLSys 2019*.  
    https://doi.org/10.1145/3315508.3329973

[5] Verda AI Platform — Spot Instances and volume persistence.  
    https://docs.verda.ai/compute/spot-instances

[6] The Open Group Base Specifications Issue 7, 2018 Edition (POSIX.1-2017).
    `rename()` — §2.7: "If the link named by the new argument exists,
    it shall be removed and old renamed to new atomically."  
    https://pubs.opengroup.org/onlinepubs/9699919799/functions/rename.html

[7] NVIDIA RTX PRO 6000 Blackwell Server GPU Product Brief, 2025.
    GB202 die, Compute Capability 12.2 (sm_122), 96 GB GDDR7.  
    https://www.nvidia.com/en-us/design-visualization/rtx-pro-6000/

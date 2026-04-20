# CFWrinklePINN — Workspace Setup

Clean rebuild project. Raw data is referenced in place — do not copy it.

## Directory Structure

```
CFWrinklePINN/
├── SETUP.md                    ← this file
├── ANIFORM_REFERENCE.md        ← WP1 field archaeology reference
├── config/
│   ├── pipeline_config.yaml    ← all paths (created in WP2)
│   └── field_registry.yaml     ← locked after WP1
├── io/                         ← WP2/3 deliverables
├── mesh/                       ← WP3 deliverables
├── features/                   ← WP4 deliverables
├── targets/                    ← WP4 deliverables
├── training/                   ← WP5/7 deliverables
├── model/                      ← WP6/7 (do not create until WP5 gate)
├── validation/                 ← WP1/3 deliverables
├── tests/
├── reports/
├── data/
│   └── dataset.h5              ← created empty in WP2, filled in WP3
├── wp1_survey/                 ← WP1 archaeology scripts (disposable)
│   ├── probe_afr_fields.py
│   ├── parse_all_afi.py
│   └── read_track_force.py
├── orchestrator.py
├── requirements.txt
└── .venv/                      ← local venv (not committed)
```

Raw data stays at its original location — set `data_root` in `pipeline_config.yaml`.

## Create the venv

```powershell
cd "C:\Users\ellis\Documents\VS Code\CFWrinklePINN"
python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install numpy scipy h5py pyyaml tqdm
pip install pytest pytest-cov ruff

# PyTorch ROCm (after GPU env vars are set)
# pip install --no-cache-dir --index-url https://repo.radeon.com/rocm/windows/rocm-rel-7.2/ torch
```

Use a clean venv (no `--system-site-packages`) so ROCm CLI tools resolve from the active environment.

## GPU environment (AMD RX 7900 XT)

```powershell
$env:TORCH_BLAS_PREFER_HIPBLASLT = "1"
$env:PYTORCH_TUNABLE_OP_ENABLED = "1"
$env:PATH = "C:\Users\ellis\AppData\Local\Programs\Python\Python312\Scripts;$env:PATH"
```

Avoid forcing `HSA_OVERRIDE_GFX_VERSION` and `PYTORCH_ROCM_ARCH` unless AMD docs explicitly require it for your driver/toolchain combination.

## Radeon AI bundle PyTorch (recommended Windows path)

If Radeon AI bundle PyTorch is already installed globally for Python 3.12, use:

```powershell
cd "C:\Users\ellis\Documents\VS Code\CFWrinklePINN"
py -3.12 -m venv --system-site-packages .venv
.\activate-amd-bundle.ps1
```

This keeps the project isolated while reusing the installed AMD bundle distribution.

## WSL2 ROCm path (recommended for training stability)

When native Windows ROCm training is unstable, use WSL2 with a Linux venv stored on the Linux filesystem (not `/mnt/c/...`).

```powershell
wsl -d Ubuntu-24.04 --cd "/mnt/c/Users/ellis/Documents/VS Code/CFWrinklePINN" bash -lc "./wsl_setup_env.sh"
```

This creates/updates:
- venv: `/home/ellis/venvs/cfwrinkle`
- PyTorch ROCm wheels (default: ROCm 7.1 index)
- project requirements + `pandas`
- a quick torch/HIP probe printout

`wsl_setup_env.sh` and `wsl_gpu_smoke.sh` also apply a runtime-link workaround by default:

```bash
export ROCM_RUNTIME_LIB=/opt/rocm-7.2.0/lib/libamdhip64.so
```

On some WSL driver/runtime combinations, this is required for PyTorch to enumerate the GPU (`torch.cuda.is_available()`), even when `rocminfo` already sees `gfx1100`.

Run the project test suite from WSL:

```powershell
wsl -d Ubuntu-24.04 --cd "/mnt/c/Users/ellis/Documents/VS Code/CFWrinklePINN" bash -lc "source /home/ellis/venvs/cfwrinkle/bin/activate && export PYTHONPATH=\"$(pwd)\" && python -m pytest tests/ -q"
```

Run one-epoch GPU smoke training from WSL:

```powershell
wsl -d Ubuntu-24.04 --cd "/mnt/c/Users/ellis/Documents/VS Code/CFWrinklePINN" bash -lc "./wsl_gpu_smoke.sh"
```

`wsl_gpu_smoke.sh` defaults to `SMOKE_HIDDEN_DIM=16` to reduce OOM risk on busy desktop GPUs. Override if needed:

```bash
SMOKE_HIDDEN_DIM=32 ./wsl_gpu_smoke.sh
```

Run the progressive WP7 training gate (with resource preflight before any training starts):

```powershell
wsl -d Ubuntu-24.04 --cd "/mnt/c/Users/ellis/Documents/VS Code/CFWrinklePINN" bash -lc "./wsl_progressive_suite.sh --preflight-only"
wsl -d Ubuntu-24.04 --cd "/mnt/c/Users/ellis/Documents/VS Code/CFWrinklePINN" bash -lc "./wsl_progressive_suite.sh --max-level 3"
```

Track B Level 3 (`run_cross_scale_level3.sh`) now uses a staged profile strategy.

Default (gate-first pilot) profile:
- `LEVEL3_PROFILE=pilot` (default)
- `PILOT_MAX_TRAIN_SIMS=8`, `PILOT_MAX_VAL_SIMS=1`
- `PILOT_EPOCHS=12`
- `PILOT_MAX_TIMESTEPS=32`
- `PILOT_ATTN_BATCH_NODES=96`
- `PILOT_DECODER_CHUNK_T=8`

Run the full mini-train profile after pilot completes:

```bash
LEVEL3_PROFILE=mini ./run_cross_scale_level3.sh --resume
```

Full mini profile defaults:
- `MINI_MAX_TRAIN_SIMS=13`, `MINI_MAX_VAL_SIMS=2`
- `MINI_EPOCHS=50`
- `MINI_MAX_TIMESTEPS=64`
- `MINI_ATTN_BATCH_NODES=128`
- `MINI_DECODER_CHUNK_T=16`

Optional stall bound for either profile:

```bash
TRAIN_TIMEOUT_SEC=3600 ./run_cross_scale_level3.sh
```

If interrupted, failed, or timed out, the script now writes an explicit blocked gate report to `REPORT_PATH` (instead of leaving stale pass/fail ambiguity).

## Track B Level 4 safe pause / night resume (current operating mode)

The Level 4 all-fold run supports safe pause/resume via checkpoint state per fold.

Current known paused state:
- `fold_0`: complete at epoch 50 (`latest.pt`, `best.pt`, `history.json`)
- `fold_1`: partial at epoch 21 (`latest.pt`, `best.pt`, `history.json`)
- normalization state preserved (`fine_input/normalized=1.0`)

Resume command:

```bash
AUTO_RESUME=1 ./run_cross_scale_level4.sh
```

Do not lower fidelity constraints when resuming:
- keep `MAX_TIMESTEPS >= 96`
- keep fine-feature normalization ON

Fallback retry (tighter mini profile) if Level 3 still OOMs:

```bash
LEVEL3_PROFILE=mini \
MINI_MAX_TIMESTEPS=48 \
MINI_ATTN_BATCH_NODES=96 \
MINI_DECODER_CHUNK_T=8 \
MINI_PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.5,max_split_size_mb:32 \
./run_cross_scale_level3.sh --resume
```

Default preflight thresholds are:
- CPU cores >= 8
- total RAM >= 12 GB
- available RAM >= 6 GB
- swap >= 8 GB
- free disk >= 40 GB
- free GPU memory >= 6 GB (`>= 10 GB` for full CV)

Override thresholds per run if needed:

```bash
MIN_MEM_TOTAL_GB=16 MIN_GPU_FREE_GB=8 ./wsl_progressive_suite.sh --max-level 2
```

If preflight fails due WSL resource caps, set `C:\Users\ellis\.wslconfig` and restart WSL:

```ini
[wsl2]
memory=24GB
processors=12
swap=16GB
```

```powershell
wsl --shutdown
```

If `rocminfo` sees `gfx1100` but PyTorch still reports `cuda=False`, this is a host/runtime compatibility issue (ROCDXG + driver/toolchain combo), not a project code issue. In that case:
- verify host Radeon driver/ROCDXG support matrix alignment for your WSL target stack,
- keep using native Windows Radeon AI bundle path temporarily for GPU work,
- or run CPU smoke in WSL until the host stack is aligned.

## What gets copied from CFWrinklePredict2

Only the Aniform readers — nothing else.

```powershell
$src = "C:\Users\ellis\Documents\VS Code\CFWrinklePredict2\data\aniform_readers"
$dst = "C:\Users\ellis\Documents\VS Code\CFWrinklePINN\io\aniform_readers"
cp -r $src $dst
```

Everything else is rewritten from scratch per the WP plan.

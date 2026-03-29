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

## GPU environment (AMD RX 7900 XT)

```powershell
$env:HIP_VISIBLE_DEVICES = "1"
$env:HSA_OVERRIDE_GFX_VERSION = "11.0.0"
$env:TORCH_BLAS_PREFER_HIPBLASLT = "1"
$env:PYTORCH_TUNABLE_OP_ENABLED = "1"
```

## What gets copied from CFWrinklePredict2

Only the Aniform readers — nothing else.

```powershell
$src = "C:\Users\ellis\Documents\VS Code\CFWrinklePredict2\data\aniform_readers"
$dst = "C:\Users\ellis\Documents\VS Code\CFWrinklePINN\io\aniform_readers"
cp -r $src $dst
```

Everything else is rewritten from scratch per the WP plan.

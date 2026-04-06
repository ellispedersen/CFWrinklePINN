# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CFWrinklePINN predicts wrinkle formation in composite sheet molding using Physics-Informed Neural Networks. It processes AniForm FEA simulation data through a staged work-package pipeline (WP1→WP7). Raw data lives in `CFWrinklePredict2/` and is referenced in place — never copy it.

Two material batches: **Batch A** (21 UD thermoplastic pairs, 2 plies) and **Batch B** (45 Twintex 2×2 twill pairs, 3 plies). Total: 66 simulation pairs, each with coarse and fine mesh variants.

## Environment Setup

```powershell
cd "C:\Users\ellis\Documents\VS Code\CFWrinklePINN"
# Venv is Python 3.12 with --system-site-packages (inherits PyTorch+ROCm)
.\.venv\Scripts\Activate.ps1
```

**GPU**: AMD RX 7900 XT, 21.5 GB VRAM, ROCm 7.2, PyTorch 2.9.1+rocmsdk.
All critical GNN ops verified: GRU, scatter_add, LayerNorm, MultiheadAttention.
No PyTorch Geometric — use custom message passing with scatter ops.

### WSL2 ROCm Training Path (preferred when Windows stack is unstable)

- WSL distro: `Ubuntu-24.04`
- Linux-native venv: `/home/ellis/venvs/cfwrinkle`
- Runtime-link workaround may be required for HIP visibility in WSL:
  - `LD_PRELOAD=/opt/rocm-7.2.0/lib/libamdhip64.so`

Use the helper scripts:

```powershell
wsl -d Ubuntu-24.04 --cd "/mnt/c/Users/ellis/Documents/VS Code/CFWrinklePINN" bash -lc "./wsl_setup_env.sh"
wsl -d Ubuntu-24.04 --cd "/mnt/c/Users/ellis/Documents/VS Code/CFWrinklePINN" bash -lc "./wsl_gpu_smoke.sh"
```

Progressive WP7 suite with preflight resource gate:

```powershell
wsl -d Ubuntu-24.04 --cd "/mnt/c/Users/ellis/Documents/VS Code/CFWrinklePINN" bash -lc "./wsl_progressive_suite.sh --preflight-only"
wsl -d Ubuntu-24.04 --cd "/mnt/c/Users/ellis/Documents/VS Code/CFWrinklePINN" bash -lc "./wsl_progressive_suite.sh --max-level 3"
```

Default preflight requirements (override via env vars):
- CPU cores >= 8
- RAM total >= 12 GB, RAM available >= 6 GB
- Swap >= 8 GB
- Disk free >= 40 GB
- GPU free >= 6 GB (`>= 10 GB` for full CV)

## Commands

```bash
# WP1 survey scripts (run in order)
python -m wp1_survey.probe_afr_fields      # → reports/afr_field_survey.json
python -m wp1_survey.parse_all_afi         # → reports/input_param_registry.json
python -m wp1_survey.read_track_force      # → reports/force_stroke_summary.json
python -m validation.field_survey          # → reports/field_survey_detail.json
python -m validation.wrinkle_detector      # → reports/wrinkle_onset_registry.json

# Tests and lint
pytest tests/
ruff check .
```

## Architecture

```
io/aniform_readers/    — Binary AniForm file readers (DO NOT MODIFY)
wp1_survey/            — Disposable archaeology scripts (WP1 complete)
validation/            — Field survey + wrinkle detector
wp2_build/             — HDF5 dataset builder (schema, build, validate)
wp3_features/          — Feature extraction, graph construction, targets (WP3)
config/                — pipeline_config.yaml (paths/params) + field_registry.yaml (field metadata)
reports/               — JSON outputs + gate checklists
data/                  — cfwrinkle_dataset.h5 (68 GB, gitignored)
data/                  — HDF5 dataset (WP2+, gitignored)
```

WP progression: WP1 (archaeology) → WP2 (HDF5 schema) → WP3 (extraction) → WP4 (features) → WP5 (CV) → WP6 (model) → WP7 (training). Each WP has a gate checklist in `reports/`. Do not begin WP(N+1) until WP(N) gate passes.

## Critical AniForm Reader APIs

**ReadAFResult** — reads `.afr` binary result files:
```python
(ResultsIncr, Groups, Increments, res_type, indices_included, res_id) = ReadAFResult(filename, elemGrNrs=[], IncsToExport=[], silent=True)
# ResultsIncr[incr_nr][group_id] → ndarray (n_nodes, n_components+1), col 0 = node index
# silent=False prints the header Name string — use this to identify unknown fields
```

**ReadAFSFile** — reads `.afs` solution files; returns **pandas DataFrames** (not dicts):
```python
(IncrementInfo, Name, Version) = ReadAFSFile(filename)
# Access: IncrementInfo.loc['t_end'].values.astype(float)  — NOT dict-style iteration
```

**ReadAFMesh** — reads `.afm` reference mesh:
```python
(Nodes, Elements) = ReadAFMesh(filename, elemGrNrs=[])
# Nodes[group] → (n_nodes, 3), Elements[group] → (n_elements, 3)
```

## Critical Data Conventions

**Batch A ply groups:** `[6, 10]` (kinematics) / `[6, 7, 10, 11]` (tensor fields with bending sub-elements)
**Batch B ply groups:** `[4, 8, 12]` (kinematics) / `[4, 5, 8, 9, 12, 13]` (tensor fields)
Contact/tool groups (exclude from features): Batch A `[8, 9, 13]`, Batch B `[6, 7, 11, 15]`

**Coarse vs fine mesh:** Identify by node count from displacement field — Batch A fine ~75,000 nodes, coarse ~3,700; Batch B fine ~40,000.

## Field Identity (confirmed via ReadAFResult silent=False)

The field_registry.yaml was initially populated with wrong assumptions. The correct mapping (from `reports/SESSION_HANDOFF.md`) is:

| AFR file | Actual Name | Type | Wrinkle relevance |
|----------|-------------|------|-------------------|
| model_40_1 | Displacement | Vec3 | dz variance = wrinkle detection |
| model_44_1 | Temperature | Vec3 | Thermal history |
| model_102_1 | Green-Lagrange strain | Tensor3 | [E11,E22,E12] elastic strains |
| model_105_1 | Thickness | Scalar | Thinning indicator |
| model_106_1 | Eq shear rate | Scalar | Viscous flow rate |
| model_200_1 | Stress | Tensor3 | [s11,s22,s12] Cauchy stress |
| model_204_1 | Shear angle f1_f2 | Scalar | Primary woven wrinkle indicator |
| model_205_1/2 | Fiber strain 1/2 | Scalar | Stretch ratio per fiber family |
| model_206_1/2 | Fiber stress 1/2 | Scalar | **PRIMARY wrinkle precursor** (0→80% compressive) |
| model_214_1 | Nakamura crystallinity | Scalar | Batch A only |
| model_302_1 | Penetration depth | Scalar | Contact field — exclude |
| model_304_1 | Slip path length | Scalar | Contact field — exclude |
| model_401_1 | Traction | Vec3 | All zeros — exclude |

Fields 42 (Rotation), 48 (Temperature 1), 49 (Temperature 2) are secondary.

## Key Conventions

- All paths should come from `config/pipeline_config.yaml` — avoid hardcoding.
- `field_registry.yaml` is locked after WP1 gate. Changes require version bump in notes.
- Raw data in `CFWrinklePredict2/` is read-only. Never modify or copy it.
- Reports JSON files are gitignored — they are regenerated by scripts.
- Windows environment with bash shell in terminal. Use forward slashes in Python paths.
- When reading AniForm files, always pass `elemGrNrs=[]` first to discover available groups, then filter.

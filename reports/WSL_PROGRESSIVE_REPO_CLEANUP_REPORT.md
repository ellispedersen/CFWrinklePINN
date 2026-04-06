# WSL Progressive Suite + Repo Cleanup Report

## Scope completed

1. Added and documented WSL progressive training orchestration with preflight resource gating.
2. Cleaned transient local artifacts from the working tree.
3. Updated guidance docs to make WSL startup checks explicit before training.

## Key implementation

- Added `wsl_progressive_suite.sh`:
  - Preflight checks for CPU/RAM/swap/disk/GPU memory
  - Hard fail before training if system resources are below thresholds
  - Progressive levels:
    - Level 0: dataset + unit/smoke tests
    - Level 1: smoke train
    - Level 2: overfit train
    - Level 3: mini-train
    - Level 4: full CV (optional)

- Updated `SETUP.md`:
  - New preflight and progressive run commands
  - `.wslconfig` tuning guidance
  - Threshold override examples

- Updated `CLAUDE.md`:
  - WSL ROCm path and helper scripts
  - Progressive preflight commands and defaults

## Repository cleanup done

- Removed transient local artifacts:
  - `.venv312_clean/`
  - `.venv_native/`
  - `.venv_wsl/`
  - `checkpoints/`
  - `.pytest_cache/`
  - `.ruff_cache/`

- Updated `.gitignore` to reduce future repo noise:
  - `.venv*/`
  - `checkpoints/`
  - `.pytest_cache/`

## Current environment preflight snapshot (latest run)

- CPU cores: 16
- RAM total: 27.41 GB
- RAM available: 26.72 GB
- Swap total: 8.00 GB
- Disk free: 201.21 GB
- GPU free memory: 18.96 GB

Result: preflight passes for progressive execution.

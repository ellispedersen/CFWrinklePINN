#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${VENV_PATH:-/home/ellis/venvs/cfwrinkle}"
ROCM_RUNTIME_LIB="${ROCM_RUNTIME_LIB:-/opt/rocm-7.2.0/lib/libamdhip64.so}"
SMOKE_HIDDEN_DIM="${SMOKE_HIDDEN_DIM:-16}"

if [[ ! -f "$VENV_PATH/bin/activate" ]]; then
  echo "ERROR: Venv not found at $VENV_PATH"
  echo "Run ./wsl_setup_env.sh first."
  exit 1
fi

source "$VENV_PATH/bin/activate"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR"
if [[ -f "$ROCM_RUNTIME_LIB" ]]; then
  export LD_LIBRARY_PATH="$(dirname "$ROCM_RUNTIME_LIB"):${LD_LIBRARY_PATH:-}"
  export LD_PRELOAD="$ROCM_RUNTIME_LIB${LD_PRELOAD:+:$LD_PRELOAD}"
fi

python - <<'PY'
import torch, sys
print("torch", torch.__version__)
print("hip", torch.version.hip)
print("cuda", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
if not torch.cuda.is_available():
    print("ERROR: No HIP/CUDA-visible GPU for PyTorch in WSL.")
    print("Hint: if rocminfo sees GPU but torch does not, host ROCDXG/driver stack is misaligned.")
    sys.exit(2)
print("device", torch.cuda.get_device_name(0))
PY

python -m training.train \
  --sim-ids geom_0_2 \
  --epochs 1 \
  --hidden-dim "$SMOKE_HIDDEN_DIM" \
  --device cuda \
  --no-normalize \
  --output checkpoints/smoke_wsl_gpu

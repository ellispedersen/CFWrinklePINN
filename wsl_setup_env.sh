#!/usr/bin/env bash
set -euo pipefail

if ! grep -qi microsoft /proc/version 2>/dev/null; then
  echo "ERROR: This script is intended for WSL2."
  exit 1
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${VENV_PATH:-/home/ellis/venvs/cfwrinkle}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/rocm7.1}"
TORCH_VERSION="${TORCH_VERSION:-2.11.0+rocm7.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.26.0+rocm7.1}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.11.0+rocm7.1}"
ROCM_RUNTIME_LIB="${ROCM_RUNTIME_LIB:-/opt/rocm-7.2.0/lib/libamdhip64.so}"

mkdir -p "$(dirname "$VENV_PATH")"
if [[ ! -d "$VENV_PATH" ]]; then
  python3 -m venv "$VENV_PATH"
fi

source "$VENV_PATH/bin/activate"
python -m pip install -U pip --progress-bar off
python -m pip install \
  "torch==$TORCH_VERSION" \
  "torchvision==$TORCHVISION_VERSION" \
  "torchaudio==$TORCHAUDIO_VERSION" \
  --index-url "$TORCH_INDEX_URL" \
  --progress-bar off
python -m pip install -r "$REPO_DIR/requirements.txt" --progress-bar off
python -m pip install pandas --progress-bar off

if [[ -f "$ROCM_RUNTIME_LIB" ]]; then
  export LD_LIBRARY_PATH="$(dirname "$ROCM_RUNTIME_LIB"):${LD_LIBRARY_PATH:-}"
  export LD_PRELOAD="$ROCM_RUNTIME_LIB${LD_PRELOAD:+:$LD_PRELOAD}"
fi

python - <<'PY'
import torch
print("torch", torch.__version__)
print("hip", torch.version.hip)
print("cuda", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
PY

echo "WSL environment setup complete."

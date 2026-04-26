#!/usr/bin/env bash
# RunPod environment setup for RTX Pro 6000 Blackwell (CUDA).
# Run once after pod creation, before starting training.
# Assumes network volume is mounted at /workspace.
set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/repo}"
VENV_PATH="${VENV_PATH:-/workspace/venv}"
DATA_DIR="${DATA_DIR:-/workspace/data}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/workspace/checkpoints}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/workspace/.triton_cache}"
LOG_DIR="${LOG_DIR:-/workspace/logs}"

echo "=== RunPod CUDA setup (RTX Pro 6000 Blackwell) ==="
echo "Network volume root: /workspace"
echo ""

# ── 1. Verify GPU ─────────────────────────────────────────────────────────────
echo "[1/6] GPU check..."
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python3 -c "import torch; print(f'PyTorch {torch.__version__}  CUDA {torch.version.cuda}  GPU: {torch.cuda.get_device_name(0)}')"
echo ""

# ── 2. PyTorch version gate ───────────────────────────────────────────────────
echo "[2/6] Checking PyTorch version (need >= 2.1)..."
python3 - <<'PYCHECK'
import sys, torch
v = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
if v < (2, 1):
    print(f"ERROR: PyTorch {torch.__version__} is too old. Need >= 2.1 for stable torch.compile.")
    print("       Use a newer RunPod template or custom Docker image.")
    sys.exit(1)
else:
    print(f"OK: PyTorch {torch.__version__}")
PYCHECK
echo ""

# ── 3. Directory structure ─────────────────────────────────────────────────────
echo "[3/6] Creating directories..."
mkdir -p "$DATA_DIR" "$CHECKPOINT_DIR" "$TRITON_CACHE_DIR" "$LOG_DIR"
mkdir -p "$REPO_DIR"
echo "  Data:        $DATA_DIR"
echo "  Checkpoints: $CHECKPOINT_DIR"
echo "  Triton cache: $TRITON_CACHE_DIR  (persistent — survives pod restarts)"
echo "  Logs:        $LOG_DIR"
echo ""

# ── 4. Python venv ───────────────────────────────────────────────────────────
echo "[4/6] Setting up Python venv..."
if [[ ! -f "$VENV_PATH/bin/activate" ]]; then
  python3 -m venv "$VENV_PATH" --system-site-packages
  echo "  Created venv at $VENV_PATH (inherits system PyTorch)"
else
  echo "  Venv already exists at $VENV_PATH"
fi
source "$VENV_PATH/bin/activate"

# Install extra dependencies (no PyTorch — comes from system/Docker)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQUIREMENTS="$SCRIPT_DIR/../requirements_runpod.txt"
if [[ -f "$REQUIREMENTS" ]]; then
  pip install -q --no-cache-dir -r "$REQUIREMENTS"
  echo "  Installed requirements from $REQUIREMENTS"
else
  # Fallback: install known essentials
  pip install -q --no-cache-dir h5py numpy scipy tqdm
  echo "  Installed fallback requirements (h5py, numpy, scipy, tqdm)"
fi
echo ""

# ── 5. Data transfer instructions ─────────────────────────────────────────────
echo "[5/6] Data files..."
WP2_H5="$DATA_DIR/cfwrinkle_dataset.h5"
WP3_H5="$DATA_DIR/cfwrinkle_wp3_features.h5"

if [[ -f "$WP2_H5" && -f "$WP3_H5" ]]; then
  WP2_SIZE=$(du -sh "$WP2_H5" | cut -f1)
  WP3_SIZE=$(du -sh "$WP3_H5" | cut -f1)
  echo "  WP2 HDF5 found: $WP2_H5 ($WP2_SIZE)"
  echo "  WP3 HDF5 found: $WP3_H5 ($WP3_SIZE)"
else
  echo "  DATA FILES NOT FOUND — transfer them from WSL before running training."
  echo ""
  echo "  From WSL, run:"
  echo "    POD_IP=<your-runpod-ip>"
  echo "    POD_PORT=<your-runpod-ssh-port>"
  echo "    rsync -avz --progress \\"
  echo "      /home/ellis/cfwrinkle/data/cfwrinkle_dataset.h5 \\"
  echo "      /home/ellis/cfwrinkle/data/cfwrinkle_wp3_features.h5 \\"
  echo "      root@\${POD_IP}:\${POD_PORT}:$DATA_DIR/"
  echo ""
  echo "  (RunPod SSH details are in the pod dashboard under 'Connect')"
fi
echo ""

# ── 6. Quick smoke test ───────────────────────────────────────────────────────
echo "[6/6] CUDA smoke test..."
python3 - <<'SMOKE'
import sys
import torch
import time

assert torch.cuda.is_available(), "CUDA not available"
dev = torch.device("cuda")
props = torch.cuda.get_device_properties(dev)
sm_major, sm_minor = torch.cuda.get_device_capability(dev)

print(f"  GPU:         {props.name}")
print(f"  VRAM:        {props.total_memory / 1024**3:.1f} GB")
print(f"  SM count:    {props.multi_processor_count}")
print(f"  Capability:  sm_{sm_major}{sm_minor}", end="")
if sm_major >= 10:
    print("  (Blackwell — FA3 eligible)")
elif sm_major >= 9:
    print("  (Hopper — FA2)")
elif sm_major >= 8:
    print("  (Ampere/Ada — FA2)")
else:
    print("  (WARNING: pre-Ampere)")

cuda_ver = torch.version.cuda or "unknown"
print(f"  CUDA:        {cuda_ver}", end="")
cuda_parts = [int(x) for x in cuda_ver.split(".")]
if cuda_parts >= [12, 8]:
    print("  (FA3 fully supported)")
elif cuda_parts >= [12, 4]:
    print("  (FA3 not available — upgrade CUDA for Blackwell FA3)")
else:
    print("  (WARNING: old CUDA — recommend 12.8+)")

print(f"  bfloat16:    {torch.cuda.is_bf16_supported()}")
print(f"  PyTorch:     {torch.__version__}")

# BF16 matmul timing
a = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(10):
    _ = a @ a.T
torch.cuda.synchronize()
elapsed = (time.time() - t0) / 10 * 1000
print(f"  BF16 matmul 4096×4096: {elapsed:.1f} ms/call")

# Flash attention check
try:
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
        q = torch.randn(1, 8, 512, 64, device=dev, dtype=torch.bfloat16)
        out = torch.nn.functional.scaled_dot_product_attention(q, q, q)
    print("  Flash attention:       available")
except Exception as e:
    print(f"  Flash attention:       {e}")

print("  Smoke test PASSED")
SMOKE

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Transfer data files if not already present (see [5] above)"
echo "  2. Clone/sync repo to $REPO_DIR"
echo "     e.g.: rsync -avz --progress /mnt/c/Users/ellis/Documents/VS\ Code/CFWrinklePINN/ $REPO_DIR/"
echo "  3. Start training:"
echo "     cd $REPO_DIR && bash scripts/resume_cross_scale_level4_cuda.sh"
echo "     tmux attach -t level4_trackc_cuda"

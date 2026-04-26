#!/usr/bin/env bash
set -euo pipefail

if ! grep -qi microsoft /proc/version 2>/dev/null; then
  echo "ERROR: This script is intended for WSL2."
  exit 1
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${VENV_PATH:-/home/ellis/venvs/cfwrinkle}"
ROCM_RUNTIME_LIB="${ROCM_RUNTIME_LIB:-/opt/rocm-7.2.0/lib/libamdhip64.so}"

# WSL-native directories for large files (data + checkpoints).
# Code stays on the Windows filesystem; only big I/O goes through native ext4.
# Override with e.g. WSL_DATA_DIR=/data/cfwrinkle ./wsl_progressive_suite.sh
WSL_DATA_DIR="${WSL_DATA_DIR:-/home/ellis/cfwrinkle/data}"
WSL_CHECKPOINT_DIR="${WSL_CHECKPOINT_DIR:-/home/ellis/cfwrinkle/checkpoints}"

# Tunable resource gates (can be overridden via env vars).
MIN_CPU_CORES="${MIN_CPU_CORES:-8}"
MIN_MEM_TOTAL_GB="${MIN_MEM_TOTAL_GB:-12}"
MIN_MEM_AVAILABLE_GB="${MIN_MEM_AVAILABLE_GB:-6}"
MIN_SWAP_TOTAL_GB="${MIN_SWAP_TOTAL_GB:-8}"
MIN_DISK_FREE_GB="${MIN_DISK_FREE_GB:-40}"
MIN_GPU_FREE_GB="${MIN_GPU_FREE_GB:-6}"
FULL_CV_MIN_GPU_FREE_GB="${FULL_CV_MIN_GPU_FREE_GB:-10}"

MAX_LEVEL=3
PRECHECK_ONLY=0
INCLUDE_FULL_CV=0
DEVICE="${DEVICE:-cuda}"
SMOKE_EPOCHS="${SMOKE_EPOCHS:-1}"
OVERFIT_EPOCHS="${OVERFIT_EPOCHS:-200}"
MINI_EPOCHS="${MINI_EPOCHS:-50}"
FULL_CV_EPOCHS="${FULL_CV_EPOCHS:-100}"
SMOKE_HIDDEN_DIM="${SMOKE_HIDDEN_DIM:-16}"
TRAIN_HIDDEN_DIM="${TRAIN_HIDDEN_DIM:-64}"
ATTN_BATCH_NODES="${ATTN_BATCH_NODES:-512}"
MINI_GPU_RETRY_ATTN_BATCH_NODES="${MINI_GPU_RETRY_ATTN_BATCH_NODES:-128}"
# Temporal subsampling per level — reduces memory ~8x per halving of T
SMOKE_MAX_TIMESTEPS="${SMOKE_MAX_TIMESTEPS:-16}"
OVERFIT_MAX_TIMESTEPS="${OVERFIT_MAX_TIMESTEPS:-32}"
MINI_MAX_TIMESTEPS="${MINI_MAX_TIMESTEPS:-256}"
FULL_CV_MAX_TIMESTEPS="${FULL_CV_MAX_TIMESTEPS:-256}"
ALLOW_CPU_FALLBACK="${ALLOW_CPU_FALLBACK:-0}"
MINI_FOLD="${MINI_FOLD:-auto}"
SMOKE_SIM_ID="${SMOKE_SIM_ID:-}"
OVERFIT_SIM_IDS="${OVERFIT_SIM_IDS:-}"

usage() {
  cat <<'EOF'
Usage: ./wsl_progressive_suite.sh [options]

Options:
  --preflight-only      Run resource and environment checks only.
  --max-level N         Run through level N (0-4). Default: 3.
  --include-full-cv     Include level 4 full CV (equivalent to --max-level 4).
  --device DEVICE       auto|cpu|cuda (default: cuda).
  -h, --help            Show this help text.

Levels:
  0 = dataset and unit/smoke test checks
  1 = 1-sim smoke train
  2 = 3-sim overfit train
  3 = mini-train (fold 0, 13/2 split)
  4 = full 5-fold CV
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --preflight-only)
      PRECHECK_ONLY=1
      shift
      ;;
    --max-level)
      MAX_LEVEL="$2"
      shift 2
      ;;
    --include-full-cv)
      INCLUDE_FULL_CV=1
      MAX_LEVEL=4
      shift
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ ! -f "$VENV_PATH/bin/activate" ]]; then
  echo "ERROR: Venv not found at $VENV_PATH"
  echo "Run ./wsl_setup_env.sh first."
  exit 1
fi

if [[ ! "$MAX_LEVEL" =~ ^[0-4]$ ]]; then
  echo "ERROR: --max-level must be an integer from 0 to 4."
  exit 1
fi

source "$VENV_PATH/bin/activate"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR"

# Validate and export WSL-native data paths
if [[ ! -d "$WSL_DATA_DIR" ]]; then
  echo "ERROR: WSL data directory not found: $WSL_DATA_DIR"
  echo "Create it and copy your HDF5 files there:"
  echo "  mkdir -p $WSL_DATA_DIR"
  echo "  cp /mnt/c/Users/ellis/Documents/VS\\ Code/CFWrinklePINN/data/cfwrinkle_dataset.h5 $WSL_DATA_DIR/"
  echo "  cp /mnt/c/Users/ellis/Documents/VS\\ Code/CFWrinklePINN/data/cfwrinkle_wp3_features.h5 $WSL_DATA_DIR/"
  echo "Or override with: WSL_DATA_DIR=/your/path ./wsl_progressive_suite.sh"
  exit 1
fi
export WP3_H5="$WSL_DATA_DIR/cfwrinkle_wp3_features.h5"
export WP2_H5="$WSL_DATA_DIR/cfwrinkle_dataset.h5"
if [[ ! -f "$WP3_H5" ]]; then
  echo "ERROR: WP3 features file not found: $WP3_H5"
  exit 1
fi
if [[ ! -f "$WP2_H5" ]]; then
  echo "ERROR: WP2 dataset file not found: $WP2_H5"
  exit 1
fi
mkdir -p "$WSL_CHECKPOINT_DIR"
echo "Data dir:        $WSL_DATA_DIR"
echo "Checkpoint dir:  $WSL_CHECKPOINT_DIR"

if [[ -f "$ROCM_RUNTIME_LIB" ]]; then
  export LD_LIBRARY_PATH="$(dirname "$ROCM_RUNTIME_LIB"):${LD_LIBRARY_PATH:-}"
  export LD_PRELOAD="$ROCM_RUNTIME_LIB${LD_PRELOAD:+:$LD_PRELOAD}"
fi

to_gib_from_kib() {
  awk -v kib="$1" 'BEGIN { printf "%.2f", kib / 1024 / 1024 }'
}

to_gib_from_bytes() {
  awk -v bytes="$1" 'BEGIN { printf "%.2f", bytes / 1024 / 1024 / 1024 }'
}

require_minimum() {
  local label="$1"
  local value="$2"
  local min="$3"
  awk -v value="$value" -v min="$min" 'BEGIN { exit !(value + 0 >= min + 0) }'
  local ok=$?
  if [[ $ok -ne 0 ]]; then
    echo "FAIL: ${label}=${value} is below required minimum ${min}"
    return 1
  fi
  echo "OK:   ${label}=${value} (min ${min})"
}

run_step() {
  local name="$1"
  shift
  echo
  echo "=== ${name} ==="
  "$@"
}

print_wslconfig_hint() {
  echo
  echo "If limits are too low, set Windows-side WSL resources in C:\\Users\\ellis\\.wslconfig and restart WSL:"
  cat <<'EOF'
[wsl2]
memory=24GB
processors=12
swap=16GB
EOF
  echo "Apply with: wsl --shutdown"
}

echo "=== WSL resource preflight ==="
if [[ -f "/mnt/c/Users/ellis/.wslconfig" ]]; then
  echo "Detected C:\\Users\\ellis\\.wslconfig:"
  grep -E '^\s*(memory|processors|swap)\s*=' /mnt/c/Users/ellis/.wslconfig || true
fi

MEM_TOTAL_KIB="$(awk '/MemTotal:/ {print $2}' /proc/meminfo)"
MEM_AVAIL_KIB="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
SWAP_TOTAL_KIB="$(awk '/SwapTotal:/ {print $2}' /proc/meminfo)"
CPU_CORES="$(nproc)"
DISK_FREE_KIB="$(df -Pk "$REPO_DIR" | awk 'NR==2 {print $4}')"

MEM_TOTAL_GB="$(to_gib_from_kib "$MEM_TOTAL_KIB")"
MEM_AVAIL_GB="$(to_gib_from_kib "$MEM_AVAIL_KIB")"
SWAP_TOTAL_GB="$(to_gib_from_kib "$SWAP_TOTAL_KIB")"
DISK_FREE_GB="$(to_gib_from_kib "$DISK_FREE_KIB")"

resource_fail=0
require_minimum "cpu_cores" "$CPU_CORES" "$MIN_CPU_CORES" || resource_fail=1
require_minimum "mem_total_gb" "$MEM_TOTAL_GB" "$MIN_MEM_TOTAL_GB" || resource_fail=1
require_minimum "mem_available_gb" "$MEM_AVAIL_GB" "$MIN_MEM_AVAILABLE_GB" || resource_fail=1
require_minimum "swap_total_gb" "$SWAP_TOTAL_GB" "$MIN_SWAP_TOTAL_GB" || resource_fail=1
require_minimum "disk_free_gb" "$DISK_FREE_GB" "$MIN_DISK_FREE_GB" || resource_fail=1

GPU_FREE_GB=""
if [[ "$DEVICE" != "cpu" ]]; then
  GPU_PROBE_OUT="$(python - <<'PY'
import json
import torch
out = {"cuda": bool(torch.cuda.is_available()), "count": int(torch.cuda.device_count())}
if out["cuda"] and out["count"] > 0:
    free_b, total_b = torch.cuda.mem_get_info(0)
    out["free_bytes"] = int(free_b)
    out["total_bytes"] = int(total_b)
    out["name"] = torch.cuda.get_device_name(0)
print(json.dumps(out))
PY
)"
  CUDA_OK="$(python - <<'PY' "$GPU_PROBE_OUT"
import json, sys
print("1" if json.loads(sys.argv[1]).get("cuda") else "0")
PY
)"
  if [[ "$CUDA_OK" != "1" ]]; then
    echo "FAIL: torch.cuda.is_available() is false; cannot run GPU progressive tests."
    resource_fail=1
  else
    GPU_FREE_BYTES="$(python - <<'PY' "$GPU_PROBE_OUT"
import json, sys
print(json.loads(sys.argv[1]).get("free_bytes", 0))
PY
)"
    GPU_NAME="$(python - <<'PY' "$GPU_PROBE_OUT"
import json, sys
print(json.loads(sys.argv[1]).get("name", "unknown"))
PY
)"
    GPU_FREE_GB="$(to_gib_from_bytes "$GPU_FREE_BYTES")"
    echo "GPU: ${GPU_NAME}"
    require_minimum "gpu_free_gb" "$GPU_FREE_GB" "$MIN_GPU_FREE_GB" || resource_fail=1
    if [[ "$MAX_LEVEL" -ge 4 ]]; then
      require_minimum "gpu_free_gb(full_cv)" "$GPU_FREE_GB" "$FULL_CV_MIN_GPU_FREE_GB" || resource_fail=1
    fi
  fi
fi

if [[ "$resource_fail" -ne 0 ]]; then
  echo
  echo "Preflight failed. Adjust system resources before starting progressive training."
  print_wslconfig_hint
  exit 2
fi

echo "Preflight passed."
if [[ "$PRECHECK_ONLY" -eq 1 ]]; then
  exit 0
fi

PREFERRED_IDS="geom_0_2_pair1,geom_0_2,geom_0_0_pair1,geom_0_0_pair2"

SIM_SELECTION_JSON="$(python - <<'PY' "$SMOKE_SIM_ID" "$OVERFIT_SIM_IDS" "$PREFERRED_IDS"
import json
import sys
import h5py

smoke_override = sys.argv[1].strip()
overfit_override = sys.argv[2].strip()
preferred = [x.strip() for x in sys.argv[3].split(",") if x.strip()]

import os
with h5py.File(os.environ.get("WP3_H5", "data/cfwrinkle_wp3_features.h5"), "r") as f:
    sims = sorted(f["simulations"].keys())

if smoke_override:
    smoke_id = smoke_override
    if smoke_id not in sims:
        raise SystemExit(f"SMOKE_SIM_ID not found in dataset: {smoke_id}")
else:
    smoke_id = next((s for s in preferred if s in sims), sims[0])

if overfit_override:
    overfit_ids = [s.strip() for s in overfit_override.split(",") if s.strip()]
    missing = [s for s in overfit_ids if s not in sims]
    if missing:
        raise SystemExit(f"OVERFIT_SIM_IDS contain missing simulations: {missing}")
    if len(overfit_ids) < 3:
        raise SystemExit("OVERFIT_SIM_IDS must include at least 3 simulations.")
else:
    overfit_ids = sims[:3]

print(json.dumps({"smoke_id": smoke_id, "overfit_ids": overfit_ids, "sim_count": len(sims)}))
PY
)"

SMOKE_ID="$(python - <<'PY' "$SIM_SELECTION_JSON"
import json, sys
print(json.loads(sys.argv[1])["smoke_id"])
PY
)"
OVERFIT_IDS="$(python - <<'PY' "$SIM_SELECTION_JSON"
import json, sys
print(",".join(json.loads(sys.argv[1])["overfit_ids"]))
PY
)"

MINI_FOLD_SELECTED="$(python - <<'PY' "$MINI_FOLD"
import sys
import h5py
from model.dataset import load_fold_sim_ids

import os
mini_fold_arg = sys.argv[1].strip().lower()
wp3 = os.environ.get("WP3_H5", "data/cfwrinkle_wp3_features.h5")
wp2 = os.environ.get("WP2_H5", "data/cfwrinkle_dataset.h5")

with h5py.File(wp3, "r") as f:
    available = set(f["simulations"].keys())

valid_folds = []
for fold in range(5):
    train_ids, val_ids = load_fold_sim_ids(wp3, fold, wp2_h5_path=wp2)
    missing = [sid for sid in (train_ids + val_ids) if sid not in available]
    if not missing:
        valid_folds.append(fold)

if mini_fold_arg == "auto":
    if not valid_folds:
        raise SystemExit("No valid fold found where all split IDs exist in WP3 features.")
    print(valid_folds[0])
else:
    try:
        selected = int(mini_fold_arg)
    except ValueError as exc:
        raise SystemExit(f"MINI_FOLD must be 0-4 or 'auto', got: {mini_fold_arg}") from exc
    if selected < 0 or selected > 4:
        raise SystemExit(f"MINI_FOLD out of range: {selected}")
    train_ids, val_ids = load_fold_sim_ids(wp3, selected, wp2_h5_path=wp2)
    missing = [sid for sid in (train_ids + val_ids) if sid not in available]
    if missing:
        raise SystemExit(
            f"MINI_FOLD={selected} has IDs missing in WP3 features: {missing[:5]}"
            + (" ..." if len(missing) > 5 else "")
        )
    print(selected)
PY
)"

echo
echo "Selected smoke sim: $SMOKE_ID"
echo "Selected overfit sims: $OVERFIT_IDS"
echo "Selected mini-train fold: $MINI_FOLD_SELECTED"

if [[ "$MAX_LEVEL" -ge 0 ]]; then
  run_step "Level 0: dataset + test gate" python - <<'PY'
import h5py
from pathlib import Path

import os
path = Path(os.environ.get("WP3_H5", "data/cfwrinkle_wp3_features.h5"))
if not path.exists():
    raise SystemExit(f"Missing dataset: {path}")
with h5py.File(path, "r") as f:
    sims = sorted(f["simulations"].keys())
    if len(sims) != 65:
        raise SystemExit(f"Expected 65 simulations, found {len(sims)}")
    sample = f["simulations"][sims[0]]
    if "coarse_fields_resampled" in sample:
        x = sample["coarse_fields_resampled"].shape
    else:
        x = sample["coarse"]["resampled"]["fields"].shape
    y = sample["targets"]["wrinkle_severity"].shape
    print(f"sim_count={len(sims)}")
    print(f"sample_features_shape={x}")
    print(f"sample_wrinkle_target_shape={y}")
PY
  run_step "Level 0: unit/smoke tests" python -m pytest tests/test_model_unit.py tests/test_training_smoke.py -q
fi

if [[ "$MAX_LEVEL" -ge 1 ]]; then
  run_step "Level 1: smoke train" \
    env PYTHONUNBUFFERED=1 \
    python -u -m training.train \
      --sim-ids "$SMOKE_ID" \
      --epochs "$SMOKE_EPOCHS" \
      --hidden-dim "$SMOKE_HIDDEN_DIM" \
      --attn-batch-nodes "$ATTN_BATCH_NODES" \
      --max-timesteps "$SMOKE_MAX_TIMESTEPS" \
      --amp \
      --device "$DEVICE" \
      --output "$WSL_CHECKPOINT_DIR/progressive/level1_smoke"
  run_step "Level 1: gate check" \
    python -m training.gate_check \
      --level 1 \
      --run-dir "$WSL_CHECKPOINT_DIR/progressive/level1_smoke" \
      --report-path "$REPO_DIR/reports/wp7_gate_level1.json"
fi

if [[ "$MAX_LEVEL" -ge 2 ]]; then
  run_step "Level 2: overfit train" \
    env PYTHONUNBUFFERED=1 \
    python -u -m training.train \
      --sim-ids "$OVERFIT_IDS" \
      --epochs "$OVERFIT_EPOCHS" \
      --hidden-dim "$TRAIN_HIDDEN_DIM" \
      --attn-batch-nodes "$ATTN_BATCH_NODES" \
      --max-timesteps "$OVERFIT_MAX_TIMESTEPS" \
      --amp \
      --device "$DEVICE" \
      --output "$WSL_CHECKPOINT_DIR/progressive/level2_overfit"
  run_step "Level 2: gate check" \
    python -m training.gate_check \
      --level 2 \
      --run-dir "$WSL_CHECKPOINT_DIR/progressive/level2_overfit" \
      --report-path "$REPO_DIR/reports/wp7_gate_level2.json"
fi

if [[ "$MAX_LEVEL" -ge 3 ]]; then
  echo
  echo "=== Level 3: mini-train ==="
  MINI_LOG="$WSL_CHECKPOINT_DIR/progressive/level3_mini.run.log"
  mkdir -p "$(dirname "$MINI_LOG")"
  set +e
  PYTHONUNBUFFERED=1 \
  PYTORCH_HIP_ALLOC_CONF="garbage_collection_threshold:0.8,max_split_size_mb:512" \
  python -u -m training.train \
    --fold "$MINI_FOLD_SELECTED" \
    --max-train-sims 13 \
    --max-val-sims 2 \
    --epochs "$MINI_EPOCHS" \
    --hidden-dim "$TRAIN_HIDDEN_DIM" \
    --attn-batch-nodes "$ATTN_BATCH_NODES" \
    --max-timesteps "$MINI_MAX_TIMESTEPS" \
    --amp \
    --device "$DEVICE" \
    --output "$WSL_CHECKPOINT_DIR/progressive/level3_mini" \
    2>&1 | tee "$MINI_LOG"
  mini_rc=${PIPESTATUS[0]}
  set -e
  if [[ $mini_rc -ne 0 ]]; then
    if [[ "$DEVICE" != "cpu" ]] && grep -Eq "OutOfMemoryError|CUDA out of memory|HSA exception: MemoryRegion::BlockAllocator::alloc failed" "$MINI_LOG"; then
      echo "Level 3 GPU mini-train hit OOM; retrying on GPU with tighter memory settings."
      MINI_RETRY_LOG="$WSL_CHECKPOINT_DIR/progressive/level3_mini.retry-gpu.log"
      set +e
      PYTHONUNBUFFERED=1 \
      PYTORCH_HIP_ALLOC_CONF="garbage_collection_threshold:0.6,max_split_size_mb:64" \
      python -u -m training.train \
        --fold "$MINI_FOLD_SELECTED" \
        --max-train-sims 13 \
        --max-val-sims 2 \
        --epochs "$MINI_EPOCHS" \
        --hidden-dim "$TRAIN_HIDDEN_DIM" \
        --attn-batch-nodes "$MINI_GPU_RETRY_ATTN_BATCH_NODES" \
        --max-timesteps "$MINI_MAX_TIMESTEPS" \
        --amp \
        --device "$DEVICE" \
        --resume \
        --output "$WSL_CHECKPOINT_DIR/progressive/level3_mini" \
        2>&1 | tee "$MINI_RETRY_LOG"
      mini_retry_rc=${PIPESTATUS[0]}
      set -e
      if [[ $mini_retry_rc -ne 0 ]]; then
        if [[ "$ALLOW_CPU_FALLBACK" == "1" ]]; then
          echo "GPU retry failed; falling back to CPU because ALLOW_CPU_FALLBACK=1."
          run_step "Level 3: mini-train (CPU fallback)" \
            python -m training.train \
              --fold "$MINI_FOLD_SELECTED" \
              --max-train-sims 13 \
              --max-val-sims 2 \
              --epochs "$MINI_EPOCHS" \
              --hidden-dim "$TRAIN_HIDDEN_DIM" \
              --attn-batch-nodes "$MINI_GPU_RETRY_ATTN_BATCH_NODES" \
              --max-timesteps "$MINI_MAX_TIMESTEPS" \
              --amp \
              --device cpu \
              --output "$WSL_CHECKPOINT_DIR/progressive/level3_mini_cpu"
        else
          echo "GPU retry failed and CPU fallback is disabled (ALLOW_CPU_FALLBACK=0)."
          echo "See $MINI_RETRY_LOG"
          exit $mini_retry_rc
        fi
      fi
    else
      echo "Level 3 mini-train failed. See $MINI_LOG"
      exit $mini_rc
    fi
  fi
  # Gate check: primary and GPU retry both write to level3_mini (--resume).
  # CPU fallback writes to level3_mini_cpu as a separate run.
  LEVEL3_DIR=""
  for candidate in \
      "$WSL_CHECKPOINT_DIR/progressive/level3_mini" \
      "$WSL_CHECKPOINT_DIR/progressive/level3_mini_cpu"; do
    if [[ -f "$candidate/history.json" ]]; then
      LEVEL3_DIR="$candidate"
      break
    fi
  done
  if [[ -n "$LEVEL3_DIR" ]]; then
    run_step "Level 3: gate check" \
      python -m training.gate_check \
        --level 3 \
        --run-dir "$LEVEL3_DIR" \
        --report-path "$REPO_DIR/reports/wp7_gate_level3.json"
  else
    echo "WARNING: No Level 3 history.json found; skipping gate check."
  fi
fi

if [[ "$MAX_LEVEL" -ge 4 || "$INCLUDE_FULL_CV" -eq 1 ]]; then
  run_step "Level 4: full CV" \
    env PYTHONUNBUFFERED=1 PYTORCH_HIP_ALLOC_CONF="garbage_collection_threshold:0.8,max_split_size_mb:512" \
    python -u -m training.train \
      --all-folds \
      --epochs "$FULL_CV_EPOCHS" \
      --hidden-dim "$TRAIN_HIDDEN_DIM" \
      --attn-batch-nodes "$ATTN_BATCH_NODES" \
      --max-timesteps "$FULL_CV_MAX_TIMESTEPS" \
      --amp \
      --device "$DEVICE" \
      --output "$WSL_CHECKPOINT_DIR/progressive/level4_full_cv"
  run_step "Level 4: gate check" \
    python -m training.gate_check \
      --level 4 \
      --run-dir "$WSL_CHECKPOINT_DIR/progressive/level4_full_cv" \
      --report-path "$REPO_DIR/reports/wp7_gate_level4.json"
fi

echo
echo "=== Gate Reports ==="
ls -la "$REPO_DIR/reports/wp7_gate_level"*.json 2>/dev/null || echo "(no gate reports yet)"
echo
echo "Progressive suite complete through level $MAX_LEVEL."

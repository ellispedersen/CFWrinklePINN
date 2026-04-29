#!/usr/bin/env bash
# Track C — CUDA variant: RTX Pro 6000 Blackwell, 96 GB GDDR7, 188 GB DDR5
# Target: RTX Pro 6000 Blackwell (sm_122, CC 12.2) on Verda. DO NOT use on ROCm/AMD.
# Handles 1 or 2 GPUs automatically: single GPU runs all 5 folds serially;
# dual GPU splits folds 0,2,4 → GPU 0 and folds 1,3 → GPU 1 in parallel (~1.67× speedup).
#
# Tuned for 96 GB GDDR7 vs 7900 XT 7900 XT 20 GB:
#   hidden_dim    64  → 96    (no HSA pool limit; ~42 GB VRAM without grad checkpoint)
#   MAX_TIMESTEPS 96  → 128   (188 GB DDR5 handles preload; no WSL2 ceiling)
#   AMP_DTYPE     float16 → bfloat16  (Blackwell native; TF32 cores used by matmul)
#   gradient checkpoint   ON → OFF    (96 GB holds T=128 activations at hidden_dim=96)
#   preload       OFF → ON            (188 GB DDR5 handles Batch B 30k × 40 × T=128 ≈ 41 GB)
#   ATTN_BATCH_NODES 512 → 1024       (Blackwell 184 SMs fill more efficiently at larger batch)
#   DECODER_CHUNK_T  6 → 24           (96 GB headroom; larger chunk = fewer backward iterations)
#   FINE_LOSS_CHUNK_ELEMS 512 → 2048  (restored; no VRAM pressure)
#   torch.compile mode → max-autotune (stable on CUDA; tunes all kernel shapes on first run)
#   TRITON cache on network volume     (compile once per pod session, survives restarts)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${VENV_PATH:-/workspace/venv}"
DATA_DIR="${DATA_DIR:-/workspace/data}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/workspace/checkpoints}"
WP2_H5="${WP2_H5:-$DATA_DIR/cfwrinkle_dataset.h5}"
WP3_H5="${WP3_H5:-$DATA_DIR/cfwrinkle_wp3_features.h5}"
RUN_DIR="${RUN_DIR:-$CHECKPOINT_DIR/progressive/cross_scale_level4_cv_trackc_cuda}"
REPORT_PATH="${REPORT_PATH:-$REPO_DIR/reports/wp7_gate_level4_cross_scale_trackc_cuda.json}"

# ── Training hyperparameters ──────────────────────────────────────────────────
EPOCHS="${EPOCHS:-50}"
HIDDEN_DIM="${HIDDEN_DIM:-96}"
TEMPORAL_STRATEGY="${TEMPORAL_STRATEGY:-tail}"
DEVICE="${DEVICE:-cuda}"
AUTO_RESUME="${AUTO_RESUME:-1}"

# ── Blackwell hardware config (sm_122, 96 GB GDDR7 per device, 188 GB DDR5) ──
MAX_TIMESTEPS="${MAX_TIMESTEPS:-128}"
ATTN_BATCH_NODES="${ATTN_BATCH_NODES:-1024}"    # Blackwell 184 SMs; 512 was RDNA3-specific
DECODER_CHUNK_T="${DECODER_CHUNK_T:-24}"         # 96 GB headroom allows longer GRU chunks
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"              # Blackwell native; TF32 path for matmul
FINE_FEATURE_NORMALIZE="${FINE_FEATURE_NORMALIZE:-1}"

# ── Track C: fine_mp + dz weight + compile ────────────────────────────────────
USE_FINE_MP="${USE_FINE_MP:-1}"
USE_TORCH_COMPILE="${USE_TORCH_COMPILE:-1}"
TORCH_COMPILE_MODE="${TORCH_COMPILE_MODE:-max-autotune}"
CFWRINKLE_FINE_DZ_WEIGHT="${CFWRINKLE_FINE_DZ_WEIGHT:-4.0}"

# ── Memory / physics hooks ────────────────────────────────────────────────────
CFWRINKLE_PHYSICS_WARMUP_EPOCHS="${CFWRINKLE_PHYSICS_WARMUP_EPOCHS:-5}"
CFWRINKLE_PHYSICS_RAMP_EPOCHS="${CFWRINKLE_PHYSICS_RAMP_EPOCHS:-15}"
CFWRINKLE_FINE_ELEM_REGIONS="${CFWRINKLE_FINE_ELEM_REGIONS:-4}"
CFWRINKLE_FINE_LOSS_CHUNK_ELEMS="${CFWRINKLE_FINE_LOSS_CHUNK_ELEMS:-2048}"
CFWRINKLE_AUX_LOSS_INTERVAL="${CFWRINKLE_AUX_LOSS_INTERVAL:-1}"
CFWRINKLE_DISABLE_FINE_COHERENCE="${CFWRINKLE_DISABLE_FINE_COHERENCE:-0}"
CFWRINKLE_DISABLE_FINE_COUPLING="${CFWRINKLE_DISABLE_FINE_COUPLING:-0}"
CFWRINKLE_DISABLE_FINE_BUCKLING="${CFWRINKLE_DISABLE_FINE_BUCKLING:-0}"
CFWRINKLE_DISABLE_FINE_DZ_MONO="${CFWRINKLE_DISABLE_FINE_DZ_MONO:-0}"
RESUME=0

usage() {
  cat <<EOF
Usage: ./run_cross_scale_level4_cuda.sh [--resume] [--help]

Track C CUDA: hidden_dim=96, T=128, bfloat16, no gradient checkpointing.
Target: RTX Pro 6000 Blackwell on RunPod (96 GB GDDR7). DO NOT use on ROCm/AMD.

Key env overrides:
  DATA_DIR, CHECKPOINT_DIR, RUN_DIR, REPORT_PATH
  VENV_PATH              (default: /workspace/venv)
  HIDDEN_DIM             (default: 96)
  MAX_TIMESTEPS          (default: 128)
  ATTN_BATCH_NODES       (default: 1024; try 2048 if VRAM allows)
  DECODER_CHUNK_T        (default: 24)
  TORCH_COMPILE_MODE     (default: max-autotune; try reduce-overhead for short runs)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --resume) RESUME=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: Unknown option: $1"; usage; exit 1 ;;
  esac
done

[[ -f "$VENV_PATH/bin/activate" ]] || { echo "ERROR: venv not found at $VENV_PATH — run scripts/runpod_cuda_setup.sh first"; exit 1; }
[[ -f "$WP2_H5" ]] || { echo "ERROR: WP2_H5 not found: $WP2_H5"; exit 1; }
[[ -f "$WP3_H5" ]] || { echo "ERROR: WP3_H5 not found: $WP3_H5"; exit 1; }

mkdir -p "$RUN_DIR" "$(dirname "$REPORT_PATH")"
LOG_PATH="$RUN_DIR/run.log"

source "$VENV_PATH/bin/activate"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR"
export PYTHONUNBUFFERED=1

# CUDA allocator config: no max_split_size_mb — 96 GB has no fragmentation pressure at our scale.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.9}"

# Triton kernel cache on persistent volume — survives pod restarts, avoids re-compile.
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/workspace/.triton_cache}"
mkdir -p "$TRITON_CACHE_DIR"

export TORCHINDUCTOR_MAX_AUTOTUNE="${TORCHINDUCTOR_MAX_AUTOTUNE:-1}"
export TORCH_COMPILE_MODE

export CFWRINKLE_FINE_LOSS_CHUNK_ELEMS CFWRINKLE_AUX_LOSS_INTERVAL
export CFWRINKLE_FINE_ELEM_REGIONS
export CFWRINKLE_DISABLE_FINE_COHERENCE CFWRINKLE_DISABLE_FINE_COUPLING
export CFWRINKLE_DISABLE_FINE_BUCKLING CFWRINKLE_DISABLE_FINE_DZ_MONO
export CFWRINKLE_PHYSICS_WARMUP_EPOCHS CFWRINKLE_PHYSICS_RAMP_EPOCHS
export CFWRINKLE_FINE_DZ_WEIGHT
export WP2_H5 WP3_H5

if [[ "$AUTO_RESUME" == "1" ]]; then
  # Scan all fold directories — not just fold_0.
  # Dual-GPU runs may complete fold_1/fold_3 (GPU 1) before fold_0 writes its
  # first checkpoint (GPU 0), so checking only fold_0 would miss existing work.
  for _fold_check in 0 1 2 3 4; do
    if [[ -f "$RUN_DIR/fold_${_fold_check}/latest.pt" ]]; then
      RESUME=1
      break
    fi
  done
fi

CMD=(
  python -u -m training.train
  --model-type cross-scale
  --epochs "$EPOCHS"
  --hidden-dim "$HIDDEN_DIM"
  --attn-batch-nodes "$ATTN_BATCH_NODES"
  --max-timesteps "$MAX_TIMESTEPS"
  --decoder-chunk-t "$DECODER_CHUNK_T"
  --temporal-strategy "$TEMPORAL_STRATEGY"
  --amp
  --amp-dtype "$AMP_DTYPE"
  --device "$DEVICE"
  --output "$RUN_DIR"
)
if [[ "$RESUME" == "1" ]]; then
  CMD+=(--resume --allow-resume-mismatch)
fi
if [[ "$FINE_FEATURE_NORMALIZE" == "1" ]]; then
  CMD+=(--normalize-fine-features)
fi
if [[ "$USE_FINE_MP" == "1" ]]; then
  CMD+=(--use-fine-mp)
fi
if [[ "$USE_TORCH_COMPILE" == "1" ]]; then
  CMD+=(--torch-compile)
fi
# Gradient checkpointing OFF: 96 GB GDDR7 holds all T=128 × hidden_dim=96 fine activations (~42-55 GB).
# The 7900 XT OOM'd at 18.78 GB (hidden_dim=64, T=96) without checkpointing. Scaling:
#   (96/64)^2 × (128/96) × 18.78 ≈ 56 GB — within 96 GB with comfortable margin.
CMD+=(--no-checkpoint)
# Preload ON: 188 GB DDR5 handles Batch B 30k-node × 40 sims × T=128 ≈ 41 GB RAM.

# ── Detect available GPUs ─────────────────────────────────────────────────────
N_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "1")
[[ "$N_GPUS" =~ ^[1-9][0-9]*$ ]] || N_GPUS=1  # guard against non-integer output

echo "=== Track C CUDA (cross-scale full CV, fine_mp=ON, compile=ON, no-checkpoint) ===" | tee "$LOG_PATH"
echo "Run dir:       $RUN_DIR" | tee -a "$LOG_PATH"
echo "Gate report:   $REPORT_PATH" | tee -a "$LOG_PATH"
echo "GPUs:          $N_GPUS" | tee -a "$LOG_PATH"
echo "Resume:        $RESUME" | tee -a "$LOG_PATH"
echo "hidden_dim:    $HIDDEN_DIM" | tee -a "$LOG_PATH"
echo "T:             $MAX_TIMESTEPS" | tee -a "$LOG_PATH"
echo "AMP dtype:     $AMP_DTYPE" | tee -a "$LOG_PATH"
echo "compile mode:  $TORCH_COMPILE_MODE" | tee -a "$LOG_PATH"
echo "attn batch:    $ATTN_BATCH_NODES nodes" | tee -a "$LOG_PATH"
echo "decoder chunk: $DECODER_CHUNK_T" | tee -a "$LOG_PATH"
echo "fine_dz wt:    $CFWRINKLE_FINE_DZ_WEIGHT" | tee -a "$LOG_PATH"
echo "loss chunk:    $CFWRINKLE_FINE_LOSS_CHUNK_ELEMS elems" | tee -a "$LOG_PATH"
echo "physics ramp:  warmup=$CFWRINKLE_PHYSICS_WARMUP_EPOCHS ramp=$CFWRINKLE_PHYSICS_RAMP_EPOCHS" | tee -a "$LOG_PATH"
echo "triton cache:  $TRITON_CACHE_DIR" | tee -a "$LOG_PATH"

if [[ "$N_GPUS" -ge 2 ]]; then
  # ── Dual-GPU mode ───────────────────────────────────────────────────────────
  # Each GPU runs its fold subset sequentially. The two subsets run in parallel.
  #   GPU 0: folds 0,2,4  (3 folds — bounds wall-clock; ~3 × epoch_time)
  #   GPU 1: folds 1,3    (2 folds — finishes first, then idles)
  # CUDA_VISIBLE_DEVICES restricts each subprocess to one physical GPU;
  # both see it as cuda:0 — no code changes required in train.py.
  # Triton compiles are per-GPU but share the same cache directory (read-safe on NVMe).
  echo "Dual-GPU mode: GPU 0 → folds 0,2,4 | GPU 1 → folds 1,3 (~1.67× speedup)" | tee -a "$LOG_PATH"
  LOG_GPU0="${RUN_DIR}/run_gpu0.log"
  LOG_GPU1="${RUN_DIR}/run_gpu1.log"

  set +e
  CUDA_VISIBLE_DEVICES=0 "${CMD[@]}" --folds 0,2,4 2>&1 | tee -a "$LOG_GPU0" &
  PID0=$!
  CUDA_VISIBLE_DEVICES=1 "${CMD[@]}" --folds 1,3   2>&1 | tee -a "$LOG_GPU1" &
  PID1=$!

  wait $PID0; rc0=$?
  wait $PID1; rc1=$?
  set -e

  if [[ $rc0 -ne 0 || $rc1 -ne 0 ]]; then
    echo "Training failed — GPU 0 exit=$rc0, GPU 1 exit=$rc1" | tee -a "$LOG_PATH"
    echo "  GPU 0 log: $LOG_GPU0"
    echo "  GPU 1 log: $LOG_GPU1"
    exit 1
  fi

  # Both GPU processes skip writing summary.json to avoid a write race.
  # Merge fold histories into a single top-level summary after both complete.
  python3 - <<PYMERGE
import json, pathlib
run_dir = pathlib.Path("$RUN_DIR")
results = []
for fold in range(5):
    history_path = run_dir / f"fold_{fold}" / "history.json"
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
        last = history[-1] if history else {}
        results.append({
            "fold": fold,
            "best_val_loss": last.get("best_val_loss"),
            "epochs_trained": len(history),
        })
with open(run_dir / "summary.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"summary.json merged: {len(results)} folds")
PYMERGE

else
  # ── Single-GPU mode ─────────────────────────────────────────────────────────
  echo "Single-GPU mode: folds 0-4 sequential on cuda:0" | tee -a "$LOG_PATH"
  set +e
  CUDA_VISIBLE_DEVICES=0 "${CMD[@]}" --all-folds 2>&1 | tee -a "$LOG_PATH"
  train_rc=${PIPESTATUS[0]}
  set -e
  if [[ $train_rc -ne 0 ]]; then
    echo "Training failed (exit $train_rc). See $LOG_PATH"
    exit $train_rc
  fi
fi

python -m training.gate_check \
  --level 4 \
  --run-dir "$RUN_DIR" \
  --report-path "$REPORT_PATH" | tee -a "$LOG_PATH"

echo "Artifacts:"
echo "  $RUN_DIR/fold_*/history.json"
echo "  $RUN_DIR/fold_*/best.pt"
echo "  $REPORT_PATH"

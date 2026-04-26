#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${VENV_PATH:-/home/ellis/venvs/cfwrinkle}"
ROCM_RUNTIME_LIB="${ROCM_RUNTIME_LIB:-/opt/rocm-7.2.0/lib/libamdhip64.so}"
WSL_DATA_DIR="${WSL_DATA_DIR:-/home/ellis/cfwrinkle/data}"
WSL_CHECKPOINT_DIR="${WSL_CHECKPOINT_DIR:-/home/ellis/cfwrinkle/checkpoints}"
WP2_H5="${WP2_H5:-$WSL_DATA_DIR/cfwrinkle_dataset.h5}"
WP3_H5="${WP3_H5:-$WSL_DATA_DIR/cfwrinkle_wp3_features.h5}"
RUN_DIR="${RUN_DIR:-$WSL_CHECKPOINT_DIR/progressive/cross_scale_level4_cv}"
REPORT_PATH="${REPORT_PATH:-$REPO_DIR/reports/wp7_gate_level4_cross_scale.json}"
EPOCHS="${EPOCHS:-50}"
HIDDEN_DIM="${HIDDEN_DIM:-64}"
ATTN_BATCH_NODES="${ATTN_BATCH_NODES:-512}"
MAX_TIMESTEPS="${MAX_TIMESTEPS:-128}"
DECODER_CHUNK_T="${DECODER_CHUNK_T:-16}"
TEMPORAL_STRATEGY="${TEMPORAL_STRATEGY:-tail}"
DEVICE="${DEVICE:-cuda}"
AUTO_RESUME="${AUTO_RESUME:-1}"
FINE_FEATURE_NORMALIZE="${FINE_FEATURE_NORMALIZE:-0}"
AMP_DTYPE="${AMP_DTYPE:-${CFWRINKLE_AMP_DTYPE:-float16}}"
CFWRINKLE_PHYSICS_WARMUP_EPOCHS="${CFWRINKLE_PHYSICS_WARMUP_EPOCHS:-0}"
CFWRINKLE_PHYSICS_RAMP_EPOCHS="${CFWRINKLE_PHYSICS_RAMP_EPOCHS:-20}"
CFWRINKLE_FINE_LOSS_CHUNK_ELEMS="${CFWRINKLE_FINE_LOSS_CHUNK_ELEMS:-4096}"
CFWRINKLE_AUX_LOSS_INTERVAL="${CFWRINKLE_AUX_LOSS_INTERVAL:-1}"
CFWRINKLE_DISABLE_FINE_COHERENCE="${CFWRINKLE_DISABLE_FINE_COHERENCE:-0}"
CFWRINKLE_DISABLE_FINE_COUPLING="${CFWRINKLE_DISABLE_FINE_COUPLING:-0}"
CFWRINKLE_DISABLE_FINE_BUCKLING="${CFWRINKLE_DISABLE_FINE_BUCKLING:-0}"
CFWRINKLE_DISABLE_FINE_DZ_MONO="${CFWRINKLE_DISABLE_FINE_DZ_MONO:-0}"
PYTORCH_NO_CUDA_MEMORY_CACHING="${PYTORCH_NO_CUDA_MEMORY_CACHING:-0}"
RESUME=0

usage() {
  cat <<EOF
Usage: ./run_cross_scale_level4.sh [--resume] [--help]

Environment overrides:
  VENV_PATH, WSL_DATA_DIR, WSL_CHECKPOINT_DIR, WP2_H5, WP3_H5
  RUN_DIR, REPORT_PATH
  EPOCHS, HIDDEN_DIM, ATTN_BATCH_NODES, MAX_TIMESTEPS, DECODER_CHUNK_T, TEMPORAL_STRATEGY, DEVICE
  AUTO_RESUME=1|0, FINE_FEATURE_NORMALIZE=1|0
  AMP_DTYPE=float16|bfloat16
  CFWRINKLE_PHYSICS_WARMUP_EPOCHS, CFWRINKLE_PHYSICS_RAMP_EPOCHS
  CFWRINKLE_FINE_LOSS_CHUNK_ELEMS, CFWRINKLE_AUX_LOSS_INTERVAL
  CFWRINKLE_DISABLE_FINE_COHERENCE=1|0, CFWRINKLE_DISABLE_FINE_COUPLING=1|0
  CFWRINKLE_DISABLE_FINE_BUCKLING=1|0, CFWRINKLE_DISABLE_FINE_DZ_MONO=1|0
  PYTORCH_NO_CUDA_MEMORY_CACHING=1|0
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --resume) RESUME=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: Unknown option: $1"; usage; exit 1 ;;
  esac
done

[[ -f "$VENV_PATH/bin/activate" ]] || { echo "ERROR: venv not found at $VENV_PATH"; exit 1; }
[[ -f "$WP2_H5" ]] || { echo "ERROR: WP2_H5 not found: $WP2_H5"; exit 1; }
[[ -f "$WP3_H5" ]] || { echo "ERROR: WP3_H5 not found: $WP3_H5"; exit 1; }

mkdir -p "$RUN_DIR" "$(dirname "$REPORT_PATH")"
LOG_PATH="$RUN_DIR/run.log"

source "$VENV_PATH/bin/activate"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR"
export PYTHONUNBUFFERED=1
export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-garbage_collection_threshold:0.7,max_split_size_mb:512,roundup_power2_divisions:8}"
export CFWRINKLE_FINE_LOSS_CHUNK_ELEMS CFWRINKLE_AUX_LOSS_INTERVAL
export CFWRINKLE_DISABLE_FINE_COHERENCE CFWRINKLE_DISABLE_FINE_COUPLING
export CFWRINKLE_DISABLE_FINE_BUCKLING CFWRINKLE_DISABLE_FINE_DZ_MONO
export CFWRINKLE_PHYSICS_WARMUP_EPOCHS CFWRINKLE_PHYSICS_RAMP_EPOCHS
if [[ "$PYTORCH_NO_CUDA_MEMORY_CACHING" == "1" ]]; then
  export PYTORCH_NO_CUDA_MEMORY_CACHING=1
else
  unset PYTORCH_NO_CUDA_MEMORY_CACHING
fi
NO_ALLOC_CACHE_EFFECTIVE="${PYTORCH_NO_CUDA_MEMORY_CACHING:-0}"
export WP2_H5 WP3_H5

if [[ -f "$ROCM_RUNTIME_LIB" ]]; then
  export LD_LIBRARY_PATH="$(dirname "$ROCM_RUNTIME_LIB"):${LD_LIBRARY_PATH:-}"
  export LD_PRELOAD="$ROCM_RUNTIME_LIB${LD_PRELOAD:+:$LD_PRELOAD}"
fi

if [[ "$AUTO_RESUME" == "1" && -f "$RUN_DIR/fold_0/latest.pt" ]]; then
  RESUME=1
fi

CMD=(
  python -u -m training.train
  --model-type cross-scale
  --all-folds
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
  CMD+=(--resume)
fi
if [[ "$FINE_FEATURE_NORMALIZE" == "1" ]]; then
  CMD+=(--normalize-fine-features)
fi

echo "=== Track B Level 4 (cross-scale full CV) ===" | tee "$LOG_PATH"
echo "Run dir:    $RUN_DIR" | tee -a "$LOG_PATH"
echo "Gate report:$REPORT_PATH" | tee -a "$LOG_PATH"
echo "Resume:     $RESUME" | tee -a "$LOG_PATH"
echo "Fine norm:  $FINE_FEATURE_NORMALIZE" | tee -a "$LOG_PATH"
echo "Loss chunk: $CFWRINKLE_FINE_LOSS_CHUNK_ELEMS elems" | tee -a "$LOG_PATH"
echo "AMP dtype:  $AMP_DTYPE" | tee -a "$LOG_PATH"
echo "Physics ramp: warmup=$CFWRINKLE_PHYSICS_WARMUP_EPOCHS ramp=$CFWRINKLE_PHYSICS_RAMP_EPOCHS" | tee -a "$LOG_PATH"
echo "Aux interval:$CFWRINKLE_AUX_LOSS_INTERVAL" | tee -a "$LOG_PATH"
echo "Disable aux: coherence=$CFWRINKLE_DISABLE_FINE_COHERENCE coupling=$CFWRINKLE_DISABLE_FINE_COUPLING buckling=$CFWRINKLE_DISABLE_FINE_BUCKLING dz_mono=$CFWRINKLE_DISABLE_FINE_DZ_MONO" | tee -a "$LOG_PATH"
echo "No allocator cache: $NO_ALLOC_CACHE_EFFECTIVE" | tee -a "$LOG_PATH"
set +e
"${CMD[@]}" 2>&1 | tee -a "$LOG_PATH"
train_rc=${PIPESTATUS[0]}
set -e
if [[ $train_rc -ne 0 ]]; then
  echo "Training failed (exit $train_rc). See $LOG_PATH"
  exit $train_rc
fi

python -m training.gate_check \
  --level 4 \
  --run-dir "$RUN_DIR" \
  --report-path "$REPORT_PATH" | tee -a "$LOG_PATH"

echo "Artifacts:"
echo "  $RUN_DIR/summary.json"
echo "  $RUN_DIR/fold_*/history.json"
echo "  $RUN_DIR/fold_*/best.pt"
echo "  $RUN_DIR/fold_*/latest.pt"
echo "  $REPORT_PATH"

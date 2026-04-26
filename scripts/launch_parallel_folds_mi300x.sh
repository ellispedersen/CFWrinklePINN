#!/usr/bin/env bash
# Parallel fold training for MI300X — saturates 192 GB HBM3 with concurrent folds.
# Runs N folds simultaneously then batches remaining folds.
#
# VRAM budget per fold (hidden_dim=96, T=128, no-checkpoint):
#   ~42-55 GB  →  N=2: ~84-110 GB of 192 GB (always safe)
#                 N=3: ~126-165 GB of 192 GB (aggressive; watch first run)
#
# Fold batches:
#   N=2: [0,1] → [2,3] → [4]    3 rounds, est. ~10 hr, ~$20
#   N=3: [0,1,2] → [3,4]        2 rounds, est. ~8 hr,  ~$16
#
# Docker image required (RunPod ROCm 5.7 template is incompatible):
#   rocm/pytorch:rocm6.2.4_ubuntu22.04_py3.10_pytorch_release_2.3.0
#   or: rocm/pytorch:rocm6.3.1_ubuntu22.04_py3.10_pytorch_release_2.4.0
#
# Each fold gets an isolated Triton cache to prevent concurrent write races.
set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/repo}"
VENV_PATH="${VENV_PATH:-/workspace/venv}"
DATA_DIR="${DATA_DIR:-/workspace/data}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/workspace/checkpoints}"
LOG_DIR="${LOG_DIR:-/workspace/logs}"
N_PARALLEL="${N_PARALLEL:-2}"   # 2 = conservative (always safe); 3 = aggressive
AUTO_RESUME="${AUTO_RESUME:-1}"

RUN_DIR="${RUN_DIR:-$CHECKPOINT_DIR/progressive/cross_scale_level4_cv_trackc_mi300x}"
WP2_H5="${WP2_H5:-$DATA_DIR/cfwrinkle_dataset.h5}"
WP3_H5="${WP3_H5:-$DATA_DIR/cfwrinkle_wp3_features.h5}"
REPORT_PATH="${REPORT_PATH:-$REPO_DIR/reports/wp7_gate_level4_cross_scale_trackc_mi300x.json}"
ROCM_RUNTIME_LIB="${ROCM_RUNTIME_LIB:-/opt/rocm/lib/libamdhip64.so}"

# ── Track C MI300X hyperparameters ────────────────────────────────────────────
EPOCHS="${EPOCHS:-50}"
HIDDEN_DIM="${HIDDEN_DIM:-96}"
MAX_TIMESTEPS="${MAX_TIMESTEPS:-128}"
# Conservative batch sizes for parallel execution: avoids simultaneous peak spikes
# when both processes are in the GRU/attention backward at the same tick.
ATTN_BATCH_NODES="${ATTN_BATCH_NODES:-512}"
DECODER_CHUNK_T="${DECODER_CHUNK_T:-12}"
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"
FINE_FEATURE_NORMALIZE="${FINE_FEATURE_NORMALIZE:-1}"
USE_FINE_MP="${USE_FINE_MP:-1}"
USE_TORCH_COMPILE="${USE_TORCH_COMPILE:-1}"
TORCH_ROCM_FA_PREFER_CK="${TORCH_ROCM_FA_PREFER_CK:-1}"
INDUCTOR_GEMM_BACKENDS="${INDUCTOR_GEMM_BACKENDS:-ATEN,TRITON}"
CFWRINKLE_FINE_DZ_WEIGHT="${CFWRINKLE_FINE_DZ_WEIGHT:-4.0}"
CFWRINKLE_PHYSICS_WARMUP_EPOCHS="${CFWRINKLE_PHYSICS_WARMUP_EPOCHS:-5}"
CFWRINKLE_PHYSICS_RAMP_EPOCHS="${CFWRINKLE_PHYSICS_RAMP_EPOCHS:-15}"
CFWRINKLE_FINE_ELEM_REGIONS="${CFWRINKLE_FINE_ELEM_REGIONS:-4}"
CFWRINKLE_FINE_LOSS_CHUNK_ELEMS="${CFWRINKLE_FINE_LOSS_CHUNK_ELEMS:-2048}"
CFWRINKLE_AUX_LOSS_INTERVAL="${CFWRINKLE_AUX_LOSS_INTERVAL:-1}"
CFWRINKLE_DISABLE_FINE_COHERENCE="${CFWRINKLE_DISABLE_FINE_COHERENCE:-0}"
CFWRINKLE_DISABLE_FINE_COUPLING="${CFWRINKLE_DISABLE_FINE_COUPLING:-0}"
CFWRINKLE_DISABLE_FINE_BUCKLING="${CFWRINKLE_DISABLE_FINE_BUCKLING:-0}"
CFWRINKLE_DISABLE_FINE_DZ_MONO="${CFWRINKLE_DISABLE_FINE_DZ_MONO:-0}"

# ── Validation ────────────────────────────────────────────────────────────────
[[ -f "$VENV_PATH/bin/activate" ]] || { echo "ERROR: venv not found at $VENV_PATH"; exit 1; }
[[ -f "$WP2_H5" ]]                || { echo "ERROR: WP2_H5 not found: $WP2_H5"; exit 1; }
[[ -f "$WP3_H5" ]]                || { echo "ERROR: WP3_H5 not found: $WP3_H5"; exit 1; }
[[ "$N_PARALLEL" =~ ^[23]$ ]]     || { echo "ERROR: N_PARALLEL must be 2 or 3"; exit 1; }

mkdir -p "$RUN_DIR" "$LOG_DIR" "$(dirname "$REPORT_PATH")"

source "$VENV_PATH/bin/activate"
cd "$REPO_DIR"

if [[ -f "$ROCM_RUNTIME_LIB" ]]; then
  export LD_LIBRARY_PATH="$(dirname "$ROCM_RUNTIME_LIB"):${LD_LIBRARY_PATH:-}"
  export LD_PRELOAD="$ROCM_RUNTIME_LIB${LD_PRELOAD:+:$LD_PRELOAD}"
fi

# ── Common environment for all fold processes ─────────────────────────────────
export PYTHONPATH="$REPO_DIR"
export PYTHONUNBUFFERED=1
export WP2_H5 WP3_H5
export AMP_DTYPE HIDDEN_DIM MAX_TIMESTEPS ATTN_BATCH_NODES DECODER_CHUNK_T
export CFWRINKLE_FINE_DZ_WEIGHT CFWRINKLE_FINE_LOSS_CHUNK_ELEMS
export CFWRINKLE_AUX_LOSS_INTERVAL CFWRINKLE_FINE_ELEM_REGIONS
export CFWRINKLE_PHYSICS_WARMUP_EPOCHS CFWRINKLE_PHYSICS_RAMP_EPOCHS
export CFWRINKLE_DISABLE_FINE_COHERENCE CFWRINKLE_DISABLE_FINE_COUPLING
export CFWRINKLE_DISABLE_FINE_BUCKLING CFWRINKLE_DISABLE_FINE_DZ_MONO
export TORCH_ROCM_FA_PREFER_CK INDUCTOR_GEMM_BACKENDS
export TORCHINDUCTOR_MAX_AUTOTUNE=1
export HSA_DISABLE_FRAGMENT_ALLOCATOR=1   # safe: ~100-165 GB peak << 192 GB
export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.9}"

# ── launch_fold: starts one fold as a background process ─────────────────────
launch_fold() {
  local fold_id="$1"
  local fold_dir="$RUN_DIR/fold_${fold_id}"
  local fold_log="$LOG_DIR/fold_${fold_id}_mi300x.log"
  # Isolated Triton cache per fold: prevents concurrent write races on shared kernels.
  local triton_cache="/workspace/.triton_cache/fold_${fold_id}"

  mkdir -p "$fold_dir" "$triton_cache"

  local cmd=(
    python -u -m training.train
    --fold "$fold_id"
    --model-type cross-scale
    --epochs "$EPOCHS"
    --hidden-dim "$HIDDEN_DIM"
    --attn-batch-nodes "$ATTN_BATCH_NODES"
    --max-timesteps "$MAX_TIMESTEPS"
    --decoder-chunk-t "$DECODER_CHUNK_T"
    --temporal-strategy tail
    --amp --amp-dtype "$AMP_DTYPE"
    --device cuda
    --output "$fold_dir"
    --no-checkpoint
    --normalize-fine-features
    --use-fine-mp
  )
  [[ "$USE_TORCH_COMPILE" == "1" ]] && cmd+=(--torch-compile)
  if [[ "$AUTO_RESUME" == "1" && -f "$fold_dir/latest.pt" ]]; then
    cmd+=(--resume --allow-resume-mismatch)
    echo "[fold $fold_id] Resuming from $fold_dir/latest.pt"
  fi

  echo "[fold $fold_id] Starting → log: $fold_log"
  TRITON_CACHE_DIR="$triton_cache" "${cmd[@]}" 2>&1 | tee "$fold_log" &
  echo $!   # return PID to caller
}

# ── wait_pids: waits for all PIDs, exits 1 if any failed ─────────────────────
wait_pids() {
  local pids=("$@")
  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      echo "ERROR: process $pid failed"
      failed=1
    fi
  done
  [[ $failed -eq 0 ]] || { echo "One or more folds failed — stopping."; exit 1; }
}

# ── Fold batch schedule ───────────────────────────────────────────────────────
echo "=== MI300X parallel fold training: N_PARALLEL=$N_PARALLEL ==="
echo "Run dir:    $RUN_DIR"
echo "HBM3 est:   ~$((N_PARALLEL * 50)) GB peak of 192 GB"
echo "Rounds:     $(( (5 + N_PARALLEL - 1) / N_PARALLEL ))"
echo ""

if [[ "$N_PARALLEL" == "3" ]]; then
  BATCHES=("0 1 2" "3 4")
else
  BATCHES=("0 1" "2 3" "4")
fi

round=1
for batch in "${BATCHES[@]}"; do
  read -ra fold_ids <<< "$batch"
  echo "── Round $round: folds [${fold_ids[*]}] ────────────────────────────────"
  pids=()
  for fid in "${fold_ids[@]}"; do
    pid=$(launch_fold "$fid")
    pids+=("$pid")
  done
  echo "   PIDs: ${pids[*]} — waiting for round $round to complete..."
  wait_pids "${pids[@]}"
  echo "   Round $round complete."
  echo ""
  (( round++ ))
done

# ── Gate check across all 5 folds ────────────────────────────────────────────
echo "=== All folds complete — running gate check ==="
python -m training.gate_check \
  --level 4 \
  --run-dir "$RUN_DIR" \
  --report-path "$REPORT_PATH"

echo ""
echo "Artifacts:"
echo "  $RUN_DIR/fold_*/history.json"
echo "  $RUN_DIR/fold_*/best.pt"
echo "  $REPORT_PATH"

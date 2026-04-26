#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${VENV_PATH:-/home/ellis/venvs/cfwrinkle}"
ROCM_RUNTIME_LIB="${ROCM_RUNTIME_LIB:-/opt/rocm-7.2.0/lib/libamdhip64.so}"
WSL_DATA_DIR="${WSL_DATA_DIR:-/home/ellis/cfwrinkle/data}"
WSL_CHECKPOINT_DIR="${WSL_CHECKPOINT_DIR:-/home/ellis/cfwrinkle/checkpoints}"
WP2_H5="${WP2_H5:-$WSL_DATA_DIR/cfwrinkle_dataset.h5}"
WP3_H5="${WP3_H5:-$WSL_DATA_DIR/cfwrinkle_wp3_features.h5}"
RUN_DIR="${RUN_DIR:-$WSL_CHECKPOINT_DIR/progressive/cross_scale_level2}"
REPORT_PATH="${REPORT_PATH:-$REPO_DIR/reports/wp7_gate_level2_cross_scale.json}"
DEFAULT_SIM_IDS="geom_0_0_pair1,geom_0_0_pair2"
SIM_IDS="${SIM_IDS:-$DEFAULT_SIM_IDS}"
EPOCHS="${EPOCHS:-25}"
HIDDEN_DIM="${HIDDEN_DIM:-64}"
ATTN_BATCH_NODES="${ATTN_BATCH_NODES:-512}"
MAX_TIMESTEPS="${MAX_TIMESTEPS:-64}"
DECODER_CHUNK_T="${DECODER_CHUNK_T:-16}"
TEMPORAL_STRATEGY="${TEMPORAL_STRATEGY:-tail}"
DEVICE="${DEVICE:-cuda}"
AUTO_RESUME="${AUTO_RESUME:-1}"
FINE_FEATURE_NORMALIZE="${FINE_FEATURE_NORMALIZE:-0}"
HEARTBEAT_INTERVAL_SEC="${HEARTBEAT_INTERVAL_SEC:-120}"
RESUME=0

usage() {
  cat <<EOF
Usage: ./run_cross_scale_level2.sh [--resume] [--help]

Environment overrides:
  VENV_PATH, WSL_DATA_DIR, WSL_CHECKPOINT_DIR, WP2_H5, WP3_H5
  RUN_DIR, REPORT_PATH, SIM_IDS
  EPOCHS, HIDDEN_DIM, ATTN_BATCH_NODES, MAX_TIMESTEPS, DECODER_CHUNK_T, TEMPORAL_STRATEGY, DEVICE
  AUTO_RESUME=1|0, FINE_FEATURE_NORMALIZE=1|0, HEARTBEAT_INTERVAL_SEC

Operational defaults (safer first pass):
  SIM_IDS=$DEFAULT_SIM_IDS
  EPOCHS=$EPOCHS
  MAX_TIMESTEPS=$MAX_TIMESTEPS

Restore full Level 2 load via environment overrides:
  SIM_IDS=geom_0_0_pair1,geom_0_0_pair2,geom_0_10 EPOCHS=50 MAX_TIMESTEPS=128 ./run_cross_scale_level2.sh
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
LOG_PATH="${LOG_PATH:-$RUN_DIR/run_level2.log}"

source "$VENV_PATH/bin/activate"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR"
export PYTHONUNBUFFERED=1
export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-garbage_collection_threshold:0.8,max_split_size_mb:512}"
export WP2_H5 WP3_H5

if [[ -f "$ROCM_RUNTIME_LIB" ]]; then
  export LD_LIBRARY_PATH="$(dirname "$ROCM_RUNTIME_LIB"):${LD_LIBRARY_PATH:-}"
  export LD_PRELOAD="$ROCM_RUNTIME_LIB${LD_PRELOAD:+:$LD_PRELOAD}"
fi

if [[ "$AUTO_RESUME" == "1" && -f "$RUN_DIR/latest.pt" ]]; then
  RESUME=1
fi

CMD=(
  python -u -m training.train
  --model-type cross-scale
  --sim-ids "$SIM_IDS"
  --epochs "$EPOCHS"
  --hidden-dim "$HIDDEN_DIM"
  --attn-batch-nodes "$ATTN_BATCH_NODES"
  --max-timesteps "$MAX_TIMESTEPS"
  --decoder-chunk-t "$DECODER_CHUNK_T"
  --temporal-strategy "$TEMPORAL_STRATEGY"
  --amp
  --device "$DEVICE"
  --output "$RUN_DIR"
)
if [[ "$RESUME" == "1" ]]; then
  CMD+=(--resume)
fi
if [[ "$FINE_FEATURE_NORMALIZE" == "1" ]]; then
  CMD+=(--normalize-fine-features)
fi

echo "=== Track B Level 2 (cross-scale overfit) ===" | tee "$LOG_PATH"
echo "Run dir:    $RUN_DIR" | tee -a "$LOG_PATH"
echo "Gate report:$REPORT_PATH" | tee -a "$LOG_PATH"
echo "Resume:     $RESUME" | tee -a "$LOG_PATH"
echo "Fine norm:  $FINE_FEATURE_NORMALIZE" | tee -a "$LOG_PATH"
echo "Sim IDs:    $SIM_IDS" | tee -a "$LOG_PATH"
echo "Epochs:     $EPOCHS" | tee -a "$LOG_PATH"
echo "Max steps:  $MAX_TIMESTEPS" | tee -a "$LOG_PATH"
echo "Heartbeat:  every ${HEARTBEAT_INTERVAL_SEC}s while training runs" | tee -a "$LOG_PATH"
echo "Progress note: first epoch can run for a long time before first metrics are emitted." | tee -a "$LOG_PATH"
echo "If no epoch logs yet, heartbeat lines confirm training is still alive." | tee -a "$LOG_PATH"
echo "Log tail command: tail -f \"$LOG_PATH\"" | tee -a "$LOG_PATH"
train_start_epoch="$(date +%s)"
set +e
("${CMD[@]}" 2>&1 | tee -a "$LOG_PATH") &
train_pid=$!
while kill -0 "$train_pid" 2>/dev/null; do
  sleep "$HEARTBEAT_INTERVAL_SEC"
  if kill -0 "$train_pid" 2>/dev/null; then
    now_epoch="$(date +%s)"
    elapsed_sec="$((now_epoch - train_start_epoch))"
    echo "[$(date -Iseconds)] progress: training still running (elapsed ${elapsed_sec}s)." | tee -a "$LOG_PATH"
  fi
done
wait "$train_pid"
train_rc=$?
set -e
train_end_epoch="$(date +%s)"
train_elapsed_sec="$((train_end_epoch - train_start_epoch))"
echo "Training command elapsed: ${train_elapsed_sec}s" | tee -a "$LOG_PATH"
if [[ $train_rc -ne 0 ]]; then
  echo "Training failed (exit $train_rc). Writing blocked gate report to $REPORT_PATH." | tee -a "$LOG_PATH"
  python -m training.gate_check \
    --level 2 \
    --run-dir "$RUN_DIR" \
    --report-path "$REPORT_PATH" \
    --blocked-reason train_failed \
    --blocked-note "training.train exited with code $train_rc; see $LOG_PATH" | tee -a "$LOG_PATH"
  exit $train_rc
fi

if [[ ! -f "$RUN_DIR/history.json" ]]; then
  echo "Missing $RUN_DIR/history.json after successful training. Writing blocked gate report." | tee -a "$LOG_PATH"
  python -m training.gate_check \
    --level 2 \
    --run-dir "$RUN_DIR" \
    --report-path "$REPORT_PATH" \
    --blocked-reason history_missing \
    --blocked-note "history.json was not produced; skipping gate check" | tee -a "$LOG_PATH"
  exit 2
fi

if [[ ! -f "$RUN_DIR/summary.json" ]]; then
  echo "Missing $RUN_DIR/summary.json after successful training. Writing blocked gate report." | tee -a "$LOG_PATH"
  python -m training.gate_check \
    --level 2 \
    --run-dir "$RUN_DIR" \
    --report-path "$REPORT_PATH" \
    --blocked-reason summary_missing \
    --blocked-note "summary.json was not produced; skipping gate check" | tee -a "$LOG_PATH"
  exit 2
fi

python -m training.gate_check \
  --level 2 \
  --run-dir "$RUN_DIR" \
  --report-path "$REPORT_PATH" | tee -a "$LOG_PATH"

echo "Artifacts:"
echo "  $RUN_DIR/history.json"
echo "  $RUN_DIR/summary.json"
echo "  $RUN_DIR/best.pt"
echo "  $RUN_DIR/latest.pt"
echo "  $REPORT_PATH"

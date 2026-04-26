#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${VENV_PATH:-/home/ellis/venvs/cfwrinkle}"
ROCM_RUNTIME_LIB="${ROCM_RUNTIME_LIB:-/opt/rocm-7.2.0/lib/libamdhip64.so}"
WSL_DATA_DIR="${WSL_DATA_DIR:-/home/ellis/cfwrinkle/data}"
WSL_CHECKPOINT_DIR="${WSL_CHECKPOINT_DIR:-/home/ellis/cfwrinkle/checkpoints}"
WP2_H5="${WP2_H5:-$WSL_DATA_DIR/cfwrinkle_dataset.h5}"
WP3_H5="${WP3_H5:-$WSL_DATA_DIR/cfwrinkle_wp3_features.h5}"
RUN_DIR="${RUN_DIR:-$WSL_CHECKPOINT_DIR/progressive/cross_scale_level3}"
REPORT_PATH="${REPORT_PATH:-$REPO_DIR/reports/wp7_gate_level3_cross_scale.json}"
LEVEL3_PROFILE="${LEVEL3_PROFILE:-pilot}"
FOLD="${FOLD:-3}"
PILOT_MAX_TRAIN_SIMS="${PILOT_MAX_TRAIN_SIMS:-8}"
PILOT_MAX_VAL_SIMS="${PILOT_MAX_VAL_SIMS:-1}"
PILOT_EPOCHS="${PILOT_EPOCHS:-12}"
MINI_MAX_TRAIN_SIMS="${MINI_MAX_TRAIN_SIMS:-13}"
MINI_MAX_VAL_SIMS="${MINI_MAX_VAL_SIMS:-2}"
MINI_EPOCHS="${MINI_EPOCHS:-50}"
HIDDEN_DIM="${HIDDEN_DIM:-64}"
PILOT_ATTN_BATCH_NODES="${PILOT_ATTN_BATCH_NODES:-96}"
PILOT_MAX_TIMESTEPS="${PILOT_MAX_TIMESTEPS:-32}"
PILOT_DECODER_CHUNK_T="${PILOT_DECODER_CHUNK_T:-8}"
MINI_ATTN_BATCH_NODES="${MINI_ATTN_BATCH_NODES:-128}"
MINI_MAX_TIMESTEPS="${MINI_MAX_TIMESTEPS:-64}"
MINI_DECODER_CHUNK_T="${MINI_DECODER_CHUNK_T:-16}"
PILOT_PYTORCH_HIP_ALLOC_CONF="${PILOT_PYTORCH_HIP_ALLOC_CONF:-garbage_collection_threshold:0.7,max_split_size_mb:512,roundup_power2_divisions:8}"
MINI_PYTORCH_HIP_ALLOC_CONF="${MINI_PYTORCH_HIP_ALLOC_CONF:-garbage_collection_threshold:0.7,max_split_size_mb:512,roundup_power2_divisions:8}"
TEMPORAL_STRATEGY="${TEMPORAL_STRATEGY:-tail}"
DEVICE="${DEVICE:-cuda}"
AUTO_RESUME="${AUTO_RESUME:-1}"
FINE_FEATURE_NORMALIZE="${FINE_FEATURE_NORMALIZE:-${LEVEL3_FINE_NORM:-0}}"
AMP_DTYPE="${AMP_DTYPE:-${CFWRINKLE_AMP_DTYPE:-float16}}"
CFWRINKLE_PHYSICS_WARMUP_EPOCHS="${CFWRINKLE_PHYSICS_WARMUP_EPOCHS:-0}"
CFWRINKLE_PHYSICS_RAMP_EPOCHS="${CFWRINKLE_PHYSICS_RAMP_EPOCHS:-20}"
TRAIN_TIMEOUT_SEC="${TRAIN_TIMEOUT_SEC:-0}"
RESUME=0

usage() {
  cat <<EOF
Usage: ./run_cross_scale_level3.sh [--resume] [--help]

Environment overrides:
  VENV_PATH, WSL_DATA_DIR, WSL_CHECKPOINT_DIR, WP2_H5, WP3_H5
  RUN_DIR, REPORT_PATH, LEVEL3_PROFILE=pilot|mini, FOLD
  PILOT_MAX_TRAIN_SIMS, PILOT_MAX_VAL_SIMS, PILOT_EPOCHS
  PILOT_ATTN_BATCH_NODES, PILOT_MAX_TIMESTEPS, PILOT_DECODER_CHUNK_T, PILOT_PYTORCH_HIP_ALLOC_CONF
  MINI_MAX_TRAIN_SIMS, MINI_MAX_VAL_SIMS, MINI_EPOCHS
  MINI_ATTN_BATCH_NODES, MINI_MAX_TIMESTEPS, MINI_DECODER_CHUNK_T, MINI_PYTORCH_HIP_ALLOC_CONF
  HIDDEN_DIM, TEMPORAL_STRATEGY, DEVICE, TRAIN_TIMEOUT_SEC
  PYTORCH_HIP_ALLOC_CONF
  AUTO_RESUME=1|0, FINE_FEATURE_NORMALIZE=1|0
  AMP_DTYPE=float16|bfloat16
  CFWRINKLE_PHYSICS_WARMUP_EPOCHS, CFWRINKLE_PHYSICS_RAMP_EPOCHS

Timeout behavior:
  TRAIN_TIMEOUT_SEC defaults to 0 (disabled).
  Set TRAIN_TIMEOUT_SEC>0 to enable the wrapper timeout.

Default staged strategy:
  LEVEL3_PROFILE=pilot uses a bounded gate-first profile (8/1 split, 12 epochs, 32 timesteps).
  Run full mini-train profile afterward with:
    LEVEL3_PROFILE=mini ./run_cross_scale_level3.sh --resume
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --resume) RESUME=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: Unknown option: $1"; usage; exit 1 ;;
  esac
done

if [[ "$LEVEL3_PROFILE" != "pilot" && "$LEVEL3_PROFILE" != "mini" ]]; then
  echo "ERROR: LEVEL3_PROFILE must be 'pilot' or 'mini' (got: $LEVEL3_PROFILE)"
  exit 1
fi
if ! [[ "$TRAIN_TIMEOUT_SEC" =~ ^[0-9]+$ ]]; then
  echo "ERROR: TRAIN_TIMEOUT_SEC must be a non-negative integer (got: $TRAIN_TIMEOUT_SEC)"
  exit 1
fi
# Guard: the mini profile needs ~4 hours. Reject unsafe short timeouts early.
if [[ "$LEVEL3_PROFILE" == "mini" && "$TRAIN_TIMEOUT_SEC" -gt 0 && "$TRAIN_TIMEOUT_SEC" -lt 14400 ]]; then
  echo "ERROR: TRAIN_TIMEOUT_SEC=$TRAIN_TIMEOUT_SEC is too short for the mini profile."
  echo "  The mini profile (~50 epochs, 13 sims) requires at least 4 hours (14400s)."
  echo "  Set TRAIN_TIMEOUT_SEC=0 to disable the timeout, or TRAIN_TIMEOUT_SEC>=14400."
  exit 1
fi

if [[ "$LEVEL3_PROFILE" == "pilot" ]]; then
  PROFILE_RUN_DIR="${PROFILE_RUN_DIR:-$RUN_DIR/pilot/fold_${FOLD}}"
  MAX_TRAIN_SIMS="$PILOT_MAX_TRAIN_SIMS"
  MAX_VAL_SIMS="$PILOT_MAX_VAL_SIMS"
  EPOCHS="$PILOT_EPOCHS"
  ATTN_BATCH_NODES="$PILOT_ATTN_BATCH_NODES"
  MAX_TIMESTEPS="$PILOT_MAX_TIMESTEPS"
  DECODER_CHUNK_T="$PILOT_DECODER_CHUNK_T"
  PROFILE_ALLOC_CONF_DEFAULT="$PILOT_PYTORCH_HIP_ALLOC_CONF"
else
  PROFILE_RUN_DIR="${PROFILE_RUN_DIR:-$RUN_DIR/mini/fold_${FOLD}}"
  MAX_TRAIN_SIMS="$MINI_MAX_TRAIN_SIMS"
  MAX_VAL_SIMS="$MINI_MAX_VAL_SIMS"
  EPOCHS="$MINI_EPOCHS"
  ATTN_BATCH_NODES="$MINI_ATTN_BATCH_NODES"
  MAX_TIMESTEPS="$MINI_MAX_TIMESTEPS"
  DECODER_CHUNK_T="$MINI_DECODER_CHUNK_T"
  PROFILE_ALLOC_CONF_DEFAULT="$MINI_PYTORCH_HIP_ALLOC_CONF"
fi

mkdir -p "$PROFILE_RUN_DIR" "$(dirname "$REPORT_PATH")"
LOG_PATH="${LOG_PATH:-$PROFILE_RUN_DIR/run_${LEVEL3_PROFILE}_fold${FOLD}.log}"
RUN_STARTED_UTC="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

cd "$REPO_DIR"

write_blocked_report() {
  local reason="$1"
  local note="$2"
  PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}" python -m training.gate_check \
    --level 3 \
    --run-dir "$PROFILE_RUN_DIR" \
    --report-path "$REPORT_PATH" \
    --blocked-reason "$reason" \
    --blocked-note "$note" | tee -a "$LOG_PATH" || true
}

block_and_exit() {
  local reason="$1"
  local note="$2"
  local code="${3:-1}"
  echo "$note" | tee -a "$LOG_PATH"
  write_blocked_report "$reason" "$note"
  exit "$code"
}

[[ -f "$VENV_PATH/bin/activate" ]] || block_and_exit "venv_missing" "ERROR: venv not found at $VENV_PATH"
[[ -f "$WP2_H5" ]] || block_and_exit "wp2_missing" "ERROR: WP2_H5 not found: $WP2_H5"
[[ -f "$WP3_H5" ]] || block_and_exit "wp3_missing" "ERROR: WP3_H5 not found: $WP3_H5"

source "$VENV_PATH/bin/activate"
export PYTHONPATH="$REPO_DIR"
export PYTHONUNBUFFERED=1
export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-$PROFILE_ALLOC_CONF_DEFAULT}"
export CFWRINKLE_PHYSICS_WARMUP_EPOCHS CFWRINKLE_PHYSICS_RAMP_EPOCHS
export WP2_H5 WP3_H5

if [[ -f "$ROCM_RUNTIME_LIB" ]]; then
  export LD_LIBRARY_PATH="$(dirname "$ROCM_RUNTIME_LIB"):${LD_LIBRARY_PATH:-}"
  export LD_PRELOAD="$ROCM_RUNTIME_LIB${LD_PRELOAD:+:$LD_PRELOAD}"
fi

if [[ "$AUTO_RESUME" == "1" && -f "$PROFILE_RUN_DIR/latest.pt" ]]; then
  RESUME=1
fi

on_interrupt() {
  local signal_name="$1"
  echo "Received ${signal_name}; writing blocked gate report to $REPORT_PATH." | tee -a "$LOG_PATH"
  write_blocked_report "interrupted" "received ${signal_name} while LEVEL3_PROFILE=${LEVEL3_PROFILE}; see $LOG_PATH"
  exit 130
}

trap 'on_interrupt SIGINT' INT
trap 'on_interrupt SIGTERM' TERM

CMD=(
  python -u -m training.train
  --model-type cross-scale
  --fold "$FOLD"
  --max-train-sims "$MAX_TRAIN_SIMS"
  --max-val-sims "$MAX_VAL_SIMS"
  --epochs "$EPOCHS"
  --hidden-dim "$HIDDEN_DIM"
  --attn-batch-nodes "$ATTN_BATCH_NODES"
  --max-timesteps "$MAX_TIMESTEPS"
  --decoder-chunk-t "$DECODER_CHUNK_T"
  --temporal-strategy "$TEMPORAL_STRATEGY"
  --amp
  --amp-dtype "$AMP_DTYPE"
  --device "$DEVICE"
  --output "$PROFILE_RUN_DIR"
)
if [[ "$RESUME" == "1" ]]; then
  CMD+=(--resume)
fi
if [[ "$FINE_FEATURE_NORMALIZE" == "1" ]]; then
  CMD+=(--normalize-fine-features)
fi

echo "=== Track B Level 3 (cross-scale, profile: $LEVEL3_PROFILE) ===" | tee "$LOG_PATH"
echo "Run dir:    $PROFILE_RUN_DIR" | tee -a "$LOG_PATH"
echo "Gate report:$REPORT_PATH" | tee -a "$LOG_PATH"
echo "Resume:     $RESUME" | tee -a "$LOG_PATH"
echo "Fine norm:  $FINE_FEATURE_NORMALIZE" | tee -a "$LOG_PATH"
echo "Split:      train=$MAX_TRAIN_SIMS val=$MAX_VAL_SIMS" | tee -a "$LOG_PATH"
echo "Epochs:     $EPOCHS" | tee -a "$LOG_PATH"
echo "OOM profile:MAX_TIMESTEPS=$MAX_TIMESTEPS ATTN_BATCH_NODES=$ATTN_BATCH_NODES DECODER_CHUNK_T=$DECODER_CHUNK_T" | tee -a "$LOG_PATH"
echo "Allocator:  PYTORCH_HIP_ALLOC_CONF=$PYTORCH_HIP_ALLOC_CONF" | tee -a "$LOG_PATH"
echo "AMP dtype:  $AMP_DTYPE" | tee -a "$LOG_PATH"
echo "Physics ramp: warmup=$CFWRINKLE_PHYSICS_WARMUP_EPOCHS ramp=$CFWRINKLE_PHYSICS_RAMP_EPOCHS" | tee -a "$LOG_PATH"
if [[ "$TRAIN_TIMEOUT_SEC" -gt 0 ]]; then
  echo "Timeout:    ${TRAIN_TIMEOUT_SEC}s" | tee -a "$LOG_PATH"
else
  echo "Timeout:    disabled" | tee -a "$LOG_PATH"
fi
rm -f "$REPORT_PATH"
write_blocked_report "in_progress" "LEVEL3_PROFILE=${LEVEL3_PROFILE} fold=${FOLD} started_at=${RUN_STARTED_UTC}; run_dir=$PROFILE_RUN_DIR; log=$LOG_PATH"
set +e
if [[ "$TRAIN_TIMEOUT_SEC" -gt 0 ]]; then
  timeout --signal=INT --kill-after=60 "$TRAIN_TIMEOUT_SEC" "${CMD[@]}" 2>&1 | tee -a "$LOG_PATH"
else
  "${CMD[@]}" 2>&1 | tee -a "$LOG_PATH"
fi
train_rc=${PIPESTATUS[0]}
set -e
if [[ $train_rc -ne 0 ]]; then
  final_rc=$train_rc
  if [[ "$TRAIN_TIMEOUT_SEC" -gt 0 && ( $train_rc -eq 124 || $train_rc -eq 137 || $train_rc -eq 143 ) ]]; then
    echo "Training timed out after ${TRAIN_TIMEOUT_SEC}s; writing blocked gate report." | tee -a "$LOG_PATH"
    write_blocked_report "stalled_timeout" "timeout wrapper reached TRAIN_TIMEOUT_SEC=${TRAIN_TIMEOUT_SEC}; training was interrupted; see $LOG_PATH"
    final_rc=124
  elif [[ $train_rc -eq 130 || $train_rc -eq 143 ]]; then
    echo "Training interrupted (exit $train_rc); writing blocked gate report." | tee -a "$LOG_PATH"
    write_blocked_report "interrupted" "training interrupted by signal (exit=${train_rc}); likely user/system interrupt; see $LOG_PATH"
    final_rc=130
  else
    echo "Training failed (exit $train_rc). Writing blocked gate report." | tee -a "$LOG_PATH"
    write_blocked_report "train_failed" "training.train exited with code $train_rc; see $LOG_PATH"
  fi
  exit $final_rc
fi

if [[ ! -f "$PROFILE_RUN_DIR/history.json" ]]; then
  echo "Missing $PROFILE_RUN_DIR/history.json after successful training. Writing blocked gate report." | tee -a "$LOG_PATH"
  write_blocked_report "history_missing" "history.json was not produced; skipping gate check"
  exit 2
fi

if ! PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}" python - "$PROFILE_RUN_DIR/history.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except Exception as exc:
    raise SystemExit(f"invalid_json: {exc}")
if not isinstance(payload, list) or not payload:
    raise SystemExit("history_not_nonempty_list")
if not isinstance(payload[-1], dict):
    raise SystemExit("history_last_row_not_object")
PY
then
  echo "Invalid $PROFILE_RUN_DIR/history.json after successful training. Writing blocked gate report." | tee -a "$LOG_PATH"
  write_blocked_report "history_invalid" "history.json is missing/invalid/non-empty-list contract; see $LOG_PATH"
  exit 2
fi

if [[ ! -f "$PROFILE_RUN_DIR/summary.json" ]]; then
  echo "Missing $PROFILE_RUN_DIR/summary.json after successful training. Writing blocked gate report." | tee -a "$LOG_PATH"
  write_blocked_report "summary_missing" "summary.json was not produced; skipping gate check"
  exit 2
fi

if ! PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}" python - "$PROFILE_RUN_DIR/summary.json" "$FOLD" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected_fold = int(sys.argv[2])
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except Exception as exc:
    raise SystemExit(f"invalid_json: {exc}")
if not isinstance(payload, list) or not payload:
    raise SystemExit("summary_not_nonempty_list")
first = payload[0]
if not isinstance(first, dict):
    raise SystemExit("summary_first_item_not_object")
if int(first.get("fold", -1)) != expected_fold:
    raise SystemExit(f"summary_fold_mismatch:{first.get('fold')}")
if not isinstance(first.get("history"), list) or not first["history"]:
    raise SystemExit("summary_history_missing_or_empty")
PY
then
  echo "Invalid $PROFILE_RUN_DIR/summary.json after successful training. Writing blocked gate report." | tee -a "$LOG_PATH"
  write_blocked_report "summary_invalid" "summary.json is missing/invalid/non-empty-list contract or fold mismatch; see $LOG_PATH"
  exit 2
fi

python -m training.gate_check \
  --level 3 \
  --run-dir "$PROFILE_RUN_DIR" \
  --report-path "$REPORT_PATH" | tee -a "$LOG_PATH"

trap - INT TERM
echo "Artifacts:"
echo "  $PROFILE_RUN_DIR/history.json"
echo "  $PROFILE_RUN_DIR/summary.json"
echo "  $PROFILE_RUN_DIR/best.pt"
echo "  $PROFILE_RUN_DIR/latest.pt"
echo "  $REPORT_PATH"

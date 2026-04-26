#!/usr/bin/env bash
# Track C — CrossScaleNet Level 4 full CV
# Changes from Track B: use_fine_mp=True, fine_dz weight 1.5→4.0
# Track B reference checkpoints preserved at: cross_scale_level4_cv/
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${VENV_PATH:-/home/ellis/venvs/cfwrinkle}"
ROCM_RUNTIME_LIB="${ROCM_RUNTIME_LIB:-/opt/rocm-7.2.0/lib/libamdhip64.so}"
WSL_DATA_DIR="${WSL_DATA_DIR:-/home/ellis/cfwrinkle/data}"
WSL_CHECKPOINT_DIR="${WSL_CHECKPOINT_DIR:-/home/ellis/cfwrinkle/checkpoints}"
WP2_H5="${WP2_H5:-$WSL_DATA_DIR/cfwrinkle_dataset.h5}"
WP3_H5="${WP3_H5:-$WSL_DATA_DIR/cfwrinkle_wp3_features.h5}"
RUN_DIR="${RUN_DIR:-$WSL_CHECKPOINT_DIR/progressive/cross_scale_level4_cv_trackc}"
REPORT_PATH="${REPORT_PATH:-$REPO_DIR/reports/wp7_gate_level4_cross_scale_trackc.json}"

# ── Training hyperparameters ──────────────────────────────────────────────────
EPOCHS="${EPOCHS:-50}"
HIDDEN_DIM="${HIDDEN_DIM:-64}"
TEMPORAL_STRATEGY="${TEMPORAL_STRATEGY:-tail}"
DEVICE="${DEVICE:-cuda}"
AUTO_RESUME="${AUTO_RESUME:-1}"

# ── ROCm hard constraints (7900 XT — do not change) ──────────────────────────
MAX_TIMESTEPS="${MAX_TIMESTEPS:-96}"           # hard floor; T=128 preload exceeds WSL2 27 GB RAM ceiling (tested 2026-04-25)
ATTN_BATCH_NODES="${ATTN_BATCH_NODES:-512}"   # 512 fills GPU kernels (RDNA3 native)
DECODER_CHUNK_T="${DECODER_CHUNK_T:-6}"    # halved: reduces GRU BPTT activation memory per chunk
AMP_DTYPE="${AMP_DTYPE:-float16}"             # RDNA3 gfx1100 native fp16
FINE_FEATURE_NORMALIZE="${FINE_FEATURE_NORMALIZE:-1}"  # must remain ON

# ── Track C: fine_mp + dz weight + compile ────────────────────────────────────
USE_FINE_MP="${USE_FINE_MP:-1}"
USE_TORCH_COMPILE="${USE_TORCH_COMPILE:-1}"
TORCH_ROCM_FA_PREFER_CK="${TORCH_ROCM_FA_PREFER_CK:-0}"       # CK targets Wave64 (MI series); gfx1100 is Wave32 — causes extra Triton kernels with torch.compile
CFWRINKLE_FINE_DZ_WEIGHT="${CFWRINKLE_FINE_DZ_WEIGHT:-4.0}"   # up from Track B 1.5

# ── Memory / physics hooks (proven stable from Track B) ──────────────────────
CFWRINKLE_PHYSICS_WARMUP_EPOCHS="${CFWRINKLE_PHYSICS_WARMUP_EPOCHS:-5}"
CFWRINKLE_PHYSICS_RAMP_EPOCHS="${CFWRINKLE_PHYSICS_RAMP_EPOCHS:-15}"
CFWRINKLE_FINE_ELEM_REGIONS="${CFWRINKLE_FINE_ELEM_REGIONS:-4}"
CFWRINKLE_FINE_LOSS_CHUNK_ELEMS="${CFWRINKLE_FINE_LOSS_CHUNK_ELEMS:-512}"   # reduced from 2048: fewer fine elements per loss chunk
CFWRINKLE_AUX_LOSS_INTERVAL="${CFWRINKLE_AUX_LOSS_INTERVAL:-1}"
CFWRINKLE_DISABLE_FINE_COHERENCE="${CFWRINKLE_DISABLE_FINE_COHERENCE:-0}"
CFWRINKLE_DISABLE_FINE_COUPLING="${CFWRINKLE_DISABLE_FINE_COUPLING:-0}"
CFWRINKLE_DISABLE_FINE_BUCKLING="${CFWRINKLE_DISABLE_FINE_BUCKLING:-0}"
CFWRINKLE_DISABLE_FINE_DZ_MONO="${CFWRINKLE_DISABLE_FINE_DZ_MONO:-0}"
PYTORCH_NO_CUDA_MEMORY_CACHING="${PYTORCH_NO_CUDA_MEMORY_CACHING:-0}"
RESUME=0

usage() {
  cat <<EOF
Usage: ./run_cross_scale_level4_trackc.sh [--resume] [--help]

Track C: use_fine_mp=True, CFWRINKLE_FINE_DZ_WEIGHT=4.0 (up from Track B 1.5)
Track B reference: checkpoints/progressive/cross_scale_level4_cv/ (do not modify)

Key env overrides:
  RUN_DIR, REPORT_PATH
  USE_FINE_MP=1|0  CFWRINKLE_FINE_DZ_WEIGHT=<float>
  EPOCHS, HIDDEN_DIM, MAX_TIMESTEPS (>=96), AMP_DTYPE (float16 for RDNA3/gfx1100)
  CFWRINKLE_FINE_LOSS_CHUNK_ELEMS, CFWRINKLE_PHYSICS_WARMUP_EPOCHS
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
export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.8}"
export CFWRINKLE_FINE_LOSS_CHUNK_ELEMS CFWRINKLE_AUX_LOSS_INTERVAL
export CFWRINKLE_FINE_ELEM_REGIONS
export CFWRINKLE_DISABLE_FINE_COHERENCE CFWRINKLE_DISABLE_FINE_COUPLING
export CFWRINKLE_DISABLE_FINE_BUCKLING CFWRINKLE_DISABLE_FINE_DZ_MONO
export CFWRINKLE_PHYSICS_WARMUP_EPOCHS CFWRINKLE_PHYSICS_RAMP_EPOCHS
export CFWRINKLE_FINE_DZ_WEIGHT
export TORCH_ROCM_FA_PREFER_CK
# HSA_DISABLE_FRAGMENT_ALLOCATOR intentionally NOT set.
# With it enabled: allocation failures become infinite retry loops → GPU hang mid-epoch.
# Without it: failures retry with fragmentation and eventually succeed (transient HSA
# exceptions visible in logs are benign — they appeared in completed epochs too).
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
# Preload disabled: Batch B coarse mesh is 30k nodes; 40 sims × T=96 preload needs ~31 GB,
# which exceeds the WSL2 27 GB ceiling. Fine stats are persisted in HDF5 (one-time cost done).
CMD+=(--no-preload-fold)
# Gradient checkpointing must stay ON: without it fine_mp = 18.78 GB VRAM OOM.
# Per-epoch empty_cache() is in train.py epoch loop — keeps pool below 15 GB at each
# epoch start so Triton kernel-load hipMalloc calls succeed under HSA_DISABLE_FRAGMENT_ALLOCATOR=1.

echo "=== Track C Level 4 (cross-scale full CV, fine_mp=ON, compile=ON) ===" | tee "$LOG_PATH"
echo "Run dir:      $RUN_DIR" | tee -a "$LOG_PATH"
echo "Gate report:  $REPORT_PATH" | tee -a "$LOG_PATH"
echo "Resume:       $RESUME" | tee -a "$LOG_PATH"
echo "Fine norm:    $FINE_FEATURE_NORMALIZE" | tee -a "$LOG_PATH"
echo "Use fine MP:  $USE_FINE_MP" | tee -a "$LOG_PATH"
echo "torch.compile:$USE_TORCH_COMPILE" | tee -a "$LOG_PATH"
echo "CK flash attn:$TORCH_ROCM_FA_PREFER_CK" | tee -a "$LOG_PATH"
echo "fine_dz wt:   $CFWRINKLE_FINE_DZ_WEIGHT  (Track B was 1.5)" | tee -a "$LOG_PATH"
echo "Loss chunk:   $CFWRINKLE_FINE_LOSS_CHUNK_ELEMS elems" | tee -a "$LOG_PATH"
echo "AMP dtype:    $AMP_DTYPE" | tee -a "$LOG_PATH"
echo "Physics ramp: warmup=$CFWRINKLE_PHYSICS_WARMUP_EPOCHS ramp=$CFWRINKLE_PHYSICS_RAMP_EPOCHS" | tee -a "$LOG_PATH"
echo "Aux interval: $CFWRINKLE_AUX_LOSS_INTERVAL" | tee -a "$LOG_PATH"
echo "Disable aux:  coherence=$CFWRINKLE_DISABLE_FINE_COHERENCE coupling=$CFWRINKLE_DISABLE_FINE_COUPLING buckling=$CFWRINKLE_DISABLE_FINE_BUCKLING dz_mono=$CFWRINKLE_DISABLE_FINE_DZ_MONO" | tee -a "$LOG_PATH"
echo "No alloc cache: $NO_ALLOC_CACHE_EFFECTIVE" | tee -a "$LOG_PATH"
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
echo "  $RUN_DIR/fold_*/history.json"
echo "  $RUN_DIR/fold_*/best.pt"
echo "  $REPORT_PATH"

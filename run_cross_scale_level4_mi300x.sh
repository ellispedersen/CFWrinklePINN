#!/usr/bin/env bash
# Track C — MI300X variant: gfx942/CDNA3, 192 GB HBM3, 283 GB DDR5
# DO NOT use on gfx1100/RDNA3 — settings are hardware-specific (bfloat16, CK, no-checkpoint).
#
# Key differences from 7900 XT Track C (run_cross_scale_level4_trackc.sh):
#   hidden_dim       64  → 96    (HSA pool exhaustion absent on gfx942)
#   MAX_TIMESTEPS    96  → 128   (DDR5 RAM supports preload; no WSL2 ceiling)
#   AMP_DTYPE        float16 → bfloat16  (CDNA3 native; faster than fp16 on MI300X)
#   TORCH_ROCM_FA_PREFER_CK  0 → 1      (CK targets Wave64 = CDNA3 native)
#   gradient checkpoint   ON → OFF       (192 GB VRAM holds T=128 activations; +40% backward speed)
#   preload          OFF → ON            (283 GB DDR5 handles Batch B 30k × 40 × T=128 ≈ 41 GB)
#   INDUCTOR_GEMM_BACKENDS  ATEN → ATEN,TRITON  (Triton GEMM stable on gfx942)
#   HSA_DISABLE_FRAGMENT_ALLOCATOR  unset → 1   (safe: model <60 GB of 192 GB VRAM)
#   DECODER_CHUNK_T  6 → 12             (restored; 192 GB handles full GRU backward)
#   CFWRINKLE_FINE_LOSS_CHUNK_ELEMS  512 → 2048 (restored; no VRAM pressure)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${VENV_PATH:-/workspace/venv}"
ROCM_RUNTIME_LIB="${ROCM_RUNTIME_LIB:-/opt/rocm/lib/libamdhip64.so}"
DATA_DIR="${DATA_DIR:-/workspace/data}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/workspace/checkpoints}"
WP2_H5="${WP2_H5:-$DATA_DIR/cfwrinkle_dataset.h5}"
WP3_H5="${WP3_H5:-$DATA_DIR/cfwrinkle_wp3_features.h5}"
RUN_DIR="${RUN_DIR:-$CHECKPOINT_DIR/progressive/cross_scale_level4_cv_trackc_mi300x}"
REPORT_PATH="${REPORT_PATH:-$REPO_DIR/reports/wp7_gate_level4_cross_scale_trackc_mi300x.json}"

# ── Training hyperparameters ──────────────────────────────────────────────────
EPOCHS="${EPOCHS:-50}"
HIDDEN_DIM="${HIDDEN_DIM:-96}"
TEMPORAL_STRATEGY="${TEMPORAL_STRATEGY:-tail}"
DEVICE="${DEVICE:-cuda}"
AUTO_RESUME="${AUTO_RESUME:-1}"

# ── MI300X hardware config (gfx942 / CDNA3 / Wave64) ─────────────────────────
MAX_TIMESTEPS="${MAX_TIMESTEPS:-128}"
ATTN_BATCH_NODES="${ATTN_BATCH_NODES:-512}"      # benchmark 1024/2048 — 512 is safe starting point
DECODER_CHUNK_T="${DECODER_CHUNK_T:-12}"
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"               # CDNA3 native; faster than float16 on MI300X
FINE_FEATURE_NORMALIZE="${FINE_FEATURE_NORMALIZE:-1}"

# ── Track C: fine_mp + dz weight + compile ────────────────────────────────────
USE_FINE_MP="${USE_FINE_MP:-1}"
USE_TORCH_COMPILE="${USE_TORCH_COMPILE:-1}"
TORCH_ROCM_FA_PREFER_CK="${TORCH_ROCM_FA_PREFER_CK:-1}"        # CK is native on Wave64/CDNA3
INDUCTOR_GEMM_BACKENDS="${INDUCTOR_GEMM_BACKENDS:-ATEN,TRITON}" # Triton GEMM stable on gfx942
CFWRINKLE_FINE_DZ_WEIGHT="${CFWRINKLE_FINE_DZ_WEIGHT:-4.0}"    # Track C target

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
Usage: ./run_cross_scale_level4_mi300x.sh [--resume] [--help]

Track C MI300X: hidden_dim=96, T=128, bfloat16, CK flash-attn, no gradient checkpointing.
Target: gfx942/CDNA3 (RunPod MI300X). DO NOT run on gfx1100/RDNA3.

Key env overrides:
  DATA_DIR, CHECKPOINT_DIR, RUN_DIR, REPORT_PATH
  VENV_PATH                (default: /workspace/venv)
  HIDDEN_DIM               (default: 96)
  MAX_TIMESTEPS            (default: 128)
  ATTN_BATCH_NODES         (default: 512; benchmark 1024/2048 on MI300X)
  INDUCTOR_GEMM_BACKENDS   (default: ATEN,TRITON)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --resume) RESUME=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: Unknown option: $1"; usage; exit 1 ;;
  esac
done

[[ -f "$VENV_PATH/bin/activate" ]] || { echo "ERROR: venv not found at $VENV_PATH — run scripts/runpod_mi300x_setup.sh first"; exit 1; }
[[ -f "$WP2_H5" ]] || { echo "ERROR: WP2_H5 not found: $WP2_H5"; exit 1; }
[[ -f "$WP3_H5" ]] || { echo "ERROR: WP3_H5 not found: $WP3_H5"; exit 1; }

mkdir -p "$RUN_DIR" "$(dirname "$REPORT_PATH")"
LOG_PATH="$RUN_DIR/run.log"

source "$VENV_PATH/bin/activate"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR"
export PYTHONUNBUFFERED=1
# Relaxed allocator config: no max_split_size_mb — 192 GB VRAM has no fragmentation pressure.
export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.9}"
export CFWRINKLE_FINE_LOSS_CHUNK_ELEMS CFWRINKLE_AUX_LOSS_INTERVAL
export CFWRINKLE_FINE_ELEM_REGIONS
export CFWRINKLE_DISABLE_FINE_COHERENCE CFWRINKLE_DISABLE_FINE_COUPLING
export CFWRINKLE_DISABLE_FINE_BUCKLING CFWRINKLE_DISABLE_FINE_DZ_MONO
export CFWRINKLE_PHYSICS_WARMUP_EPOCHS CFWRINKLE_PHYSICS_RAMP_EPOCHS
export CFWRINKLE_FINE_DZ_WEIGHT
export TORCH_ROCM_FA_PREFER_CK
export INDUCTOR_GEMM_BACKENDS
export WP2_H5 WP3_H5
# HSA fragment allocator is safe to enable on MI300X: model peak VRAM < 60 GB of 192 GB.
# This prevents heap fragmentation during Triton compilation; the mid-epoch deadlock that
# triggered on 7900 XT (pool at 95%+ capacity) cannot occur here.
export HSA_DISABLE_FRAGMENT_ALLOCATOR=1

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
# Gradient checkpointing OFF: 192 GB VRAM holds all T=128 × hidden_dim=96 fine activations (~50-60 GB).
# Eliminates 128 backward re-materializations per training sample — significant epoch speedup.
CMD+=(--no-checkpoint)
# Preload ON: 283 GB DDR5 handles Batch B 30k-node coarse × 40 sims × T=128 = ~41 GB RAM.
# (--no-preload-fold is intentionally absent; contrast with 7900 XT WSL2 script)

echo "=== Track C MI300X (cross-scale full CV, fine_mp=ON, compile=ON, no-checkpoint) ===" | tee "$LOG_PATH"
echo "Run dir:       $RUN_DIR" | tee -a "$LOG_PATH"
echo "Gate report:   $REPORT_PATH" | tee -a "$LOG_PATH"
echo "Resume:        $RESUME" | tee -a "$LOG_PATH"
echo "hidden_dim:    $HIDDEN_DIM" | tee -a "$LOG_PATH"
echo "T:             $MAX_TIMESTEPS" | tee -a "$LOG_PATH"
echo "AMP dtype:     $AMP_DTYPE" | tee -a "$LOG_PATH"
echo "CK flash attn: $TORCH_ROCM_FA_PREFER_CK" | tee -a "$LOG_PATH"
echo "GEMM backends: $INDUCTOR_GEMM_BACKENDS" | tee -a "$LOG_PATH"
echo "fine_dz wt:    $CFWRINKLE_FINE_DZ_WEIGHT" | tee -a "$LOG_PATH"
echo "Loss chunk:    $CFWRINKLE_FINE_LOSS_CHUNK_ELEMS elems" | tee -a "$LOG_PATH"
echo "Physics ramp:  warmup=$CFWRINKLE_PHYSICS_WARMUP_EPOCHS ramp=$CFWRINKLE_PHYSICS_RAMP_EPOCHS" | tee -a "$LOG_PATH"
echo "Aux interval:  $CFWRINKLE_AUX_LOSS_INTERVAL" | tee -a "$LOG_PATH"
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

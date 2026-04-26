#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/mnt/c/Users/ellis/Documents/VS Code/CFWrinklePINN"
RUN_DIR="/home/ellis/cfwrinkle/checkpoints/progressive/cross_scale_level4_cv"
TRAIN_LOG="/home/ellis/cfwrinkle/logs/level4_trackb_main.log"
RUNNER_LOG="/home/ellis/cfwrinkle/logs/fleet_b4b6_after_b3.nohup.log"

while true; do
  if [[ -s "$RUN_DIR/summary.json" ]]; then
    echo "[$(date -Iseconds)] summary-ready"
    exit 0
  fi

  if ! pgrep -A -f -- '^python(3)? -u -m training\.train --model-type cross-scale --all-folds' >/dev/null; then
    echo "[$(date -Iseconds)] relaunch-b3"
    tmux kill-session -t level4_trackb_main 2>/dev/null || true
    tmux new-session -d -s level4_trackb_main \
      "cd \"$REPO_DIR\" && source /home/ellis/venvs/cfwrinkle/bin/activate && \
       export PYTHONPATH=\"$REPO_DIR\" && \
       export WP2_H5=/home/ellis/cfwrinkle/data/cfwrinkle_dataset.h5 && \
       export WP3_H5=/home/ellis/cfwrinkle/data/cfwrinkle_wp3_features.h5 && \
       export LD_PRELOAD=/opt/rocm-7.2.0/lib/libamdhip64.so && \
        export PYTHONUNBUFFERED=1 && \
        export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:512 && \
        export EPOCHS=30 ATTN_BATCH_NODES=96 MAX_TIMESTEPS=96 DECODER_CHUNK_T=8 TEMPORAL_STRATEGY=tail FINE_FEATURE_NORMALIZE=1 AUTO_RESUME=1 HIPBLAS_WORKSPACE_CONFIG=:4096:2:16:8 CFWRINKLE_FINE_LOSS_CHUNK_ELEMS=2048 CFWRINKLE_AUX_LOSS_INTERVAL=2 CFWRINKLE_DISABLE_FINE_COHERENCE=1 PYTORCH_NO_CUDA_MEMORY_CACHING=0 && \
        ./run_cross_scale_level4.sh --resume 2>&1 | tee \"$TRAIN_LOG\""
  fi

  if ! pgrep -A -f -- '^bash scripts/fleet_b4b6_after_b3\.sh$' >/dev/null; then
    echo "[$(date -Iseconds)] relaunch-b4b6-runner"
    nohup bash "$REPO_DIR/scripts/fleet_b4b6_after_b3.sh" >"$RUNNER_LOG" 2>&1 &
  fi

  sleep 180
done

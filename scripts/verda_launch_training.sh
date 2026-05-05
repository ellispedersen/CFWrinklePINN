#!/usr/bin/env bash
# Verda training launcher — RTX Pro 6000 Blackwell.
#
# Wraps run_cross_scale_level4_cuda.sh inside the Docker container, mapping the
# NVMe block volume paths into the /workspace layout the script expects.
# run_cross_scale_level4_cuda.sh requires zero changes.
#
# Prerequisites:
#   - verda_gpu_setup.sh has completed (volume mounted, image pulled, smoke test passed)
#   - /mnt/data/cfwrinkle_dataset.h5 and cfwrinkle_wp3_features.h5 present
#
# On spot eviction:
#   1. Volume retained (on_spot_discontinue: keep_detached)
#   2. Spin up new RTX Pro 6000 instance, attach cfwrinkle-data volume
#   3. Run verda_gpu_setup.sh  (mounts volume, pulls image, smoke test)
#   4. Run this script again — AUTO_RESUME detects latest.pt and resumes
#
set -euo pipefail

IMAGE="${IMAGE:-vccr.io/20175b95-1ac5-4808-89b0-b08dc612c71e/cfwrinkle-train:pt2110}"
VOLUME_ROOT="${VOLUME_ROOT:-/mnt/cfwrinkle-data}"

# Verify volume is mounted and data is present before launching.
[[ -f "$VOLUME_ROOT/cfwrinkle_dataset.h5" ]] || {
  echo "ERROR: WP2 HDF5 not found at $VOLUME_ROOT/cfwrinkle_dataset.h5"
  echo "       Is the volume mounted? Run verda_gpu_setup.sh first."
  exit 1
}
[[ -f "$VOLUME_ROOT/cfwrinkle_wp3_features.h5" ]] || {
  echo "ERROR: WP3 HDF5 not found at $VOLUME_ROOT/cfwrinkle_wp3_features.h5"
  echo "       Run verda_cpu_rebuild.sh on a CPU instance first."
  exit 1
}

echo "=== Verda training launch ==="
echo "Image:       $IMAGE"
echo "Volume root: $VOLUME_ROOT"
echo "WP2:         $(du -sh "$VOLUME_ROOT/cfwrinkle_dataset.h5" | cut -f1)"
echo "WP3:         $(du -sh "$VOLUME_ROOT/cfwrinkle_wp3_features.h5" | cut -f1)"
echo ""

# The /workspace layout inside the container mirrors the RunPod convention.
# run_cross_scale_level4_cuda.sh reads these paths from env vars — no script edits needed.
#
# Triton cache is mounted from the volume so torch.compile max-autotune kernel
# compilation (10-20 min first run) is cached across spot evictions.

docker run --rm --gpus all \
  -v "$VOLUME_ROOT:/workspace/data_vol" \
  -v "$VOLUME_ROOT/checkpoints:/workspace/checkpoints" \
  -v "$VOLUME_ROOT/logs:/workspace/logs" \
  -v "$VOLUME_ROOT/.triton_cache:/workspace/.triton_cache" \
  -e REPO_DIR=/workspace/repo \
  -e DATA_DIR=/workspace/data_vol \
  -e WP2_H5=/workspace/data_vol/cfwrinkle_dataset.h5 \
  -e WP3_H5=/workspace/data_vol/cfwrinkle_wp3_features.h5 \
  -e CHECKPOINT_DIR=/workspace/checkpoints \
  -e LOG_PATH=/workspace/logs/level4_trackc_verda.log \
  -e RUN_DIR=/workspace/checkpoints/progressive/cross_scale_level4_cv_trackc_verda \
  -e REPORT_PATH=/workspace/checkpoints/progressive/cross_scale_level4_cv_trackc_verda/gate_report.json \
  -e TRITON_CACHE_DIR=/workspace/.triton_cache \
  -e PYTHONUNBUFFERED=1 \
  -e USE_TORCH_COMPILE="${USE_TORCH_COMPILE:-1}" \
  -e USE_GRAD_CHECKPOINT="${USE_GRAD_CHECKPOINT:-0}" \
  -e HIDDEN_DIM="${HIDDEN_DIM:-96}" \
  -e MAX_TIMESTEPS="${MAX_TIMESTEPS:-128}" \
  -e EPOCHS="${EPOCHS:-50}" \
  -e ATTN_BATCH_NODES="${ATTN_BATCH_NODES:-1024}" \
  -e DECODER_CHUNK_T="${DECODER_CHUNK_T:-24}" \
  -e AMP_DTYPE="${AMP_DTYPE:-bfloat16}" \
  -e TORCH_COMPILE_MODE="${TORCH_COMPILE_MODE:-max-autotune}" \
  -e AUTO_RESUME="${AUTO_RESUME:-1}" \
  -e B300_PARALLEL="${B300_PARALLEL:-1}" \
  -e CFWRINKLE_FINE_LOSS_CHUNK_ELEMS="${CFWRINKLE_FINE_LOSS_CHUNK_ELEMS:-2048}" \
  -e CFWRINKLE_AUX_LOSS_INTERVAL="${CFWRINKLE_AUX_LOSS_INTERVAL:-1}" \
  -e CFWRINKLE_FINE_DZ_WEIGHT="${CFWRINKLE_FINE_DZ_WEIGHT:-4.0}" \
  -e CFWRINKLE_DISABLE_FINE_COHERENCE="${CFWRINKLE_DISABLE_FINE_COHERENCE:-0}" \
  -e CFWRINKLE_DISABLE_FINE_COUPLING="${CFWRINKLE_DISABLE_FINE_COUPLING:-0}" \
  -e CFWRINKLE_DISABLE_FINE_BUCKLING="${CFWRINKLE_DISABLE_FINE_BUCKLING:-0}" \
  -e CFWRINKLE_DISABLE_FINE_DZ_MONO="${CFWRINKLE_DISABLE_FINE_DZ_MONO:-0}" \
  "$IMAGE" \
  bash /workspace/repo/run_cross_scale_level4_cuda.sh

echo ""
echo "=== Training complete ==="
echo "Gate report: $VOLUME_ROOT/checkpoints/progressive/cross_scale_level4_cv_trackc_verda/gate_report.json"
echo ""
echo "Download checkpoints before terminating the instance:"
echo "  rsync -avz root@<VERDA_IP>:$VOLUME_ROOT/checkpoints/progressive/ ./checkpoints/"

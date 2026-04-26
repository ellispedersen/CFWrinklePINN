#!/usr/bin/env bash
set -e
source /home/ellis/venvs/cfwrinkle/bin/activate
export LD_PRELOAD=/opt/rocm-7.2.0/lib/libamdhip64.so
export PYTHONUNBUFFERED=1
export PYTORCH_HIP_ALLOC_CONF="garbage_collection_threshold:0.8,max_split_size_mb:512"

python -u -m training.train \
  --fold 3 \
  --max-train-sims 13 --max-val-sims 2 \
  --epochs 50 --hidden-dim 64 --attn-batch-nodes 512 \
  --max-timesteps 128 --temporal-strategy tail --amp --device cuda \
  --output /home/ellis/cfwrinkle/checkpoints/progressive/level3_mini

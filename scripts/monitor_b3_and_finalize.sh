#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/mnt/c/Users/ellis/Documents/VS Code/CFWrinklePINN"
RUN_DIR="/home/ellis/cfwrinkle/checkpoints/progressive/cross_scale_level4_cv"

summary_ready() {
  python3 - <<'PY'
import json
from pathlib import Path
p = Path("/home/ellis/cfwrinkle/checkpoints/progressive/cross_scale_level4_cv/summary.json")
if not p.exists() or p.stat().st_size == 0:
    raise SystemExit(1)
d = json.loads(p.read_text())
folds = d if isinstance(d, list) else d.get("folds", [])
raise SystemExit(0 if isinstance(folds, list) and len(folds) == 5 else 1)
PY
}

while true; do
  ts="$(date -Iseconds)"
  trainer="$(pgrep -A -f -- 'training.train --model-type cross-scale --all-folds' || true)"
  if summary_ready; then
    echo "[$ts] summary-ready"
    break
  fi
  if [[ -z "$trainer" ]]; then
    echo "[$ts] trainer-off-before-summary"
    exit 2
  fi
  echo "[$ts] trainer-running"
  tail -n 6 "$RUN_DIR/run.log" 2>/dev/null || true
  sleep 600
done

cd "$REPO_DIR"
source /home/ellis/venvs/cfwrinkle/bin/activate
export PYTHONPATH="$REPO_DIR"
export LD_PRELOAD=/opt/rocm-7.2.0/lib/libamdhip64.so

python -m training.gate_check --level 4 --run-dir "$RUN_DIR" --report-path reports/wp8_gate_level4.json
python -m training.cross_test_harness \
  --model-a-dir /home/ellis/cfwrinkle/checkpoints/progressive/level4_cv \
  --model-b-dir "$RUN_DIR" \
  --report-path reports/cross_model_comparison.json
python -m training.release_check --run-dir "$RUN_DIR" --report-path reports/release_check_trackb.json

echo "FINAL_CHAIN_DONE"

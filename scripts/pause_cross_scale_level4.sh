#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="${SESSION_NAME:-level4_finalexec}"
RUN_DIR="${RUN_DIR:-/home/ellis/cfwrinkle/checkpoints/progressive/cross_scale_level4_cv}"
WAIT_SEC="${WAIT_SEC:-60}"

echo "Pausing Track B Level 4..."
echo "  session: $SESSION_NAME"
echo "  run dir: $RUN_DIR"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  tmux send-keys -t "$SESSION_NAME" C-c
fi

mapfile -t pids < <(pgrep -f "python -u -m training.train --model-type cross-scale --all-folds" || true)
if ((${#pids[@]} > 0)); then
  echo "Sending SIGINT to trainer PIDs: ${pids[*]}"
  kill -INT "${pids[@]}"
fi

deadline=$((SECONDS + WAIT_SEC))
while pgrep -f "python -u -m training.train --model-type cross-scale --all-folds" >/dev/null 2>&1; do
  if ((SECONDS >= deadline)); then
    echo "Graceful stop timed out; sending SIGTERM."
    mapfile -t pids < <(pgrep -f "python -u -m training.train --model-type cross-scale --all-folds" || true)
    if ((${#pids[@]} > 0)); then
      kill -TERM "${pids[@]}"
    fi
    break
  fi
  sleep 2
done

sleep 2
if pgrep -f "python -u -m training.train --model-type cross-scale --all-folds" >/dev/null 2>&1; then
  echo "WARNING: trainer process still running. Check manually with pgrep."
else
  echo "Trainer stopped."
fi

RUN_DIR="$RUN_DIR" python3 - <<'PY'
import json
import os
from pathlib import Path

run_dir = Path(os.environ["RUN_DIR"])
print("Checkpoint summary:")
for fold in range(5):
    d = run_dir / f"fold_{fold}"
    if not d.exists():
        continue
    h = d / "history.json"
    if h.exists():
        rows = json.loads(h.read_text())
        last = rows[-1].get("epoch") if rows else None
        print(f"  fold_{fold}: epoch={last}, rows={len(rows)}")
    else:
        print(f"  fold_{fold}: history missing")
    for name in ("latest.pt", "best.pt"):
        print(f"    {name}: {'yes' if (d / name).exists() else 'no'}")
PY

echo "Pause complete."

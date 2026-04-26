#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/mnt/c/Users/ellis/Documents/VS Code/CFWrinklePINN"
RUN_DIR="/home/ellis/cfwrinkle/checkpoints/progressive/cross_scale_level4_cv"
LOG="/home/ellis/cfwrinkle/logs/fleet_b4b6_after_b3.log"

source /home/ellis/venvs/cfwrinkle/bin/activate
export LD_PRELOAD=/opt/rocm-7.2.0/lib/libamdhip64.so
export PYTHONUNBUFFERED=1

mkdir -p "$(dirname "$LOG")"
echo "[fleet-b4b6] started $(date -Iseconds)" | tee -a "$LOG"

summary_is_valid() {
  python3 - <<'PY'
import json, pathlib, sys
p = pathlib.Path("/home/ellis/cfwrinkle/checkpoints/progressive/cross_scale_level4_cv/summary.json")
if not p.exists() or p.stat().st_size == 0:
    raise SystemExit(1)
try:
    d = json.loads(p.read_text())
except Exception:
    raise SystemExit(1)
folds = d if isinstance(d, list) else d.get("folds", [])
ok = isinstance(folds, list) and len(folds) == 5
print(f"fold_count={len(folds) if isinstance(folds, list) else -1}")
raise SystemExit(0 if ok else 1)
PY
}

while true; do
  if summary_is_valid; then
    echo "[fleet-b4b6] summary ready $(date -Iseconds)" | tee -a "$LOG"
    break
  fi

  if ! pgrep -A -f "training.train --model-type cross-scale --all-folds" >/dev/null; then
    echo "[fleet-b4b6] B3 exited before valid summary $(date -Iseconds)" | tee -a "$LOG"
    exit 2
  fi

  tail -n 3 "$RUN_DIR/run.log" 2>/dev/null | sed 's/^/[b3] /' | tee -a "$LOG" || true
  sleep 300
done

cd "$REPO_DIR"
echo "[fleet-b4b6] running B4 gate $(date -Iseconds)" | tee -a "$LOG"
python -m training.gate_check \
  --level 4 \
  --run-dir "$RUN_DIR" \
  --report-path reports/wp8_gate_level4.json | tee -a "$LOG"

echo "[fleet-b4b6] validating A/B level4 gates $(date -Iseconds)" | tee -a "$LOG"
python3 - <<'PY' | tee -a "$LOG"
import json
a = json.load(open("reports/wp7_gate_level4.json", "r", encoding="utf-8"))
b = json.load(open("reports/wp8_gate_level4.json", "r", encoding="utf-8"))
assert a["passed"], "Track A Level 4 not passed"
assert b["passed"], "Track B Level 4 not passed"
print("Both Level 4 gates: PASS")
PY

echo "[fleet-b4b6] running B5 compare $(date -Iseconds)" | tee -a "$LOG"
python -m training.cross_test_harness \
  --model-a-dir /home/ellis/cfwrinkle/checkpoints/progressive/level4_cv \
  --model-b-dir "$RUN_DIR" \
  --report-path reports/cross_model_comparison.json | tee -a "$LOG"

echo "[fleet-b4b6] running B6 release check $(date -Iseconds)" | tee -a "$LOG"
python -m training.release_check \
  --run-dir "$RUN_DIR" \
  --report-path reports/release_check_trackb.json | tee -a "$LOG"

echo "[fleet-b4b6] COMPLETE $(date -Iseconds)" | tee -a "$LOG"

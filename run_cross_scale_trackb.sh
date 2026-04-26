#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FROM_LEVEL=2
TO_LEVEL=4
FORCE_RESUME=0

usage() {
  cat <<EOF
Usage: ./run_cross_scale_trackb.sh [options]

Options:
  --from-level N   Start level (2-4). Default: 2
  --to-level N     End level (2-4). Default: 4
  --resume         Pass --resume to each level script
  -h, --help       Show help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --from-level) FROM_LEVEL="$2"; shift 2 ;;
    --to-level) TO_LEVEL="$2"; shift 2 ;;
    --resume) FORCE_RESUME=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: Unknown option: $1"; usage; exit 1 ;;
  esac
done

if [[ ! "$FROM_LEVEL" =~ ^[2-4]$ || ! "$TO_LEVEL" =~ ^[2-4]$ ]]; then
  echo "ERROR: levels must be integers in [2,4]."
  exit 1
fi
if (( FROM_LEVEL > TO_LEVEL )); then
  echo "ERROR: --from-level must be <= --to-level."
  exit 1
fi

run_level() {
  local level="$1"
  local report_path="$REPO_DIR/reports/wp7_gate_level${level}_cross_scale.json"
  local script="$REPO_DIR/run_cross_scale_level${level}.sh"
  local args=()
  if (( FORCE_RESUME == 1 )); then
    args+=(--resume)
  fi
  echo
  echo "=== Running Track B Level ${level} ==="
  "$script" "${args[@]}"
  python - <<'PY' "$report_path" "$level"
import json
import pathlib
import sys

report_path = pathlib.Path(sys.argv[1])
level = sys.argv[2]
if not report_path.exists():
    raise SystemExit(f"Level {level}: gate report missing: {report_path}")
with report_path.open("r", encoding="utf-8") as f:
    report = json.load(f)
if not report.get("passed", False):
    raise SystemExit(f"Level {level}: gate failed. Report: {report_path}")
print(f"Level {level}: gate PASS ({report_path})")
PY
}

for level in 2 3 4; do
  if (( level < FROM_LEVEL || level > TO_LEVEL )); then
    continue
  fi
  run_level "$level"
done

echo
echo "Track B sequence complete for levels ${FROM_LEVEL}-${TO_LEVEL}."

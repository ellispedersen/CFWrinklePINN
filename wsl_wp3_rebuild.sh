#!/usr/bin/env bash
set -euo pipefail

if ! grep -qi microsoft /proc/version 2>/dev/null; then
  echo "ERROR: This script is intended for WSL2."
  exit 1
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${VENV_PATH:-/home/ellis/venvs/cfwrinkle}"
WSL_DATA_DIR="${WSL_DATA_DIR:-/home/ellis/cfwrinkle/data}"
WP2_H5_PATH="${WP2_H5:-$WSL_DATA_DIR/cfwrinkle_dataset.h5}"
WP3_H5_PATH="${WP3_H5:-$WSL_DATA_DIR/cfwrinkle_wp3_features.h5}"
LOG_DIR="${LOG_DIR:-/home/ellis/cfwrinkle/logs/wp3_rebuild}"
MIN_FREE_GB="${MIN_FREE_GB:-140}"
PRECHECK_ONLY=0

usage() {
  cat <<'EOF'
Usage: ./wsl_wp3_rebuild.sh [options]

Rebuild WP3 with --include-fine-features, with durable logging + pre/post checks.

Options:
  --preflight-only        Run preflight checks only (no rebuild).
  --min-free-gb N         Minimum free disk space in WP3 output filesystem (default: 140).
  --wp2 PATH              Override WP2 HDF5 path (default: $WP2_H5 or /home/ellis/cfwrinkle/data/cfwrinkle_dataset.h5).
  --wp3 PATH              Override WP3 HDF5 output path (default: $WP3_H5 or /home/ellis/cfwrinkle/data/cfwrinkle_wp3_features.h5).
  --log-dir PATH          Durable log directory (default: /home/ellis/cfwrinkle/logs/wp3_rebuild).
  -h, --help              Show help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --preflight-only)
      PRECHECK_ONLY=1
      shift
      ;;
    --min-free-gb)
      MIN_FREE_GB="$2"
      shift 2
      ;;
    --wp2)
      WP2_H5_PATH="$2"
      shift 2
      ;;
    --wp3)
      WP3_H5_PATH="$2"
      shift 2
      ;;
    --log-dir)
      LOG_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ ! -f "$VENV_PATH/bin/activate" ]]; then
  echo "ERROR: Venv not found at $VENV_PATH"
  echo "Run ./wsl_setup_env.sh first."
  exit 1
fi

mkdir -p "$LOG_DIR"
RUN_TS="$(date +%Y%m%d-%H%M%S)"
LOG_FILE="$LOG_DIR/wp3_include_fine_${RUN_TS}.log"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "WP3 include-fine rebuild run started: $(date -Is)"
echo "Repo dir:   $REPO_DIR"
echo "WP2_H5:     $WP2_H5_PATH"
echo "WP3_H5:     $WP3_H5_PATH"
echo "Log file:   $LOG_FILE"
echo "Min free:   ${MIN_FREE_GB} GB"

source "$VENV_PATH/bin/activate"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR"
export WP2_H5="$WP2_H5_PATH"
export WP3_H5="$WP3_H5_PATH"

echo
echo "=== Step 1: preflight checks ==="
python -m wp3_features.rebuild_checks preflight \
  --wp2-path "$WP2_H5_PATH" \
  --wp3-path "$WP3_H5_PATH" \
  --min-free-gb "$MIN_FREE_GB" \
  --require-fine

if [[ "$PRECHECK_ONLY" -eq 1 ]]; then
  echo "Preflight-only mode complete."
  exit 0
fi

echo
echo "=== Step 2: rebuild WP3 with include-fine-features ==="
PYTHONUNBUFFERED=1 python -u -m wp3_features.build_features --include-fine-features

echo
echo "=== Step 3: baseline WP3 validation ==="
python -m wp3_features.validate_features --path "$WP3_H5_PATH" --verbose

echo
echo "=== Step 4: include-fine post-build checks ==="
python -m wp3_features.rebuild_checks postbuild \
  --wp2-path "$WP2_H5_PATH" \
  --wp3-path "$WP3_H5_PATH" \
  --require-fine

echo
echo "WP3 include-fine rebuild complete: $(date -Is)"
echo "Durable log: $LOG_FILE"

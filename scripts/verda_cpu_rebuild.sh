#!/usr/bin/env bash
# Verda CPU instance — WP3 rebuild.
#
# Run on any Ubuntu 24.04 CPU instance in FIN-03 with the cfwrinkle-data
# volume already attached and mounted via the Verda UI.
# No GPU, no Docker, no CUDA required.
#
# Steps:
#   1  Install Python deps (no torch — pure h5py/numpy/scipy)
#   2  Clone repo from GitHub
#   3  Wait for WP2 HDF5 upload via rsync from WSL (exits with instructions if missing)
#   4  Sanity-check WP2 (sim count + fine mesh presence)
#   5  WP3 preflight check
#   6  WP3 rebuild + postbuild + semantic validation
#
# Usage:
#   Paste as Verda startup script, or SSH in and run directly.
#   Only edit the two lines in the USER CONFIG block below.
#
set -euo pipefail

# ── USER CONFIG — edit these two lines only ───────────────────────────────────
GITHUB_TOKEN="ghp_REPLACE_ME"   # GitHub PAT with repo scope
MOUNT_POINT="/mnt/data"          # match the mount path set in the Verda UI
# ─────────────────────────────────────────────────────────────────────────────

GITHUB_REPO="https://github.com/ellispedersen/CFWrinklePINN"
WP2_H5="$MOUNT_POINT/cfwrinkle_dataset.h5"
WP3_H5="$MOUNT_POINT/cfwrinkle_wp3_features.h5"
REPO_DIR="$MOUNT_POINT/repo"
VENV_DIR="$MOUNT_POINT/rebuild_venv"
LOG_DIR="$MOUNT_POINT/logs"
MIN_FREE_GB="${MIN_FREE_GB:-110}"

echo "=== Verda CPU rebuild ==="
echo "Mount:   $MOUNT_POINT"
echo "WP2_H5:  $WP2_H5"
echo "WP3_H5:  $WP3_H5"
echo "Repo:    $REPO_DIR"
echo ""

# Verify the volume is actually mounted before doing anything.
mountpoint -q "$MOUNT_POINT" || {
  echo "ERROR: $MOUNT_POINT is not a mountpoint."
  echo "  Attach and mount the cfwrinkle-data volume via the Verda UI first."
  exit 1
}
mkdir -p "$LOG_DIR" "$MOUNT_POINT/checkpoints" "$MOUNT_POINT/.triton_cache"

# ── [1/6] Install deps ────────────────────────────────────────────────────────
echo "[1/6] Installing Python deps..."
apt-get update -qq
apt-get install -y --no-install-recommends python3-pip python3-venv git curl

if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
  python3 -m venv "$VENV_DIR"
  echo "  Created venv at $VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
pip install -q --no-cache-dir "h5py>=3.8" "numpy>=1.24" "scipy>=1.10" "tqdm>=4.65"
echo "  Deps installed."
echo ""

# ── [2/6] Clone repo ──────────────────────────────────────────────────────────
echo "[2/6] Cloning repo..."

_CLONE_URL="${GITHUB_REPO/https:\/\//https:\/\/$GITHUB_TOKEN@}"

if [[ -d "$REPO_DIR/.git" ]]; then
  echo "  Repo already cloned — pulling latest."
  git -C "$REPO_DIR" pull --ff-only
else
  git clone "$_CLONE_URL" "$REPO_DIR"
fi
unset _CLONE_URL

export PYTHONPATH="$REPO_DIR"

[[ -f "$REPO_DIR/reports/input_param_registry.json" ]] || {
  echo "ERROR: reports/input_param_registry.json not found in cloned repo."
  exit 1
}
echo "  Repo ready."
echo ""

# ── [3/6] WP2 HDF5 — upload via rsync from WSL before re-running ─────────────
echo "[3/6] WP2 HDF5..."
if [[ -f "$WP2_H5" ]]; then
  WP2_SIZE=$(du -sh "$WP2_H5" | cut -f1)
  echo "  Present: $WP2_H5 ($WP2_SIZE)"
else
  echo ""
  echo "  WP2 not found at $WP2_H5"
  echo "  Steps 1-2 are complete. Upload WP2 from WSL then re-run this script:"
  echo ""
  echo "    wsl -d Ubuntu-24.04 -- rsync -avz --progress \\"
  echo "      /home/ellis/cfwrinkle/data/cfwrinkle_dataset.h5 \\"
  echo "      root@<VERDA_IP>:$WP2_H5"
  echo ""
  exit 1
fi
echo ""

# ── [4/6] Sanity-check WP2 ───────────────────────────────────────────────────
echo "[4/6] Sanity-checking WP2..."
python3 - <<PYCHECK
import h5py, sys

with h5py.File("$WP2_H5", "r") as f:
    if "simulations" not in f:
        print("ERROR: 'simulations' group missing from WP2 HDF5")
        sys.exit(1)
    sims = list(f["simulations"].keys())
    if len(sims) < 60:
        print(f"ERROR: expected ~66 simulations, found {len(sims)}")
        sys.exit(1)
    s0 = f[f"simulations/{sims[0]}"]
    if "mesh/fine/nodes" not in s0:
        print(f"ERROR: mesh/fine/nodes missing from {sims[0]} — WP2 predates fine mesh")
        print("       Ensure the WSL copy at /home/ellis/cfwrinkle/data/ includes fine mesh data.")
        sys.exit(1)
    fn_shape = s0["mesh/fine/nodes"].shape
    if fn_shape[0] < 10000:
        print(f"ERROR: fine nodes suspiciously small: {fn_shape} — check WP2 version")
        sys.exit(1)
    print(f"OK: {len(sims)} simulations, sample fine nodes={fn_shape}")
PYCHECK
echo ""

# ── [5/6] Preflight check ────────────────────────────────────────────────────
echo "[5/6] WP3 preflight check..."
cd "$REPO_DIR"
export WP2_H5 WP3_H5
python -m wp3_features.rebuild_checks preflight \
  --wp2-path "$WP2_H5" \
  --wp3-path "$WP3_H5" \
  --min-free-gb "$MIN_FREE_GB" \
  --require-fine
echo ""

# ── [6/6] Rebuild + validate ─────────────────────────────────────────────────
LOG_FILE="$LOG_DIR/wp3_rebuild_$(date +%Y%m%d_%H%M%S).log"
echo "[6/6] Rebuilding WP3 (3-4 hr) — log: $LOG_FILE"
python -u -m wp3_features.build_features --include-fine-features \
  2>&1 | tee "$LOG_FILE"

echo ""
echo "--- Postbuild schema check ---"
python -m wp3_features.rebuild_checks postbuild \
  --wp2-path "$WP2_H5" \
  --wp3-path "$WP3_H5" \
  --require-fine

echo ""
echo "--- Semantic validation (7 checks) ---"
python -m wp3_features.validate_features --verbose

echo ""
echo "--- Simulation count ---"
python3 - <<SIMCOUNT
import h5py
with h5py.File("$WP3_H5", "r") as f:
    n = len(f["simulations"])
    print(f"WP3 built: {n} simulations")
    assert n >= 60, f"Expected >= 60, got {n}"
SIMCOUNT

echo ""
echo "=== WP3 rebuild complete ==="
echo ""
echo "Next: detach cfwrinkle-data volume in Verda console, terminate this instance."
echo "      Then spin up RTX Pro 6000 spot instance and attach the same volume."

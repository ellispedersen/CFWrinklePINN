#!/usr/bin/env bash
# Verda CPU instance — WP3 rebuild.
#
# Run on any Ubuntu 24.04 CPU instance in FIN-03 with the cfwrinkle-data
# NVMe block volume attached.  No GPU, no Docker, no CUDA required.
#
# Steps:
#   1  Mount NVMe volume (formats on first use, mounts on subsequent uses)
#   2  Install Python deps (no torch — pure h5py/numpy/scipy)
#   3  Clone repo from GitHub
#   4  Download WP2 HDF5 from SharePoint (rclone or curl — see [4] below)
#   5  Sanity-check WP2 (sim count + fine mesh presence)
#   6  WP3 preflight check
#   7  WP3 rebuild + postbuild + semantic validation
#
# After success: optionally back up WP3 to SharePoint, detach volume, terminate.
#
# Usage:
#   As Verda startup script OR interactively:
#     GITHUB_REPO=https://github.com/<user>/CFWrinklePINN bash verda_cpu_rebuild.sh
#
set -euo pipefail

GITHUB_REPO="${GITHUB_REPO:-https://github.com/ellispedersen/CFWrinklePINN}"
VOLUME_DEVICE="${VOLUME_DEVICE:-}"          # auto-detected if empty
MOUNT_POINT="${MOUNT_POINT:-/mnt/data}"
WP2_H5="${WP2_H5:-$MOUNT_POINT/cfwrinkle_dataset.h5}"
WP3_H5="${WP3_H5:-$MOUNT_POINT/cfwrinkle_wp3_features.h5}"
REPO_DIR="${REPO_DIR:-$MOUNT_POINT/repo}"
VENV_DIR="${VENV_DIR:-$MOUNT_POINT/rebuild_venv}"
LOG_DIR="${LOG_DIR:-$MOUNT_POINT/logs}"
MIN_FREE_GB="${MIN_FREE_GB:-110}"

echo "=== Verda CPU rebuild ==="
echo "Mount:   $MOUNT_POINT"
echo "WP2_H5:  $WP2_H5"
echo "WP3_H5:  $WP3_H5"
echo "Repo:    $REPO_DIR"
echo ""

# ── [1/7] Mount volume ────────────────────────────────────────────────────────
echo "[1/7] Mounting NVMe volume..."

if [[ -z "$VOLUME_DEVICE" ]]; then
  # Find the attached block device that is NOT the OS disk.
  # OS is typically /dev/vda or /dev/sda; data volume is /dev/vdb or /dev/sdb.
  for dev in /dev/vdb /dev/sdb /dev/xvdb; do
    if [[ -b "$dev" ]]; then
      VOLUME_DEVICE="$dev"
      break
    fi
  done
fi

[[ -n "$VOLUME_DEVICE" ]] || { echo "ERROR: could not detect volume device. Set VOLUME_DEVICE="; exit 1; }
echo "  Device: $VOLUME_DEVICE"

# Format only if no filesystem exists (first use).
if ! blkid "$VOLUME_DEVICE" &>/dev/null; then
  echo "  First use — formatting $VOLUME_DEVICE as ext4..."
  mkfs.ext4 -F "$VOLUME_DEVICE"
fi

mkdir -p "$MOUNT_POINT"
if ! mountpoint -q "$MOUNT_POINT"; then
  mount "$VOLUME_DEVICE" "$MOUNT_POINT"
  echo "  Mounted $VOLUME_DEVICE → $MOUNT_POINT"
else
  echo "  Already mounted."
fi

mkdir -p "$LOG_DIR" "$MOUNT_POINT/checkpoints" "$MOUNT_POINT/.triton_cache"
echo "  Volume ready."
echo ""

# ── [2/7] Install deps ────────────────────────────────────────────────────────
echo "[2/7] Installing Python deps..."
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

# ── [3/7] Clone repo ──────────────────────────────────────────────────────────
echo "[3/7] Cloning repo..."
if [[ -d "$REPO_DIR/.git" ]]; then
  echo "  Repo already cloned — pulling latest."
  git -C "$REPO_DIR" pull --ff-only
else
  git clone "$GITHUB_REPO" "$REPO_DIR"
fi

export PYTHONPATH="$REPO_DIR"

# Verify input_param_registry.json is present (required by build_features.py).
REG_PATH="$REPO_DIR/reports/input_param_registry.json"
[[ -f "$REG_PATH" ]] || {
  echo "ERROR: $REG_PATH not found."
  echo "  Fix: add !reports/input_param_registry.json to .gitignore, commit, push."
  exit 1
}
echo "  Repo ready. input_param_registry.json present."
echo ""

# ── [4/7] Download WP2 from SharePoint ───────────────────────────────────────
echo "[4/7] WP2 HDF5..."
if [[ -f "$WP2_H5" ]]; then
  WP2_SIZE=$(du -sh "$WP2_H5" | cut -f1)
  echo "  Already present: $WP2_H5 ($WP2_SIZE) — skipping download."
else
  echo "  WP2 not found. Download it now using one of:"
  echo ""
  echo "  Option A — rclone (OneDrive/SharePoint backend):"
  echo "    Install: curl https://rclone.org/install.sh | bash"
  echo "    Auth:    rclone config  (select 'onedrive', paste token from local machine)"
  echo "    Copy:    rclone copy 'onedrive:CFWrinkle/cfwrinkle_dataset.h5' $MOUNT_POINT/ --progress"
  echo ""
  echo "  Option B — SharePoint sharing link (Anyone with link → Can download):"
  echo "    curl -L -o $WP2_H5 'https://univ.sharepoint.com/:u:/s/...'"
  echo ""
  echo "  After download, re-run this script (steps 1-3 will be skipped)."
  exit 1
fi
echo ""

# ── [5/7] Sanity-check WP2 ───────────────────────────────────────────────────
echo "[5/7] Sanity-checking WP2..."
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
        print(f"ERROR: mesh/fine/nodes missing from {sims[0]} — this WP2 predates fine mesh")
        print("       The SharePoint copy must include fine mesh data.")
        sys.exit(1)
    fn_shape = s0["mesh/fine/nodes"].shape
    if fn_shape[0] < 10000:
        print(f"ERROR: fine nodes suspiciously small: {fn_shape} — check WP2 version")
        sys.exit(1)
    print(f"OK: {len(sims)} simulations, sample fine nodes={fn_shape}")
PYCHECK
echo ""

# ── [6/7] Preflight check ────────────────────────────────────────────────────
echo "[6/7] WP3 preflight check..."
cd "$REPO_DIR"
export WP2_H5 WP3_H5
python -m wp3_features.rebuild_checks preflight \
  --wp2-path "$WP2_H5" \
  --wp3-path "$WP3_H5" \
  --min-free-gb "$MIN_FREE_GB" \
  --require-fine
echo ""

# ── [7/7] Rebuild + validate ─────────────────────────────────────────────────
LOG_FILE="$LOG_DIR/wp3_rebuild_$(date +%Y%m%d_%H%M%S).log"
echo "[7/7] Rebuilding WP3 (3–4 hr) — log: $LOG_FILE"
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
echo "Optional: back up WP3 to SharePoint before terminating:"
echo "  rclone copy $WP3_H5 'onedrive:CFWrinkle/' --progress"
echo ""
echo "Next: detach cfwrinkle-data volume in Verda console, terminate this instance."
echo "      Then spin up RTX Pro 6000 spot instance and attach the same volume."

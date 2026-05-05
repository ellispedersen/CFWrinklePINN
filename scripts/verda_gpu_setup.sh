#!/usr/bin/env bash
# Verda GPU instance setup — RTX Pro 6000 / B200 / B300 Blackwell.
#
# Runs on ubuntu-24.04-cuda-13.0-open-docker after attaching the cfwrinkle-data
# NVMe volume (already formatted and populated by verda_cpu_rebuild.sh).
#
# Steps:
#   1  Mount NVMe volume (already formatted — no mkfs)
#   2  Pull training Docker image from Verda container registry
#   3  GPU smoke test (inside container): sm_122, SDPA Flash, BF16 matmul timing
#   4  Verify both HDF5 files present and readable
#
# After success, run: bash /mnt/data/repo/scripts/verda_launch_training.sh
#
set -euo pipefail

VOLUME_DEVICE="${VOLUME_DEVICE:-}"
MOUNT_POINT="${MOUNT_POINT:-/mnt/data}"
IMAGE="${IMAGE:-vccr.io/20175b95-1ac5-4808-89b0-b08dc612c71e/cfwrinkle-train:pt2110}"
REGISTRY_USER="${REGISTRY_USER:-vcr-20175b95-1ac5-4808-89b0-b08dc612c71e+erjp-cred-1}"
REGISTRY_SECRET="${REGISTRY_SECRET:-fTSvVgcig4eA8T7S7TEFSc8nP3d3RnGn}"

echo "=== Verda GPU setup (RTX Pro 6000 / B200 / B300 Blackwell) ==="
echo "Image:  $IMAGE"
echo "Mount:  $MOUNT_POINT"
echo ""

# ── [1/4] Mount volume ────────────────────────────────────────────────────────
echo "[1/4] Mounting NVMe volume..."

if [[ -z "$VOLUME_DEVICE" ]]; then
  for dev in /dev/nvme1n1 /dev/nvme0n1 /dev/vdb /dev/sdb /dev/xvdb; do
    if [[ -b "$dev" ]]; then
      VOLUME_DEVICE="$dev"
      break
    fi
  done
fi
[[ -n "$VOLUME_DEVICE" ]] || { echo "ERROR: volume device not found. Set VOLUME_DEVICE="; exit 1; }

# Volume is already formatted from the CPU rebuild phase — do not mkfs.
mkdir -p "$MOUNT_POINT"
if ! mountpoint -q "$MOUNT_POINT"; then
  mount "$VOLUME_DEVICE" "$MOUNT_POINT"
  echo "  Mounted $VOLUME_DEVICE → $MOUNT_POINT"
else
  echo "  Already mounted."
fi

# Quick data presence check before spending time pulling Docker image.
WP2="$MOUNT_POINT/cfwrinkle_dataset.h5"
WP3="$MOUNT_POINT/cfwrinkle_wp3_features.h5"
[[ -f "$WP2" ]] || { echo "ERROR: WP2 not found at $WP2 — run verda_cpu_rebuild.sh first"; exit 1; }
[[ -f "$WP3" ]] || { echo "ERROR: WP3 not found at $WP3 — run verda_cpu_rebuild.sh first"; exit 1; }
WP2_SIZE=$(du -sh "$WP2" | cut -f1)
WP3_SIZE=$(du -sh "$WP3" | cut -f1)
echo "  WP2: $WP2 ($WP2_SIZE)"
echo "  WP3: $WP3 ($WP3_SIZE)"
mkdir -p "$MOUNT_POINT/logs" "$MOUNT_POINT/checkpoints" "$MOUNT_POINT/.triton_cache"
echo ""

# ── [2/4] Pull Docker image ───────────────────────────────────────────────────
echo "[2/4] Pulling Docker image..."
if [[ -n "$REGISTRY_SECRET" ]]; then
  echo "$REGISTRY_SECRET" | docker login "$(echo "$IMAGE" | cut -d/ -f1)" \
    -u "$REGISTRY_USER" --password-stdin
else
  echo "  REGISTRY_SECRET not set — attempting pull without explicit login."
  echo "  If this fails: set REGISTRY_SECRET=<your-vcr-secret> and re-run."
fi
docker pull "$IMAGE"
echo "  Image ready."
echo ""

# ── [3/4] GPU smoke test ─────────────────────────────────────────────────────
echo "[3/4] GPU smoke test (inside container)..."
docker run --rm --gpus all "$IMAGE" python3 - <<'SMOKE'
import torch, time, sys

assert torch.cuda.is_available(), "CUDA not available"

n_gpus = torch.cuda.device_count()
print(f"  GPU count: {n_gpus}")
if n_gpus >= 2:
    print(f"  Dual-GPU mode will be active: folds 0,2,4 → GPU 0 | folds 1,3 → GPU 1")

# Test every visible GPU.
for gpu_idx in range(n_gpus):
    dev = torch.device(f"cuda:{gpu_idx}")
    props = torch.cuda.get_device_properties(gpu_idx)
    sm_maj, sm_min = torch.cuda.get_device_capability(gpu_idx)

    print(f"\n  --- GPU {gpu_idx} ---")
    vram_gb = props.total_memory / 1024**3
    # Tier by VRAM: B300 ≥ 200 GB; B200 ≥ 160 GB; RTX Pro 6000 ≥ 80 GB GDDR7.
    if vram_gb >= 200:
        gpu_tier = "B300"
        matmul_threshold_ms = 3.0
    elif vram_gb >= 160:
        gpu_tier = "B200"
        matmul_threshold_ms = 3.0
    elif vram_gb >= 80:
        gpu_tier = "RTX-Pro-6000"
        matmul_threshold_ms = 5.0
    else:
        gpu_tier = "other"
        matmul_threshold_ms = 20.0

    print(f"  Name:      {props.name}")
    print(f"  VRAM:      {vram_gb:.1f} GB  [{gpu_tier}]")
    print(f"  SMs:       {props.multi_processor_count}")
    print(f"  Capability: sm_{sm_maj}{sm_min}", end="")
    # sm_122 = RTX Pro 6000 Blackwell (GB202, CC 12.2) — primary target.
    # sm_10x = B300/B200/B100 data-centre Blackwell — supported fallback.
    if sm_maj == 12:
        print("  ← Blackwell GB202 (RTX Pro 6000) ✓")
    elif sm_maj == 10:
        print("  ← Blackwell data-centre (B300/B200/B100) ✓")
    elif sm_maj == 9:
        print("  ← Hopper (H100/H200)")
    else:
        print("  ← unexpected architecture")
        print(f"  WARNING: expected sm_12x (RTX Pro 6000) or sm_10x (B300). Verify instance type.")

    assert torch.cuda.is_bf16_supported(), f"GPU {gpu_idx}: bfloat16 not supported"

    cuda_ver = torch.version.cuda or "unknown"
    print(f"  CUDA:      {cuda_ver}")
    print(f"  PyTorch:   {torch.__version__}")

    # SDPA — PyTorch dispatches FA2-compatible kernels on sm_10x and sm_12x Blackwell.
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            q = torch.randn(1, 8, 512, 64, device=dev, dtype=torch.bfloat16)
            torch.nn.functional.scaled_dot_product_attention(q, q, q)
        print("  SDPA Flash: available ✓")
    except Exception as e:
        print(f"  SDPA Flash: {e} (will fall back to efficient/math kernel)")

    # BF16 matmul timing — RTX Pro 6000 < 5 ms; B200/B300 < 3 ms.
    a = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16)
    torch.cuda.synchronize(dev)
    t0 = time.time()
    for _ in range(10):
        _ = a @ a.T
    torch.cuda.synchronize(dev)
    elapsed_ms = (time.time() - t0) / 10 * 1000
    print(f"  BF16 matmul 4096×4096: {elapsed_ms:.1f} ms/call", end="")
    if elapsed_ms < matmul_threshold_ms:
        print("  ✓")
    else:
        print(f"  (slower than expected for {gpu_tier} — target < {matmul_threshold_ms:.0f} ms)")

print("")
print(f"  SMOKE TEST PASSED ({n_gpus} GPU{'s' if n_gpus > 1 else ''})")
SMOKE

echo ""

# ── [4/4] HDF5 readability check ─────────────────────────────────────────────
echo "[4/4] Verifying HDF5 files are readable..."
docker run --rm --gpus all \
  -v "$MOUNT_POINT:/mnt/data:ro" \
  "$IMAGE" python3 - <<'HDFCHECK'
import h5py, sys

for path, label in [
    ("/mnt/data/cfwrinkle_dataset.h5",       "WP2"),
    ("/mnt/data/cfwrinkle_wp3_features.h5",  "WP3"),
]:
    with h5py.File(path, "r") as f:
        n = len(f["simulations"])
        assert n >= 60, f"{label}: expected >= 60 sims, got {n}"
        print(f"  {label}: {n} simulations ✓")
HDFCHECK

echo ""
echo "=== Setup complete ==="
echo ""
echo "Run training with:"
echo "  bash /mnt/data/repo/scripts/verda_launch_training.sh"
echo ""
echo "On eviction: re-attach cfwrinkle-data volume, re-run this script,"
echo "  then re-run verda_launch_training.sh — AUTO_RESUME will pick up latest.pt."

# ROCm/PyTorch Optimization Handoff (2026-04-18)

## Objective and hard constraints
- Keep **Track B quality-critical settings fixed**:
  - `MAX_TIMESTEPS >= 96`
  - `--normalize-fine-features` ON
- Find memory/perf optimizations that avoid unacceptable accuracy loss.

## What was changed in code
1. `model/loss.py`
   - Added chunked fine-supervision path to avoid materializing large fine target gathers:
     - `CFWRINKLE_FINE_LOSS_CHUNK_ELEMS` (defaulted and wired at runtime).
   - Added optional auxiliary fine-loss throttling/disable knobs:
     - `CFWRINKLE_AUX_LOSS_INTERVAL`
     - `CFWRINKLE_DISABLE_FINE_COHERENCE`
     - `CFWRINKLE_DISABLE_FINE_COUPLING`
     - `CFWRINKLE_DISABLE_FINE_BUCKLING`
     - `CFWRINKLE_DISABLE_FINE_DZ_MONO`
2. `training/train.py`
   - Passes per-step index into `cross_scale_loss(...)` so interval-based aux-loss gating is deterministic.
3. `run_cross_scale_level4.sh`
   - Exposes/logs the new env knobs.
   - Fixed allocator-cache logging to be safe with `set -u`.
4. `scripts/fleet_b3_watchdog.sh`
   - Launch profile updated during tuning (memory-conservative settings).

## Trial outcomes (high signal only)
- Pre-change and early profiles repeatedly failed with HIP allocator/OOM (`HSA ... alloc failed`, `torch.OutOfMemoryError`) at required settings.
- After loss chunking + aux throttling, short probes became stable:
  - Probe A (`attn=96`, `chunk_t=12`, fine-loss chunking + aux throttling): completed 1 epoch on small subset.
  - Probe B (`attn=64`, `chunk_t=8`, stronger aux reduction): completed 1 epoch on larger subset (`train=10`, `val=3`).
- Full all-fold B3 at required settings remained unstable in long-running attempts:
  - recurrent allocator failures and stalls (long periods with no epoch completion).

## Why trials were stopped
- Per user direction, active trials were stopped to hand off further exploration to Claude.

## Recommended next exploration for Claude
1. **Loss-path memory accounting**
   - Add lightweight per-step memory telemetry around forward/loss/backward to isolate exact spike stage and tensor family.
2. **Fine supervision sampling strategy (accuracy-preserving)**
   - Investigate deterministic per-step fine-element sampling with full coverage over epoch (instead of disabling many aux terms).
3. **Allocator strategy deep-dive (ROCm-specific)**
   - Systematically benchmark combinations of:
     - `PYTORCH_HIP_ALLOC_CONF` split/GC values
     - `PYTORCH_NO_CUDA_MEMORY_CACHING`
     - hipBLAS workspace bounds
   - Measure both throughput and stall/OOM incidence.
4. **Aux-loss schedule redesign**
   - Replace global disable flags with warmup/curriculum scheduling so full objective is restored after stability phase.
5. **WSL/driver interaction checks**
   - Correlate stalls with ROCm/WSL runtime signals (`dmesg`, `dxgkio_make_resident` style failures) to separate model issues from runtime residency failures.

## Current state at handoff
- No active Track B training trials.
- Codebase contains the above memory-control hooks and is ready for deeper ROCm strategy exploration.

"""Run inspector / diagnostics for CFWrinklePINN training outputs.

Usage:
    python -m training.inspect_run --run-dir checkpoints/progressive/level3_mini
    python -m training.inspect_run --run-dir /home/ellis/cfwrinkle/checkpoints/progressive/level3_mini --verbose

Reads history.json, best.pt/latest.pt, and any gate reports in reports/.
Prints a structured diagnostic summary and exits 0 (pass) or 1 (anomaly detected).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> list | dict | None:
    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"  WARNING: could not read {path}: {e}")
        return None


def _load_checkpoint_meta(path: Path) -> dict | None:
    """Load only the non-tensor metadata from a checkpoint without loading model weights."""
    if not path.exists():
        return None
    try:
        import torch
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        return {
            "epoch": ckpt.get("epoch"),
            "val_loss": ckpt.get("val_loss"),
            "val_metrics": ckpt.get("val_metrics", {}),
            "model_config": ckpt.get("model_config", {}),
            "model_type": ckpt.get("model_type", "coarse"),
            "model_version": ckpt.get("model_version"),
            "checkpoint_format_version": ckpt.get("checkpoint_format_version"),
        }
    except Exception as e:
        print(f"  WARNING: could not read {path.name}: {e}")
        return None


def _field(history: list[dict], key: str) -> list[float]:
    """Extract a numeric field from history, skipping NaN/None entries."""
    values = []
    for row in history:
        v = row.get(key)
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            values.append(float(v))
    return values


# ---------------------------------------------------------------------------
# Anomaly checks
# ---------------------------------------------------------------------------

def _check_nan_inf(history: list[dict]) -> list[str]:
    issues = []
    loss_keys = ["loss/total", "loss/severity", "loss/comp_frac",
                 "loss/oop", "loss/thick_var", "loss/physics", "val_loss"]
    for row in history:
        epoch = row.get("epoch", "?")
        for key in loss_keys:
            v = row.get(key)
            if v is None:
                continue
            if math.isnan(v):
                issues.append(f"NaN in {key} at epoch {epoch}")
            elif math.isinf(v):
                issues.append(f"Inf in {key} at epoch {epoch}")
    return issues


def _check_flat_loss(history: list[dict], window: int = 10) -> str | None:
    vals = _field(history, "loss/total")
    if len(vals) < window:
        return None
    tail = vals[-window:]
    mean = sum(tail) / len(tail)
    variance = sum((v - mean) ** 2 for v in tail) / len(tail)
    if variance < 1e-6 and mean > 0.01:
        return (f"Loss/total variance={variance:.2e} over last {window} epochs "
                f"(mean={mean:.4f}) — model may be stuck")
    return None


def _check_divergence(history: list[dict], window: int = 5) -> str | None:
    vals = _field(history, "loss/total")
    if len(vals) < window + 1:
        return None
    recent = vals[-(window + 1):]
    # Check if monotonically increasing
    if all(recent[i] < recent[i + 1] for i in range(len(recent) - 1)):
        return (f"Loss/total increased for {window} consecutive epochs "
                f"({recent[0]:.4f} → {recent[-1]:.4f}) — possible divergence")
    return None


def _check_val_loss_flat(history: list[dict]) -> str | None:
    """Detect the val_loss=7.5 self-normalisation bug (all val_loss values identical)."""
    vals = _field(history, "val_loss")
    if len(vals) < 3:
        return None
    if len(set(round(v, 4) for v in vals)) == 1:
        return (f"val_loss is constant at {vals[0]:.4f} across all epochs — "
                "likely using grad_total instead of raw_total for early stopping")
    return None


# ---------------------------------------------------------------------------
# Gate report lookup
# ---------------------------------------------------------------------------

def _find_gate_report(run_dir: Path, repo_root: Path) -> tuple[int | None, dict | None]:
    """Find the gate report most likely to correspond to this run directory."""
    reports_dir = repo_root / "reports"
    if not reports_dir.exists():
        return None, None
    # Infer level from run dir name
    name = run_dir.name.lower()
    level = None
    if "level1" in name or "smoke" in name:
        level = 1
    elif "level2" in name or "overfit" in name:
        level = 2
    elif "level3" in name or "mini" in name:
        level = 3
    elif "level4" in name or "full_cv" in name or "cv" in name:
        level = 4

    if level is None:
        # Try all gate reports
        for lvl in [4, 3, 2, 1]:
            candidates = sorted(reports_dir.glob(f"wp7_gate_level{lvl}*.json"),
                                 key=lambda p: p.stat().st_mtime, reverse=True)
            if candidates:
                return lvl, _load_json(candidates[0])
        return None, None

    candidates = sorted(reports_dir.glob(f"wp7_gate_level{level}*.json"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        return level, _load_json(candidates[0])
    return level, None


# ---------------------------------------------------------------------------
# Main inspector
# ---------------------------------------------------------------------------

def inspect(run_dir: Path, verbose: bool = False) -> bool:
    """Inspect a training run directory. Returns True if no anomalies found."""
    print(f"\n{'='*60}")
    print(f"Run Inspector: {run_dir}")
    print(f"{'='*60}")

    history_path = run_dir / "history.json"
    best_path = run_dir / "best.pt"
    latest_path = run_dir / "latest.pt"

    history = _load_json(history_path)
    best_meta = _load_checkpoint_meta(best_path)
    latest_meta = _load_checkpoint_meta(latest_path)

    # -- Status --
    has_history = isinstance(history, list) and len(history) > 0
    has_checkpoint = best_path.exists() or latest_path.exists()

    print("\n[Status]")
    if not has_history and not has_checkpoint:
        print("  NOT STARTED — no history.json and no checkpoint found")
        return True
    elif not has_history and has_checkpoint:
        print("  CRASHED — checkpoint exists but no history.json (process likely killed mid-epoch)")
    elif has_history:
        n_epochs = len(history)
        last_epoch = history[-1].get("epoch", n_epochs)
        stopped_early = (n_epochs > 0 and
                         history[-1].get("epoch", 0) < history[-1].get("epoch", 0) + 1 and
                         n_epochs < last_epoch)
        # Check if early stopping message would apply
        print(f"  COMPLETE — {n_epochs} epochs recorded (last epoch={last_epoch})")

    # -- Checkpoint metadata --
    print("\n[Checkpoint]")
    for label, meta in [("best.pt", best_meta), ("latest.pt", latest_meta)]:
        if meta:
            print(f"  {label}: epoch={meta['epoch']}, val_loss={meta['val_loss']:.4f}, "
                  f"model_type={meta['model_type']}, "
                  f"model_version={meta['model_version']}, "
                  f"ckpt_format={meta['checkpoint_format_version']}, "
                  f"model_config={meta['model_config']}")
        else:
            if (run_dir / label).exists():
                print(f"  {label}: exists but could not be read")
            else:
                print(f"  {label}: not found")

    if not has_history:
        return not has_checkpoint  # crashed if checkpoint with no history

    # -- Training summary --
    print("\n[Training Summary]")
    val_losses = _field(history, "val_loss")
    train_losses = _field(history, "loss/total")
    detection_rates = _field(history, "detection_rate")
    f1s = _field(history, "f1")

    if val_losses:
        best_val = min(val_losses)
        best_val_epoch = history[val_losses.index(best_val)].get("epoch", "?")
        print(f"  Best val_loss: {best_val:.4f} at epoch {best_val_epoch}")
    if train_losses:
        print(f"  Train loss: {train_losses[0]:.4f} → {train_losses[-1]:.4f} "
              f"({'↓' if train_losses[-1] < train_losses[0] else '↑'})")
    if detection_rates:
        print(f"  Detection rate: max={max(detection_rates):.3f}, "
              f"final={detection_rates[-1]:.3f}")
    if f1s:
        print(f"  F1: max={max(f1s):.3f}, final={f1s[-1]:.3f}")
    if history:
        elapsed = sum(r.get("elapsed_s", 0) for r in history)
        print(f"  Total time: {elapsed/60:.1f} min ({elapsed/3600:.2f} h)")

    # VRAM tracking
    vram_vals = _field(history, "peak_vram_gb")
    if vram_vals:
        print(f"  Peak VRAM: max={max(vram_vals):.2f} GB, final={vram_vals[-1]:.2f} GB")

    if verbose:
        print("\n[Per-epoch detail]")
        for row in history:
            print(f"  epoch={row.get('epoch'):3d} | "
                  f"train={row.get('loss/total', float('nan')):.4f} | "
                  f"val={row.get('val_loss', float('nan')):.4f} | "
                  f"det={row.get('detection_rate', float('nan')):.3f} | "
                  f"f1={row.get('f1', float('nan')):.3f}")

    # -- Anomaly scan --
    print("\n[Anomaly Scan]")
    anomalies: list[str] = []

    nan_issues = _check_nan_inf(history)
    anomalies.extend(nan_issues)

    flat = _check_flat_loss(history)
    if flat:
        anomalies.append(flat)

    div = _check_divergence(history)
    if div:
        anomalies.append(div)

    val_flat = _check_val_loss_flat(history)
    if val_flat:
        anomalies.append(val_flat)

    if anomalies:
        for a in anomalies:
            print(f"  ⚠  {a}")
    else:
        print("  No anomalies detected")

    # -- Gate status --
    # Walk up from run_dir to find the repo root (contains training/ directory)
    repo_root = run_dir
    for _ in range(8):
        if (repo_root / "training").exists():
            break
        repo_root = repo_root.parent

    gate_level, gate_report = _find_gate_report(run_dir, repo_root)
    print("\n[Gate Status]")
    if gate_report is None:
        if gate_level:
            print(f"  Level {gate_level} gate report not found in reports/")
        else:
            print("  No gate report found")
    else:
        passed = gate_report.get("passed", False)
        status = "PASS" if passed else "FAIL"
        print(f"  Level {gate_level} gate: {status}")
        for check in gate_report.get("checks", []):
            mark = "✓" if check.get("passed") else "✗"
            print(f"    {mark} {check.get('name')}: {check.get('note', '')}")

    # -- Recommendations --
    print("\n[Recommendations]")
    recommendations: list[str] = []
    if not has_history and has_checkpoint:
        recommendations.append(
            "Run crashed before writing history. Use --resume to continue from last checkpoint."
        )
    if val_flat:
        recommendations.append(
            "val_loss is constant — check that evaluate_split() uses raw_total (not grad_total)."
        )
    if flat:
        recommendations.append(
            "Loss is flat — try a higher learning rate or check data loading."
        )
    if div:
        recommendations.append(
            "Loss is diverging — reduce learning rate or check for NaN in input features."
        )
    if detection_rates and max(detection_rates) == 0:
        recommendations.append(
            "Detection rate never exceeded 0 — check wrinkle threshold (default 0.15) "
            "and whether any training sims have wrinkled=True labels."
        )
    if vram_vals and max(vram_vals) > 18.0:
        recommendations.append(
            f"Peak VRAM {max(vram_vals):.1f} GB is close to card capacity (21.5 GB). "
            "Consider reducing max_timesteps or enabling gradient checkpointing."
        )

    if recommendations:
        for r in recommendations:
            print(f"  → {r}")
    else:
        print("  None")

    print()
    return len(anomalies) == 0


def main() -> None:
    p = argparse.ArgumentParser(description="Inspect a CFWrinklePINN training run directory")
    p.add_argument("--run-dir", type=Path, required=True,
                   help="Path to the training output directory (contains history.json, best.pt, ...)")
    p.add_argument("--verbose", action="store_true",
                   help="Print per-epoch detail table")
    args = p.parse_args()

    if not args.run_dir.exists():
        print(f"ERROR: run-dir does not exist: {args.run_dir}")
        sys.exit(1)

    ok = inspect(args.run_dir, verbose=args.verbose)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

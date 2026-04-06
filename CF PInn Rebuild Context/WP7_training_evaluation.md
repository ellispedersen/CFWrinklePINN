# WP7 — Training, Evaluation, Visualisation & Inference
## Progressive Test Runs, Dynamic Visualisations, Single-Sim Inference

**Depends on:** WP6 gate passed — unit tests green, overfit-on-1-sim confirmed
**Status:** Implement after WP6 gate

---

## Objective

Implement the full training pipeline, visualisation suite, and single-simulation inference. Proceed through a structured ladder of test runs — each with explicit human-verifiable checkpoints — before committing to a full cross-validation run.

**Production use case:** At inference time, the user provides a single AniForm results directory (coarse mesh only). The pipeline extracts features and produces wrinkle severity predictions on the coarse mesh.

---

## File Structure to Create

```
training/
  __init__.py
  train.py          — training loop, CV harness, checkpointing
  evaluate.py       — metrics, detection rate, per-fold summary
  visualize.py      — all dynamic visualisations
  infer.py          — single-sim inference pipeline

tests/
  test_training_smoke.py  — smoke test (1 sim, 1 epoch, no crash)
```

---

## 7.1 Training Loop

**File:** `training/train.py`

```python
"""
WP7 — Training loop and CV harness.

Usage:
  # Single fold (development)
  python -m training.train --fold 0 --epochs 100 --output checkpoints/fold_0/

  # All 5 folds (full run)
  python -m training.train --all-folds --epochs 100 --output checkpoints/

  # Overfit test (3 sims, many epochs)
  python -m training.train --sim-ids geom_0_2_pair1,geom_0_3_pair1,geom_0_4_pair1 \
      --epochs 200 --output checkpoints/overfit_test/
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.optim as optim

from model.dataset import WrinkleDataset, load_fold_sim_ids
from model.gnn import FormingGraphNet
from model.loss import wrinkle_loss


WP3_H5 = Path("data/cfwrinkle_wp3_features.h5")


def train_one_epoch(
    model: FormingGraphNet,
    sim_ids: list[str],
    dataset: WrinkleDataset,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    total_log: dict[str, float] = {}
    count = 0

    # Shuffle order each epoch
    import random
    order = list(range(len(sim_ids)))
    random.shuffle(order)

    for i in order:
        batch = dataset[i]
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        optimizer.zero_grad()
        predictions = model(batch)               # (T, M, 4)
        loss, log = wrinkle_loss(predictions, batch["targets"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        for k, v in log.items():
            total_log[k] = total_log.get(k, 0.0) + v
        count += 1

    return {k: v / count for k, v in total_log.items()}


@torch.no_grad()
def evaluate_fold(
    model: FormingGraphNet,
    val_ids: list[str],
    dataset: WrinkleDataset,
    device: torch.device,
    severity_threshold: float = 0.15,
) -> dict:
    """
    Compute validation metrics.
    Primary metric: simulation-level wrinkle detection rate.
    """
    from training.evaluate import compute_metrics
    model.eval()
    all_preds = []
    all_targets = []

    for i, sid in enumerate(val_ids):
        # Find index in dataset
        idx = dataset.sim_ids.index(sid)
        batch = dataset[idx]
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        pred = model(batch)  # (T, M, 4)
        all_preds.append({"sim_id": sid, "pred": pred.cpu(), "target": batch["targets"].cpu()})

    return compute_metrics(all_preds, threshold=severity_threshold)


def train_fold(
    fold: int,
    train_ids: list[str],
    val_ids: list[str],
    config: dict,
    output_dir: Path,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== Fold {fold} | train={len(train_ids)} val={len(val_ids)} | device={device} ===")

    # Dataset — build with all sim_ids so normalisation stats are consistent
    all_ids = train_ids + val_ids
    dataset = WrinkleDataset(WP3_H5, all_ids, normalize=True, device="cpu")

    train_dataset = WrinkleDataset(WP3_H5, train_ids, normalize=True, device="cpu")
    val_dataset   = WrinkleDataset(WP3_H5, val_ids,   normalize=True, device="cpu")

    model = FormingGraphNet(**config.get("model", {})).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=config.get("lr", 1e-3), weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=config.get("T_0", 20), T_mult=2
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    history = []
    n_epochs = config.get("epochs", 100)
    patience = config.get("patience", 20)
    patience_counter = 0

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        train_log = train_one_epoch(model, train_ids, train_dataset, optimizer, device)
        val_metrics = evaluate_fold(model, val_ids, val_dataset, device)
        scheduler.step()

        val_loss = val_metrics.get("val_loss", float("inf"))
        lr = scheduler.get_last_lr()[0]

        row = {
            "epoch": epoch,
            "lr": lr,
            "elapsed_s": time.time() - t0,
            **train_log,
            **val_metrics,
        }
        history.append(row)

        print(
            f"  Epoch {epoch:3d}/{n_epochs} | "
            f"train_loss={train_log.get('loss/total', 0):.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"detect_rate={val_metrics.get('detection_rate', 0):.3f} | "
            f"lr={lr:.2e} | {time.time()-t0:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_metrics": val_metrics,
            }, output_dir / "best.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch} (patience={patience})")
                break

    # Save training history
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"  Best val_loss={best_val_loss:.4f}, saved to {output_dir}/best.pt")
    return {"fold": fold, "best_val_loss": best_val_loss, "history": history}
```

---

## 7.2 Evaluation Metrics

**File:** `training/evaluate.py`

```python
"""
WP7 — Evaluation metrics.

Primary metric: simulation-level wrinkle detection rate (sensitivity).
A missed wrinkle is expensive; a false alarm just triggers a fine-mesh run.
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch


WP2_H5 = Path("data/cfwrinkle_dataset.h5")


def compute_metrics(
    predictions: list[dict],  # [{'sim_id': str, 'pred': Tensor(T,M,4), 'target': Tensor(T,M,4)}]
    threshold: float = 0.15,
) -> dict:
    """
    Compute simulation-level wrinkle detection metrics.

    For each sim:
      - aggregate predicted severity over time and elements → scalar max
      - threshold → binary prediction
      - compare against ground-truth is_wrinkled from WP2 HDF5

    Returns dict with: detection_rate, false_alarm_rate, f1, precision, recall,
                       mean_severity_mae, val_loss
    """
    gt_labels = _load_ground_truth_labels([p["sim_id"] for p in predictions])

    preds_binary = []
    gt_binary = []
    severity_maes = []
    val_losses = []

    for p in predictions:
        sid = p["sim_id"]
        pred_sev = p["pred"][..., 0]      # (T, M) predicted severity
        tgt_sev  = p["target"][..., 0]   # (T, M) target severity

        # Simulation-level prediction: max severity across time and elements
        max_pred = float(pred_sev.nanmax()) if not torch.isnan(pred_sev).all() else 0.0
        preds_binary.append(1 if max_pred >= threshold else 0)
        gt_binary.append(gt_labels.get(sid, 0))

        # Node-level MAE (ignoring NaN boundary elements)
        mask = ~torch.isnan(tgt_sev)
        if mask.sum() > 0:
            severity_maes.append(float((pred_sev[mask] - tgt_sev[mask]).abs().mean()))

        # Val loss for this sim
        from model.loss import wrinkle_loss
        loss, _ = wrinkle_loss(p["pred"], p["target"])
        val_losses.append(float(loss))

    tp = sum(p == 1 and g == 1 for p, g in zip(preds_binary, gt_binary))
    fp = sum(p == 1 and g == 0 for p, g in zip(preds_binary, gt_binary))
    fn = sum(p == 0 and g == 1 for p, g in zip(preds_binary, gt_binary))
    tn = sum(p == 0 and g == 0 for p, g in zip(preds_binary, gt_binary))

    recall    = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-8)
    far       = fp / max(fp + tn, 1)

    return {
        "detection_rate": recall,
        "false_alarm_rate": far,
        "precision": precision,
        "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "mean_severity_mae": float(np.mean(severity_maes)) if severity_maes else 0.0,
        "val_loss": float(np.mean(val_losses)) if val_losses else 0.0,
    }


def _load_ground_truth_labels(sim_ids: list[str]) -> dict[str, int]:
    """Read is_wrinkled labels from WP2 HDF5."""
    labels = {}
    try:
        with h5py.File(WP2_H5, "r") as f:
            for sid in sim_ids:
                if f"simulations/{sid}" in f:
                    labels[sid] = int(f[f"simulations/{sid}"].attrs.get("is_wrinkled", 0))
    except Exception:
        pass
    return labels
```

---

## 7.3 Visualisation Suite

**File:** `training/visualize.py`

All functions return matplotlib Figure objects (or save to file). Use `plt.show()` for interactive display.

```python
"""
WP7 — Visualisation suite for training, evaluation, and predictions.

All functions:
  - Return a matplotlib Figure
  - Accept optional `save_path` to write PNG
  - Are callable standalone or from training scripts

Usage examples at bottom of this file.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import h5py
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.animation as animation
import numpy as np
import torch


# ── Training Curves ───────────────────────────────────────────────────────────

def plot_training_curves(
    history_path: str | Path,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Dynamic training curves: total loss, per-component losses, LR, detection rate.

    Args:
        history_path: path to history.json written by train.py
        save_path: if given, save PNG here
    """
    with open(history_path) as f:
        history = json.load(f)

    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"Training History — {Path(history_path).parent.name}", fontsize=13)

    loss_keys = ["loss/total", "loss/severity", "loss/comp_frac", "loss/oop", "loss/physics"]
    metric_keys = ["detection_rate", "val_loss", "f1", "lr"]

    panels = [
        ("Total Loss",       "loss/total",      "val_loss"),
        ("Severity Loss",    "loss/severity",   None),
        ("Comp Frac Loss",   "loss/comp_frac",  None),
        ("OOP Loss",         "loss/oop",        None),
        ("Detection Rate",   "detection_rate",  None),
        ("LR",               "lr",              None),
    ]

    for ax, (title, train_key, val_key) in zip(axes.flat, panels):
        train_vals = [h.get(train_key, float("nan")) for h in history]
        ax.plot(epochs, train_vals, label="train", color="steelblue")
        if val_key:
            val_vals = [h.get(val_key, float("nan")) for h in history]
            ax.plot(epochs, val_vals, label="val", color="orange", linestyle="--")
            ax.legend(fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig


# ── Mesh Prediction Map ───────────────────────────────────────────────────────

def plot_mesh_predictions(
    sim_id: str,
    predictions: torch.Tensor,   # (T, M, 4) or (M,) for single-timestep
    targets: torch.Tensor,        # same shape
    h5_path: str | Path,
    timestep: int = -1,           # -1 = final timestep
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Side-by-side: predicted vs actual wrinkle severity on coarse mesh.
    Elements coloured by severity [0, 1].

    Human-verifiable check: wrinkled regions should appear red in both panels.
    """
    with h5py.File(h5_path, "r") as f:
        grp = f[f"simulations/{sim_id}"]
        nodes = grp["mesh/coarse_nodes"][:]       # (N, 3)
        elements = grp["mesh/coarse_elements"][:] # (M, 3)

    # Extract severity at requested timestep
    if predictions.ndim == 3:
        pred_sev = predictions[timestep, :, 0].numpy()
        tgt_sev  = targets[timestep, :, 0].numpy()
    else:
        pred_sev = predictions[:, 0].numpy()
        tgt_sev  = targets[:, 0].numpy()

    # Replace NaN with 0 for display
    pred_sev = np.nan_to_num(pred_sev, nan=0.0)
    tgt_sev  = np.nan_to_num(tgt_sev,  nan=0.0)

    # Build triangulation from XY positions of coarse nodes
    triang = mtri.Triangulation(nodes[:, 0], nodes[:, 1], elements)

    # Compute per-node colour as mean of adjacent element severities
    def _elem_to_node(sev_elem: np.ndarray, elem: np.ndarray, n_nodes: int) -> np.ndarray:
        sev_node = np.zeros(n_nodes)
        count = np.zeros(n_nodes)
        for ei, tri in enumerate(elem):
            for ni in tri:
                sev_node[ni] += sev_elem[ei]
                count[ni] += 1
        return np.divide(sev_node, count, where=count > 0)

    n_nodes = nodes.shape[0]
    pred_node = _elem_to_node(pred_sev, elements, n_nodes)
    tgt_node  = _elem_to_node(tgt_sev,  elements, n_nodes)

    vmax = max(tgt_sev.max(), 0.01)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"{sim_id} — Wrinkle Severity (t={timestep})", fontsize=12)

    for ax, vals, label in [
        (ax1, tgt_node,  "Ground Truth"),
        (ax2, pred_node, "Predicted"),
    ]:
        tcf = ax.tricontourf(triang, vals, levels=20, cmap="RdYlGn_r", vmin=0, vmax=vmax)
        plt.colorbar(tcf, ax=ax, label="Severity")
        ax.set_title(label)
        ax.set_aspect("equal")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")

    # Error map
    err_node = np.abs(pred_node - tgt_node)
    tcf = ax3.tricontourf(triang, err_node, levels=20, cmap="Reds", vmin=0)
    plt.colorbar(tcf, ax=ax3, label="|Error|")
    ax3.set_title("Absolute Error")
    ax3.set_aspect("equal")
    ax3.set_xlabel("X (mm)")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig


# ── Temporal Evolution Animation ──────────────────────────────────────────────

def animate_severity_evolution(
    sim_id: str,
    predictions: torch.Tensor,  # (T, M, 4)
    targets: torch.Tensor,       # (T, M, 4)
    h5_path: str | Path,
    save_path: Optional[str | Path] = None,
    fps: int = 10,
) -> animation.FuncAnimation:
    """
    Animated heatmap of predicted vs actual wrinkle severity over stroke fraction.
    Saves as GIF if save_path ends with .gif, MP4 otherwise.

    Human-verifiable: watch wrinkles develop — should appear at expected location.
    """
    with h5py.File(h5_path, "r") as f:
        grp = f[f"simulations/{sim_id}"]
        nodes = grp["mesh/coarse_nodes"][:]
        elements = grp["mesh/coarse_elements"][:]

    T = predictions.shape[0]
    stroke_fracs = np.linspace(0, 1, T)
    triang = mtri.Triangulation(nodes[:, 0], nodes[:, 1], elements)

    pred_sev = np.nan_to_num(predictions[:, :, 0].numpy(), nan=0.0)  # (T, M)
    tgt_sev  = np.nan_to_num(targets[:, :, 0].numpy(),     nan=0.0)

    def _e2n(sev_t: np.ndarray) -> np.ndarray:
        n_nodes = nodes.shape[0]
        out = np.zeros(n_nodes)
        cnt = np.zeros(n_nodes)
        for ei, tri in enumerate(elements):
            for ni in tri:
                out[ni] += sev_t[ei]
                cnt[ni] += 1
        return np.divide(out, cnt, where=cnt > 0)

    vmax = max(tgt_sev.max(), 0.01)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"{sim_id} — Severity Evolution")

    tc1 = ax1.tricontourf(triang, _e2n(tgt_sev[0]),  levels=20, cmap="RdYlGn_r", vmin=0, vmax=vmax)
    tc2 = ax2.tricontourf(triang, _e2n(pred_sev[0]), levels=20, cmap="RdYlGn_r", vmin=0, vmax=vmax)
    plt.colorbar(tc1, ax=ax1); plt.colorbar(tc2, ax=ax2)
    ax1.set_title("Ground Truth"); ax2.set_title("Predicted")
    for ax in [ax1, ax2]:
        ax.set_aspect("equal")
    title = fig.suptitle(f"{sim_id} — stroke={stroke_fracs[0]:.2f}")

    def update(frame: int):
        for ax in [ax1, ax2]:
            for coll in ax.collections:
                coll.remove()
        ax1.tricontourf(triang, _e2n(tgt_sev[frame]),  levels=20, cmap="RdYlGn_r", vmin=0, vmax=vmax)
        ax2.tricontourf(triang, _e2n(pred_sev[frame]), levels=20, cmap="RdYlGn_r", vmin=0, vmax=vmax)
        title.set_text(f"{sim_id} — stroke={stroke_fracs[frame]:.2f}")
        return []

    anim = animation.FuncAnimation(fig, update, frames=T, interval=1000 // fps, blit=False)

    if save_path:
        path = Path(save_path)
        if path.suffix == ".gif":
            anim.save(path, writer="pillow", fps=fps)
        else:
            anim.save(path, writer="ffmpeg", fps=fps)
        print(f"Saved animation: {save_path}")

    return anim


# ── Cross-Fold Summary ────────────────────────────────────────────────────────

def plot_fold_summary(
    results: list[dict],  # [{'fold': int, 'detection_rate': float, 'f1': float, ...}]
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Box plots of metrics across folds. One box per metric.
    """
    import pandas as pd
    df = pd.DataFrame(results)
    metrics = ["detection_rate", "false_alarm_rate", "f1", "mean_severity_mae"]
    present = [m for m in metrics if m in df.columns]

    fig, axes = plt.subplots(1, len(present), figsize=(4 * len(present), 5))
    if len(present) == 1:
        axes = [axes]
    fig.suptitle("Cross-Fold Evaluation Summary", fontsize=13)

    for ax, metric in zip(axes, present):
        vals = [df.loc[df["fold"] == fold, metric].values[0]
                for fold in sorted(df["fold"].unique())]
        ax.bar(range(len(vals)), vals, color="steelblue", alpha=0.7)
        ax.axhline(np.mean(vals), color="red", linestyle="--", label=f"mean={np.mean(vals):.3f}")
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels([f"F{i}" for i in range(len(vals))])
        ax.set_title(metric.replace("_", " ").title())
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ── Quick-plot helper (for interactive verification) ─────────────────────────

def quick_verify_sample(
    sim_id: str,
    h5_path: str | Path,
    model_checkpoint: Optional[str | Path] = None,
) -> None:
    """
    Print dataset shapes and plot a severity map for a single sim.
    Use during smoke test and overfit test to verify data is loading correctly.

    Run interactively:
      from training.visualize import quick_verify_sample
      quick_verify_sample("geom_0_2_pair1", "data/cfwrinkle_wp3_features.h5")
    """
    from model.dataset import WrinkleDataset

    ds = WrinkleDataset(h5_path, [sim_id], normalize=False, device="cpu")
    item = ds[0]

    print(f"\n=== {sim_id} ===")
    for k, v in item.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:30s}: {tuple(v.shape)}  {v.dtype}")
        else:
            print(f"  {k:30s}: {v}")

    tgt = item["targets"]
    sev = tgt[..., 0]
    mask = ~torch.isnan(sev)
    print(f"\n  severity range (non-NaN): [{sev[mask].min():.4f}, {sev[mask].max():.4f}]")
    print(f"  NaN element fraction:     {(~mask).float().mean():.3f}")

    # Plot final-timestep ground truth
    fig = plot_mesh_predictions(sim_id, tgt, tgt, h5_path, timestep=-1)
    fig.suptitle(f"{sim_id} — Ground Truth (both panels identical for verification)")
    plt.show()
```

---

## 7.4 Single-Simulation Inference Pipeline

**File:** `training/infer.py`

```python
"""
WP7 — Single-simulation inference.

Production use case: user provides a path to one AniForm coarse-mesh results
directory. The pipeline extracts features and returns wrinkle severity predictions.

CRITICAL CONSTRAINT: only the coarse mesh is used — no fine mesh at inference.

Usage:
  python -m training.infer \
      --results-dir "C:/path/to/geom_X.Results" \
      --checkpoint checkpoints/fold_0/best.pt \
      --output reports/infer_geom_X.json \
      --visualize
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from model.gnn import FormingGraphNet
from model.dataset import WrinkleDataset


def run_inference(
    results_dir: str | Path,
    checkpoint_path: str | Path,
    output_path: Optional[str | Path] = None,
    visualize: bool = True,
    device: str = "cuda",
) -> dict:
    """
    Full inference pipeline for a single AniForm coarse-mesh results directory.

    Steps:
      1. Extract coarse-mesh fields using WP2-style readers
      2. Build graph (edge_index, edge_attr) using WP3 graph.py
      3. Compute 37 physics features using WP3 physics.py
      4. Compute 37 temporal rates using WP3 temporal.py
      5. Resample to 256 uniform stroke fractions using WP3 temporal.py
      6. Extract material card from results metadata
      7. Run model forward pass
      8. Return (and optionally visualise) predictions

    Returns:
        dict with keys:
          'sim_id': str (inferred from directory name)
          'predictions': ndarray (256, N_elem, 4)
          'max_severity': float (sim-level wrinkle risk score)
          'is_wrinkled_prediction': bool (max_severity >= 0.15)
          'wrinkle_risk_elements': list of element indices with severity > 0.1
    """
    results_dir = Path(results_dir)
    sim_id = results_dir.stem.replace(".Results", "")
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Inference on: {sim_id}  (device={device})")

    # ── Step 1-6: Feature extraction ─────────────────────────────────────────
    # Import WP2/3 modules
    from io.aniform_readers import ReadAFMesh, ReadAFResult, ReadAFSFile
    from wp3_features.graph import build_edge_index, build_edge_attr
    from wp3_features.physics import compute_feature_tensor
    from wp3_features.temporal import compute_rates, resample_to_uniform
    from wp3_features.material import extract_material_card
    import yaml

    cfg = yaml.safe_load(
        (Path("config/pipeline_config.yaml")).read_text(encoding="utf-8")
    )

    # Detect batch from directory path / naming convention
    # Batch A: "geom_0_*", Batch B: "mold set *"
    name = results_dir.name
    if name.startswith("geom"):
        batch = "A"
        ply_groups = cfg["batches"]["A"]["ply_groups"]
        bending_groups = cfg["batches"]["A"]["bending_groups"]
    else:
        batch = "B"
        ply_groups = cfg["batches"]["B"]["ply_groups"]
        bending_groups = cfg["batches"]["B"]["bending_groups"]

    # Read coarse mesh (identify coarse by smaller node count in displacement field)
    afm_path = results_dir / "model.afm"
    nodes_raw, elements_raw = ReadAFMesh(str(afm_path), elemGrNrs=[])
    # TODO: select correct group(s) and concatenate — see build_features.py for reference

    # Read fields, build features — mirror wp3_features/build_features.py logic
    # This is a simplified extraction; for production use, factor out a shared
    # extraction function from build_features.py so inference and training use
    # identical code paths.
    print("  Feature extraction not yet fully factored — see build_features.py for reference")
    print("  ACTION: Refactor common extraction logic into wp3_features/extract_single.py")

    # ── Step 7: Load model + run ──────────────────────────────────────────────
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = FormingGraphNet()
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    # Load normalisation stats
    wp3_h5 = Path("data/cfwrinkle_wp3_features.h5")
    with h5py.File(wp3_h5, "r") as f:
        feat_mean = f["metadata/feature_stats/feat_mean"][:]
        feat_std  = f["metadata/feature_stats/feat_std"][:]

    # Apply normalisation to features (same as WrinkleDataset._compute_stats_online)
    # node_features = (node_features - feat_mean) / (feat_std + 1e-8)

    # TODO: construct batch dict from extracted features and run model
    # pred = model(batch)  # (256, N_elem, 4)

    # ── Placeholder output ────────────────────────────────────────────────────
    # Remove this block once extraction is wired up
    print("\nNOTE: inference.py is a scaffold — wire up extract_single.py to complete.")
    result = {
        "sim_id": sim_id,
        "predictions": None,
        "max_severity": None,
        "is_wrinkled_prediction": None,
        "wrinkle_risk_elements": [],
    }

    if output_path:
        with open(output_path, "w") as f:
            json.dump({k: v for k, v in result.items() if not isinstance(v, np.ndarray)}, f, indent=2)

    if visualize and result["predictions"] is not None:
        from training.visualize import plot_mesh_predictions, animate_severity_evolution
        pred_t = torch.as_tensor(result["predictions"])
        # No ground truth at inference — pass predictions as both args for display
        plot_mesh_predictions(sim_id, pred_t, pred_t, wp3_h5, timestep=-1)
        import matplotlib.pyplot as plt
        plt.show()

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir",  required=True)
    parser.add_argument("--checkpoint",   required=True)
    parser.add_argument("--output",       default=None)
    parser.add_argument("--visualize",    action="store_true")
    args = parser.parse_args()
    run_inference(args.results_dir, args.checkpoint, args.output, args.visualize)
```

**NOTE:** `infer.py` is intentionally a scaffold. The extraction logic in `wp3_features/build_features.py` is written to process batches of sims from the WP2 HDF5. Before WP7 is complete, refactor the per-sim field extraction into `wp3_features/extract_single.py` so both `build_features.py` and `infer.py` call identical code.

---

## 7.5 Progressive Test Run Ladder

Run these in order. **Do not proceed to the next level until the current one passes its human-verifiable check.**

---

### Level 0 — Environment + Data Verification (before any training)

```bash
# Verify GPU
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0)); print('VRAM:', torch.cuda.get_device_properties(0).total_memory // 1024**3, 'GB')"

# Verify WP3 HDF5 is readable and has expected structure
python -c "
import h5py
with h5py.File('data/cfwrinkle_wp3_features.h5', 'r') as f:
    sims = list(f['simulations'].keys())
    print(f'Simulations: {len(sims)}')
    sid = sims[0]
    print(f'Sample sim: {sid}')
    print('  coarse_fields_resampled:', f[f'simulations/{sid}/coarse_fields_resampled'].shape)
    print('  targets/wrinkle_severity:', f[f'simulations/{sid}/targets/wrinkle_severity'].shape)
    print('  graph/edge_index:', f[f'simulations/{sid}/graph/edge_index'].shape)
    print('  material_card:', f[f'simulations/{sid}/material_card'].shape)
"
```

**Human check:** 65 simulations listed. Sample shapes should be:
- `coarse_fields_resampled`: `(256, ~3700, 37)` for Batch A
- `targets/wrinkle_severity`: `(256, ~N_elem)`
- `graph/edge_index`: `(2, ~N_edges)`

---

### Level 1 — Smoke Test (1 sim, 1 epoch)

**Purpose:** Verify the full forward/backward pass runs without crash or NaN.

```bash
python -m training.train \
    --sim-ids geom_0_2_pair1 \
    --epochs 1 \
    --output checkpoints/smoke_test/
```

**Human checks:**
1. Script prints model parameter count (expect ~100K–500K for hidden_dim=64)
2. Script prints input/output shapes — confirm (T=256, N_nodes, 74) and (256, N_elem, 4)
3. Loss value printed — must be a finite number (not NaN, not Inf)
4. No CUDA OOM error
5. `checkpoints/smoke_test/best.pt` written

**Pass criteria:** completes in < 5 minutes, loss is finite.

---

### Level 2 — Gradient + Overfit Test (3 sims, 200 epochs)

**Purpose:** Verify model can memorise a tiny dataset (loss → near 0). If it cannot overfit 3 samples, the architecture or loss is broken.

```bash
# Pick 1 wrinkled Batch A, 1 wrinkled Batch B, 1 clean sim
python -m training.train \
    --sim-ids geom_0_2_pair1,geom_0_5_pair1,geom_0_8_pair1 \
    --epochs 200 \
    --output checkpoints/overfit_test/
```

**Human checks (after run):**
1. Plot training curves: `python -c "from training.visualize import plot_training_curves; import matplotlib.pyplot as plt; plot_training_curves('checkpoints/overfit_test/history.json'); plt.show()"`
2. Final total loss should be < 0.05 (target: < 0.01 for severity component)
3. Load best checkpoint and visualise predictions on all 3 sims:
```python
from training.visualize import quick_verify_sample, plot_mesh_predictions
import torch, h5py
from model.gnn import FormingGraphNet
from model.dataset import WrinkleDataset

ckpt = torch.load("checkpoints/overfit_test/best.pt", map_location="cpu")
model = FormingGraphNet(); model.load_state_dict(ckpt["model_state_dict"]); model.eval()
ds = WrinkleDataset("data/cfwrinkle_wp3_features.h5", ["geom_0_2_pair1"], normalize=True)
item = ds[0]
with torch.no_grad():
    pred = model(item)
plot_mesh_predictions("geom_0_2_pair1", pred, item["targets"], "data/cfwrinkle_wp3_features.h5")
import matplotlib.pyplot as plt; plt.show()
```
4. Prediction map should visually match ground truth for the wrinkled sim.

**Pass criteria:** severity loss < 0.05, prediction map visually matches ground truth wrinkle location.

---

### Level 3 — Mini-Train (13 sims, 50 epochs, 2 held-out)

**Purpose:** First generalization check. Small enough to run in < 1 hour.

```bash
# Use fold 0 but truncate to 13 sims for speed
python -m training.train \
    --fold 0 \
    --max-train-sims 13 \
    --max-val-sims 2 \
    --epochs 50 \
    --output checkpoints/mini_train/
```

**Human checks:**
1. Validation loss should decrease (or plateau) — not increase — after epoch 10
2. Detection rate should be > 0 (model is learning to identify wrinkled sims)
3. Plot prediction maps for both held-out sims — do wrinkled regions appear?
4. If val loss increases while train loss decreases after epoch 20: reduce hidden_dim or increase dropout

**Pass criteria:** val loss decreases, detection_rate > 0 on held-out set.

---

### Level 4 — Full Cross-Validation (65 sims, 5-fold)

**Purpose:** Final evaluation with full dataset.

```bash
# Run all 5 folds (will take several hours)
python -m training.train \
    --all-folds \
    --epochs 100 \
    --output checkpoints/full_cv/
```

**Human checks after run:**
1. Per-fold metrics in `checkpoints/full_cv/fold_*/history.json`
2. Generate cross-fold summary:
```python
import json
from pathlib import Path
from training.visualize import plot_fold_summary
import matplotlib.pyplot as plt

results = []
for fold in range(5):
    history = json.load(open(f"checkpoints/full_cv/fold_{fold}/history.json"))
    best = min(history, key=lambda h: h.get("val_loss", float("inf")))
    results.append({"fold": fold, **best})

plot_fold_summary(results, save_path="reports/fold_summary.png")
plt.show()
print("Mean detection rate:", sum(r["detection_rate"] for r in results) / 5)
print("Mean F1:", sum(r["f1"] for r in results) / 5)
```
3. Animate severity evolution for 2 validation sims per fold:
```python
from training.visualize import animate_severity_evolution
# (run with a loaded model and a validation batch)
```

**Pass criteria (WP7 gate):**
- All 5 folds complete without crash
- Mean detection rate (sensitivity) > 0.5 across folds
- At least 1 fold with detection_rate > 0.7
- Mean F1 > 0.4

---

## 7.6 Smoke Test

**File:** `tests/test_training_smoke.py`

```python
"""
Smoke test — full forward pass in training mode, 1 sim, 1 step.
Does NOT require real HDF5 data — uses synthetic tensors.
Run: pytest tests/test_training_smoke.py -v -s
"""
import torch
import pytest
from model.gnn import FormingGraphNet
from model.loss import wrinkle_loss

T, N, M, E = 256, 200, 150, 500   # realistic sizes


def _synthetic_batch(device="cpu"):
    elements = torch.randint(0, N, (M, 3))
    return {
        "sim_id": "synthetic_test",
        "node_features":  torch.randn(T, N, 74),
        "edge_index":     torch.randint(0, N, (2, E), dtype=torch.int64),
        "edge_attr":      torch.randn(E, 4),
        "material_card":  torch.randn(8),
        "elements":       elements,
        "targets":        torch.rand(T, M, 4),
    }


def test_full_training_step():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FormingGraphNet(hidden_dim=32).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
             for k, v in _synthetic_batch().items()}

    model.train()
    optimizer.zero_grad()
    pred = model(batch)
    assert pred.shape == (T, M, 4), f"Wrong shape: {pred.shape}"

    loss, log = wrinkle_loss(pred, batch["targets"])
    assert not torch.isnan(loss), "Loss is NaN"
    assert torch.isfinite(loss), "Loss is infinite"

    loss.backward()
    optimizer.step()

    # Check gradients flowed
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No grad: {name}"

    print(f"\nSmoke test passed on {device}")
    print(f"  Output shape: {pred.shape}")
    for k, v in log.items():
        print(f"  {k}: {v:.6f}")
```

---

## WP7 Gate Checklist

```
Training Infrastructure
  [ ] test_training_smoke.py passes on GPU
  [ ] Training loop completes 1 epoch on the largest sim without OOM
  [ ] Checkpointing: best.pt written and loadable
  [ ] history.json contains all expected keys

Progressive Tests
  [ ] Level 0: 65 sims confirmed in WP3 HDF5, shapes correct
  [ ] Level 1 (smoke): 1 sim, 1 epoch — finite loss, no crash, < 5 min
  [ ] Level 2 (overfit): 3 sims, 200 epochs — severity loss < 0.05
  [ ] Level 3 (mini-train): val loss decreasing, detection_rate > 0
  [ ] Level 4 (full CV): all 5 folds complete

Visualisation
  [ ] plot_training_curves() renders without error on real history.json
  [ ] plot_mesh_predictions() shows plausible severity maps (wrinkled regions visible)
  [ ] animate_severity_evolution() produces a viewable animation
  [ ] plot_fold_summary() renders cross-fold bar chart

Metrics (final gate)
  [ ] Mean detection rate (sensitivity) > 0.5 across 5 folds
  [ ] Mean F1 > 0.4
  [ ] No fold with detection_rate = 0

Inference Scaffold
  [ ] infer.py runs without import error
  [ ] extract_single.py created (factored from build_features.py)
  [ ] End-to-end inference tested on 1 held-out sim
```

---

## Commands Reference

```bash
# Activate environment
.\.venv\Scripts\Activate.ps1

# Run all unit tests
pytest tests/test_model_unit.py tests/test_training_smoke.py -v

# Level 1: smoke test (1 sim, 1 epoch)
python -m training.train --sim-ids geom_0_2_pair1 --epochs 1 --output checkpoints/smoke/

# Level 2: overfit test
python -m training.train --sim-ids geom_0_2_pair1,geom_0_5_pair1,geom_0_8_pair1 --epochs 200 --output checkpoints/overfit/

# Level 3: mini-train
python -m training.train --fold 0 --max-train-sims 13 --max-val-sims 2 --epochs 50 --output checkpoints/mini/

# Level 4: full CV
python -m training.train --all-folds --epochs 100 --output checkpoints/full_cv/

# Interactive visualisation (in Python REPL)
from training.visualize import quick_verify_sample
quick_verify_sample("geom_0_2_pair1", "data/cfwrinkle_wp3_features.h5")

# Single-sim inference
python -m training.infer \
    --results-dir "C:/path/to/geom_X.Results" \
    --checkpoint checkpoints/full_cv/fold_0/best.pt \
    --visualize
```

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import h5py
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import torch


def plot_training_curves(history_path: str | Path, save_path: Optional[str | Path] = None) -> plt.Figure:
    with open(history_path, encoding="utf-8") as f:
        history = json.load(f)

    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"Training History - {Path(history_path).parent.name}", fontsize=13)
    panels = [
        ("Total Loss", "loss/total", "val_loss"),
        ("Severity Loss", "loss/severity", None),
        ("Comp Frac Loss", "loss/comp_frac", None),
        ("OOP Loss", "loss/oop", None),
        ("Detection Rate", "detection_rate", None),
        ("LR", "lr", None),
    ]
    for ax, (title, train_key, val_key) in zip(axes.flat, panels):
        train_vals = [h.get(train_key, float("nan")) for h in history]
        ax.plot(epochs, train_vals, label="train", color="steelblue")
        if val_key is not None:
            val_vals = [h.get(val_key, float("nan")) for h in history]
            ax.plot(epochs, val_vals, label="val", color="orange", linestyle="--")
            ax.legend(fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def _load_mesh(
    sim_id: str,
    h5_path: str | Path,
    wp2_path: str | Path = "data/cfwrinkle_dataset.h5",
) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(h5_path, "r") as f:
        grp = f[f"simulations/{sim_id}"]
        if "mesh/coarse_nodes" in grp and "mesh/coarse_elements" in grp:
            return grp["mesh/coarse_nodes"][:], grp["mesh/coarse_elements"][:]
        if "coarse/mesh_elements" in grp:
            elements = grp["coarse/mesh_elements"][:]
        else:
            raise KeyError(f"Could not find coarse element connectivity for sim {sim_id}")
    with h5py.File(wp2_path, "r") as wp2:
        nodes = wp2[f"simulations/{sim_id}/mesh/coarse/nodes"][:]
    return nodes, elements


def _elem_to_node(sev_elem: np.ndarray, elements: np.ndarray, n_nodes: int) -> np.ndarray:
    sev_node = np.zeros(n_nodes, dtype=np.float64)
    count = np.zeros(n_nodes, dtype=np.float64)
    for ei, tri in enumerate(elements):
        for ni in tri:
            sev_node[int(ni)] += float(sev_elem[ei])
            count[int(ni)] += 1.0
    return np.divide(sev_node, count, where=count > 0).astype(np.float32)


def plot_mesh_predictions(
    sim_id: str,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    h5_path: str | Path,
    timestep: int = -1,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    nodes, elements = _load_mesh(sim_id, h5_path)
    pred_np = predictions.detach().cpu().numpy()
    tgt_np = targets.detach().cpu().numpy()
    if pred_np.ndim == 3:
        pred_sev = pred_np[timestep, :, 0]
        tgt_sev = tgt_np[timestep, :, 0]
    else:
        pred_sev = pred_np[:, 0]
        tgt_sev = tgt_np[:, 0]
    pred_sev = np.nan_to_num(pred_sev, nan=0.0)
    tgt_sev = np.nan_to_num(tgt_sev, nan=0.0)

    triang = mtri.Triangulation(nodes[:, 0], nodes[:, 1], elements)
    n_nodes = int(nodes.shape[0])
    pred_node = _elem_to_node(pred_sev, elements, n_nodes)
    tgt_node = _elem_to_node(tgt_sev, elements, n_nodes)
    vmax = max(float(tgt_sev.max()), 0.01)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"{sim_id} - Wrinkle Severity (t={timestep})", fontsize=12)
    for ax, vals, label in [(ax1, tgt_node, "Ground Truth"), (ax2, pred_node, "Predicted")]:
        tcf = ax.tricontourf(triang, vals, levels=20, cmap="RdYlGn_r", vmin=0.0, vmax=vmax)
        plt.colorbar(tcf, ax=ax, label="Severity")
        ax.set_title(label)
        ax.set_aspect("equal")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
    err_node = np.abs(pred_node - tgt_node)
    tcf = ax3.tricontourf(triang, err_node, levels=20, cmap="Reds", vmin=0.0)
    plt.colorbar(tcf, ax=ax3, label="|Error|")
    ax3.set_title("Absolute Error")
    ax3.set_aspect("equal")
    ax3.set_xlabel("X (mm)")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def animate_severity_evolution(
    sim_id: str,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    h5_path: str | Path,
    save_path: Optional[str | Path] = None,
    fps: int = 10,
) -> animation.FuncAnimation:
    nodes, elements = _load_mesh(sim_id, h5_path)
    pred = np.nan_to_num(predictions.detach().cpu().numpy()[:, :, 0], nan=0.0)
    tgt = np.nan_to_num(targets.detach().cpu().numpy()[:, :, 0], nan=0.0)
    t = pred.shape[0]
    triang = mtri.Triangulation(nodes[:, 0], nodes[:, 1], elements)
    stroke = np.linspace(0.0, 1.0, t)
    vmax = max(float(tgt.max()), 0.01)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.set_title("Ground Truth")
    ax2.set_title("Predicted")
    for ax in [ax1, ax2]:
        ax.set_aspect("equal")
    title = fig.suptitle(f"{sim_id} - stroke={stroke[0]:.2f}")

    def draw(frame: int) -> None:
        for ax in [ax1, ax2]:
            ax.clear()
            ax.set_aspect("equal")
        ax1.set_title("Ground Truth")
        ax2.set_title("Predicted")
        ax1.tricontourf(
            triang,
            _elem_to_node(tgt[frame], elements, nodes.shape[0]),
            levels=20,
            cmap="RdYlGn_r",
            vmin=0.0,
            vmax=vmax,
        )
        ax2.tricontourf(
            triang,
            _elem_to_node(pred[frame], elements, nodes.shape[0]),
            levels=20,
            cmap="RdYlGn_r",
            vmin=0.0,
            vmax=vmax,
        )
        title.set_text(f"{sim_id} - stroke={stroke[frame]:.2f}")

    anim = animation.FuncAnimation(fig, draw, frames=t, interval=1000 // fps, blit=False)
    if save_path:
        save = Path(save_path)
        if save.suffix.lower() == ".gif":
            anim.save(save, writer="pillow", fps=fps)
        else:
            anim.save(save, writer="ffmpeg", fps=fps)
    return anim


def plot_fold_summary(results: list[dict], save_path: Optional[str | Path] = None) -> plt.Figure:
    metrics = ["detection_rate", "false_alarm_rate", "f1", "mean_severity_mae"]
    folds = sorted({int(r["fold"]) for r in results})
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]
    fig.suptitle("Cross-Fold Evaluation Summary", fontsize=13)
    for ax, metric in zip(axes, metrics):
        vals = [float(next(r[metric] for r in results if int(r["fold"]) == fold)) for fold in folds]
        ax.bar(range(len(vals)), vals, color="steelblue", alpha=0.7)
        ax.axhline(float(np.mean(vals)), color="red", linestyle="--", label=f"mean={np.mean(vals):.3f}")
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels([f"F{fold}" for fold in folds])
        ax.set_title(metric.replace("_", " ").title())
        ax.set_ylim(0.0, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def quick_verify_sample(
    sim_id: str,
    h5_path: str | Path,
    model_checkpoint: Optional[str | Path] = None,
) -> None:
    from model.dataset import WrinkleDataset

    ds = WrinkleDataset(h5_path, [sim_id], normalize=False, device="cpu")
    item = ds[0]
    print(f"\n=== {sim_id} ===")
    for key, value in item.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key:20s}: {tuple(value.shape)}  {value.dtype}")
        else:
            print(f"  {key:20s}: {value}")

    tgt = item["targets"]
    pred = tgt
    if model_checkpoint is not None:
        from model.gnn import FormingGraphNet

        ckpt = torch.load(model_checkpoint, map_location="cpu")
        model_cfg = ckpt.get("model_config", {})
        model = FormingGraphNet(**model_cfg)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        with torch.no_grad():
            pred = model(item)
    fig = plot_mesh_predictions(sim_id, pred, tgt, h5_path, timestep=-1)
    fig.suptitle(f"{sim_id} - Quick Verification")
    plt.show()


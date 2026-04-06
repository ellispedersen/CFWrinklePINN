from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.optim as optim

from model.dataset import WrinkleDataset, load_fold_sim_ids
from model.gnn import FormingGraphNet
from model.loss import wrinkle_loss
from training.evaluate import compute_metrics

WP3_H5 = Path("data/cfwrinkle_wp3_features.h5")


def _move_to_device(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def train_one_epoch(
    model: FormingGraphNet,
    dataset: WrinkleDataset,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    order = list(range(len(dataset)))
    random.shuffle(order)
    totals: dict[str, float] = {}
    n = 0

    for idx in order:
        batch = _move_to_device(dataset[idx], device)
        optimizer.zero_grad()
        pred = model(batch)
        loss, log = wrinkle_loss(pred, batch["targets"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        for key, value in log.items():
            totals[key] = totals.get(key, 0.0) + value
        n += 1
    return {key: value / max(n, 1) for key, value in totals.items()}


@torch.no_grad()
def evaluate_split(
    model: FormingGraphNet,
    dataset: WrinkleDataset,
    sim_ids: list[str],
    device: torch.device,
    severity_threshold: float = 0.15,
) -> dict[str, float]:
    model.eval()
    preds: list[dict] = []
    for sid in sim_ids:
        item = _move_to_device(dataset[dataset.index_of(sid)], device)
        pred = model(item)
        preds.append({"sim_id": sid, "pred": pred.cpu(), "target": item["targets"].cpu()})
    return compute_metrics(preds, threshold=severity_threshold)


def train_fold(
    fold: int,
    train_ids: list[str],
    val_ids: list[str],
    config: dict,
    output_dir: Path,
) -> dict:
    device_name = str(config.get("device", "auto"))
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    print(f"\n=== Fold {fold} | train={len(train_ids)} val={len(val_ids)} | device={device} ===")

    normalize = bool(config.get("normalize", True))
    train_ds = WrinkleDataset(WP3_H5, train_ids, normalize=normalize, device="cpu")
    val_ds = WrinkleDataset(WP3_H5, val_ids, normalize=normalize, device="cpu")

    model_cfg = config.get("model", {})
    model = FormingGraphNet(**model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=float(config.get("lr", 1e-3)), weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=int(config.get("T_0", 20)),
        T_mult=int(config.get("T_mult", 2)),
    )
    n_epochs = int(config.get("epochs", 100))
    patience = int(config.get("patience", 20))

    output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        train_log = train_one_epoch(model, train_ds, optimizer, device)
        val_metrics = evaluate_split(model, val_ds, val_ids, device)
        scheduler.step()
        lr = float(scheduler.get_last_lr()[0])
        val_loss = float(val_metrics.get("val_loss", float("inf")))
        row = {"epoch": epoch, "lr": lr, "elapsed_s": time.time() - t0, **train_log, **val_metrics}
        history.append(row)
        print(
            f"  Epoch {epoch:3d}/{n_epochs} | "
            f"train_loss={train_log.get('loss/total', 0.0):.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"detect_rate={val_metrics.get('detection_rate', 0.0):.3f} | "
            f"lr={lr:.2e} | {row['elapsed_s']:.1f}s"
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_metrics": val_metrics,
                    "model_config": model_cfg,
                },
                output_dir / "best.pt",
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch} (patience={patience})")
                break

    with (output_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"  Best val_loss={best_val_loss:.4f}, saved to {output_dir / 'best.pt'}")
    return {"fold": fold, "best_val_loss": best_val_loss, "history": history}


def _read_all_sim_ids() -> list[str]:
    import h5py

    with h5py.File(WP3_H5, "r") as f:
        return sorted(f["simulations"].keys())


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="WP7 training loop")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--all-folds", action="store_true")
    p.add_argument("--sim-ids", type=str, default=None, help="Comma-separated sim IDs for custom run")
    p.add_argument("--max-train-sims", type=int, default=None)
    p.add_argument("--max-val-sims", type=int, default=None)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--no-normalize", action="store_true")
    p.add_argument("--output", type=Path, default=Path("checkpoints"))
    return p.parse_args()


def _run_one(
    fold: int,
    train_ids: list[str],
    val_ids: list[str],
    args: argparse.Namespace,
    out_dir: Path,
) -> dict:
    if args.max_train_sims is not None:
        train_ids = train_ids[: args.max_train_sims]
    if args.max_val_sims is not None:
        val_ids = val_ids[: args.max_val_sims]
    cfg = {
        "epochs": args.epochs,
        "patience": args.patience,
        "lr": args.lr,
        "device": args.device,
        "normalize": not args.no_normalize,
        "model": {"hidden_dim": args.hidden_dim},
    }
    return train_fold(fold, train_ids, val_ids, cfg, out_dir)


def main() -> None:
    args = _parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    if args.sim_ids:
        sim_ids = [s.strip() for s in args.sim_ids.split(",") if s.strip()]
        if not sim_ids:
            raise ValueError("No valid sim IDs provided in --sim-ids.")
        result = _run_one(0, sim_ids, sim_ids, args, args.output)
        with (args.output / "summary.json").open("w", encoding="utf-8") as f:
            json.dump([result], f, indent=2)
        return

    if args.all_folds:
        fold_ids = [0, 1, 2, 3, 4]
        results = []
        for fold in fold_ids:
            train_ids, val_ids = load_fold_sim_ids(WP3_H5, fold)
            results.append(_run_one(fold, train_ids, val_ids, args, args.output / f"fold_{fold}"))
        with (args.output / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        return

    all_ids = _read_all_sim_ids()
    train_ids, val_ids = load_fold_sim_ids(WP3_H5, args.fold)
    missing_train = [sid for sid in train_ids if sid not in all_ids]
    missing_val = [sid for sid in val_ids if sid not in all_ids]
    if missing_train or missing_val:
        raise RuntimeError(f"Missing sims in WP3 file. train_missing={missing_train} val_missing={missing_val}")
    result = _run_one(args.fold, train_ids, val_ids, args, args.output)
    with (args.output / "summary.json").open("w", encoding="utf-8") as f:
        json.dump([result], f, indent=2)


if __name__ == "__main__":
    main()


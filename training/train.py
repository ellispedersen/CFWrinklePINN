from __future__ import annotations

import argparse
import json
import math
import random
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import torch
import torch.optim as optim

from model.cross_scale import CrossScaleNet
from model.contracts import (
    add_checkpoint_contract_metadata,
    validate_checkpoint_compatibility,
    validate_contract_invariants,
)
from model.dataset import WrinkleDataset, load_fold_sim_ids
from model.gnn import FormingGraphNet
from model.labels import N_COARSE_TARGETS, N_FINE_TARGETS, require_last_dim
from model.loss import cross_scale_loss, wrinkle_loss
from training.evaluate import (
    _accumulate_fine_metrics,
    _finalize_fine_metric_agg,
    _init_fine_metric_agg,
    compute_metrics,
)

import os as _os
WP3_H5 = Path(_os.environ.get("WP3_H5", "data/cfwrinkle_wp3_features.h5"))
WP2_H5 = Path(_os.environ.get("WP2_H5", "data/cfwrinkle_dataset.h5"))
PHYSICS_WARMUP_EPOCHS = max(0, int(_os.environ.get("CFWRINKLE_PHYSICS_WARMUP_EPOCHS", "0")))
PHYSICS_RAMP_EPOCHS = max(1, int(_os.environ.get("CFWRINKLE_PHYSICS_RAMP_EPOCHS", "20")))


def _physics_scale(epoch: int) -> float:
    if PHYSICS_WARMUP_EPOCHS <= 0:
        return 1.0
    if epoch <= PHYSICS_WARMUP_EPOCHS:
        return 0.0
    elapsed = epoch - PHYSICS_WARMUP_EPOCHS
    return min(1.0, elapsed / PHYSICS_RAMP_EPOCHS)


def _move_to_device(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def _compute_loss(
    model_out,
    batch: dict,
    step_idx: int | None = None,
    physics_weight_scale: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """Dispatch to the right loss function based on model output type."""
    if isinstance(model_out, dict):
        # CrossScaleNet: dict with "coarse" and "fine"
        if "coarse" not in model_out or "fine" not in model_out:
            raise KeyError("CrossScaleNet output must include both 'coarse' and 'fine' tensors")
        require_last_dim(model_out["coarse"], N_COARSE_TARGETS, "model_out['coarse']")
        require_last_dim(model_out["fine"], N_FINE_TARGETS, "model_out['fine']")
        return cross_scale_loss(
            model_out["fine"],
            batch["fine_features"],
            batch["fine_elements"],
            model_out["coarse"],
            batch["targets"],
            fine_elem_edge_index=batch.get("fine_elem_edge_index"),
            step_idx=step_idx,
            physics_weight_scale=physics_weight_scale,
        )
    # FormingGraphNet: plain tensor
    require_last_dim(model_out, N_COARSE_TARGETS, "model_out")
    return wrinkle_loss(model_out, batch["targets"])


def train_one_epoch(
    model: FormingGraphNet | CrossScaleNet,
    dataset: WrinkleDataset,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_amp: bool = False,
    scaler: torch.amp.GradScaler | None = None,
    empty_cache_each_step: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    physics_weight_scale: float = 1.0,
) -> dict[str, float]:
    model.train()
    order = list(range(len(dataset)))
    random.shuffle(order)
    totals: dict[str, float] = {}
    n = 0
    _log_vram_peak = device.type == "cuda"
    fine_elem_regions = max(1, int(_os.environ.get("CFWRINKLE_FINE_ELEM_REGIONS", "1")))
    accum_steps = fine_elem_regions
    optimizer.zero_grad()

    for sample_i, idx in enumerate(order):
        batch = _move_to_device(dataset[idx], device)
        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=amp_dtype)
            if use_amp and device.type == "cuda"
            else nullcontext()
        )
        with amp_ctx:
            model_out = model(batch)
            loss, log = _compute_loss(
                model_out,
                batch,
                step_idx=n,
                physics_weight_scale=physics_weight_scale,
            )
            step_loss = loss / accum_steps if accum_steps > 1 else loss
        should_step = ((sample_i + 1) % accum_steps == 0) or (sample_i == len(order) - 1)
        if scaler is not None and scaler.is_enabled():
            scaler.scale(step_loss).backward()
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            step_loss.backward()
            if should_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
        for key, value in log.items():
            totals[key] = totals.get(key, 0.0) + value
        n += 1
        if _log_vram_peak and n == 1:
            alloc_gb = torch.cuda.memory_allocated(device) / 1024 ** 3
            reserved_gb = torch.cuda.memory_reserved(device) / 1024 ** 3
            total_gb = torch.cuda.get_device_properties(device).total_memory / 1024 ** 3
            pct = 100.0 * reserved_gb / total_gb
            print(f"  [VRAM] After first sample: alloc={alloc_gb:.2f} GB  reserved={reserved_gb:.2f}/{total_gb:.2f} GB ({pct:.1f}%)")
            if pct > 85:
                print("  [VRAM] WARNING: >85% VRAM reserved after first sample — OOM risk.")
            _log_vram_peak = False
        del batch, model_out, loss, step_loss
        if device.type == "cuda" and empty_cache_each_step:
            torch.cuda.empty_cache()
    if device.type == "cuda":
        peak_vram = torch.cuda.max_memory_allocated(device) / 1024 ** 3
        torch.cuda.reset_peak_memory_stats(device)
        totals["peak_vram_gb"] = peak_vram
    return {key: value / max(n, 1) for key, value in totals.items()}


@torch.no_grad()
def evaluate_split(
    model: FormingGraphNet | CrossScaleNet,
    dataset: WrinkleDataset,
    sim_ids: list[str],
    device: torch.device,
    severity_threshold: float = 0.15,
    use_amp: bool = False,
    empty_cache_each_step: bool = False,
    amp_dtype: torch.dtype = torch.float16,
) -> dict[str, float]:
    model.eval()
    preds: list[dict] = []
    fine_metric_agg = _init_fine_metric_agg()
    fine_pred_count = 0
    fine_channel_sum = torch.zeros(4, dtype=torch.float64)
    fine_channel_sq_sum = torch.zeros(4, dtype=torch.float64)
    fine_channel_count = torch.zeros(4, dtype=torch.float64)
    fine_channel_names = ("fs1", "fs2", "dz", "thick")
    for sid in sim_ids:
        item = _move_to_device(dataset[dataset.index_of(sid)], device)
        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=amp_dtype)
            if use_amp and device.type == "cuda"
            else nullcontext()
        )
        with amp_ctx:
            model_out = model(item)
        # For CrossScaleNet, use the coarse head for existing gate-compatible evaluation.
        if isinstance(model_out, dict):
            pred = model_out["coarse"]
            fine_features_cpu = item["fine_features"].detach().cpu().to(torch.float64)
            flat = fine_features_cpu.reshape(-1, 4)
            valid = torch.isfinite(flat)
            fine_channel_sum += torch.where(valid, flat, torch.zeros_like(flat)).sum(dim=0)
            fine_channel_sq_sum += torch.where(valid, flat.square(), torch.zeros_like(flat)).sum(dim=0)
            fine_channel_count += valid.sum(dim=0).to(torch.float64)
            _accumulate_fine_metrics(fine_metric_agg, {
                "fine_pred": model_out["fine"].cpu(),
                "fine_features": fine_features_cpu.to(torch.float32),
                "fine_elements": item["fine_elements"].cpu(),
            })
            fine_pred_count += 1
        else:
            pred = model_out
        preds.append({"sim_id": sid, "pred": pred.cpu(), "target": item["targets"].cpu()})
        del item, model_out
        if device.type == "cuda" and empty_cache_each_step:
            torch.cuda.empty_cache()
    metrics = compute_metrics(preds, threshold=severity_threshold)
    if fine_pred_count > 0:
        metrics.update(_finalize_fine_metric_agg(fine_metric_agg))
        metrics["fine_input/normalized"] = float(getattr(dataset, "fine_normalize", False))
        if torch.all(fine_channel_count > 0):
            mean = fine_channel_sum / fine_channel_count
            var = torch.clamp(fine_channel_sq_sum / fine_channel_count - mean.square(), min=0.0)
            std = torch.sqrt(var)
            for i, name in enumerate(fine_channel_names):
                metrics[f"fine_input/{name}_mean"] = float(mean[i].item())
                metrics[f"fine_input/{name}_std"] = float(std[i].item())
    return metrics


def train_fold(
    fold: int,
    train_ids: list[str],
    val_ids: list[str],
    config: dict,
    output_dir: Path,
) -> dict:
    validate_contract_invariants()
    device_name = str(config.get("device", "auto"))
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    print(f"\n=== Fold {fold} | train={len(train_ids)} val={len(val_ids)} | device={device} ===")

    normalize = bool(config.get("normalize", True))
    fine_normalize = bool(config.get("normalize_fine_features", False))
    max_timesteps = config.get("max_timesteps", None)
    temporal_strategy = str(config.get("temporal_strategy", "tail"))
    model_type = str(config.get("model_type", "coarse"))
    include_fine = model_type == "cross-scale"
    static_graph_cache = bool(config.get("dataset_static_cache", True))
    empty_cache_each_step = bool(config.get("cuda_empty_cache_per_sample", False))
    if fine_normalize and not include_fine:
        raise ValueError("normalize_fine_features requires model_type='cross-scale'")
    train_ds = WrinkleDataset(WP3_H5, train_ids, normalize=normalize, device="cpu",
                               max_timesteps=max_timesteps, temporal_strategy=temporal_strategy,
                               include_fine=include_fine, fine_normalize=fine_normalize,
                               cache_static_graph=static_graph_cache)
    get_fine_stats = getattr(train_ds, "get_fine_norm_stats", None)
    train_fine_stats = get_fine_stats() if callable(get_fine_stats) else None
    val_ds = WrinkleDataset(WP3_H5, val_ids, normalize=normalize, device="cpu",
                               max_timesteps=max_timesteps, temporal_strategy=temporal_strategy,
                               include_fine=include_fine, fine_normalize=fine_normalize,
                               fine_norm_stats=train_fine_stats,
                               cache_static_graph=static_graph_cache)

    model_cfg = dict(config.get("model", {}))
    if model_type == "cross-scale":
        model: FormingGraphNet | CrossScaleNet = CrossScaleNet(**model_cfg).to(device)
    else:
        model = FormingGraphNet(**model_cfg).to(device)
    use_amp = bool(config.get("amp", False)) and device.type == "cuda"
    amp_dtype_name = str(config.get("amp_dtype", "float16")).lower()
    if amp_dtype_name not in {"float16", "bfloat16"}:
        raise ValueError(f"Unsupported amp_dtype: {amp_dtype_name!r}")
    amp_dtype = torch.bfloat16 if amp_dtype_name == "bfloat16" else torch.float16
    use_scaler = use_amp and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler) if device.type == "cuda" else None
    if config.get("torch_compile", False) and hasattr(torch, "compile"):
        import torch._dynamo as _dynamo
        _dynamo.config.capture_scalar_outputs = True
        import torch._inductor.config as _ind_cfg
        _is_rocm = hasattr(torch.version, "hip")
        _gemm_backends = _os.environ.get("INDUCTOR_GEMM_BACKENDS", "ATEN" if _is_rocm else "ATEN,TRITON")
        if hasattr(_ind_cfg, "max_autotune_gemm_backends"):
            _ind_cfg.max_autotune_gemm_backends = _gemm_backends
        _ind_cfg.memory_planning = True
        _ind_cfg.inplace_buffers = True
        if _is_rocm:
            # ROCm mitigations: avoid HSA code-object pool exhaustion on gfx1100/RDNA3.
            # gfx942/MI300X is more stable; set INDUCTOR_GEMM_BACKENDS=ATEN,TRITON there.
            _ind_cfg.coordinate_descent_tuning = False   # reduces HSA pool pressure
            _ind_cfg.triton.cudagraphs = False           # cudagraph capture deadlocks on ROCm
            _ind_cfg.triton.cudagraph_trees = False
            _compile_mode = "default"                    # NEVER reduce-overhead on ROCm
        else:
            # CUDA (Ampere+ / Blackwell) — full inductor feature set available.
            _ind_cfg.coordinate_descent_tuning = True
            _ind_cfg.triton.cudagraphs = True
            torch.backends.cuda.matmul.allow_tf32 = True  # already default in PyTorch 1.12+; explicit for clarity
            torch.backends.cudnn.allow_tf32 = True
            _compile_mode = _os.environ.get("TORCH_COMPILE_MODE", "max-autotune")
        compile_kwargs: dict = dict(backend="inductor", fullgraph=False, dynamic=True, mode=_compile_mode)
        model = torch.compile(model, **compile_kwargs)
        _autotune = _os.environ.get("TORCHINDUCTOR_MAX_AUTOTUNE", "1") != "0"
        print(f"torch.compile: ON (inductor, dynamic=True, mode={_compile_mode}, autotune={'ON' if _autotune else 'OFF'}, gemm_backends={_gemm_backends})")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    if include_fine:
        print(f"Fine feature normalization: {'ON' if fine_normalize else 'OFF'}")
    print(f"Dataset static cache: {'ON' if static_graph_cache else 'OFF'}")
    print(f"CUDA empty_cache per sample: {'ON' if empty_cache_each_step else 'OFF'}")
    if config.get("preload_fold", True) and hasattr(train_ds, "preload_fold"):
        print("Preloading fold feature data into RAM...")
        train_ds.preload_fold()
        print("Preload complete.")
    if use_amp:
        print(f"AMP dtype: {amp_dtype_name}")

    optimizer = optim.AdamW(model.parameters(), lr=float(config.get("lr", 1e-3)), weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=int(config.get("T_0", 20)),
        T_mult=int(config.get("T_mult", 2)),
    )
    n_epochs = int(config.get("epochs", 100))
    patience = int(config.get("patience", 20))

    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.json"
    history: list[dict] = []
    best_val_loss = float("inf")
    patience_counter = 0
    start_epoch = 1

    # Resume from a previous run if requested and a checkpoint exists.
    resume_ckpt = output_dir / "latest.pt"
    if not resume_ckpt.exists():
        resume_ckpt = output_dir / "best.pt"
    if config.get("resume") and resume_ckpt.exists():
        ckpt = torch.load(resume_ckpt, map_location=device, weights_only=False)
        validate_checkpoint_compatibility(ckpt, expected_model_type=model_type, require_contract_fields=False)
        ckpt_model_cfg = ckpt.get("model_config", {})
        if isinstance(ckpt_model_cfg, dict) and not bool(config.get("allow_resume_mismatch", False)):
            mismatches = {
                key: (ckpt_model_cfg[key], model_cfg[key])
                for key in sorted(set(ckpt_model_cfg).intersection(model_cfg))
                if ckpt_model_cfg[key] != model_cfg[key]
            }
            if mismatches:
                mismatch_str = ", ".join(
                    f"{k}: checkpoint={v[0]!r} current={v[1]!r}" for k, v in mismatches.items()
                )
                raise ValueError(
                    "Refusing to resume with mismatched model config. "
                    f"{mismatch_str}. "
                    "Use --allow-resume-mismatch to override."
                )
        sd = ckpt["model_state_dict"]
        # Normalize key prefix mismatch between compiled and uncompiled checkpoints.
        # torch.compile wraps the model as OptimizedModule, prepending _orig_mod. to all keys.
        # Checkpoints saved from a compiled model have _orig_mod.; plain saves don't.
        _is_compiled = hasattr(model, "_orig_mod")
        _sd_has_prefix = any(k.startswith("_orig_mod.") for k in sd)
        if _is_compiled and not _sd_has_prefix:
            sd = {"_orig_mod." + k: v for k, v in sd.items()}
        elif not _is_compiled and _sd_has_prefix:
            sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
        model.load_state_dict(sd)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        best_val_loss = float(ckpt.get("best_val_loss", ckpt.get("val_loss", float("inf"))))
        start_epoch = int(ckpt["epoch"]) + 1
        if history_path.exists():
            with history_path.open(encoding="utf-8") as f:
                history = json.load(f)
            if history:
                if int(history[-1].get("epoch", 0)) > int(ckpt["epoch"]):
                    history = [row for row in history if int(row.get("epoch", 0)) <= int(ckpt["epoch"])]
                    print(f"  Trimmed history to epoch {ckpt['epoch']} to match resumed checkpoint.")
                hist_vals = [float(row.get("val_loss", float("inf"))) for row in history]
                finite_hist_vals = [v for v in hist_vals if math.isfinite(v)]
                if finite_hist_vals:
                    best_val_loss = min(best_val_loss, min(finite_hist_vals))
        print(f"  Resumed from {resume_ckpt.name} at epoch {ckpt['epoch']} "
               f"(best_val_loss={best_val_loss:.4f}); continuing from epoch {start_epoch}.")
    elif config.get("resume"):
        print(f"  Resume requested, but no checkpoint found in {output_dir}; starting from scratch.")

    history_path = output_dir / "history.json"
    if not history_path.exists():
        with history_path.open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    if start_epoch > n_epochs:
        print(f"  Resume epoch ({start_epoch}) exceeds configured epochs ({n_epochs}); no training steps run.")
        return {"fold": fold, "best_val_loss": best_val_loss, "history": history}

    for epoch in range(start_epoch, n_epochs + 1):
        # Release the PyTorch VRAM pool at the start of each epoch. Without this, the
        # caching allocator retains peak allocations from the previous epoch's backward
        # passes (growing from ~12 GB to ~19 GB over one epoch with Batch B fine_mp).
        # At 95%+ reserved, HSA_DISABLE_FRAGMENT_ALLOCATOR=1 causes Triton kernel-load
        # hipMalloc calls to hang indefinitely. One empty_cache per epoch costs <1 ms.
        if device.type == "cuda":
            torch.cuda.empty_cache()
        t0 = time.time()
        physics_weight_scale = _physics_scale(epoch)
        train_log = train_one_epoch(
            model,
            train_ds,
            optimizer,
            device,
            use_amp=use_amp,
            scaler=scaler,
            empty_cache_each_step=empty_cache_each_step,
            amp_dtype=amp_dtype,
            physics_weight_scale=physics_weight_scale,
        )
        val_metrics = evaluate_split(
            model,
            val_ds,
            val_ids,
            device,
            use_amp=use_amp,
            empty_cache_each_step=empty_cache_each_step,
            amp_dtype=amp_dtype,
        )
        scheduler.step()
        lr = float(scheduler.get_last_lr()[0])
        val_loss = float(val_metrics.get("val_loss", float("inf")))
        row = {"epoch": epoch, "lr": lr, "elapsed_s": time.time() - t0, **train_log, **val_metrics}
        history.append(row)
        # Write after every epoch so history is never lost on crash.
        with history_path.open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        ts = datetime.now().strftime("%H:%M:%S")
        print(
            f"  [{ts}] Epoch {epoch:3d}/{n_epochs} | "
            f"train_loss={train_log.get('loss/total', 0.0):.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"detect_rate={val_metrics.get('detection_rate', 0.0):.3f} | "
            f"wr_frac={val_metrics.get('mean_wrinkled_frac_pred', 0.0):.3f} "
            f"(tgt={val_metrics.get('mean_wrinkled_frac_target', 0.0):.3f}) | "
            f"fine_match={val_metrics.get('fine/wrinkled_match_rate', 0.0):.3f} | "
            f"lr={lr:.2e} | {row['elapsed_s']:.1f}s"
        )
        ckpt_best_val = min(best_val_loss, val_loss)
        ckpt_dict = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": val_loss,
            "best_val_loss": ckpt_best_val,
            "val_metrics": val_metrics,
            "model_config": model_cfg,
        }
        add_checkpoint_contract_metadata(ckpt_dict, model_type=model_type)
        # Atomic write: save to .tmp then rename so a spot-eviction mid-write
        # never leaves a corrupted checkpoint that breaks resume.
        _ckpt_tmp = output_dir / "latest.pt.tmp"
        torch.save(ckpt_dict, _ckpt_tmp)
        _ckpt_tmp.rename(output_dir / "latest.pt")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            _best_tmp = output_dir / "best.pt.tmp"
            torch.save(ckpt_dict, _best_tmp)
            _best_tmp.rename(output_dir / "best.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch} (patience={patience})")
                break

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
    p.add_argument("--folds", type=str, default=None,
                   help="Comma-separated fold indices to run, e.g. '0,2,4'. "
                        "Runs the subset sequentially in this process; use "
                        "CUDA_VISIBLE_DEVICES to parallelize across GPUs.")
    p.add_argument("--sim-ids", type=str, default=None, help="Comma-separated sim IDs for custom run")
    p.add_argument("--max-train-sims", type=int, default=None)
    p.add_argument("--max-val-sims", type=int, default=None)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--attn-batch-nodes", type=int, default=512)
    p.add_argument("--max-timesteps", type=int, default=None, help="Subsample T axis to this many timesteps (None = use all 256)")
    p.add_argument("--temporal-strategy", type=str, default="tail", choices=["tail", "stride", "linspace"],
                   help="How to select timesteps: tail=last N consecutive (default), stride=every Nth, linspace=uniform")
    p.add_argument("--decoder-chunk-t", type=int, default=32,
                   help="Chunk the GRU decoder over this many timesteps at once (bounds decoder activation memory)")
    p.add_argument("--model-type", type=str, default="coarse", choices=["coarse", "cross-scale"],
                   help="coarse=FormingGraphNet (Model A, default); cross-scale=CrossScaleNet (Model B)")
    p.add_argument("--use-fine-mp", action="store_true",
                   help="Enable fine-mesh message passing in CrossScaleNet (Phase 2 architecture)")
    p.add_argument("--torch-compile", action="store_true",
                   help="Apply torch.compile (inductor, dynamic=True, mode=default). "
                        "First epoch ~2-10min slower (Triton compilation). ROCm-safe.")
    p.add_argument("--amp", action="store_true", help="Enable mixed precision on CUDA/HIP")
    p.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    p.add_argument(
        "--amp-dtype",
        type=str,
        default=_os.environ.get("CFWRINKLE_AMP_DTYPE", "float16"),
        choices=["float16", "bfloat16"],
        help="AMP dtype; bfloat16 can be more stable for ROCm memory pressure.",
    )
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--no-normalize", action="store_true")
    p.add_argument("--normalize-fine-features", action="store_true",
                   help="Apply per-channel z-score normalization to fine features (fs1, fs2, dz, thick)")
    p.add_argument("--resume", action="store_true",
                   help="Resume training from latest.pt (or best.pt) in the output directory")
    p.add_argument("--allow-resume-mismatch", action="store_true",
                   help="Allow --resume even if checkpoint model_config differs from current args")
    p.add_argument("--no-checkpoint", action="store_true",
                   help="Disable activation checkpointing to speed up at the cost of memory")
    p.add_argument("--no-dataset-static-cache", action="store_true",
                   help="Disable in-process caching of static graph tensors loaded from HDF5")
    p.add_argument("--cuda-empty-cache-per-sample", action="store_true",
                   help="Force torch.cuda.empty_cache() after each sample (slower; for diagnostics only)")
    p.add_argument("--no-preload-fold", action="store_true",
                   help="Disable per-fold preload of all feature tensors into RAM. "
                        "Required when per-fold preload exceeds available RAM (e.g. Batch B 30k-node "
                        "coarse mesh × 40 sims × T=96 = ~31 GB, exceeding WSL2 27 GB ceiling).")
    p.add_argument("--dry-run", action="store_true",
                   help="Run one forward pass (no_grad) on the first sim and report peak VRAM, then exit")
    p.add_argument("--output", type=Path, default=Path("checkpoints"))
    return p.parse_args()


def _run_one(
    fold: int,
    train_ids: list[str],
    val_ids: list[str],
    args: argparse.Namespace,
    out_dir: Path,
) -> dict:
    # Filter to sims that actually exist in WP3 (e.g. geom_0_6 is excluded).
    wp3_ids = set(_read_all_sim_ids())
    dropped_train = [s for s in train_ids if s not in wp3_ids]
    dropped_val = [s for s in val_ids if s not in wp3_ids]
    if dropped_train or dropped_val:
        print(f"  [fold {fold}] Skipping sims not in WP3: {dropped_train + dropped_val}")
    train_ids = [s for s in train_ids if s in wp3_ids]
    val_ids = [s for s in val_ids if s in wp3_ids]
    if not train_ids:
        raise RuntimeError(f"Fold {fold}: no valid training sims after WP3 filter.")
    if not val_ids:
        raise RuntimeError(f"Fold {fold}: no valid validation sims after WP3 filter.")

    if args.max_train_sims is not None:
        train_ids = train_ids[: args.max_train_sims]
    if args.max_val_sims is not None:
        val_ids = val_ids[: args.max_val_sims]
    normalize_fine_features = bool(getattr(args, "normalize_fine_features", False))
    static_graph_cache = not bool(getattr(args, "no_dataset_static_cache", False))
    empty_cache_each_step = bool(getattr(args, "cuda_empty_cache_per_sample", False))
    if normalize_fine_features and args.model_type != "cross-scale":
        raise ValueError("--normalize-fine-features requires --model-type cross-scale")
    model_cfg: dict[str, int | bool] = {
        "hidden_dim": args.hidden_dim,
        "attn_batch_nodes": args.attn_batch_nodes,
        "decoder_chunk_t": args.decoder_chunk_t,
        "use_checkpoint": not args.no_checkpoint,
    }
    if args.model_type == "cross-scale":
        model_cfg["use_fine_mp"] = args.use_fine_mp
    cfg = {
        "epochs": args.epochs,
        "patience": args.patience,
        "lr": args.lr,
        "device": args.device,
        "normalize": not args.no_normalize,
        "normalize_fine_features": normalize_fine_features,
        "amp": args.amp and not args.no_amp,
        "amp_dtype": getattr(args, "amp_dtype", "float16"),
        "max_timesteps": args.max_timesteps,
        "temporal_strategy": args.temporal_strategy,
        "resume": args.resume,
        "allow_resume_mismatch": args.allow_resume_mismatch,
        "model_type": args.model_type,
        "dataset_static_cache": static_graph_cache,
        "cuda_empty_cache_per_sample": empty_cache_each_step,
        "torch_compile": bool(getattr(args, "torch_compile", False)),
        "preload_fold": not bool(getattr(args, "no_preload_fold", False)),
        "model": model_cfg,
    }
    return train_fold(fold, train_ids, val_ids, cfg, out_dir)


def _dry_run(args: argparse.Namespace) -> None:
    """Single no_grad forward pass on the first available sim; reports peak VRAM."""
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    all_sim_ids = _read_all_sim_ids()
    if not all_sim_ids:
        raise RuntimeError(f"No simulations found in WP3 feature file: {WP3_H5}")
    sim_id = all_sim_ids[0]
    include_fine = args.model_type == "cross-scale"
    normalize_fine_features = bool(getattr(args, "normalize_fine_features", False))
    if normalize_fine_features and not include_fine:
        raise ValueError("--normalize-fine-features requires --model-type cross-scale")
    print(f"Dry-run: loading sim {sim_id!r} with max_timesteps={args.max_timesteps}, model_type={args.model_type}")
    ds = WrinkleDataset(WP3_H5, [sim_id], normalize=not args.no_normalize,
                        device="cpu", max_timesteps=args.max_timesteps, include_fine=include_fine,
                        fine_normalize=normalize_fine_features)
    model_cfg: dict[str, int | bool] = {
        "hidden_dim": args.hidden_dim,
        "attn_batch_nodes": args.attn_batch_nodes,
        "decoder_chunk_t": args.decoder_chunk_t,
        "use_checkpoint": not args.no_checkpoint,
    }
    if args.model_type == "cross-scale":
        model_cfg["use_fine_mp"] = args.use_fine_mp
    if args.model_type == "cross-scale":
        model: FormingGraphNet | CrossScaleNet = CrossScaleNet(**model_cfg).to(device)
    else:
        model = FormingGraphNet(**model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    batch = _move_to_device(ds[0], device)
    with torch.no_grad():
        model_out = model(batch)
    if isinstance(model_out, dict):
        print(f"Output shapes: coarse={tuple(model_out['coarse'].shape)} fine={tuple(model_out['fine'].shape)}")
        pred = model_out["coarse"]
    else:
        print(f"Output shape: {tuple(model_out.shape)}")
        pred = model_out

    if device.type == "cuda":
        fwd_peak_gb = torch.cuda.max_memory_allocated(device) / 1024 ** 3
        total_gb = torch.cuda.get_device_properties(device).total_memory / 1024 ** 3
        est_bwd_gb = fwd_peak_gb * 2.0
        print(f"Peak VRAM (forward only): {fwd_peak_gb:.2f} GB")
        print(f"Estimated peak with backward (~2×): {est_bwd_gb:.2f} GB")
        print(f"Card capacity: {total_gb:.2f} GB")
        margin = total_gb - est_bwd_gb
        print(f"Estimated margin: {margin:.2f} GB ({'OK' if margin > 1.0 else 'TIGHT — OOM risk'})")
    else:
        print("(VRAM estimation requires CUDA/HIP device)")


def main() -> None:
    args = _parse_args()

    if args.dry_run:
        _dry_run(args)
        return

    args.output.mkdir(parents=True, exist_ok=True)

    if args.sim_ids:
        sim_ids = [s.strip() for s in args.sim_ids.split(",") if s.strip()]
        if not sim_ids:
            raise ValueError("No valid sim IDs provided in --sim-ids.")
        result = _run_one(0, sim_ids, sim_ids, args, args.output)
        with (args.output / "summary.json").open("w", encoding="utf-8") as f:
            json.dump([result], f, indent=2)
        return

    if args.all_folds or args.folds is not None:
        if args.folds is not None:
            try:
                fold_ids = [int(f.strip()) for f in args.folds.split(",") if f.strip()]
            except ValueError as exc:
                raise ValueError(f"--folds must be comma-separated integers, got: {args.folds!r}") from exc
            if not fold_ids or any(f not in range(5) for f in fold_ids):
                raise ValueError(f"--folds: all indices must be in 0–4, got {fold_ids}")
        else:
            fold_ids = [0, 1, 2, 3, 4]
        results = []
        for fold in fold_ids:
            train_ids, val_ids = load_fold_sim_ids(WP3_H5, fold, wp2_h5_path=WP2_H5)
            results.append(_run_one(fold, train_ids, val_ids, args, args.output / f"fold_{fold}"))
        with (args.output / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        return

    train_ids, val_ids = load_fold_sim_ids(WP3_H5, args.fold, wp2_h5_path=WP2_H5)
    all_ids = set(_read_all_sim_ids())
    missing_train = [sid for sid in train_ids if sid not in all_ids]
    missing_val = [sid for sid in val_ids if sid not in all_ids]
    if missing_train or missing_val:
        print(f"[fold {args.fold}] Ignoring sims missing from WP3: train={missing_train} val={missing_val}")
    result = _run_one(args.fold, train_ids, val_ids, args, args.output)
    with (args.output / "summary.json").open("w", encoding="utf-8") as f:
        json.dump([result], f, indent=2)


if __name__ == "__main__":
    main()


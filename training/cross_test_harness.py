from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from model.contracts import (
    MODEL_TYPE_COARSE,
    MODEL_TYPE_CROSS_SCALE,
    SUPPORTED_MODEL_TYPES,
    validate_checkpoint_compatibility,
)
from model.cross_scale import CrossScaleNet
from model.dataset import WrinkleDataset, load_fold_sim_ids
from model.gnn import FormingGraphNet
from training.train import WP2_H5, WP3_H5, evaluate_split

REPORT_SCHEMA_VERSION = "1.0.0"
COARSE_METRIC_COLUMNS = (
    "detection_rate",
    "false_alarm_rate",
    "precision",
    "f1",
    "tp",
    "fp",
    "fn",
    "tn",
    "mean_severity_mae",
    "val_loss",
)
FINE_METRIC_COLUMNS = (
    "fine/stress_mae",
    "fine/dz_mae",
    "fine/thick_mae",
    "fine/compressive_frac",
)
METRIC_COLUMNS = COARSE_METRIC_COLUMNS + FINE_METRIC_COLUMNS


@dataclass(frozen=True)
class ModelArtifactSpec:
    name: str
    checkpoint: Path
    model_type: str | None = None


def _coerce_metric_value(value: Any) -> tuple[float | None, str]:
    if value is None:
        return None, "na:missing"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None, "na:invalid"
    if not math.isfinite(numeric):
        return None, "na:non_finite"
    return numeric, "ok"


def standardize_metrics(raw_metrics: dict[str, Any], *, model_type: str) -> tuple[dict[str, float | None], dict[str, str]]:
    metrics: dict[str, float | None] = {}
    statuses: dict[str, str] = {}
    for column in COARSE_METRIC_COLUMNS:
        value, status = _coerce_metric_value(raw_metrics.get(column))
        metrics[column] = value
        statuses[column] = status
    for column in FINE_METRIC_COLUMNS:
        if model_type != MODEL_TYPE_CROSS_SCALE:
            metrics[column] = None
            statuses[column] = "na:not_applicable"
            continue
        value, status = _coerce_metric_value(raw_metrics.get(column))
        metrics[column] = value
        statuses[column] = status
    return metrics, statuses


def _sort_key_for_row(row: dict[str, Any]) -> tuple[Any, ...]:
    val_loss = row["metrics"].get("val_loss")
    f1 = row["metrics"].get("f1")
    val_loss_sort = float("inf") if val_loss is None else float(val_loss)
    f1_sort = float("-inf") if f1 is None else float(f1)
    return (val_loss_sort, -f1_sort, str(row["artifact_name"]))


def build_comparison_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(rows, key=_sort_key_for_row)
    table: list[dict[str, Any]] = []
    for rank, row in enumerate(ranked, start=1):
        flat = {
            "rank": rank,
            "artifact_name": row["artifact_name"],
            "model_type": row["model_type"],
            "checkpoint": row["checkpoint"],
        }
        for metric_key in METRIC_COLUMNS:
            flat[metric_key] = row["metrics"].get(metric_key)
        table.append(flat)
    return table


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _load_model(checkpoint_path: Path, *, run_device: torch.device, model_type_override: str | None, require_contract: bool):
    ckpt = torch.load(checkpoint_path, map_location=run_device, weights_only=False)
    validate_checkpoint_compatibility(
        ckpt,
        expected_model_type=model_type_override,
        require_contract_fields=require_contract,
    )
    resolved_model_type = str(ckpt.get("model_type", MODEL_TYPE_COARSE))
    model_cfg = ckpt.get("model_config", {})
    if resolved_model_type == MODEL_TYPE_CROSS_SCALE:
        model = CrossScaleNet(**model_cfg)
    elif resolved_model_type == MODEL_TYPE_COARSE:
        model = FormingGraphNet(**model_cfg)
    else:
        raise ValueError(f"Unsupported checkpoint model_type={resolved_model_type!r}")
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(run_device).eval()
    return model, ckpt, resolved_model_type


def run_cross_test_harness(
    artifacts: list[ModelArtifactSpec],
    *,
    eval_sim_ids: list[str],
    wp3_h5_path: str | Path = WP3_H5,
    device: str = "auto",
    normalize: bool = True,
    max_timesteps: int | None = None,
    temporal_strategy: str = "tail",
    severity_threshold: float = 0.15,
    require_contract: bool = False,
) -> dict[str, Any]:
    if not artifacts:
        raise ValueError("At least one model artifact is required")
    if not eval_sim_ids:
        raise ValueError("eval_sim_ids must not be empty")
    names = [a.name for a in artifacts]
    if len(set(names)) != len(names):
        raise ValueError("Artifact names must be unique")

    run_device = _resolve_device(device)
    wp3_h5 = Path(wp3_h5_path)
    datasets: dict[bool, WrinkleDataset] = {}
    artifact_rows: list[dict[str, Any]] = []

    for artifact in artifacts:
        model, ckpt, resolved_model_type = _load_model(
            artifact.checkpoint,
            run_device=run_device,
            model_type_override=artifact.model_type,
            require_contract=require_contract,
        )
        include_fine = resolved_model_type == MODEL_TYPE_CROSS_SCALE
        if include_fine not in datasets:
            datasets[include_fine] = WrinkleDataset(
                wp3_h5,
                eval_sim_ids,
                normalize=normalize,
                device="cpu",
                max_timesteps=max_timesteps,
                temporal_strategy=temporal_strategy,
                include_fine=include_fine,
            )
        raw_metrics = evaluate_split(
            model,
            datasets[include_fine],
            eval_sim_ids,
            run_device,
            severity_threshold=severity_threshold,
            use_amp=False,
        )
        metrics, metric_status = standardize_metrics(raw_metrics, model_type=resolved_model_type)
        artifact_rows.append(
            {
                "artifact_name": artifact.name,
                "checkpoint": str(artifact.checkpoint),
                "model_type": resolved_model_type,
                "model_version": ckpt.get("model_version"),
                "checkpoint_format_version": ckpt.get("checkpoint_format_version"),
                "metrics": metrics,
                "metric_status": metric_status,
            }
        )

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "eval_context": {
            "sim_ids": list(eval_sim_ids),
            "sim_count": len(eval_sim_ids),
            "wp3_h5_path": str(wp3_h5),
            "device": str(run_device),
            "normalize": bool(normalize),
            "max_timesteps": max_timesteps,
            "temporal_strategy": temporal_strategy,
            "severity_threshold": float(severity_threshold),
            "require_contract": bool(require_contract),
        },
        "metric_columns": {
            "coarse": list(COARSE_METRIC_COLUMNS),
            "fine_cross_scale_only": list(FINE_METRIC_COLUMNS),
            "all": list(METRIC_COLUMNS),
        },
        "artifacts": artifact_rows,
        "comparison_table": build_comparison_table(artifact_rows),
    }
    return report


def _parse_artifact(text: str) -> ModelArtifactSpec:
    if "=" not in text:
        raise ValueError(f"Invalid --artifact value {text!r}; expected name=checkpoint_path")
    name, checkpoint = text.split("=", 1)
    name = name.strip()
    checkpoint = checkpoint.strip()
    if not name or not checkpoint:
        raise ValueError(f"Invalid --artifact value {text!r}; expected non-empty name and checkpoint path")
    return ModelArtifactSpec(name=name, checkpoint=Path(checkpoint))


def _parse_artifact_type(text: str) -> tuple[str, str]:
    if "=" not in text:
        raise ValueError(f"Invalid --artifact-model-type value {text!r}; expected name=model_type")
    name, model_type = text.split("=", 1)
    name = name.strip()
    model_type = model_type.strip()
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(f"Unsupported model type {model_type!r}; expected one of {SUPPORTED_MODEL_TYPES}")
    return name, model_type


def _load_artifacts_from_spec(path: Path) -> list[ModelArtifactSpec]:
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError("Artifact spec JSON must be a list")
    artifacts: list[ModelArtifactSpec] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("Artifact spec entries must be objects")
        name = item.get("name")
        checkpoint = item.get("checkpoint")
        model_type = item.get("model_type")
        if not isinstance(name, str) or not isinstance(checkpoint, str):
            raise ValueError("Artifact spec entries require string fields 'name' and 'checkpoint'")
        if model_type is not None and model_type not in SUPPORTED_MODEL_TYPES:
            raise ValueError(f"Artifact {name!r} has unsupported model_type {model_type!r}")
        artifacts.append(ModelArtifactSpec(name=name, checkpoint=Path(checkpoint), model_type=model_type))
    return artifacts


def _resolve_eval_ids(sim_ids_arg: str | None, *, fold: int, wp3_h5_path: Path, wp2_h5_path: Path) -> list[str]:
    if sim_ids_arg:
        return [sid.strip() for sid in sim_ids_arg.split(",") if sid.strip()]
    _, val_ids = load_fold_sim_ids(wp3_h5_path, fold, wp2_h5_path=wp2_h5_path)
    return val_ids


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cross-test harness for model artifacts")
    p.add_argument("--artifact", action="append", default=[], help="Model artifact entry: name=checkpoint_path")
    p.add_argument("--artifact-model-type", action="append", default=[], help="Optional override: name=model_type")
    p.add_argument("--artifact-spec", type=Path, default=None, help="JSON list of artifact objects")
    p.add_argument("--sim-ids", type=str, default=None, help="Comma-separated eval sim IDs")
    p.add_argument("--fold", type=int, default=0, help="Validation fold used when --sim-ids is omitted")
    p.add_argument("--wp3-h5", type=Path, default=WP3_H5)
    p.add_argument("--wp2-h5", type=Path, default=WP2_H5)
    p.add_argument("--output-report", type=Path, required=True)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--no-normalize", action="store_true")
    p.add_argument("--max-timesteps", type=int, default=None)
    p.add_argument("--temporal-strategy", type=str, default="tail", choices=["tail", "stride", "linspace"])
    p.add_argument("--severity-threshold", type=float, default=0.15)
    p.add_argument("--require-contract", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    artifacts: list[ModelArtifactSpec] = []
    if args.artifact_spec is not None:
        artifacts.extend(_load_artifacts_from_spec(args.artifact_spec))
    artifacts.extend(_parse_artifact(item) for item in args.artifact)
    if not artifacts:
        raise SystemExit("At least one artifact is required via --artifact and/or --artifact-spec")

    overrides = dict(_parse_artifact_type(item) for item in args.artifact_model_type)
    artifacts = [
        ModelArtifactSpec(
            name=artifact.name,
            checkpoint=artifact.checkpoint,
            model_type=overrides.get(artifact.name, artifact.model_type),
        )
        for artifact in artifacts
    ]

    eval_sim_ids = _resolve_eval_ids(
        args.sim_ids,
        fold=args.fold,
        wp3_h5_path=Path(args.wp3_h5),
        wp2_h5_path=Path(args.wp2_h5),
    )
    report = run_cross_test_harness(
        artifacts,
        eval_sim_ids=eval_sim_ids,
        wp3_h5_path=args.wp3_h5,
        device=args.device,
        normalize=not args.no_normalize,
        max_timesteps=args.max_timesteps,
        temporal_strategy=args.temporal_strategy,
        severity_threshold=args.severity_threshold,
        require_contract=args.require_contract,
    )
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    with args.output_report.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    print(json.dumps(report["comparison_table"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

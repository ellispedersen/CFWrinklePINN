from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from model.contracts import (
    MODEL_TYPE_COARSE,
    MODEL_TYPE_CROSS_SCALE,
    SUPPORTED_MODEL_TYPES,
    validate_checkpoint_compatibility,
    validate_contract_invariants,
)
from model.cross_scale import CrossScaleNet
from model.gnn import FormingGraphNet
from model.labels import COARSE_TARGET_INDEX
from wp3_features.extract_single import extract_single_simulation

INFERENCE_OUTPUT_SCHEMA_VERSION = "1.0.0"
INFERENCE_RISK_THRESHOLD = 0.15


def _validate_required_array(
    extracted: dict[str, Any],
    *,
    key: str,
    dtype: np.dtype[Any],
    ndim: int,
) -> np.ndarray:
    if key not in extracted:
        raise KeyError(f"Extracted data missing required key {key!r}")
    value = extracted[key]
    if not isinstance(value, np.ndarray):
        raise TypeError(f"Extracted key {key!r} must be numpy.ndarray, got {type(value).__name__}")
    if value.dtype != np.dtype(dtype):
        raise TypeError(f"Extracted key {key!r} must have dtype={np.dtype(dtype)}, got {value.dtype}")
    if value.ndim != ndim:
        raise ValueError(f"Extracted key {key!r} must have ndim={ndim}, got shape={tuple(value.shape)}")
    return value


def validate_extracted_schema(extracted: dict[str, Any], *, model_type: str) -> None:
    node_features = _validate_required_array(extracted, key="node_features", dtype=np.float32, ndim=3)
    edge_index = _validate_required_array(extracted, key="edge_index", dtype=np.int64, ndim=2)
    edge_attr = _validate_required_array(extracted, key="edge_attr", dtype=np.float32, ndim=2)
    material_card = _validate_required_array(extracted, key="material_card", dtype=np.float32, ndim=1)
    elements = _validate_required_array(extracted, key="elements", dtype=np.int64, ndim=2)

    if edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must have shape (2, E), got {tuple(edge_index.shape)}")
    if edge_attr.shape[0] != edge_index.shape[1]:
        raise ValueError(
            f"edge_attr row count must match edge_index columns, got {edge_attr.shape[0]} != {edge_index.shape[1]}"
        )
    if material_card.shape[0] != 8:
        raise ValueError(f"material_card must have shape (8,), got {tuple(material_card.shape)}")
    if elements.shape[1] != 3:
        raise ValueError(f"elements must have shape (M, 3), got {tuple(elements.shape)}")
    if node_features.shape[0] <= 0 or node_features.shape[1] <= 0:
        raise ValueError(f"node_features must have non-zero T and N dims, got {tuple(node_features.shape)}")

    if model_type == MODEL_TYPE_CROSS_SCALE:
        coarse_to_fine = _validate_required_array(extracted, key="coarse_to_fine", dtype=np.int64, ndim=2)
        fine_elements = _validate_required_array(extracted, key="fine_elements", dtype=np.int64, ndim=2)
        if coarse_to_fine.shape[0] != 2:
            raise ValueError(f"coarse_to_fine must have shape (2, K), got {tuple(coarse_to_fine.shape)}")
        if fine_elements.shape[1] != 3:
            raise ValueError(f"fine_elements must have shape (M_fine, 3), got {tuple(fine_elements.shape)}")

        if "fine_node_coarse_map" in extracted:
            _validate_required_array(extracted, key="fine_node_coarse_map", dtype=np.int64, ndim=1)
        if "fine_edge_index" in extracted:
            fine_edge_index = _validate_required_array(extracted, key="fine_edge_index", dtype=np.int64, ndim=2)
            if fine_edge_index.shape[0] != 2:
                raise ValueError(f"fine_edge_index must have shape (2, E_fine), got {tuple(fine_edge_index.shape)}")
        if "fine_edge_attr" in extracted:
            fine_edge_attr = _validate_required_array(extracted, key="fine_edge_attr", dtype=np.float32, ndim=2)
            if "fine_edge_index" in extracted and fine_edge_attr.shape[0] != extracted["fine_edge_index"].shape[1]:
                raise ValueError(
                    "fine_edge_attr row count must match fine_edge_index columns, "
                    f"got {fine_edge_attr.shape[0]} != {extracted['fine_edge_index'].shape[1]}"
                )


def _build_batch(extracted: dict[str, Any], *, run_device: torch.device, model_type: str) -> dict[str, torch.Tensor]:
    validate_extracted_schema(extracted, model_type=model_type)
    batch = {
        "node_features": torch.as_tensor(extracted["node_features"], dtype=torch.float32, device=run_device),
        "edge_index": torch.as_tensor(extracted["edge_index"], dtype=torch.int64, device=run_device),
        "edge_attr": torch.as_tensor(extracted["edge_attr"], dtype=torch.float32, device=run_device),
        "material_card": torch.as_tensor(extracted["material_card"], dtype=torch.float32, device=run_device),
        "elements": torch.as_tensor(extracted["elements"], dtype=torch.int64, device=run_device),
    }
    if model_type == MODEL_TYPE_CROSS_SCALE:
        batch["coarse_to_fine"] = torch.as_tensor(extracted["coarse_to_fine"], dtype=torch.int64, device=run_device)
        batch["fine_elements"] = torch.as_tensor(extracted["fine_elements"], dtype=torch.int64, device=run_device)
        if "fine_node_coarse_map" in extracted:
            batch["fine_node_coarse_map"] = torch.as_tensor(
                extracted["fine_node_coarse_map"], dtype=torch.int64, device=run_device
            )
        if "fine_edge_index" in extracted:
            batch["fine_edge_index"] = torch.as_tensor(extracted["fine_edge_index"], dtype=torch.int64, device=run_device)
        if "fine_edge_attr" in extracted:
            batch["fine_edge_attr"] = torch.as_tensor(extracted["fine_edge_attr"], dtype=torch.float32, device=run_device)
    return batch


def run_inference(
    results_dir: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path | None = None,
    visualize: bool = False,
    device: str = "cuda",
    model_type: str | None = None,
    require_contract: bool = False,
    wp3_h5_path: str | Path | None = None,
) -> dict[str, Any]:
    validate_contract_invariants()
    results_dir = Path(results_dir)
    sim_id = results_dir.stem.replace(".Results", "")
    run_device = torch.device(device if torch.cuda.is_available() else "cpu")

    if model_type is not None and model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(f"Unsupported model_type override {model_type!r}; expected one of {SUPPORTED_MODEL_TYPES}")

    ckpt = torch.load(checkpoint_path, map_location=run_device, weights_only=False)
    validate_checkpoint_compatibility(
        ckpt,
        expected_model_type=model_type,
        require_contract_fields=require_contract,
    )
    resolved_model_type = str(ckpt.get("model_type", MODEL_TYPE_COARSE))
    model_cfg = ckpt.get("model_config", {})

    extracted = extract_single_simulation(results_dir, sim_id=sim_id, wp3_h5_path=wp3_h5_path)
    batch = _build_batch(extracted, run_device=run_device, model_type=resolved_model_type)

    if resolved_model_type == MODEL_TYPE_CROSS_SCALE:
        model = CrossScaleNet(**model_cfg)
    elif resolved_model_type == MODEL_TYPE_COARSE:
        model = FormingGraphNet(**model_cfg)
    else:
        raise ValueError(f"Unsupported checkpoint model_type={resolved_model_type!r}")
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(run_device).eval()
    with torch.no_grad():
        model_out = model(batch)
        pred = model_out["coarse"] if isinstance(model_out, dict) else model_out

    severity = pred[..., COARSE_TARGET_INDEX["severity"]]
    max_severity = float(torch.nan_to_num(severity, nan=0.0).max().detach().cpu())
    risk_threshold = INFERENCE_RISK_THRESHOLD
    risk_elements = (
        torch.where(torch.nan_to_num(severity[-1], nan=0.0) >= 0.1)[0].detach().cpu().numpy().astype(int).tolist()
    )
    risk_elements.sort()
    result = {
        "schema_version": INFERENCE_OUTPUT_SCHEMA_VERSION,
        "sim_id": sim_id,
        "source": {
            "results_dir": str(results_dir),
            "checkpoint": str(checkpoint_path),
            "device": str(run_device),
        },
        "model": {
            "type": resolved_model_type,
            "config": model_cfg,
        },
        "contract": {
            "checkpoint_format_version": ckpt.get("checkpoint_format_version"),
            "model_version": ckpt.get("model_version"),
            "metadata": ckpt.get("contract"),
            "require_contract": bool(require_contract),
        },
        "prediction": {
            "max_severity": max_severity,
            "is_wrinkled_prediction": bool(max_severity >= risk_threshold),
            "wrinkle_risk_elements": risk_elements,
            "prediction_shape": [int(x) for x in pred.shape],
            "risk_threshold": risk_threshold,
        },
    }

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, sort_keys=True)

    if visualize:
        from training.visualize import plot_mesh_predictions

        if "targets" in extracted:
            tgt = torch.as_tensor(extracted["targets"], dtype=torch.float32)
            plot_mesh_predictions(sim_id, pred.cpu(), tgt, extracted.get("h5_path", "data/cfwrinkle_wp3_features.h5"))

    return result


def run_inference_batch(
    results_dirs: Sequence[str | Path],
    checkpoint_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    visualize: bool = False,
    device: str = "cuda",
    model_type: str | None = None,
    require_contract: bool = False,
    wp3_h5_path: str | Path | None = None,
    continue_on_error: bool = False,
) -> dict[str, Any]:
    resolved_dirs = sorted((Path(p) for p in results_dirs), key=lambda p: p.name)
    out_dir = Path(output_dir) if output_dir else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for results_dir in resolved_dirs:
        try:
            per_output = None
            if out_dir is not None:
                sim_id = results_dir.stem.replace(".Results", "")
                per_output = out_dir / f"{sim_id}.inference.json"
            run_result = run_inference(
                results_dir=results_dir,
                checkpoint_path=checkpoint_path,
                output_path=per_output,
                visualize=visualize,
                device=device,
                model_type=model_type,
                require_contract=require_contract,
                wp3_h5_path=wp3_h5_path,
            )
            results.append(run_result)
        except Exception as exc:
            if not continue_on_error:
                raise
            failures.append({"results_dir": str(results_dir), "error": str(exc)})

    return {
        "schema_version": INFERENCE_OUTPUT_SCHEMA_VERSION,
        "batch_size": len(resolved_dirs),
        "results": results,
        "failures": failures,
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Single-simulation inference")
    p.add_argument("--results-dir", default=None)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", default=None)
    p.add_argument("--visualize", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--model-type",
        choices=("auto", *SUPPORTED_MODEL_TYPES),
        default="auto",
        help="Optional model type override. Default 'auto' uses checkpoint metadata.",
    )
    p.add_argument("--require-contract", action="store_true", help="Fail if checkpoint is missing contract metadata.")
    p.add_argument("--wp3-h5", default=None, help="Override WP3 HDF5 path used by extraction.")
    p.add_argument("--batch-root", default=None, help="Optional root directory containing multiple *.Results folders.")
    p.add_argument("--batch-pattern", default="*.Results", help="Glob pattern under --batch-root for results folders.")
    p.add_argument("--batch-output-dir", default=None, help="Optional directory for per-simulation output JSON files.")
    p.add_argument("--continue-on-error", action="store_true", help="Batch mode only: continue after per-simulation failures.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    model_type_override = None if args.model_type == "auto" else args.model_type
    if args.batch_root:
        batch_root = Path(args.batch_root)
        results_dirs = sorted(p for p in batch_root.glob(args.batch_pattern) if p.is_dir())
        result = run_inference_batch(
            results_dirs=results_dirs,
            checkpoint_path=args.checkpoint,
            output_dir=args.batch_output_dir,
            visualize=args.visualize,
            device=args.device,
            model_type=model_type_override,
            require_contract=args.require_contract,
            wp3_h5_path=args.wp3_h5,
            continue_on_error=args.continue_on_error,
        )
    else:
        if not args.results_dir:
            raise SystemExit("--results-dir is required unless --batch-root is provided")
        result = run_inference(
            args.results_dir,
            args.checkpoint,
            args.output,
            args.visualize,
            args.device,
            model_type=model_type_override,
            require_contract=args.require_contract,
            wp3_h5_path=args.wp3_h5,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


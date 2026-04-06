from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from model.gnn import FormingGraphNet


def run_inference(
    results_dir: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path | None = None,
    visualize: bool = False,
    device: str = "cuda",
) -> dict[str, Any]:
    results_dir = Path(results_dir)
    sim_id = results_dir.stem.replace(".Results", "")
    run_device = torch.device(device if torch.cuda.is_available() else "cpu")

    from wp3_features.extract_single import extract_single_simulation

    extracted = extract_single_simulation(results_dir, sim_id=sim_id)
    batch = {
        "sim_id": sim_id,
        "node_features": torch.as_tensor(extracted["node_features"], dtype=torch.float32, device=run_device),
        "edge_index": torch.as_tensor(extracted["edge_index"], dtype=torch.int64, device=run_device),
        "edge_attr": torch.as_tensor(extracted["edge_attr"], dtype=torch.float32, device=run_device),
        "material_card": torch.as_tensor(extracted["material_card"], dtype=torch.float32, device=run_device),
        "elements": torch.as_tensor(extracted["elements"], dtype=torch.int64, device=run_device),
    }

    ckpt = torch.load(checkpoint_path, map_location=run_device)
    model_cfg = ckpt.get("model_config", {})
    model = FormingGraphNet(**model_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(run_device).eval()
    with torch.no_grad():
        pred = model(batch)

    severity = pred[..., 0]
    max_severity = float(torch.nan_to_num(severity, nan=0.0).max().detach().cpu())
    risk_threshold = 0.15
    risk_elements = (
        torch.where(torch.nan_to_num(severity[-1], nan=0.0) >= 0.1)[0].detach().cpu().numpy().astype(int).tolist()
    )
    result = {
        "sim_id": sim_id,
        "max_severity": max_severity,
        "is_wrinkled_prediction": bool(max_severity >= risk_threshold),
        "wrinkle_risk_elements": risk_elements,
        "prediction_shape": tuple(int(x) for x in pred.shape),
    }

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    if visualize:
        from training.visualize import plot_mesh_predictions

        if "targets" in extracted:
            tgt = torch.as_tensor(extracted["targets"], dtype=torch.float32)
            plot_mesh_predictions(sim_id, pred.cpu(), tgt, extracted.get("h5_path", "data/cfwrinkle_wp3_features.h5"))

    return result


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Single-simulation inference")
    p.add_argument("--results-dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", default=None)
    p.add_argument("--visualize", action="store_true")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_inference(args.results_dir, args.checkpoint, args.output, args.visualize, args.device)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()


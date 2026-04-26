from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from model.contracts import MODEL_TYPE_COARSE, MODEL_TYPE_CROSS_SCALE, add_checkpoint_contract_metadata
from training import infer

torch = pytest.importorskip("torch")


def _coarse_extracted() -> dict:
    return {
        "node_features": np.zeros((2, 5, 74), dtype=np.float32),
        "edge_index": np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64),
        "edge_attr": np.zeros((3, 4), dtype=np.float32),
        "material_card": np.zeros((8,), dtype=np.float32),
        "elements": np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]], dtype=np.int64),
    }


class _FakeCoarseModel:
    def __init__(self, **_: object) -> None:
        pass

    def load_state_dict(self, _: dict) -> None:
        return None

    def to(self, _: torch.device):  # type: ignore[name-defined]
        return self

    def eval(self):
        return self

    def __call__(self, _: dict) -> torch.Tensor:
        pred = torch.zeros((2, 3, 4), dtype=torch.float32)
        pred[1, 1, 0] = 0.2
        pred[1, 2, 0] = 0.1
        return pred


def test_run_inference_includes_deterministic_contract_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    ckpt = {"model_config": {}, "model_state_dict": {}, "model_type": MODEL_TYPE_COARSE}
    add_checkpoint_contract_metadata(ckpt, model_type=MODEL_TYPE_COARSE)
    monkeypatch.setattr(infer.torch, "load", lambda *args, **kwargs: ckpt)
    monkeypatch.setattr(infer, "extract_single_simulation", lambda *args, **kwargs: _coarse_extracted())
    monkeypatch.setattr(infer, "FormingGraphNet", _FakeCoarseModel)

    out = infer.run_inference("sim_a.Results", "model.pt", model_type=MODEL_TYPE_COARSE, require_contract=True, device="cpu")

    assert out["schema_version"] == infer.INFERENCE_OUTPUT_SCHEMA_VERSION
    assert out["model"]["type"] == MODEL_TYPE_COARSE
    assert out["prediction"]["wrinkle_risk_elements"] == [1, 2]
    assert out["prediction"]["prediction_shape"] == [2, 3, 4]
    assert out["contract"]["metadata"]["model_type"] == MODEL_TYPE_COARSE


def test_run_inference_rejects_model_type_override_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    ckpt = {"model_config": {}, "model_state_dict": {}, "model_type": MODEL_TYPE_COARSE}
    add_checkpoint_contract_metadata(ckpt, model_type=MODEL_TYPE_COARSE)
    monkeypatch.setattr(infer.torch, "load", lambda *args, **kwargs: ckpt)

    with pytest.raises(ValueError, match="model_type mismatch"):
        infer.run_inference("sim_a.Results", "model.pt", model_type=MODEL_TYPE_CROSS_SCALE, device="cpu")


def test_run_inference_cross_scale_requires_cross_scale_extraction_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    ckpt = {"model_config": {}, "model_state_dict": {}, "model_type": MODEL_TYPE_CROSS_SCALE}
    add_checkpoint_contract_metadata(ckpt, model_type=MODEL_TYPE_CROSS_SCALE)
    monkeypatch.setattr(infer.torch, "load", lambda *args, **kwargs: ckpt)
    monkeypatch.setattr(infer, "extract_single_simulation", lambda *args, **kwargs: _coarse_extracted())

    with pytest.raises(KeyError, match="coarse_to_fine"):
        infer.run_inference("sim_a.Results", "model.pt", device="cpu")


def test_run_inference_batch_processes_results_dirs_in_sorted_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "b.Results").mkdir()
    (tmp_path / "a.Results").mkdir()
    calls: list[str] = []

    def _fake_run(results_dir: str | Path, *args, **kwargs) -> dict:
        sim_id = Path(results_dir).stem.replace(".Results", "")
        calls.append(sim_id)
        return {"sim_id": sim_id}

    monkeypatch.setattr(infer, "run_inference", _fake_run)
    out = infer.run_inference_batch(
        results_dirs=[tmp_path / "b.Results", tmp_path / "a.Results"],
        checkpoint_path="model.pt",
    )

    assert calls == ["a", "b"]
    assert [r["sim_id"] for r in out["results"]] == ["a", "b"]

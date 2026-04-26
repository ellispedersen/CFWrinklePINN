from __future__ import annotations

import sys

import pytest

from model.contracts import MODEL_TYPE_COARSE, add_checkpoint_contract_metadata
from training import validate_contracts as validate_mod


def test_preflight_main_without_checkpoint(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["validate_contracts"])
    validate_mod.main()
    out = capsys.readouterr().out
    assert "Contract validation passed." in out


def test_preflight_main_with_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    torch = pytest.importorskip("torch")
    ckpt = {"model_config": {"hidden_dim": 32}, "model_state_dict": {}}
    add_checkpoint_contract_metadata(ckpt, model_type=MODEL_TYPE_COARSE)
    path = tmp_path / "ckpt.pt"

    torch.save(ckpt, path)
    monkeypatch.setattr(sys, "argv", ["validate_contracts", "--checkpoint", str(path)])
    validate_mod.main()
    out = capsys.readouterr().out
    assert "Contract validation passed." in out

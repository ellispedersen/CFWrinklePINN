from __future__ import annotations

import argparse
import json

import pytest
import torch

from model.gnn import FormingGraphNet
from training import train as train_mod


class _DummyDataset:
    def __init__(self, _h5_path, sim_ids, **_kwargs) -> None:
        self.sim_ids = list(sim_ids)

    def __len__(self) -> int:
        return len(self.sim_ids)

    def __getitem__(self, idx: int) -> dict:
        return {"sim_id": self.sim_ids[idx]}

    def index_of(self, sim_id: str) -> int:
        return self.sim_ids.index(sim_id)


def _make_checkpoint(model_cfg: dict, epoch: int, val_loss: float, best_val_loss: float | None = None) -> dict:
    model = FormingGraphNet(**model_cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "val_loss": float(val_loss),
        "best_val_loss": float(best_val_loss if best_val_loss is not None else val_loss),
        "val_metrics": {"detection_rate": 0.1},
        "model_config": dict(model_cfg),
    }


def _patch_fast_loop(monkeypatch: pytest.MonkeyPatch, val_losses: list[float] | None = None) -> None:
    monkeypatch.setattr(train_mod, "WrinkleDataset", _DummyDataset)
    monkeypatch.setattr(train_mod, "train_one_epoch", lambda *a, **k: {"loss/total": 1.0})
    values = list(val_losses or [0.9])
    state = {"i": 0}

    def _fake_eval(*_a, **_k) -> dict[str, float]:
        i = state["i"]
        state["i"] += 1
        return {"val_loss": values[min(i, len(values) - 1)], "detection_rate": 0.0}

    monkeypatch.setattr(train_mod, "evaluate_split", _fake_eval)


def test_resume_prefers_latest_checkpoint_and_continues_epoch(tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_fast_loop(monkeypatch, val_losses=[0.25])
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    model_cfg = {"hidden_dim": 32, "attn_batch_nodes": 64, "decoder_chunk_t": 8}
    torch.save(_make_checkpoint(model_cfg, epoch=3, val_loss=0.4, best_val_loss=0.3), out_dir / "latest.pt")
    torch.save(_make_checkpoint(model_cfg, epoch=10, val_loss=0.2, best_val_loss=0.2), out_dir / "best.pt")
    with (out_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump([{"epoch": 1, "val_loss": 0.7}, {"epoch": 2, "val_loss": 0.5}, {"epoch": 3, "val_loss": 0.4}], f)

    result = train_mod.train_fold(
        fold=0,
        train_ids=["sim_a"],
        val_ids=["sim_a"],
        config={"epochs": 4, "patience": 10, "lr": 1e-3, "resume": True, "model": model_cfg},
        output_dir=out_dir,
    )

    epochs = [row["epoch"] for row in result["history"]]
    assert epochs == [1, 2, 3, 4]


def test_resume_mismatched_model_config_raises_clear_error(
    tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_fast_loop(monkeypatch)
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    ckpt_model_cfg = {"hidden_dim": 32, "attn_batch_nodes": 64, "decoder_chunk_t": 8}
    run_model_cfg = {"hidden_dim": 16, "attn_batch_nodes": 64, "decoder_chunk_t": 8}
    torch.save(_make_checkpoint(ckpt_model_cfg, epoch=1, val_loss=0.9), out_dir / "latest.pt")

    with pytest.raises(ValueError, match="mismatched model config"):
        train_mod.train_fold(
            fold=0,
            train_ids=["sim_a"],
            val_ids=["sim_a"],
            config={"epochs": 2, "patience": 10, "lr": 1e-3, "resume": True, "model": run_model_cfg},
            output_dir=out_dir,
        )


def test_resume_trims_history_ahead_of_checkpoint(tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_fast_loop(monkeypatch, val_losses=[0.3])
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    model_cfg = {"hidden_dim": 32, "attn_batch_nodes": 64, "decoder_chunk_t": 8}
    torch.save(_make_checkpoint(model_cfg, epoch=2, val_loss=0.5), out_dir / "latest.pt")
    with (out_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(
            [{"epoch": 1, "val_loss": 0.8}, {"epoch": 2, "val_loss": 0.5}, {"epoch": 3, "val_loss": 0.4}],
            f,
        )

    result = train_mod.train_fold(
        fold=0,
        train_ids=["sim_a"],
        val_ids=["sim_a"],
        config={"epochs": 3, "patience": 10, "lr": 1e-3, "resume": True, "model": model_cfg},
        output_dir=out_dir,
    )
    assert [row["epoch"] for row in result["history"]] == [1, 2, 3]


def test_resume_falls_back_to_best_checkpoint_when_latest_missing(
    tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_fast_loop(monkeypatch)
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    best_cfg = {"hidden_dim": 16, "attn_batch_nodes": 64, "decoder_chunk_t": 8}
    run_cfg = {"hidden_dim": 32, "attn_batch_nodes": 64, "decoder_chunk_t": 8}
    torch.save(_make_checkpoint(best_cfg, epoch=1, val_loss=0.9), out_dir / "best.pt")

    with pytest.raises(ValueError, match="mismatched model config"):
        train_mod.train_fold(
            fold=0,
            train_ids=["sim_a"],
            val_ids=["sim_a"],
            config={"epochs": 2, "patience": 10, "lr": 1e-3, "resume": True, "model": run_cfg},
            output_dir=out_dir,
        )


def test_resume_epoch_after_configured_epochs_skips_training_steps(
    tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(train_mod, "WrinkleDataset", _DummyDataset)
    model_cfg = {"hidden_dim": 32, "attn_batch_nodes": 64, "decoder_chunk_t": 8}
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    torch.save(_make_checkpoint(model_cfg, epoch=5, val_loss=0.4, best_val_loss=0.3), out_dir / "latest.pt")
    with (out_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump([{"epoch": 1, "val_loss": 0.6}, {"epoch": 5, "val_loss": 0.4}], f)

    monkeypatch.setattr(train_mod, "train_one_epoch", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("unexpected")))
    monkeypatch.setattr(train_mod, "evaluate_split", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("unexpected")))

    result = train_mod.train_fold(
        fold=0,
        train_ids=["sim_a"],
        val_ids=["sim_a"],
        config={"epochs": 3, "patience": 10, "lr": 1e-3, "resume": True, "model": model_cfg},
        output_dir=out_dir,
    )
    assert [row["epoch"] for row in result["history"]] == [1, 5]


def test_run_one_wires_efficiency_flags(monkeypatch: pytest.MonkeyPatch, tmp_path: pytest.TempPathFactory) -> None:
    captured: dict = {}

    monkeypatch.setattr(train_mod, "_read_all_sim_ids", lambda: ["sim_a"])

    def _fake_train_fold(_fold, _train_ids, _val_ids, cfg, _out_dir):
        captured["cfg"] = cfg
        return {"ok": True}

    monkeypatch.setattr(train_mod, "train_fold", _fake_train_fold)
    args = argparse.Namespace(
        max_train_sims=None,
        max_val_sims=None,
        hidden_dim=32,
        attn_batch_nodes=123,
        decoder_chunk_t=7,
        model_type="coarse",
        use_fine_mp=False,
        epochs=2,
        patience=1,
        lr=1e-3,
        device="cpu",
        no_normalize=False,
        amp=False,
        no_amp=False,
        max_timesteps=None,
        temporal_strategy="tail",
        resume=True,
        allow_resume_mismatch=True,
        no_checkpoint=True,
        normalize_fine_features=False,
    )

    train_mod._run_one(0, ["sim_a"], ["sim_a"], args, tmp_path)
    cfg = captured["cfg"]
    assert cfg["model"]["attn_batch_nodes"] == 123
    assert cfg["model"]["decoder_chunk_t"] == 7
    assert cfg["model"]["use_checkpoint"] is False
    assert cfg["allow_resume_mismatch"] is True
    assert cfg["normalize_fine_features"] is False


def test_run_one_wires_fine_normalization_flag(monkeypatch: pytest.MonkeyPatch, tmp_path: pytest.TempPathFactory) -> None:
    captured: dict = {}
    monkeypatch.setattr(train_mod, "_read_all_sim_ids", lambda: ["sim_a"])

    def _fake_train_fold(_fold, _train_ids, _val_ids, cfg, _out_dir):
        captured["cfg"] = cfg
        return {"ok": True}

    monkeypatch.setattr(train_mod, "train_fold", _fake_train_fold)
    args = argparse.Namespace(
        max_train_sims=None,
        max_val_sims=None,
        hidden_dim=32,
        attn_batch_nodes=123,
        decoder_chunk_t=7,
        model_type="cross-scale",
        use_fine_mp=False,
        epochs=2,
        patience=1,
        lr=1e-3,
        device="cpu",
        no_normalize=False,
        normalize_fine_features=True,
        amp=False,
        no_amp=False,
        max_timesteps=None,
        temporal_strategy="tail",
        resume=False,
        allow_resume_mismatch=False,
        no_checkpoint=False,
    )

    train_mod._run_one(0, ["sim_a"], ["sim_a"], args, tmp_path)
    assert captured["cfg"]["normalize_fine_features"] is True


def test_run_one_rejects_fine_normalization_for_coarse_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pytest.TempPathFactory
) -> None:
    monkeypatch.setattr(train_mod, "_read_all_sim_ids", lambda: ["sim_a"])
    args = argparse.Namespace(
        max_train_sims=None,
        max_val_sims=None,
        hidden_dim=32,
        attn_batch_nodes=64,
        decoder_chunk_t=8,
        model_type="coarse",
        use_fine_mp=False,
        epochs=2,
        patience=1,
        lr=1e-3,
        device="cpu",
        no_normalize=False,
        normalize_fine_features=True,
        amp=False,
        no_amp=False,
        max_timesteps=None,
        temporal_strategy="tail",
        resume=False,
        allow_resume_mismatch=False,
        no_checkpoint=False,
    )
    with pytest.raises(ValueError, match="normalize-fine-features"):
        train_mod._run_one(0, ["sim_a"], ["sim_a"], args, tmp_path)


def test_dry_run_empty_dataset_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(train_mod, "_read_all_sim_ids", lambda: [])
    args = argparse.Namespace(
        device="cpu",
        model_type="coarse",
        no_normalize=False,
        max_timesteps=None,
        hidden_dim=32,
        attn_batch_nodes=64,
        decoder_chunk_t=8,
        use_fine_mp=False,
        no_checkpoint=False,
        normalize_fine_features=False,
    )
    with pytest.raises(RuntimeError, match="No simulations found"):
        train_mod._dry_run(args)


def test_dry_run_wires_no_checkpoint_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    class _DryRunDataset:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def __getitem__(self, _idx: int) -> dict:
            return {
                "node_features": torch.zeros(1, 3, 74),
                "edge_index": torch.zeros(2, 1, dtype=torch.int64),
                "edge_attr": torch.zeros(1, 4),
                "material_card": torch.zeros(8),
                "elements": torch.zeros(1, 3, dtype=torch.int64),
            }

    class _DryRunModel(torch.nn.Module):
        def __init__(self, **kwargs) -> None:
            super().__init__()
            captured["kwargs"] = kwargs

        def forward(self, _batch: dict) -> torch.Tensor:
            return torch.zeros(1, 1, 4)

    monkeypatch.setattr(train_mod, "_read_all_sim_ids", lambda: ["sim_a"])
    monkeypatch.setattr(train_mod, "WrinkleDataset", _DryRunDataset)
    monkeypatch.setattr(train_mod, "FormingGraphNet", _DryRunModel)
    args = argparse.Namespace(
        device="cpu",
        model_type="coarse",
        no_normalize=False,
        max_timesteps=None,
        hidden_dim=32,
        attn_batch_nodes=64,
        decoder_chunk_t=8,
        use_fine_mp=False,
        no_checkpoint=True,
        normalize_fine_features=False,
    )
    train_mod._dry_run(args)
    assert captured["kwargs"]["use_checkpoint"] is False

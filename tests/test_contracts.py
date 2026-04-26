from __future__ import annotations

import pytest

from model.contracts import (
    MODEL_TYPE_COARSE,
    add_checkpoint_contract_metadata,
    validate_checkpoint_compatibility,
    validate_contract_invariants,
)


def test_contract_invariants_hold() -> None:
    validate_contract_invariants()


def test_checkpoint_metadata_roundtrip_compatibility() -> None:
    ckpt = {"model_config": {"hidden_dim": 32}}
    add_checkpoint_contract_metadata(ckpt, model_type=MODEL_TYPE_COARSE)
    validate_checkpoint_compatibility(ckpt, expected_model_type=MODEL_TYPE_COARSE, require_contract_fields=True)


def test_checkpoint_compatibility_rejects_mismatched_model_type() -> None:
    ckpt = {"model_config": {"hidden_dim": 32}}
    add_checkpoint_contract_metadata(ckpt, model_type=MODEL_TYPE_COARSE)
    with pytest.raises(ValueError, match="model_type mismatch"):
        validate_checkpoint_compatibility(ckpt, expected_model_type="cross-scale", require_contract_fields=True)


def test_checkpoint_compatibility_allows_legacy_without_contract() -> None:
    legacy_ckpt = {"model_state_dict": {}, "model_config": {}}
    validate_checkpoint_compatibility(legacy_ckpt, expected_model_type=MODEL_TYPE_COARSE, require_contract_fields=False)

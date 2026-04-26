from __future__ import annotations

from typing import Any

from model.labels import (
    COARSE_TARGET_CHANNELS,
    FINE_TARGET_CHANNELS,
    N_COARSE_TARGETS,
    N_FINE_TARGETS,
)
from wp3_features.physics import FEATURE_NAMES
from wp3_features.targets import TARGET_NAMES

MODEL_FAMILY = "cfwrinkle-pinn"
MODEL_CONTRACT_VERSION = "1.0.0"
CHECKPOINT_FORMAT_VERSION = 2

MODEL_TYPE_COARSE = "coarse"
MODEL_TYPE_CROSS_SCALE = "cross-scale"
SUPPORTED_MODEL_TYPES = (MODEL_TYPE_COARSE, MODEL_TYPE_CROSS_SCALE)

COARSE_TARGET_DATASET_KEYS_BY_CHANNEL = {
    "severity": "targets/wrinkle_severity",
    "comp_frac": "targets/comp_frac_elem",
    "oop": "targets/oop_max_elem",
    "thick_var": "targets/thickness_variance_elem",
}
COARSE_TARGET_DATASET_KEYS = tuple(
    COARSE_TARGET_DATASET_KEYS_BY_CHANNEL[channel] for channel in COARSE_TARGET_CHANNELS
)
COARSE_TARGET_DATASET_NAMES = tuple(key.split("/", 1)[1] for key in COARSE_TARGET_DATASET_KEYS)


def validate_contract_invariants() -> None:
    if len(COARSE_TARGET_CHANNELS) != N_COARSE_TARGETS:
        raise ValueError("Coarse target channel constants are inconsistent")
    if len(FINE_TARGET_CHANNELS) != N_FINE_TARGETS:
        raise ValueError("Fine target channel constants are inconsistent")
    if tuple(COARSE_TARGET_DATASET_KEYS_BY_CHANNEL.keys()) != tuple(COARSE_TARGET_CHANNELS):
        raise ValueError(
            "Dataset target key mapping must follow coarse channel order "
            f"{COARSE_TARGET_CHANNELS}, got {tuple(COARSE_TARGET_DATASET_KEYS_BY_CHANNEL.keys())}"
        )
    if tuple(COARSE_TARGET_DATASET_NAMES) != tuple(TARGET_NAMES):
        raise ValueError(
            "Target names mismatch between dataset/model contracts and wp3_features.targets: "
            f"{COARSE_TARGET_DATASET_NAMES} vs {TARGET_NAMES}"
        )
    if len(FEATURE_NAMES) != 37:
        raise ValueError(f"Expected 37 canonical feature names, got {len(FEATURE_NAMES)}")


def build_contract_info(model_type: str) -> dict[str, Any]:
    validate_contract_invariants()
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(f"Unsupported model_type {model_type!r}; expected one of {SUPPORTED_MODEL_TYPES}")
    return {
        "model_family": MODEL_FAMILY,
        "model_contract_version": MODEL_CONTRACT_VERSION,
        "model_type": model_type,
        "coarse_target_channels": list(COARSE_TARGET_CHANNELS),
        "fine_target_channels": list(FINE_TARGET_CHANNELS),
        "dataset_target_names": list(COARSE_TARGET_DATASET_NAMES),
        "feature_names": list(FEATURE_NAMES),
    }


def add_checkpoint_contract_metadata(
    checkpoint: dict[str, Any],
    *,
    model_type: str,
) -> dict[str, Any]:
    checkpoint["model_type"] = model_type
    checkpoint["checkpoint_format_version"] = CHECKPOINT_FORMAT_VERSION
    checkpoint["contract"] = build_contract_info(model_type)
    checkpoint["model_version"] = MODEL_CONTRACT_VERSION
    return checkpoint


def validate_checkpoint_compatibility(
    checkpoint: dict[str, Any],
    *,
    expected_model_type: str | None = None,
    require_contract_fields: bool = False,
) -> None:
    model_type = str(checkpoint.get("model_type", MODEL_TYPE_COARSE))
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(f"Checkpoint has unsupported model_type={model_type!r}")
    if expected_model_type is not None and model_type != expected_model_type:
        raise ValueError(
            f"Checkpoint model_type mismatch: checkpoint={model_type!r} expected={expected_model_type!r}"
        )

    contract = checkpoint.get("contract")
    if contract is None:
        if require_contract_fields:
            raise ValueError("Checkpoint is missing required 'contract' metadata")
        return
    if not isinstance(contract, dict):
        raise ValueError("Checkpoint 'contract' metadata must be a dictionary")

    required = {
        "model_family",
        "model_contract_version",
        "model_type",
        "coarse_target_channels",
        "fine_target_channels",
        "dataset_target_names",
        "feature_names",
    }
    missing = sorted(required.difference(contract))
    if missing:
        raise ValueError(f"Checkpoint contract metadata missing required fields: {missing}")

    if contract["model_family"] != MODEL_FAMILY:
        raise ValueError(
            f"Checkpoint model_family mismatch: {contract['model_family']!r} != {MODEL_FAMILY!r}"
        )
    if tuple(contract["coarse_target_channels"]) != tuple(COARSE_TARGET_CHANNELS):
        raise ValueError("Checkpoint coarse_target_channels are incompatible with current contract")
    if tuple(contract["dataset_target_names"]) != tuple(COARSE_TARGET_DATASET_NAMES):
        raise ValueError("Checkpoint dataset_target_names are incompatible with current contract")
    if tuple(contract["feature_names"]) != tuple(FEATURE_NAMES):
        raise ValueError("Checkpoint feature_names are incompatible with current contract")

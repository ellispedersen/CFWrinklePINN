from __future__ import annotations

COARSE_TARGET_CHANNELS = ("severity", "comp_frac", "oop", "thick_var")
FINE_TARGET_CHANNELS = ("stress_1", "stress_2", "dz", "thickness")

COARSE_TARGET_INDEX = {name: idx for idx, name in enumerate(COARSE_TARGET_CHANNELS)}
FINE_TARGET_INDEX = {name: idx for idx, name in enumerate(FINE_TARGET_CHANNELS)}

N_COARSE_TARGETS = len(COARSE_TARGET_CHANNELS)
N_FINE_TARGETS = len(FINE_TARGET_CHANNELS)


def require_last_dim(tensor: object, expected: int, tensor_name: str) -> None:
    if not hasattr(tensor, "shape") or not hasattr(tensor, "ndim"):
        raise TypeError(f"{tensor_name} must be array-like with shape/ndim, got {type(tensor)!r}")
    if tensor.ndim < 1 or int(tensor.shape[-1]) != expected:
        raise ValueError(
            f"{tensor_name} must have last dimension {expected}, got shape {tuple(tensor.shape)}"
        )


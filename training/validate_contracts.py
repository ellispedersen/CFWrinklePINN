from __future__ import annotations

import argparse
from pathlib import Path

from model.contracts import validate_checkpoint_compatibility, validate_contract_invariants


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate canonical production contracts")
    p.add_argument("--checkpoint", type=Path, default=None, help="Optional checkpoint path to validate")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    validate_contract_invariants()
    if args.checkpoint is not None:
        import torch

        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        validate_checkpoint_compatibility(ckpt, require_contract_fields=True)
    print("Contract validation passed.")


if __name__ == "__main__":
    main()

__all__ = ["FormingGraphNet", "wrinkle_loss"]

try:
    from .gnn import FormingGraphNet
    from .loss import wrinkle_loss
except ModuleNotFoundError:
    # Allow importing non-torch submodules (e.g. model.contracts) in environments
    # where torch is intentionally absent.
    pass


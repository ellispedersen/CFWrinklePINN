"""Quick smoke test for CrossScaleNet — run directly, not via pytest."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from model.cross_scale import CrossScaleNet
from model.loss import cross_scale_loss

model = CrossScaleNet(hidden_dim=32)
print(f"CrossScaleNet params: {sum(p.numel() for p in model.parameters()):,}")

T, N, M = 8, 50, 30
N_fine_nodes, N_fine_elem = 200, 120
N_mapping = 100

batch = {
    "node_features": torch.randn(T, N, 74),
    "edge_index": torch.randint(0, N, (2, 100)),
    "edge_attr": torch.randn(100, 4),
    "material_card": torch.randn(8),
    "elements": torch.randint(0, N, (M, 3)),
    "coarse_to_fine": torch.stack([
        torch.randint(0, M, (N_mapping,)),
        torch.arange(N_mapping),
    ]),
    "fine_elements": torch.randint(0, N_fine_nodes, (N_fine_elem, 3)),
}

out = model(batch)
assert out["coarse"].shape == (T, M, 4), f"Bad coarse shape: {out['coarse'].shape}"
assert out["fine"].shape == (T, N_fine_elem, 4), f"Bad fine shape: {out['fine'].shape}"
print(f"coarse: {out['coarse'].shape}  fine: {out['fine'].shape}")

fine_features = torch.randn(T, N_fine_nodes, 4)
coarse_targets = torch.rand(T, M, 4)
loss, log = cross_scale_loss(
    out["fine"], fine_features, batch["fine_elements"], out["coarse"], coarse_targets
)
assert not torch.isnan(loss), "NaN loss"
print(f"cross_scale_loss: {loss.item():.4f}")

loss.backward()
bad = [(n, p) for n, p in model.named_parameters()
       if p.grad is None or torch.isnan(p.grad).any()]
assert not bad, f"Bad gradients: {[n for n,_ in bad]}"
print("Gradient check: OK")
print("ALL PASS")

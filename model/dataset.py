from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class WrinkleDataset(Dataset):
    TARGET_KEYS = [
        "targets/wrinkle_severity",
        "targets/comp_frac_elem",
        "targets/oop_max_elem",
        "targets/thickness_variance_elem",
    ]

    def __init__(
        self,
        h5_path: str | Path,
        sim_ids: list[str],
        normalize: bool = True,
        device: str | torch.device = "cpu",
    ) -> None:
        self.h5_path = Path(h5_path)
        self.sim_ids = list(sim_ids)
        self.normalize = normalize
        self.device = torch.device(device)
        self._stats: dict[str, np.ndarray] | None = None
        self._sim_index = {sid: i for i, sid in enumerate(self.sim_ids)}
        if self.normalize:
            self._stats = self._load_or_compute_stats()

    def __len__(self) -> int:
        return len(self.sim_ids)

    def index_of(self, sim_id: str) -> int:
        return self._sim_index[sim_id]

    def __getitem__(self, idx: int) -> dict:
        sim_id = self.sim_ids[idx]
        with h5py.File(self.h5_path, "r") as f:
            grp = f[f"simulations/{sim_id}"]
            feats = self._read_features(grp)
            rates = self._read_rates(grp)
            edge_index = grp["graph/edge_index"][:]
            edge_attr = grp["graph/edge_attr"][:]
            material = grp["material_card"][:]
            elements = self._read_elements(grp)
            targets = np.stack([grp[k][:] for k in self.TARGET_KEYS], axis=-1)

            node_features = np.concatenate([feats, rates], axis=-1).astype(np.float32)
            if self.normalize and self._stats is not None:
                mu = self._stats["feat_mean"]
                std = self._stats["feat_std"]
                node_features = (node_features - mu) / (std + 1e-8)

            n_nodes = int(node_features.shape[1])
            n_elem = int(elements.shape[0])
            n_edges = int(edge_index.shape[1])
            batch = str(grp.attrs.get("batch", "?"))
            material_name = str(grp.attrs.get("material", "?"))

        return {
            "sim_id": sim_id,
            "node_features": self._to_tensor(node_features, torch.float32),
            "edge_index": self._to_tensor(edge_index, torch.int64),
            "edge_attr": self._to_tensor(edge_attr, torch.float32),
            "material_card": self._to_tensor(material, torch.float32),
            "elements": self._to_tensor(elements, torch.int64),
            "targets": self._to_tensor(targets, torch.float32),
            "sim_meta": {
                "batch": batch,
                "material": material_name,
                "n_nodes": n_nodes,
                "n_elem": n_elem,
                "n_edges": n_edges,
            },
        }

    def _to_tensor(self, arr: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
        return torch.as_tensor(arr, dtype=dtype, device=self.device)

    @staticmethod
    def _read_features(grp: h5py.Group) -> np.ndarray:
        if "coarse_fields_resampled" in grp:
            return grp["coarse_fields_resampled"][:].astype(np.float32)
        return grp["coarse/resampled/fields"][:].astype(np.float32)

    @staticmethod
    def _read_rates(grp: h5py.Group) -> np.ndarray:
        if "coarse_rates_resampled" in grp:
            return grp["coarse_rates_resampled"][:].astype(np.float32)
        return grp["coarse/resampled/rates"][:].astype(np.float32)

    @staticmethod
    def _read_elements(grp: h5py.Group) -> np.ndarray:
        if "mesh/coarse_elements" in grp:
            return grp["mesh/coarse_elements"][:].astype(np.int64)
        return grp["coarse/mesh_elements"][:].astype(np.int64)

    def _load_or_compute_stats(self) -> dict[str, np.ndarray]:
        with h5py.File(self.h5_path, "r") as f:
            if "metadata/feature_stats/feat_mean" in f and "metadata/feature_stats/feat_std" in f:
                return {
                    "feat_mean": f["metadata/feature_stats/feat_mean"][:].astype(np.float32),
                    "feat_std": f["metadata/feature_stats/feat_std"][:].astype(np.float32),
                }
        return self._compute_stats_online()

    def _compute_stats_online(self) -> dict[str, np.ndarray]:
        count = 0
        total: np.ndarray | None = None
        total_sq: np.ndarray | None = None
        with h5py.File(self.h5_path, "r") as f:
            for sid in f["simulations"].keys():
                grp = f[f"simulations/{sid}"]
                feats = self._read_features(grp)
                rates = self._read_rates(grp)
                x = np.concatenate([feats, rates], axis=-1).reshape(-1, 74).astype(np.float64, copy=False)
                if x.size == 0:
                    continue
                if total is None or total_sq is None:
                    total = np.zeros(x.shape[1], dtype=np.float64)
                    total_sq = np.zeros(x.shape[1], dtype=np.float64)
                total += x.sum(axis=0)
                total_sq += np.square(x).sum(axis=0)
                count += int(x.shape[0])
        if total is None or total_sq is None or count == 0:
            raise RuntimeError("Could not compute feature stats from dataset.")
        mean = total / count
        var = np.maximum(total_sq / count - np.square(mean), 0.0)
        std = np.sqrt(var).astype(np.float32)
        std = np.where(std < 1e-8, 1.0, std).astype(np.float32)
        feat_mean = mean.astype(np.float32)
        feat_std = std.astype(np.float32)
        with h5py.File(self.h5_path, "a") as f:
            stats_grp = f.require_group("metadata/feature_stats")
            if "feat_mean" in stats_grp:
                del stats_grp["feat_mean"]
            if "feat_std" in stats_grp:
                del stats_grp["feat_std"]
            stats_grp.create_dataset("feat_mean", data=feat_mean, compression="lzf")
            stats_grp.create_dataset("feat_std", data=feat_std, compression="lzf")
        return {"feat_mean": feat_mean, "feat_std": feat_std}


def load_fold_sim_ids(
    h5_path: str | Path,
    fold: int,
    wp2_h5_path: str | Path | None = None,
) -> tuple[list[str], list[str]]:
    wp3 = Path(h5_path)
    if wp2_h5_path is None:
        wp2 = wp3.with_name("cfwrinkle_dataset.h5")
        if not wp2.exists():
            wp2 = Path("data/cfwrinkle_dataset.h5")
    else:
        wp2 = Path(wp2_h5_path)

    with h5py.File(wp2, "r") as f:
        grp = f[f"splits/fold_{fold}"]
        train = _decode_ids(grp["train"][:])
        val = _decode_ids(grp["val"][:])
    return train, val


def _decode_ids(arr: np.ndarray) -> list[str]:
    out: list[str] = []
    for item in arr.tolist():
        if isinstance(item, bytes):
            out.append(item.decode("utf-8"))
        else:
            out.append(str(item))
    return out


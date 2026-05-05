from __future__ import annotations

import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .contracts import COARSE_TARGET_DATASET_KEYS_BY_CHANNEL, validate_contract_invariants
from .labels import COARSE_TARGET_CHANNELS, N_COARSE_TARGETS
from wp3_features.physics import FEATURE_NAMES
from wp3_features.targets import TARGET_NAMES


class WrinkleDataset(Dataset):
    TARGET_KEYS_BY_CHANNEL = dict(COARSE_TARGET_DATASET_KEYS_BY_CHANNEL)
    TARGET_KEYS = list(map(TARGET_KEYS_BY_CHANNEL.__getitem__, COARSE_TARGET_CHANNELS))
    TARGET_NAMES = tuple(key.split("/", 1)[1] for key in TARGET_KEYS)
    FINE_FEATURE_NAMES = ("fs1", "fs2", "dz", "thick")

    def __init__(
        self,
        h5_path: str | Path,
        sim_ids: list[str],
        normalize: bool = True,
        device: str | torch.device = "cpu",
        max_timesteps: int | None = None,
        temporal_strategy: str = "tail",
        include_fine: bool = False,
        fine_normalize: bool = False,
        fine_norm_stats: dict[str, np.ndarray] | None = None,
        cache_static_graph: bool = True,
    ) -> None:
        validate_contract_invariants()
        self.h5_path = Path(h5_path)
        self.sim_ids = list(sim_ids)
        self.normalize = normalize
        self.device = torch.device(device)
        self.max_timesteps = max_timesteps
        self.temporal_strategy = temporal_strategy
        self.include_fine = include_fine
        self.fine_normalize = bool(fine_normalize)
        self.cache_static_graph = bool(cache_static_graph)
        self._stats: dict[str, np.ndarray] | None = None
        self._fine_stats: dict[str, np.ndarray] | None = None
        self._sim_index = {sid: i for i, sid in enumerate(self.sim_ids)}
        self._h5_file: h5py.File | None = None
        self._h5_pid: int | None = None
        self._static_graph_cache: dict[str, dict[str, np.ndarray | str]] = {}
        if self.normalize:
            self._stats = self._load_or_compute_stats()
        if self.fine_normalize:
            if not self.include_fine:
                raise ValueError("fine_normalize=True requires include_fine=True")
            if fine_norm_stats is not None:
                self._fine_stats = self._coerce_fine_stats(fine_norm_stats)
            else:
                self._fine_stats = self._load_or_compute_fine_stats()

    def __len__(self) -> int:
        return len(self.sim_ids)

    def index_of(self, sim_id: str) -> int:
        return self._sim_index[sim_id]

    def __getitem__(self, idx: int) -> dict:
        sim_id = self.sim_ids[idx]
        f = self._get_h5_file()
        grp = f[f"simulations/{sim_id}"]
        static_data = self._read_static_graph_data(grp, sim_id)

        # Fast path: serve pre-loaded data from RAM cache when preload_fold() was called.
        _cached = getattr(self, "_feature_cache", {}).get(sim_id)
        if _cached is not None:
            node_features = _cached["node_features"]
            targets = _cached["targets"]
        else:
            # Compute t_idx BEFORE any feature read so HDF5 fancy-indexing fetches
            # only the required rows (no T=256 intermediate allocation).
            n_t_full = self._get_n_t(grp)
            t_idx: np.ndarray | None = None
            if self.max_timesteps is not None and self.max_timesteps < n_t_full:
                t_idx = self._temporal_indices(n_t_full, self.max_timesteps, self.temporal_strategy)

            feats = self._read_features(grp, t_idx)
            rates = self._read_rates(grp, t_idx)
            self._validate_feature_schema(grp, feats, rates)
            targets = self._read_targets(grp, t_idx)

            node_features = np.concatenate([feats, rates], axis=-1).astype(np.float32)
            if self.normalize and self._stats is not None:
                mu = self._stats["feat_mean"]
                std = self._stats["feat_std"]
                if mu.shape != (node_features.shape[-1],) or std.shape != (node_features.shape[-1],):
                    raise ValueError(
                        f"Normalization stats shape mismatch for {sim_id}: "
                        f"feat_mean={mu.shape}, feat_std={std.shape}, expected=({node_features.shape[-1]},)"
                    )
                node_features = (node_features - mu) / (std + 1e-8)

        self._validate_sample_shapes(
            node_features=node_features,
            edge_index=static_data["edge_index"],
            edge_attr=static_data["edge_attr"],
            material=static_data["material"],
            elements=static_data["elements"],
            targets=targets,
        )

        n_nodes = int(node_features.shape[1])
        n_elem = int(static_data["elements"].shape[0])
        n_edges = int(static_data["edge_index"].shape[1])
        batch = str(static_data["batch"])
        material_name = str(static_data["material_name"])

        fine_data: dict = {}
        if self.include_fine:
            if _cached is not None and "fine_features" in _cached:
                # Serve from preload cache — topology (elements/nodes/edges) from static_data.
                fine_data = {
                    "features":      _cached["fine_features"],
                    "elements":      static_data["fine_elements"],
                    "nodes":         static_data["fine_nodes"],
                    "edge_index":    static_data["fine_edge_index"],
                    "edge_attr":     static_data["fine_edge_attr"],
                    "coarse_to_fine":static_data["coarse_to_fine"],
                    "node_coarse_map":static_data["fine_node_coarse_map"],
                }
                if "fine_elem_edge_index" in static_data:
                    fine_data["elem_edge_index"] = static_data["fine_elem_edge_index"]
            else:
                _t_idx = locals().get("t_idx", None)
                fine_data = self._read_fine(grp, _t_idx, static_data)
                if self.fine_normalize and self._fine_stats is not None:
                    fine_mu = self._fine_stats["fine_feat_mean"]
                    fine_std = self._fine_stats["fine_feat_std"]
                    if fine_mu.shape != (4,) or fine_std.shape != (4,):
                        raise ValueError(
                            f"Fine normalization stats shape mismatch for {sim_id}: "
                            f"fine_feat_mean={fine_mu.shape}, fine_feat_std={fine_std.shape}, expected=(4,)"
                        )
                    fine_data["features"] = (fine_data["features"] - fine_mu) / (fine_std + 1e-8)
            self._validate_fine_shapes(fine_data, node_features.shape[0], n_elem)

        result = {
            "sim_id": sim_id,
            "node_features": self._to_tensor(node_features, torch.float32),
            "edge_index": self._to_tensor(static_data["edge_index"], torch.int64),
            "edge_attr": self._to_tensor(static_data["edge_attr"], torch.float32),
            "material_card": self._to_tensor(static_data["material"], torch.float32),
            "elements": self._to_tensor(static_data["elements"], torch.int64),
            "targets": self._to_tensor(targets, torch.float32),
            "sim_meta": {
                "batch": batch,
                "material": material_name,
                "n_nodes": n_nodes,
                "n_elem": n_elem,
                "n_edges": n_edges,
            },
        }
        if fine_data:
            result["fine_features"] = self._to_tensor(fine_data["features"], torch.float32)
            result["fine_elements"] = self._to_tensor(fine_data["elements"], torch.int64)
            result["fine_edge_index"] = self._to_tensor(fine_data["edge_index"], torch.int64)
            result["fine_edge_attr"] = self._to_tensor(fine_data["edge_attr"], torch.float32)
            result["fine_nodes"] = self._to_tensor(fine_data["nodes"], torch.float32)
            result["coarse_to_fine"] = self._to_tensor(fine_data["coarse_to_fine"], torch.int64)
            result["fine_node_coarse_map"] = self._to_tensor(fine_data["node_coarse_map"], torch.int64)
            if "elem_edge_index" in fine_data:
                result["fine_elem_edge_index"] = self._to_tensor(fine_data["elem_edge_index"], torch.int64)
        return result

    def _get_h5_file(self) -> h5py.File:
        pid = os.getpid()
        if self._h5_file is None or self._h5_pid != pid:
            self.close()
            self._h5_file = h5py.File(self.h5_path, "r")
            self._h5_pid = pid
        return self._h5_file

    def close(self) -> None:
        if self._h5_file is not None:
            self._h5_file.close()
            self._h5_file = None
            self._h5_pid = None

    def __del__(self) -> None:
        self.close()

    def preload_fold(self) -> None:
        """Pre-read all time-varying feature tensors for this dataset's sim_ids into RAM.

        Call once per fold before the epoch loop. After this, __getitem__ serves
        node_features, targets, and (if include_fine) fine_features from RAM with
        no HDF5 I/O per sample. Static graph data is handled by the existing
        _static_graph_cache.

        Memory note: uses exactly the same RAM that was being allocated and freed
        on every __getitem__ call anyway — the difference is it is allocated once
        at fold start, eliminating repeated HDF5 reads and GC pressure per epoch.
        """
        self._feature_cache: dict[str, dict[str, np.ndarray]] = {}
        f = self._get_h5_file()
        for sim_id in self.sim_ids:
            grp = f[f"simulations/{sim_id}"]
            n_t = self._get_n_t(grp)
            t_idx: np.ndarray | None = None
            if self.max_timesteps is not None and self.max_timesteps < n_t:
                t_idx = self._temporal_indices(n_t, self.max_timesteps, self.temporal_strategy)

            feats = self._read_features(grp, t_idx)
            rates = self._read_rates(grp, t_idx)
            node_features = np.concatenate([feats, rates], axis=-1).astype(np.float32)
            if self.normalize and self._stats is not None:
                node_features = (node_features - self._stats["feat_mean"]) / (self._stats["feat_std"] + 1e-8)

            targets = self._read_targets(grp, t_idx)
            entry: dict[str, np.ndarray] = {"node_features": node_features, "targets": targets}

            if self.include_fine:
                fine_features = self._read_fine_feature_stack(grp, t_idx)
                if self.fine_normalize and self._fine_stats is not None:
                    fine_features = (fine_features - self._fine_stats["fine_feat_mean"]) / (self._fine_stats["fine_feat_std"] + 1e-8)
                entry["fine_features"] = fine_features.astype(np.float32)
                # Pre-warm static graph cache (fine topology) for this sim
                self._read_static_graph_data(grp, sim_id)

            self._feature_cache[sim_id] = entry

    def _read_static_graph_data(self, grp: h5py.Group, sim_id: str) -> dict[str, np.ndarray | str]:
        if self.cache_static_graph and sim_id in self._static_graph_cache:
            return self._static_graph_cache[sim_id]
        static_data: dict[str, np.ndarray | str] = {
            "edge_index": grp["graph/edge_index"][:].astype(np.int64, copy=False),
            "edge_attr": grp["graph/edge_attr"][:].astype(np.float32, copy=False),
            "material": grp["material_card"][:].astype(np.float32, copy=False),
            "elements": self._read_elements(grp),
            "batch": str(grp.attrs.get("batch", "?")),
            "material_name": str(grp.attrs.get("material", "?")),
        }
        if self.include_fine:
            coarse_to_fine = grp["graph/coarse_to_fine_index"][:].astype(np.int64)
            fine = grp["fine"]
            fine_elements = fine["mesh_elements"][:].astype(np.int64)
            nodes = fine["mesh_nodes"][:].astype(np.float32)
            n_fine_nodes = int(nodes.shape[0])
            node_coarse_map = np.full(n_fine_nodes, -1, dtype=np.int64)
            c_idx_arr = coarse_to_fine[0]
            f_idx_arr = coarse_to_fine[1]
            for i in range(c_idx_arr.shape[0]):
                ci = int(c_idx_arr[i])
                fi = int(f_idx_arr[i])
                if fi < 0 or fi >= fine_elements.shape[0]:
                    continue
                for ni in fine_elements[fi]:
                    ni_int = int(ni)
                    if 0 <= ni_int < n_fine_nodes and node_coarse_map[ni_int] < 0:
                        node_coarse_map[ni_int] = ci
            static_data["coarse_to_fine"] = coarse_to_fine
            static_data["fine_elements"] = fine_elements
            static_data["fine_nodes"] = nodes
            static_data["fine_edge_index"] = fine["edge_index"][:].astype(np.int64)
            static_data["fine_edge_attr"] = fine["edge_attr"][:].astype(np.float32)
            static_data["fine_node_coarse_map"] = node_coarse_map
            if "element_edge_index" in fine:
                static_data["fine_elem_edge_index"] = fine["element_edge_index"][:].astype(np.int64)
        if self.cache_static_graph:
            self._static_graph_cache[sim_id] = static_data
        return static_data

    def _to_tensor(self, arr: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
        return torch.as_tensor(arr, dtype=dtype, device=self.device)

    @staticmethod
    def _validate_sample_shapes(
        *,
        node_features: np.ndarray,
        edge_index: np.ndarray,
        edge_attr: np.ndarray,
        material: np.ndarray,
        elements: np.ndarray,
        targets: np.ndarray,
    ) -> None:
        if node_features.ndim != 3:
            raise ValueError(f"node_features must have shape (T, N, F), got {node_features.shape}")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(f"edge_index must have shape (2, E), got {edge_index.shape}")
        if edge_attr.ndim != 2 or edge_attr.shape != (edge_index.shape[1], 4):
            raise ValueError(
                f"edge_attr must have shape (E, 4) with E={edge_index.shape[1]}, got {edge_attr.shape}"
            )
        if elements.ndim != 2 or elements.shape[1] != 3:
            raise ValueError(f"elements must have shape (M, 3), got {elements.shape}")
        if targets.ndim != 3 or targets.shape[2] != N_COARSE_TARGETS:
            raise ValueError(f"targets must have shape (T, M, {N_COARSE_TARGETS}), got {targets.shape}")
        if targets.shape[0] != node_features.shape[0]:
            raise ValueError(
                f"targets timestep mismatch: node_features T={node_features.shape[0]}, targets T={targets.shape[0]}"
            )
        if targets.shape[1] != elements.shape[0]:
            raise ValueError(
                f"targets element mismatch: elements M={elements.shape[0]}, targets M={targets.shape[1]}"
            )
        if material.ndim != 1:
            raise ValueError(f"material_card must be a 1D vector, got {material.shape}")
        if not np.isfinite(node_features).all():
            raise ValueError("node_features contains non-finite values")
        if not np.isfinite(edge_attr).all():
            raise ValueError("edge_attr contains non-finite values")
        if not np.isfinite(material).all():
            raise ValueError("material_card contains non-finite values")
        if edge_index.size > 0:
            n_nodes = int(node_features.shape[1])
            if edge_index.min() < 0 or edge_index.max() >= n_nodes:
                raise ValueError(
                    f"edge_index contains out-of-range node ids for N={n_nodes}: "
                    f"min={int(edge_index.min())}, max={int(edge_index.max())}"
                )

    @staticmethod
    def _validate_fine_shapes(fine_data: dict, n_t: int, n_coarse_elem: int) -> None:
        features = fine_data["features"]
        nodes = fine_data["nodes"]
        elements = fine_data["elements"]
        edge_index = fine_data["edge_index"]
        edge_attr = fine_data["edge_attr"]
        coarse_to_fine = fine_data["coarse_to_fine"]
        node_coarse_map = fine_data["node_coarse_map"]
        if features.ndim != 3 or features.shape[0] != n_t or features.shape[2] != 4:
            raise ValueError(f"fine features must have shape (T, N_fine_nodes, 4), got {features.shape}")
        if nodes.ndim != 2 or nodes.shape[1] != 3:
            raise ValueError(f"fine nodes must have shape (N_fine_nodes, 3), got {nodes.shape}")
        if elements.ndim != 2 or elements.shape[1] != 3:
            raise ValueError(f"fine elements must have shape (N_fine_elem, 3), got {elements.shape}")
        if features.shape[1] != nodes.shape[0]:
            raise ValueError(
                f"fine feature/node mismatch: features N={features.shape[1]}, nodes N={nodes.shape[0]}"
            )
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(f"fine edge_index must have shape (2, E), got {edge_index.shape}")
        if edge_attr.ndim != 2 or edge_attr.shape != (edge_index.shape[1], 4):
            raise ValueError(
                f"fine edge_attr must have shape (E, 4) with E={edge_index.shape[1]}, got {edge_attr.shape}"
            )
        if edge_index.size > 0 and (edge_index.min() < 0 or edge_index.max() >= nodes.shape[0]):
            raise ValueError("fine edge_index contains out-of-range node ids")
        if coarse_to_fine.ndim != 2 or coarse_to_fine.shape[0] != 2:
            raise ValueError(f"coarse_to_fine must have shape (2, K), got {coarse_to_fine.shape}")
        if node_coarse_map.ndim != 1 or node_coarse_map.shape[0] != nodes.shape[0]:
            raise ValueError(
                f"fine node_coarse_map must have shape (N_fine_nodes,), got {node_coarse_map.shape}"
            )
        if coarse_to_fine.shape[1] > 0:
            coarse_idx = coarse_to_fine[0]
            fine_idx = coarse_to_fine[1]
            if coarse_idx.min() < 0 or coarse_idx.max() >= n_coarse_elem:
                raise ValueError("coarse_to_fine coarse indices out of range")
            if fine_idx.min() < 0 or fine_idx.max() >= elements.shape[0]:
                raise ValueError("coarse_to_fine fine indices out of range")
        valid_map = node_coarse_map[node_coarse_map >= 0]
        if valid_map.size > 0 and valid_map.max() >= n_coarse_elem:
            raise ValueError("fine node_coarse_map contains out-of-range coarse element ids")

    @staticmethod
    def _get_n_t(grp: h5py.Group) -> int:
        """Return the number of timesteps by reading shape metadata only (no data load)."""
        key = "coarse_fields_resampled" if "coarse_fields_resampled" in grp else "coarse/resampled/fields"
        return int(grp[key].shape[0])

    @staticmethod
    def _read_features(grp: h5py.Group, t_idx: np.ndarray | None = None) -> np.ndarray:
        ds = grp["coarse_fields_resampled"] if "coarse_fields_resampled" in grp \
             else grp["coarse/resampled/fields"]
        return (ds[t_idx] if t_idx is not None else ds[:]).astype(np.float32)

    @staticmethod
    def _read_rates(grp: h5py.Group, t_idx: np.ndarray | None = None) -> np.ndarray:
        ds = grp["coarse_rates_resampled"] if "coarse_rates_resampled" in grp \
             else grp["coarse/resampled/rates"]
        return (ds[t_idx] if t_idx is not None else ds[:]).astype(np.float32)

    @staticmethod
    def _read_targets(grp: h5py.Group, t_idx: np.ndarray | None = None) -> np.ndarray:
        target_group = grp["targets"]
        expected_names = list(TARGET_NAMES)
        if list(WrinkleDataset.TARGET_NAMES) != expected_names:
            raise RuntimeError(
                f"Dataset target schema mismatch between model and builder: "
                f"model={WrinkleDataset.TARGET_NAMES}, builder={tuple(expected_names)}"
            )
        names = list(WrinkleDataset.TARGET_NAMES)
        if "target_names" in target_group.attrs:
            raw = target_group.attrs["target_names"]
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            if not isinstance(raw, str):
                raise ValueError(f"targets/target_names must be a JSON string, got {type(raw)!r}")
            parsed = json.loads(raw)
            if not isinstance(parsed, list) or not all(isinstance(x, str) for x in parsed):
                raise ValueError(f"targets/target_names must decode to list[str], got {parsed!r}")
            names = parsed
        if names != list(WrinkleDataset.TARGET_NAMES):
            raise ValueError(f"Target names mismatch: expected={list(WrinkleDataset.TARGET_NAMES)}, found={names}")
        for name in names:
            if name not in target_group:
                raise ValueError(f"Missing target dataset: targets/{name}")
        if t_idx is None:
            targets = np.stack([target_group[name][:].astype(np.float32, copy=False) for name in names], axis=-1)
        else:
            targets = np.stack([target_group[name][t_idx].astype(np.float32, copy=False) for name in names], axis=-1)
        if targets.shape[-1] != N_COARSE_TARGETS:
            raise ValueError(f"targets channel mismatch: expected {N_COARSE_TARGETS} got {targets.shape[-1]}")
        return targets

    @staticmethod
    def _validate_feature_schema(grp: h5py.Group, feats: np.ndarray, rates: np.ndarray) -> None:
        expected_feature_names = list(FEATURE_NAMES)
        if feats.shape[-1] != len(expected_feature_names):
            raise ValueError(
                f"Feature channel mismatch: expected={len(expected_feature_names)} got={feats.shape[-1]}"
            )
        if rates.shape[-1] != len(expected_feature_names):
            raise ValueError(f"Rate channel mismatch: expected={len(expected_feature_names)} got={rates.shape[-1]}")
        if "coarse" not in grp or "resampled" not in grp["coarse"]:
            return
        attrs = grp["coarse/resampled"].attrs
        if "feature_names" in attrs:
            raw = attrs["feature_names"]
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            if not isinstance(raw, str):
                raise ValueError(f"coarse/resampled feature_names must be JSON string, got {type(raw)!r}")
            parsed = json.loads(raw)
            if parsed != expected_feature_names:
                raise ValueError(f"Feature names mismatch: expected={expected_feature_names}, found={parsed}")
        if "rate_feature_names" in attrs:
            raw = attrs["rate_feature_names"]
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            if not isinstance(raw, str):
                raise ValueError(f"coarse/resampled rate_feature_names must be JSON string, got {type(raw)!r}")
            parsed = json.loads(raw)
            expected_rate_names = [f"d_dt:{name}" for name in expected_feature_names]
            if parsed != expected_rate_names:
                raise ValueError(f"Rate feature names mismatch: expected={expected_rate_names}, found={parsed}")

    @staticmethod
    def _read_fine_feature_stack(grp: h5py.Group, t_idx: np.ndarray | None = None) -> np.ndarray:
        fine = grp["fine/resampled"]
        def _r(key: str) -> np.ndarray:
            ds = fine[key]
            return (ds[t_idx] if t_idx is not None else ds[:]).astype(np.float32)
        return np.stack([_r("fiber_stress_1"), _r("fiber_stress_2"),
                         _r("displacement_z"), _r("thickness")], axis=-1)

    @staticmethod
    def _read_elements(grp: h5py.Group) -> np.ndarray:
        if "mesh/coarse_elements" in grp:
            return grp["mesh/coarse_elements"][:].astype(np.int64)
        return grp["coarse/mesh_elements"][:].astype(np.int64)

    @staticmethod
    def _read_fine(grp: h5py.Group, t_idx: np.ndarray | None, static_data: dict[str, np.ndarray | str]) -> dict:
        features = WrinkleDataset._read_fine_feature_stack(grp, t_idx)
        result = {
            "features": features,
            "elements": static_data["fine_elements"],      # (N_fine_elem, 3)
            "nodes": static_data["fine_nodes"],            # (N_fine_nodes, 3)
            "edge_index": static_data["fine_edge_index"],
            "edge_attr": static_data["fine_edge_attr"],
            "coarse_to_fine": static_data["coarse_to_fine"],
            "node_coarse_map": static_data["fine_node_coarse_map"],
        }
        if "fine_elem_edge_index" in static_data:
            result["elem_edge_index"] = static_data["fine_elem_edge_index"]
        return result

    @staticmethod
    def _temporal_indices(n_t: int, max_t: int, strategy: str) -> np.ndarray:
        """Select which timesteps to keep.

        Strategies:
          tail   — last max_t consecutive steps (wrinkle-active region)
          stride — every (n_t // max_t)-th step from the full range
          linspace — uniformly spaced indices (may hit interpolated steps)
        """
        if strategy == "tail":
            return np.arange(n_t - max_t, n_t)
        elif strategy == "stride":
            step = max(1, n_t // max_t)
            return np.arange(0, n_t, step)[:max_t]
        elif strategy == "linspace":
            return np.linspace(0, n_t - 1, max_t, dtype=int)
        else:
            raise ValueError(f"Unknown temporal_strategy: {strategy!r}")

    def _load_or_compute_stats(self) -> dict[str, np.ndarray]:
        with h5py.File(self.h5_path, "r") as f:
            if "metadata/feature_stats/feat_mean" in f and "metadata/feature_stats/feat_std" in f:
                return {
                    "feat_mean": f["metadata/feature_stats/feat_mean"][:].astype(np.float32),
                    "feat_std": f["metadata/feature_stats/feat_std"][:].astype(np.float32),
                }
        return self._compute_stats_online()

    def _load_or_compute_fine_stats(self) -> dict[str, np.ndarray]:
        with h5py.File(self.h5_path, "r") as f:
            if "metadata/fine_feature_stats/fine_feat_mean" in f and "metadata/fine_feature_stats/fine_feat_std" in f:
                return self._coerce_fine_stats(
                    {
                        "fine_feat_mean": f["metadata/fine_feature_stats/fine_feat_mean"][:].astype(np.float32),
                        "fine_feat_std": f["metadata/fine_feature_stats/fine_feat_std"][:].astype(np.float32),
                    }
                )
        return self._compute_fine_stats_online()

    @staticmethod
    def _coerce_fine_stats(stats: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if "fine_feat_mean" not in stats or "fine_feat_std" not in stats:
            raise ValueError("fine_norm_stats must contain fine_feat_mean and fine_feat_std")
        mean = np.asarray(stats["fine_feat_mean"], dtype=np.float32).reshape(-1)
        std = np.asarray(stats["fine_feat_std"], dtype=np.float32).reshape(-1)
        if mean.shape != (4,) or std.shape != (4,):
            raise ValueError(
                f"fine_norm_stats shapes must be (4,), got fine_feat_mean={mean.shape}, fine_feat_std={std.shape}"
            )
        std = np.where(std < 1e-8, 1.0, std).astype(np.float32)
        return {"fine_feat_mean": mean.astype(np.float32), "fine_feat_std": std}

    def _compute_fine_stats_online(self) -> dict[str, np.ndarray]:
        count = 0
        total = np.zeros(4, dtype=np.float64)
        total_sq = np.zeros(4, dtype=np.float64)
        with h5py.File(self.h5_path, "r") as f:
            for sid in self.sim_ids:
                grp = f[f"simulations/{sid}"]
                features = self._read_fine_feature_stack(grp).reshape(-1, 4).astype(np.float64, copy=False)
                if features.size == 0:
                    continue
                total += features.sum(axis=0)
                total_sq += np.square(features).sum(axis=0)
                count += int(features.shape[0])
        if count == 0:
            raise RuntimeError("Could not compute fine feature stats from dataset.")
        mean = (total / count).astype(np.float32)
        var = np.maximum(total_sq / count - np.square(mean.astype(np.float64)), 0.0)
        std = np.sqrt(var).astype(np.float32)
        std = np.where(std < 1e-8, 1.0, std).astype(np.float32)
        return {"fine_feat_mean": mean, "fine_feat_std": std}

    def get_fine_norm_stats(self) -> dict[str, np.ndarray] | None:
        if self._fine_stats is None:
            return None
        return {
            "fine_feat_mean": self._fine_stats["fine_feat_mean"].copy(),
            "fine_feat_std": self._fine_stats["fine_feat_std"].copy(),
        }

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
        try:
            with h5py.File(self.h5_path, "a") as f:
                stats_grp = f.require_group("metadata/feature_stats")
                if "feat_mean" in stats_grp:
                    del stats_grp["feat_mean"]
                if "feat_std" in stats_grp:
                    del stats_grp["feat_std"]
                stats_grp.create_dataset("feat_mean", data=feat_mean, compression="lzf")
                stats_grp.create_dataset("feat_std", data=feat_std, compression="lzf")
        except BlockingIOError:
            pass  # another parallel process holds the write lock; stats are in memory
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


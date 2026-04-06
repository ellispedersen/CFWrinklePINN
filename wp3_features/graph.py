from __future__ import annotations

import numpy as np


def build_edge_index(elements: np.ndarray) -> np.ndarray:
    if elements.size == 0:
        return np.zeros((2, 0), dtype=np.int32)
    e = elements.astype(np.int64)
    pairs = np.concatenate(
        [
            e[:, [0, 1]],
            e[:, [1, 0]],
            e[:, [1, 2]],
            e[:, [2, 1]],
            e[:, [0, 2]],
            e[:, [2, 0]],
        ],
        axis=0,
    )
    pairs = np.unique(pairs, axis=0)
    return pairs.T.astype(np.int32)


def build_edge_attr(edge_index: np.ndarray, node_coords: np.ndarray) -> np.ndarray:
    if edge_index.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    src = edge_index[0]
    dst = edge_index[1]
    d = node_coords[dst] - node_coords[src]
    dist = np.linalg.norm(d, axis=1, keepdims=True)
    return np.concatenate([d.astype(np.float32), dist.astype(np.float32)], axis=1)


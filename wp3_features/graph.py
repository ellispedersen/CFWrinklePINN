from __future__ import annotations

import numpy as np


def _validate_elements(elements: np.ndarray) -> np.ndarray:
    elems = np.asarray(elements)
    if elems.ndim != 2 or elems.shape[1] != 3:
        raise ValueError(f"elements must have shape (N, 3), got {elems.shape}")
    if not np.issubdtype(elems.dtype, np.integer):
        raise ValueError(f"elements must be integer typed, got {elems.dtype}")
    return elems.astype(np.int64, copy=False)


def build_edge_index(elements: np.ndarray) -> np.ndarray:
    e = _validate_elements(elements)
    if e.size == 0:
        return np.zeros((2, 0), dtype=np.int64)
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
    return pairs.T.astype(np.int64, copy=False)


def build_edge_attr(edge_index: np.ndarray, node_coords: np.ndarray) -> np.ndarray:
    ei = np.asarray(edge_index)
    nodes = np.asarray(node_coords)
    if ei.ndim != 2 or ei.shape[0] != 2:
        raise ValueError(f"edge_index must have shape (2, E), got {ei.shape}")
    if nodes.ndim != 2 or nodes.shape[1] != 3:
        raise ValueError(f"node_coords must have shape (N, 3), got {nodes.shape}")
    if ei.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    if not np.issubdtype(ei.dtype, np.integer):
        raise ValueError(f"edge_index must be integer typed, got {ei.dtype}")
    src = ei[0].astype(np.int64, copy=False)
    dst = ei[1].astype(np.int64, copy=False)
    if src.min(initial=0) < 0 or dst.min(initial=0) < 0:
        raise ValueError("edge_index contains negative node indices")
    n_nodes = int(nodes.shape[0])
    if src.max(initial=-1) >= n_nodes or dst.max(initial=-1) >= n_nodes:
        raise ValueError("edge_index contains out-of-range node indices")
    d = nodes[dst] - nodes[src]
    dist = np.linalg.norm(d, axis=1, keepdims=True)
    return np.concatenate([d.astype(np.float32), dist.astype(np.float32)], axis=1)


def build_element_adjacency(elements: np.ndarray) -> np.ndarray:
    """Build directed element adjacency for triangle mesh.

    Two elements are adjacent if they share an edge (2 common nodes).
    Returns shape (2, E_adj) int64 with both directions included.
    """
    elems = _validate_elements(elements)
    if elems.size == 0:
        return np.zeros((2, 0), dtype=np.int64)

    from collections import defaultdict

    edge_to_elems: dict[tuple[int, int], list[int]] = defaultdict(list)
    for ei, tri in enumerate(elems):
        for i in range(3):
            a = int(tri[i])
            b = int(tri[(i + 1) % 3])
            edge_to_elems[(min(a, b), max(a, b))].append(ei)

    src: list[int] = []
    dst: list[int] = []
    for sharing in edge_to_elems.values():
        if len(sharing) == 2:
            e0, e1 = sharing
            src.extend([e0, e1])
            dst.extend([e1, e0])

    if not src:
        return np.zeros((2, 0), dtype=np.int64)
    return np.array([src, dst], dtype=np.int64)


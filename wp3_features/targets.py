from __future__ import annotations

import numpy as np


def _node_element_index(n_nodes: int, elements: np.ndarray) -> list[np.ndarray]:
    buckets: list[list[int]] = [[] for _ in range(n_nodes)]
    for ei, tri in enumerate(elements):
        for n in tri.tolist():
            buckets[int(n)].append(ei)
    return [np.array(v, dtype=np.int32) for v in buckets]


def compute_wrinkle_targets(
    coarse_to_fine: dict[str, np.ndarray],
    fine_fiber_stress: np.ndarray,
    fine_displacement: np.ndarray,
    fine_thickness: np.ndarray,
    fine_elements: np.ndarray,
) -> dict[str, np.ndarray]:
    n_t = fine_fiber_stress.shape[0]
    n_ce = int(coarse_to_fine["n_fine_per_coarse"].shape[0])
    sev = np.full((n_t, n_ce), np.nan, dtype=np.float32)
    comp = np.full((n_t, n_ce), np.nan, dtype=np.float32)
    oop = np.full((n_t, n_ce), np.nan, dtype=np.float32)
    tv = np.full((n_t, n_ce), np.nan, dtype=np.float32)

    cidx = coarse_to_fine["coarse_index"]
    fidx = coarse_to_fine["fine_index"]
    by_coarse: list[list[int]] = [[] for _ in range(n_ce)]
    for ci, fi in zip(cidx.tolist(), fidx.tolist()):
        by_coarse[int(ci)].append(int(fi))

    for ci, fe_ids in enumerate(by_coarse):
        if not fe_ids:
            continue
        node_ids = np.unique(fine_elements[np.array(fe_ids, dtype=np.int32)].reshape(-1))
        fs = fine_fiber_stress[:, node_ids]
        dz = fine_displacement[:, node_ids, 2]
        th = fine_thickness[:, node_ids]
        comp_ci = (fs < -0.05).mean(axis=1).astype(np.float32)
        oop_ci = np.max(np.abs(dz - dz.mean(axis=1, keepdims=True)), axis=1).astype(np.float32)
        tv_ci = np.var(th, axis=1).astype(np.float32)
        sev_ci = np.clip(0.7 * comp_ci + 0.3 * np.clip(oop_ci / 5.0, 0.0, 1.0), 0.0, 1.0).astype(np.float32)
        comp[:, ci], oop[:, ci], tv[:, ci], sev[:, ci] = comp_ci, oop_ci, tv_ci, sev_ci

    return {
        "wrinkle_severity": sev,
        "comp_frac_elem": comp,
        "oop_max_elem": oop,
        "thickness_variance_elem": tv,
    }


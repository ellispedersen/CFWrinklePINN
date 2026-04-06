from __future__ import annotations

import numpy as np
from scipy.spatial import KDTree


def _tri_area_2d(points: np.ndarray) -> float:
    a, b, c = points
    return 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))


def _contains_2d(tri: np.ndarray, p: np.ndarray) -> bool:
    a, b, c = tri
    v0, v1, v2 = c - a, b - a, p - a
    den = v0[0] * v1[1] - v1[0] * v0[1]
    if abs(den) < 1e-12:
        return False
    u = (v2[0] * v1[1] - v1[0] * v2[1]) / den
    v = (v0[0] * v2[1] - v2[0] * v0[1]) / den
    return (u >= -1e-8) and (v >= -1e-8) and (u + v <= 1.0 + 1e-8)


def build_coarse_to_fine_map(
    coarse_nodes: np.ndarray,
    coarse_elements: np.ndarray,
    fine_nodes: np.ndarray,
    fine_elements: np.ndarray,
    radius_factor: float = 1.5,
) -> dict[str, np.ndarray]:
    nc = coarse_elements.shape[0]
    nf = fine_elements.shape[0]
    coarse_tri = coarse_nodes[coarse_elements][:, :, :2]
    fine_tri = fine_nodes[fine_elements][:, :, :2]
    coarse_cent = coarse_tri.mean(axis=1)
    fine_cent = fine_tri.mean(axis=1)

    buckets: list[list[int]] = [[] for _ in range(nc)]
    fallback_hits = np.zeros((nc,), dtype=np.int32)
    coarse_area = np.array([_tri_area_2d(t) for t in coarse_tri], dtype=np.float64)
    fine_area = np.array([_tri_area_2d(t) for t in fine_tri], dtype=np.float64)
    coarse_char = np.sqrt(np.maximum(coarse_area, 1e-12))

    tree = KDTree(coarse_cent)
    primary_hit = np.zeros((nc,), dtype=np.int32)
    for i in range(nf):
        p = fine_cent[i]
        nn = tree.query(p, k=min(16, max(1, nc)))[1]
        if np.isscalar(nn):
            nn = [int(nn)]
        assigned = False
        for idx in nn:
            if _contains_2d(coarse_tri[int(idx)], p):
                buckets[int(idx)].append(i)
                primary_hit[int(idx)] += 1
                assigned = True
                break
        if assigned:
            continue
        d, idx = tree.query(p, k=1)
        idx = int(idx)
        if float(d) <= float(radius_factor * coarse_char[idx]):
            buckets[idx].append(i)
            fallback_hits[idx] = 1

    pre_fallback_counts = np.array([len(b) for b in buckets], dtype=np.int32)
    pre_fallback_coverage = float((pre_fallback_counts > 0).mean()) if nc else 0.0

    # Guarantee coarse coverage: any uncovered coarse element gets its nearest
    # fine centroid assigned as a boundary fallback.
    mapped_counts = pre_fallback_counts.copy()
    if nc > 0 and nf > 0:
        fine_tree = KDTree(fine_cent)
        for ci, cnt in enumerate(mapped_counts.tolist()):
            if cnt > 0:
                continue
            _, fi = fine_tree.query(coarse_cent[ci], k=1)
            fi = int(fi)
            buckets[ci].append(fi)
            fallback_hits[ci] = 1
        mapped_counts = np.array([len(b) for b in buckets], dtype=np.int32)

    covered = mapped_counts > 0
    coverage = float(covered.mean()) if nc else 0.0

    cov_frac = np.zeros((nc,), dtype=np.float32)
    for i, b in enumerate(buckets):
        if not b or coarse_area[i] <= 0.0:
            continue
        cov_frac[i] = float(np.sum(fine_area[np.array(b, dtype=np.int32)]) / coarse_area[i])

    coarse_index = np.concatenate(
        [np.full((len(b),), i, dtype=np.int32) for i, b in enumerate(buckets) if b], axis=0
    ) if np.any(mapped_counts > 0) else np.zeros((0,), dtype=np.int32)
    fine_index = np.concatenate(
        [np.array(b, dtype=np.int32) for b in buckets if b], axis=0
    ) if np.any(mapped_counts > 0) else np.zeros((0,), dtype=np.int32)

    return {
        "coarse_index": coarse_index,
        "fine_index": fine_index,
        "n_fine_per_coarse": mapped_counts,
        "coverage_fraction_per_elem": cov_frac,
        "is_boundary": fallback_hits.astype(np.bool_),
        "primary_hit_count": primary_hit,
        "pre_fallback_coverage": np.float32(pre_fallback_coverage),
        "mapping_coverage": np.float32(coverage),
        "mean_refinement_ratio": np.float32(mapped_counts.mean() if nc else 0.0),
        "boundary_elem_frac": np.float32(fallback_hits.mean() if nc else 0.0),
    }


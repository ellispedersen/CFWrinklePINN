from __future__ import annotations

import numpy as np

TARGET_NAMES: tuple[str, ...] = (
    "wrinkle_severity",
    "comp_frac_elem",
    "oop_max_elem",
    "thickness_variance_elem",
)


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
    if fine_fiber_stress.ndim != 2:
        raise ValueError(f"fine_fiber_stress must have shape (T, N), got {fine_fiber_stress.shape}")
    if fine_displacement.ndim != 3 or fine_displacement.shape[2] < 3:
        raise ValueError(f"fine_displacement must have shape (T, N, 3+), got {fine_displacement.shape}")
    if fine_thickness.ndim != 2:
        raise ValueError(f"fine_thickness must have shape (T, N), got {fine_thickness.shape}")
    if fine_elements.ndim != 2 or fine_elements.shape[1] != 3:
        raise ValueError(f"fine_elements must have shape (N_fine_elem, 3), got {fine_elements.shape}")
    if "n_fine_per_coarse" not in coarse_to_fine or "coarse_index" not in coarse_to_fine or "fine_index" not in coarse_to_fine:
        raise ValueError("coarse_to_fine must include n_fine_per_coarse, coarse_index, fine_index")

    n_t = fine_fiber_stress.shape[0]
    if fine_displacement.shape[0] != n_t or fine_thickness.shape[0] != n_t:
        raise ValueError(
            f"Timestep mismatch across fine fields: stress={fine_fiber_stress.shape[0]}, "
            f"displacement={fine_displacement.shape[0]}, thickness={fine_thickness.shape[0]}"
        )
    n_ce = int(coarse_to_fine["n_fine_per_coarse"].shape[0])
    sev = np.full((n_t, n_ce), np.nan, dtype=np.float32)
    comp = np.full((n_t, n_ce), np.nan, dtype=np.float32)
    oop = np.full((n_t, n_ce), np.nan, dtype=np.float32)
    tv = np.full((n_t, n_ce), np.nan, dtype=np.float32)

    cidx = coarse_to_fine["coarse_index"]
    fidx = coarse_to_fine["fine_index"]
    if cidx.shape != fidx.shape:
        raise ValueError(f"coarse_index and fine_index shape mismatch: {cidx.shape} vs {fidx.shape}")
    if cidx.size > 0 and (np.min(cidx) < 0 or np.max(cidx) >= n_ce):
        raise ValueError("coarse_index contains out-of-range coarse element ids")
    if fidx.size > 0 and (np.min(fidx) < 0 or np.max(fidx) >= fine_elements.shape[0]):
        raise ValueError("fine_index contains out-of-range fine element ids")
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

    out = {
        "wrinkle_severity": sev,
        "comp_frac_elem": comp,
        "oop_max_elem": oop,
        "thickness_variance_elem": tv,
    }
    if tuple(out.keys()) != TARGET_NAMES:
        raise RuntimeError(f"Target order mismatch: got={tuple(out.keys())} expected={TARGET_NAMES}")
    for name in TARGET_NAMES:
        if out[name].shape != (n_t, n_ce):
            raise RuntimeError(f"{name} has invalid shape {out[name].shape}; expected {(n_t, n_ce)}")
    return out


from __future__ import annotations

import numpy as np

FEATURE_NAMES: tuple[str, ...] = (
    "dx",
    "dy",
    "dz",
    "temperature",
    "eq_shear_rate",
    "fiber_dir_1_x",
    "fiber_dir_1_y",
    "fiber_dir_1_z",
    "fiber_dir_2_x",
    "fiber_dir_2_y",
    "fiber_dir_2_z",
    "E11",
    "E22",
    "E12",
    "eps_1",
    "eps_2",
    "s11",
    "s22",
    "s12",
    "sigma_1",
    "sigma_2",
    "sigma_comp",
    "fiber_stress_1",
    "fiber_stress_2",
    "fiber_strain_1",
    "fiber_strain_2",
    "thickness",
    "thickness_ratio",
    "shear_angle",
    "locking_proximity",
    "area_change_ratio",
    "draw_in_distance",
    "fiber_comp_indicator",
    "bending_energy_proxy",
    "thickness_rate",
    "shear_rate",
    "fiber_stress_rate",
)


def _align_nodes(arr: np.ndarray, n_nodes: int) -> np.ndarray:
    if arr.shape[1] == n_nodes:
        return arr
    if arr.shape[1] % n_nodes == 0:
        k = arr.shape[1] // n_nodes
        if arr.ndim == 3:
            return arr.reshape(arr.shape[0], n_nodes, k, arr.shape[2]).mean(axis=2)
        return arr.reshape(arr.shape[0], n_nodes, k).mean(axis=2)
    if arr.shape[1] > n_nodes:
        return arr[:, :n_nodes]
    pad_shape = (arr.shape[0], n_nodes - arr.shape[1]) + tuple(arr.shape[2:])
    pad = np.zeros(pad_shape, dtype=arr.dtype)
    return np.concatenate([arr, pad], axis=1)


def principal_components_2d(s11: np.ndarray, s22: np.ndarray, s12: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = 0.5 * (s11 + s22)
    r = np.sqrt(np.maximum(0.0, ((s11 - s22) * 0.5) ** 2 + s12**2))
    return (m + r).astype(np.float32), (m - r).astype(np.float32)


def _require_shape(arr: np.ndarray, expected: tuple[int, ...], name: str) -> None:
    if arr.ndim != len(expected):
        raise ValueError(f"{name} must have {len(expected)} dims, got shape {arr.shape}")
    for i, exp in enumerate(expected):
        if exp >= 0 and arr.shape[i] != exp:
            raise ValueError(f"{name} has invalid shape {arr.shape}; expected dim {i} == {exp}")


def compute_feature_tensor(
    coarse_fields: dict[str, np.ndarray],
    times_s: np.ndarray,
    batch: str,
) -> tuple[list[str], np.ndarray]:
    disp = coarse_fields["displacement"].astype(np.float32)
    if str(batch) not in {"A", "B"}:
        raise ValueError(f"batch must be 'A' or 'B', got {batch!r}")
    if disp.ndim != 3 or disp.shape[2] != 3:
        raise ValueError(f"displacement must have shape (T, N, 3), got {disp.shape}")
    n_t, n_nodes, _ = disp.shape
    _require_shape(times_s, (n_t,), "times_s")

    temp = coarse_fields["temperature"].astype(np.float32)
    if temp.ndim == 3:
        temp = temp.mean(axis=-1)
    _require_shape(temp, (n_t, n_nodes), "temperature")
    thick = coarse_fields["thickness"].astype(np.float32)
    _require_shape(thick, (n_t, n_nodes), "thickness")
    eq_shear = coarse_fields["eq_shear_rate"].astype(np.float32)
    _require_shape(eq_shear, (n_t, n_nodes), "eq_shear_rate")
    dir1 = coarse_fields["fiber_dir_1"].astype(np.float32)
    _require_shape(dir1, (n_t, n_nodes, 3), "fiber_dir_1")
    dir2 = coarse_fields["fiber_dir_2"].astype(np.float32)
    _require_shape(dir2, (n_t, n_nodes, 3), "fiber_dir_2")
    shear = coarse_fields["shear_angle"].astype(np.float32)
    _require_shape(shear, (n_t, n_nodes), "shear_angle")
    fstr1 = coarse_fields["fiber_strain_1"].astype(np.float32)
    _require_shape(fstr1, (n_t, n_nodes), "fiber_strain_1")
    fstr2 = coarse_fields["fiber_strain_2"].astype(np.float32)
    _require_shape(fstr2, (n_t, n_nodes), "fiber_strain_2")
    fst1 = coarse_fields["fiber_stress_1"].astype(np.float32)
    _require_shape(fst1, (n_t, n_nodes), "fiber_stress_1")
    fst2 = coarse_fields["fiber_stress_2"].astype(np.float32)
    _require_shape(fst2, (n_t, n_nodes), "fiber_stress_2")
    gl = _align_nodes(coarse_fields["gl_strain"].astype(np.float32), n_nodes)
    st = _align_nodes(coarse_fields["stress"].astype(np.float32), n_nodes)
    _require_shape(gl, (n_t, n_nodes, 3), "gl_strain")
    _require_shape(st, (n_t, n_nodes, 3), "stress")

    e11, e22, e12 = gl[..., 0], gl[..., 1], gl[..., 2]
    s11, s22, s12 = st[..., 0], st[..., 1], st[..., 2]
    eps1, eps2 = principal_components_2d(e11, e22, e12)
    sig1, sig2 = principal_components_2d(s11, s22, s12)
    sig_comp = np.maximum(np.maximum(-np.minimum(sig1, 0.0), -np.minimum(sig2, 0.0)), 0.0).astype(np.float32)

    t0 = np.maximum(thick[0], 1e-8)
    thick_ratio = thick / t0
    lock = 90.0 if batch == "A" else 45.0
    lock_prox = np.clip(np.abs(shear) / max(lock, 1e-8), 0.0, 1.0).astype(np.float32)
    area_ratio = (1.0 + e11) * (1.0 + e22) - e12 * e12
    draw_in = np.linalg.norm(disp[:, :, :2], axis=-1).astype(np.float32)
    fiber_comp = (fst1 < -0.05).astype(np.float32)
    bend_proxy = ((eps1 - eps2) ** 2 * np.maximum(thick, 0.0) ** 3).astype(np.float32)

    def _rates(a: np.ndarray) -> np.ndarray:
        out = np.zeros_like(a, dtype=np.float32)
        if n_t <= 1:
            return out
        dt = np.diff(times_s.astype(np.float64))
        dt = np.where(np.abs(dt) < 1e-8, 1e-8, dt)
        out[1:] = (np.diff(a.astype(np.float64), axis=0) / dt[:, None]).astype(np.float32)
        return out

    thick_rate = _rates(thick)
    shear_rate = _rates(shear)
    fst1_rate = _rates(fst1)

    names = list(FEATURE_NAMES)
    feats = np.stack(
        [
            disp[..., 0],
            disp[..., 1],
            disp[..., 2],
            temp,
            eq_shear,
            dir1[..., 0],
            dir1[..., 1],
            dir1[..., 2],
            dir2[..., 0],
            dir2[..., 1],
            dir2[..., 2],
            e11,
            e22,
            e12,
            eps1,
            eps2,
            s11,
            s22,
            s12,
            sig1,
            sig2,
            sig_comp,
            fst1,
            fst2,
            fstr1,
            fstr2,
            thick,
            thick_ratio,
            shear,
            lock_prox,
            area_ratio.astype(np.float32),
            draw_in,
            fiber_comp,
            bend_proxy,
            thick_rate,
            shear_rate,
            fst1_rate,
        ],
        axis=-1,
    ).astype(np.float32)
    if feats.shape[-1] != len(FEATURE_NAMES):
        raise RuntimeError(
            f"Feature channel count mismatch: features={feats.shape[-1]} expected={len(FEATURE_NAMES)}"
        )
    return names, feats


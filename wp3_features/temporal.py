from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d


def compute_rates(times_s: np.ndarray, fields: np.ndarray) -> np.ndarray:
    rates = np.zeros_like(fields, dtype=np.float32)
    if fields.shape[0] <= 1:
        return rates
    dt = np.diff(times_s.astype(np.float64))
    dt = np.where(np.abs(dt) < 1e-8, 1e-8, dt)
    delta = np.diff(fields.astype(np.float64), axis=0)
    rates[1:] = (delta / dt.reshape((-1,) + (1,) * (fields.ndim - 1))).astype(np.float32)
    return rates


def resample_to_uniform(
    stroke_fracs: np.ndarray,
    data: np.ndarray,
    n_points: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    if data.shape[0] == 0:
        return np.linspace(0.0, 1.0, n_points, dtype=np.float32), np.zeros(
            (n_points,) + tuple(data.shape[1:]), dtype=np.float32
        )
    x = np.asarray(stroke_fracs, dtype=np.float64)
    y = np.asarray(data, dtype=np.float32)
    x_unique, unique_idx = np.unique(x, return_index=True)
    y = y[unique_idx]
    if x_unique.size == 1:
        target = np.linspace(0.0, 1.0, n_points, dtype=np.float32)
        out = np.repeat(y[:1], n_points, axis=0)
        return target, out.astype(np.float32)
    target = np.linspace(0.0, 1.0, n_points, dtype=np.float64)
    f = interp1d(
        x_unique,
        y,
        axis=0,
        kind="linear",
        bounds_error=False,
        fill_value=(y[0], y[-1]),
        assume_sorted=True,
    )
    out = f(target).astype(np.float32)
    return target.astype(np.float32), out


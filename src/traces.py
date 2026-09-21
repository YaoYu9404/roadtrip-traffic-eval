"""Align two vehicle traces on a common distance grid and score their agreement.

This is the "dual-sensor" analysis: the Civic and the Crosstrek drive the same
road, so their reconstructed speed-vs-distance profiles should agree to within
GPS noise. Systematic disagreement is signal, not noise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .route import Route, add_route_distance, METERS_PER_MILE


def resample_by_distance(df: pd.DataFrame, grid_mi: np.ndarray) -> np.ndarray:
    """Interpolate (smoothed) speed onto a regular distance grid (miles)."""
    scol = "speed_mph_s" if "speed_mph_s" in df else "speed_mph"
    d = df[["route_mi", scol]].dropna().sort_values("route_mi")
    d = d.groupby("route_mi", as_index=False)[scol].mean()  # collapse stops
    return np.interp(grid_mi, d["route_mi"], d[scol], left=np.nan, right=np.nan)


def align_dual(
    trace_a: pd.DataFrame,
    trace_b: pd.DataFrame,
    route: Route,
    step_mi: float = 0.1,
) -> pd.DataFrame:
    """Return a DataFrame [route_mi, speed_a, speed_b, diff] on a shared grid."""
    a = add_route_distance(trace_a, route)
    b = add_route_distance(trace_b, route)
    grid = np.arange(0, route.length_mi + step_mi, step_mi)
    sa = resample_by_distance(a, grid)
    sb = resample_by_distance(b, grid)
    return pd.DataFrame({"route_mi": grid, "speed_a": sa, "speed_b": sb, "diff": sa - sb})


def agreement_stats(aligned: pd.DataFrame) -> dict:
    """Bland-Altman-style agreement between the two vehicles' speed profiles."""
    d = aligned["diff"].dropna().to_numpy()
    mean_a = aligned["speed_a"].dropna().to_numpy()
    if d.size == 0:
        return {"n": 0}
    bias = float(np.mean(d))
    sd = float(np.std(d, ddof=1))
    return {
        "n": int(d.size),
        "mean_bias_mph": bias,
        "rmse_mph": float(np.sqrt(np.mean(d ** 2))),
        "sd_mph": sd,
        "loa_low_mph": bias - 1.96 * sd,   # 95% limits of agreement
        "loa_high_mph": bias + 1.96 * sd,
        "corr": float(np.corrcoef(
            aligned.dropna(subset=["speed_a", "speed_b"])["speed_a"],
            aligned.dropna(subset=["speed_a", "speed_b"])["speed_b"],
        )[0, 1]),
    }

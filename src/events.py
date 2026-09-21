"""Detect rare on-road events and estimate their rate with confidence intervals.

Two event types, both framed as the posting's "rate estimation with rare events":
  * hard-brake events  (sustained deceleration below a threshold)
  * congestion-onset events (a transition from free-flow into slow traffic)

Rates are reported per 100 miles with an exact Poisson CI and a bootstrap CI.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .geo import MPS_TO_MPH


def _speed_col(df: pd.DataFrame, prefer_smooth=True) -> str:
    if prefer_smooth and "speed_mps_s" in df:
        return "speed_mps_s"
    return "speed_mps"


def add_accel_jerk(df: pd.DataFrame, smooth_s: float = 2.0) -> pd.DataFrame:
    """Add smoothed longitudinal acceleration (m/s^2) and jerk (m/s^3).

    Acceleration is computed from the *raw* speed with a light 3 s smoothing:
    the heavier smoothing used for agreement would erase real braking peaks.
    """
    df = df.copy()
    t = (df["time"] - df["time"].iloc[0]).dt.total_seconds().to_numpy()
    # Guard the derivative: np.gradient divides by point spacing, so any
    # non-increasing timestamp (duplicate-second fixes) would blow up. clean_trace
    # normally removes these; nudge any residual ties to keep t strictly increasing.
    if len(t) > 1:
        eps = 1e-3
        for i in range(1, len(t)):
            if t[i] <= t[i - 1]:
                t[i] = t[i - 1] + eps
    v = df["speed_mps"].to_numpy()
    med_dt = np.nanmedian(np.diff(t)) or 1.0
    win = max(1, int(round(smooth_s / med_dt)))
    v_s = pd.Series(v).rolling(win, center=True, min_periods=1).mean().to_numpy()
    a = np.gradient(v_s, t)
    j = np.gradient(a, t)
    df["accel_mps2"] = a
    df["jerk_mps3"] = j
    return df


def _runs(mask: np.ndarray, min_gap: int = 1, min_len: int = 1) -> list[tuple[int, int]]:
    """Boolean mask -> (start, end) runs, merging gaps < min_gap and dropping
    runs shorter than min_len. This de-bounces noisy threshold crossings."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    runs, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev <= min_gap:
            prev = i
        else:
            runs.append((start, prev))
            start = prev = i
    runs.append((start, prev))
    return [(s, e) for s, e in runs if (e - s + 1) >= min_len]


def _mile(seg: pd.DataFrame) -> float:
    return float((seg["route_mi"] if "route_mi" in seg else seg["dist_mi"]).iloc[0])


def detect_hard_brakes(
    df: pd.DataFrame, decel_thresh_mps2: float = -3.0, min_drop_mph: float = 8.0
) -> pd.DataFrame:
    """Hard brake: sustained deceleration past threshold AND a real speed drop."""
    mask = (df["accel_mps2"] <= decel_thresh_mps2).to_numpy()
    rows = []
    for s, e in _runs(mask, min_gap=3, min_len=2):
        seg = df.iloc[s : e + 1]
        drop = float(seg["speed_mph_s"].iloc[0] - seg["speed_mph_s"].min()) if "speed_mph_s" in seg \
            else float(seg["speed_mph"].iloc[0] - seg["speed_mph"].min())
        if drop < min_drop_mph:
            continue
        rows.append({
            "type": "hard_brake",
            "route_mi": _mile(seg),
            "time": seg["time"].iloc[0],
            "min_accel_mps2": float(seg["accel_mps2"].min()),
            "speed_drop_mph": drop,
        })
    return pd.DataFrame(rows)


def detect_congestion_onset(
    df: pd.DataFrame, slow_mph: float = 30.0, free_mph: float = 55.0,
    min_dwell_s: float = 20.0,
) -> pd.DataFrame:
    """Onset = entering a sustained slow stretch after being at/above free-flow."""
    scol = "speed_mph_s" if "speed_mph_s" in df else "speed_mph"
    v = df[scol].to_numpy()
    med_dt = np.nanmedian(df["dt_s"].to_numpy()) or 1.0
    min_len = max(2, int(round(min_dwell_s / med_dt)))
    # A resume after a long stop (parking gap) reads as "slow after free-flow";
    # skip runs whose entry coincides with a flagged long gap so stops aren't
    # counted as congestion onsets.
    long_gap = df["long_gap"].to_numpy() if "long_gap" in df else np.zeros(len(df), bool)
    rows = []
    for s, e in _runs(v < slow_mph, min_gap=int(round(30 / med_dt)), min_len=min_len):
        if long_gap[s : min(len(long_gap), s + 2)].any():
            continue
        pre = df[scol].iloc[max(0, s - int(round(300 / med_dt))): s]
        if len(pre) and pre.max() >= free_mph:
            seg = df.iloc[s : e + 1]
            rows.append({
                "type": "congestion_onset",
                "route_mi": _mile(seg),
                "time": seg["time"].iloc[0],
                "min_speed_mph": float(seg[scol].min()),
            })
    return pd.DataFrame(rows)


def poisson_rate_ci(n: int, exposure_mi: float, per: float = 100.0, alpha: float = 0.05):
    """Exact Poisson CI for a rate = n / exposure, expressed per `per` miles."""
    if exposure_mi <= 0:
        return dict(rate=np.nan, low=np.nan, high=np.nan)
    lo_count = 0.0 if n == 0 else 0.5 * stats.chi2.ppf(alpha / 2, 2 * n)
    hi_count = 0.5 * stats.chi2.ppf(1 - alpha / 2, 2 * n + 2)
    scale = per / exposure_mi
    return dict(n=n, exposure_mi=exposure_mi, rate=n * scale, low=lo_count * scale, high=hi_count * scale)


def bootstrap_rate_ci(
    event_miles: np.ndarray, exposure_mi: float, per: float = 100.0,
    alpha: float = 0.05, n_boot: int = 2000, bin_mi: float = 5.0, seed: int = 0,
):
    """Block bootstrap over distance bins -> CI robust to event clustering."""
    if exposure_mi <= 0:
        return dict(rate=np.nan, low=np.nan, high=np.nan)
    rng = np.random.default_rng(seed)
    n_bins = max(1, int(np.ceil(exposure_mi / bin_mi)))
    counts, _ = np.histogram(event_miles, bins=n_bins, range=(0, exposure_mi))
    boot = [counts[rng.integers(0, n_bins, n_bins)].sum() / exposure_mi * per for _ in range(n_boot)]
    return dict(
        rate=float(counts.sum() / exposure_mi * per),
        low=float(np.percentile(boot, 100 * alpha / 2)),
        high=float(np.percentile(boot, 100 * (1 - alpha / 2))),
    )


def summarize_events(events: pd.DataFrame, exposure_mi: float) -> dict:
    """Poisson + bootstrap rate summaries for each event type present."""
    out = {}
    for etype, grp in events.groupby("type") if len(events) else []:
        miles = grp["route_mi"].to_numpy()
        out[etype] = {
            "poisson": poisson_rate_ci(len(grp), exposure_mi),
            "bootstrap": bootstrap_rate_ci(miles, exposure_mi),
        }
    return out

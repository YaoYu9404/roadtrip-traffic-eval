"""Parse GPX track files into a clean, QC'd DataFrame of trackpoints.

Uses the stdlib XML parser (no gpxpy dependency). Handles the default GPX
namespace and derives per-point speed from consecutive positions + timestamps,
so it works even for loggers that don't store <speed> tags.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

from .geo import haversine_m, MPS_TO_MPH

_GPX_NS = {"gpx": "http://www.topografix.com/GPX/1/1"}


def read_gpx(path: str | Path) -> pd.DataFrame:
    """Read a GPX file into a DataFrame with time, lat, lon, ele columns."""
    path = Path(path)
    root = ET.parse(path).getroot()

    # Strip namespace if present so we can search with or without it.
    def find_all(tag: str):
        pts = root.findall(f".//gpx:{tag}", _GPX_NS)
        if not pts:  # namespace-less GPX
            pts = root.findall(f".//{tag}")
        return pts

    rows = []
    for pt in find_all("trkpt"):
        lat = pt.get("lat")
        lon = pt.get("lon")
        if lat is None or lon is None:
            continue
        t_el = pt.find("gpx:time", _GPX_NS)
        if t_el is None:
            t_el = pt.find("time")
        e_el = pt.find("gpx:ele", _GPX_NS)
        if e_el is None:
            e_el = pt.find("ele")
        rows.append(
            {
                "time": t_el.text if t_el is not None else None,
                "lat": float(lat),
                "lon": float(lon),
                "ele": float(e_el.text) if e_el is not None and e_el.text else np.nan,
            }
        )

    if not rows:
        raise ValueError(f"No <trkpt> points found in {path}")

    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    df.attrs["source"] = path.name
    return df


def add_kinematics(df: pd.DataFrame) -> pd.DataFrame:
    """Add dt, step distance, cumulative distance, and speed columns."""
    df = df.copy()
    lat, lon = df["lat"].to_numpy(), df["lon"].to_numpy()
    step = np.zeros(len(df))
    step[1:] = haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:])
    dt = df["time"].diff().dt.total_seconds().to_numpy()

    df["dt_s"] = dt
    df["step_m"] = step
    df["dist_m"] = np.nancumsum(step)
    df["dist_mi"] = df["dist_m"] / 1609.344
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = np.where(dt > 0, step / dt, np.nan)
    df["speed_mps"] = speed
    df["speed_mph"] = speed * MPS_TO_MPH
    # Smoothed speed: GPS-derived instantaneous speed is noisy, so most analysis
    # (events, dual-sensor agreement) uses a median-then-mean smoothed channel.
    sp = pd.Series(speed)
    sp_s = sp.rolling(5, center=True, min_periods=1).median()
    sp_s = sp_s.rolling(3, center=True, min_periods=1).mean()
    df["speed_mps_s"] = sp_s.to_numpy()
    df["speed_mph_s"] = df["speed_mps_s"] * MPS_TO_MPH
    return df


def spike_mask(
    speed_mph: np.ndarray, window: int = 7, n_sigma: float = 5.0, abs_floor_mph: float = 18.0
) -> np.ndarray:
    """Flag isolated GPS speed spikes via a robust median-residual (Hampel) test.

    A single bad fix makes speed jump far above its neighbours for one sample and
    return; real braking is a *sustained* ramp the rolling median tracks, so it is
    not flagged. Pure MAD collapses to zero around an isolated spike in otherwise
    clean data, so we floor the threshold at ``abs_floor_mph`` (a residual that
    large over a single second is not physical driving).
    """
    s = pd.Series(np.asarray(speed_mph, float))
    med = s.rolling(window, center=True, min_periods=1).median()
    resid = (s - med).abs()
    mad = resid.rolling(window, center=True, min_periods=1).median()
    sigma = 1.4826 * mad
    thr = np.maximum(n_sigma * sigma, abs_floor_mph)
    return (resid > thr).fillna(False).to_numpy()


def quality_control(
    df: pd.DataFrame,
    max_speed_mph: float = 100.0,
    max_gap_s: float = 30.0,
) -> pd.DataFrame:
    """Flag duplicate/zero-dt points, implausible speeds, GPS spikes & long gaps.

    Returns a copy with a boolean ``ok`` column and a summary in ``df.attrs['qc']``.
    We flag rather than delete here; :func:`clean_trace` applies the flags (drops
    ``~ok`` points and recomputes kinematics) so downstream differentiation never
    sees a spike. Long gaps (stops) are flagged separately and kept.
    """
    df = df.copy()
    zero_dt = (df["dt_s"] <= 0) & (df.index > 0)
    over_speed = df["speed_mph"] > max_speed_mph
    spike = spike_mask(df["speed_mph"].to_numpy())
    ok = ~(zero_dt | over_speed | spike)
    long_gap = (df["dt_s"] > max_gap_s).fillna(False)
    df["ok"] = ok
    df["long_gap"] = long_gap
    df.attrs["qc"] = {
        "n_points": int(len(df)),
        "n_flagged_zero_dt": int(zero_dt.sum()),
        "n_flagged_speed": int(over_speed.sum()),
        "n_flagged_spike": int(spike.sum()),
        "n_long_gaps": int(long_gap.sum()),
        "median_dt_s": float(np.nanmedian(df["dt_s"])),
        "duration_min": float((df["time"].iloc[-1] - df["time"].iloc[0]).total_seconds() / 60),
        "distance_mi": float(df["dist_mi"].iloc[-1]),
    }
    return df


def clean_trace(df: pd.DataFrame, max_passes: int = 3) -> pd.DataFrame:
    """Drop QC-flagged points (spikes / zero-dt) and recompute kinematics.

    Iterates a few times because removing the worst spikes can expose smaller
    ones underneath. Stops (``long_gap``) are valid low-speed data and are kept.
    Records ``n_raw`` / ``n_dropped`` in ``df.attrs['qc']``.
    """
    source = df.attrs.get("source")
    n_raw = len(df)
    work = df
    for _ in range(max_passes):
        n_drop = int((~work["ok"]).sum())
        if n_drop == 0:
            break
        base = work.loc[work["ok"], ["time", "lat", "lon", "ele"]].reset_index(drop=True)
        work = quality_control(add_kinematics(base))
    qc = dict(work.attrs.get("qc", {}))
    qc["n_raw"] = n_raw
    qc["n_dropped"] = n_raw - len(work)
    work.attrs["qc"] = qc
    work.attrs["source"] = source
    return work


def load_trace(path: str | Path, clean: bool = True) -> pd.DataFrame:
    """Convenience: read + kinematics + QC in one call (cleaned by default)."""
    df = quality_control(add_kinematics(read_gpx(path)))
    return clean_trace(df) if clean else df

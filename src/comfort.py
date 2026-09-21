"""Ride-comfort dynamics from a GPS trace: longitudinal + lateral g, jerk.

Longitudinal acceleration is dv/dt; lateral (cornering) acceleration is v^2 * kappa,
where kappa is the path curvature estimated from the (lightly smoothed) trajectory
by differentiating position with respect to arc length. Values are in units of g.
Comfort context: passengers notice > ~0.15 g, find > ~0.3 g unpleasant.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .geo import to_local_xy

G = 9.80665


def add_dynamics(df: pd.DataFrame, pos_smooth: int = 5, k_smooth: int = 7) -> pd.DataFrame:
    """Add accel_g (longitudinal), lat_g (|lateral|), jerk_g_s to a kinematic trace."""
    df = df.copy()
    t = (df["time"] - df["time"].iloc[0]).dt.total_seconds().to_numpy()
    v = df["speed_mps_s"].to_numpy()
    a_lon = np.gradient(v, t)

    x, y = to_local_xy(df["lat"].to_numpy(), df["lon"].to_numpy(),
                       float(df["lat"].mean()), float(df["lon"].mean()))
    xs = pd.Series(x).rolling(pos_smooth, center=True, min_periods=1).mean().to_numpy()
    ys = pd.Series(y).rolling(pos_smooth, center=True, min_periods=1).mean().to_numpy()
    ds = np.hypot(np.diff(xs, prepend=xs[0]), np.diff(ys, prepend=ys[0]))
    s = np.maximum.accumulate(np.cumsum(ds)) + np.arange(len(ds)) * 1e-6  # strictly increasing
    xp, yp = np.gradient(xs, s), np.gradient(ys, s)
    xpp, ypp = np.gradient(xp, s), np.gradient(yp, s)
    kappa = (xp * ypp - yp * xpp) / np.power(xp ** 2 + yp ** 2, 1.5)
    a_lat = v ** 2 * kappa

    def sm(a, w):
        return pd.Series(a).rolling(w, center=True, min_periods=1).median().to_numpy()

    df["accel_g"] = sm(a_lon, 5) / G
    df["lat_g_signed"] = sm(a_lat, k_smooth) / G      # + = one turn direction
    df["lat_g"] = np.abs(df["lat_g_signed"])           # cornering intensity
    df["jerk_g_s"] = np.gradient(sm(a_lon, 5), t) / G
    return df

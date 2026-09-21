"""Linear referencing: project trackpoints onto a route polyline.

Given a route (an ordered lat/lon polyline), every trace point gets a
"distance along route" coordinate. This is what lets two vehicles' traces be
compared on a common axis and overlaid on the PeMS space-time speed field.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .geo import to_local_xy, METERS_PER_MILE


class Route:
    def __init__(self, lat, lon):
        self.lat = np.asarray(lat, float)
        self.lon = np.asarray(lon, float)
        self.lat0, self.lon0 = float(self.lat.mean()), float(self.lon.mean())
        self.x, self.y = to_local_xy(self.lat, self.lon, self.lat0, self.lon0)
        seg = np.hypot(np.diff(self.x), np.diff(self.y))
        self.cum_m = np.concatenate([[0.0], np.cumsum(seg)])  # distance at each vertex
        self.length_mi = self.cum_m[-1] / METERS_PER_MILE

    @classmethod
    def from_trace(cls, df: pd.DataFrame, decimate_m: float = 200.0) -> "Route":
        """Build a route by decimating a (clean) trace to ~one vertex / decimate_m."""
        d = df["dist_m"].to_numpy()
        keep = [0]
        last = 0.0
        for i in range(1, len(d)):
            if d[i] - last >= decimate_m:
                keep.append(i)
                last = d[i]
        keep.append(len(d) - 1)
        keep = sorted(set(keep))
        return cls(df["lat"].to_numpy()[keep], df["lon"].to_numpy()[keep])

    def project(self, lat, lon):
        """Return distance-along-route (metres) for each point via nearest segment."""
        px, py = to_local_xy(lat, lon, self.lat0, self.lon0)
        px, py = np.atleast_1d(px), np.atleast_1d(py)
        ax, ay = self.x[:-1], self.y[:-1]
        bx, by = self.x[1:], self.y[1:]
        abx, aby = bx - ax, by - ay
        ab2 = abx ** 2 + aby ** 2
        ab2[ab2 == 0] = 1e-9

        out = np.empty(len(px))
        for i in range(len(px)):
            t = ((px[i] - ax) * abx + (py[i] - ay) * aby) / ab2
            t = np.clip(t, 0.0, 1.0)
            cx, cy = ax + t * abx, ay + t * aby
            d2 = (px[i] - cx) ** 2 + (py[i] - cy) ** 2
            j = int(np.argmin(d2))
            out[i] = self.cum_m[j] + t[j] * np.sqrt(ab2[j])
        return out


def add_route_distance(df: pd.DataFrame, route: Route) -> pd.DataFrame:
    df = df.copy()
    s_m = route.project(df["lat"].to_numpy(), df["lon"].to_numpy())
    df["route_m"] = s_m
    df["route_mi"] = s_m / METERS_PER_MILE
    return df

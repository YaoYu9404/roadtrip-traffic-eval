#!/usr/bin/env python3
"""Concatenate the two legs into one trip and map GPS speed as colour on lat/lon.

Congestion (slow, red) vs. free-flow (fast, green) shown geographically, so the
LA-metro slowdown is visible in place. Uses one car (the Civic by default);
the overnight stop in Templeton is simply the gap between the two legs.

    python map_speed.py                                  # civic_sat + civic_sun
    python map_speed.py --legs data/raw/civic_sat.gpx data/raw/civic_sun.gpx
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import gpx_io

# a few cities for orientation (lat, lon)
_CITIES = {
    "San Diego": (32.72, -117.16),
    "Los Angeles": (34.05, -118.24),
    "Santa Barbara": (34.42, -119.70),
    "San Luis Obispo": (35.28, -120.66),
    "Templeton": (35.55, -120.70),
    "Cupertino": (37.32, -122.03),
}


def load_concat(paths: list[str]) -> pd.DataFrame:
    """Load + clean each leg, tag it, and concatenate in time order."""
    frames = []
    for p in paths:
        t = gpx_io.load_trace(p)
        t = t.loc[~t["long_gap"].fillna(False)].copy()  # drop the parking-gap samples
        t["leg"] = Path(p).stem
        frames.append(t)
    trip = pd.concat(frames, ignore_index=True).sort_values("time").reset_index(drop=True)
    return trip


def make_map(trip: pd.DataFrame, vmin=10.0, vmax=75.0, title="Road-trip speed map"):
    lat = trip["lat"].to_numpy()
    lon = trip["lon"].to_numpy()
    spd = trip["speed_mph_s"].to_numpy()
    ok = np.isfinite(spd)
    lat, lon, spd = lat[ok], lon[ok], spd[ok]

    # draw slowest points last so congestion sits on top and stays visible
    order = np.argsort(-spd)
    lat, lon, spd = lat[order], lon[order], spd[order]

    fig, ax = plt.subplots(figsize=(7.6, 10.5))
    sc = ax.scatter(lon, lat, c=spd, cmap="RdYlGn", vmin=vmin, vmax=vmax,
                    s=5, linewidths=0)
    cb = fig.colorbar(sc, ax=ax, shrink=0.5, pad=0.02)
    cb.set_label("GPS speed (mph)")

    for name, (cy, cx) in _CITIES.items():
        if lon.min() - 0.3 <= cx <= lon.max() + 0.3 and lat.min() - 0.3 <= cy <= lat.max() + 0.3:
            ax.plot(cx, cy, "o", ms=4, color="0.15")
            ax.annotate(name, (cx, cy), textcoords="offset points", xytext=(6, 2),
                        fontsize=8.5, color="0.15",
                        path_effects=None)

    lat0 = float(np.mean(lat))
    ax.set_aspect(1.0 / np.cos(np.radians(lat0)))  # equal ground distance per axis
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.set_title(title)
    ax.grid(True, color="0.9", lw=0.5)
    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--legs", nargs="+",
                    default=["data/raw/civic_sat.gpx", "data/raw/civic_sun.gpx"])
    ap.add_argument("--outdir", default="figures_compare")
    ap.add_argument("--title", default="Road-trip speed map: San Diego → Cupertino (Civic)")
    args = ap.parse_args()

    trip = load_concat(args.legs)
    dist_mi = float(gpx_io.add_kinematics(trip[["time", "lat", "lon", "ele"]])["dist_mi"].iloc[-1])
    dur_h = (trip["time"].iloc[-1] - trip["time"].iloc[0]).total_seconds() / 3600
    print(f"[concat] {len(trip)} points, {len(args.legs)} legs, "
          f"~{dist_mi:.0f} mi driving span, {dur_h:.1f} h wall-clock (incl. overnight)")

    outdir = Path(args.outdir); outdir.mkdir(exist_ok=True)
    fig = make_map(trip, title=args.title)
    out = outdir / "route_speed_map.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"[done] {out}")


if __name__ == "__main__":
    main()

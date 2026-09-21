#!/usr/bin/env python3
"""GPS-error geography: where consumer GPS glitched (localization / map-matching).

Before cleaning, QC flags physically impossible fixes (speed spikes, zero-dt
duplicates). Their locations are where GPS degraded — urban canyons, interchanges,
overpasses. A localization or map-matching pipeline has to reject exactly these.
Plots both cars' flagged fixes on the route.

    python analyze_gps_error.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import gpx_io
from src.basemap import fetch_basemap, flatten_ocean, pick_zoom, merc, TILES


def main():
    legs = {"civic_sat": "Civic", "crosstrek_sat": "Crosstrek",
            "civic_sun": "Civic", "crosstrek_sun": "Crosstrek"}
    route_pts, glitches, n_raw = [], [], 0
    for stem in legs:
        raw = gpx_io.load_trace(f"data/raw/{stem}.gpx", clean=False)  # keep QC flags
        n_raw += len(raw)
        # a GPS *position* glitch = a physically impossible implied speed (a jump);
        # exclude zero-dt duplicate samples, which are a logging artifact not an error.
        flagged = raw[raw["speed_mph"] > 90]
        glitches.append(flagged[["lat", "lon"]])
        clean = gpx_io.load_trace(f"data/raw/{stem}.gpx")
        route_pts.append(clean[["lat", "lon"]])
    route = pd.concat(route_pts, ignore_index=True)
    glitch = pd.concat(glitches, ignore_index=True)

    lat, lon = route["lat"].to_numpy(), route["lon"].to_numpy()
    pad = 0.08
    la0, la1, lo0, lo1 = lat.min()-pad, lat.max()+pad, lon.min()-pad, lon.max()+pad
    zoom = pick_zoom(la0, la1, lo0, lo1)
    canvas, extent, _ = fetch_basemap(la0, la1, lo0, lo1, zoom, "osm")
    canvas = flatten_ocean(canvas)

    aspect = (extent[3]-extent[2]) / (extent[1]-extent[0])
    fig, ax = plt.subplots(figsize=(9, 9*aspect))
    ax.imshow(canvas, extent=extent, origin="upper", interpolation="bilinear")
    rx, ry = merc(lat, lon)
    ax.scatter(rx, ry, s=2, color="#3a6ea5", alpha=0.35, linewidths=0, label="route")
    gx, gy = merc(glitch["lat"].to_numpy(), glitch["lon"].to_numpy())
    ax.scatter(gx, gy, s=26, color="#c0392b", alpha=0.85, edgecolor="white",
               linewidths=0.4, label="GPS glitch (rejected by QC)", zorder=5)
    ax.set_xlim(merc(la0, lo0)[0], merc(la0, lo1)[0])
    ax.set_ylim(merc(la0, lo0)[1], merc(la1, lo0)[1])
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    rate = 100 * len(glitch) / (2 * 498)  # per 100 vehicle-mi (2 cars ~498 mi each leg-set)
    ax.text(0.03, 0.05,
            f"{len(glitch)} GPS glitches rejected across both vehicles   ·   "
            f"clusters flag localization-degraded zones",
            transform=ax.transAxes, fontsize=10, fontweight="bold", color="#2a2a2a",
            va="bottom", bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#999", alpha=0.9))
    ax.set_title(f"GPS-error geography — where consumer GPS failed ({TILES['osm'][1]})", fontsize=11)
    fig.tight_layout()

    outdir = Path("figures_compare"); outdir.mkdir(exist_ok=True)
    out = outdir / "gps_error_map.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(f"[gps] {len(glitch)} glitches / {n_raw} raw fixes")
    print(f"[done] {out}")


if __name__ == "__main__":
    main()

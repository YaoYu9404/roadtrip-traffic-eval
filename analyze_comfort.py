#!/usr/bin/env python3
"""Ride-comfort map: cornering g-force on the route, with a g-g (friction-circle) inset.

Longitudinal + lateral acceleration (in g) from the GPS trajectory. The map colors
each point by cornering intensity |lateral g|; the inset g-g diagram shows the whole
ride's acceleration envelope against comfort thresholds. Frames the drive the way an
AV motion-planning / ride-quality team would evaluate it.

    python analyze_comfort.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import gpx_io
from src.comfort import add_dynamics
from src.basemap import fetch_basemap, flatten_ocean, pick_zoom, merc, TILES


def main():
    legs = ["data/raw/civic_sat.gpx", "data/raw/civic_sun.gpx"]
    d = pd.concat([add_dynamics(gpx_io.load_trace(p)) for p in legs], ignore_index=True)
    d = d[np.isfinite(d["lat_g"]) & ~d["long_gap"].fillna(False)]
    lat, lon = d["lat"].to_numpy(), d["lon"].to_numpy()
    latg = d["lat_g"].to_numpy()

    pad = 0.08
    la0, la1, lo0, lo1 = lat.min()-pad, lat.max()+pad, lon.min()-pad, lon.max()+pad
    zoom = pick_zoom(la0, la1, lo0, lo1)
    canvas, extent, _ = fetch_basemap(la0, la1, lo0, lo1, zoom, "osm")
    canvas = flatten_ocean(canvas)

    mx, my = merc(lat, lon)
    order = np.argsort(latg)  # hard corners on top
    aspect = (extent[3]-extent[2]) / (extent[1]-extent[0])
    fig, ax = plt.subplots(figsize=(9, 9*aspect))
    ax.imshow(canvas, extent=extent, origin="upper", interpolation="bilinear")
    sc = ax.scatter(mx[order], my[order], c=latg[order], cmap="inferno_r",
                    vmin=0, vmax=0.35, s=8, linewidths=0)
    cb = fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.01); cb.set_label("cornering |lateral g|")
    ax.set_xlim(merc(la0, lo0)[0], merc(la0, lo1)[0])
    ax.set_ylim(merc(la0, lo0)[1], merc(la1, lo0)[1])
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Ride comfort — cornering g-force along the route "
                 f"({TILES['osm'][1]})", fontsize=11)

    # g-g (friction-circle) inset, over the empty desert area (top-right)
    ins = ax.inset_axes([0.66, 0.60, 0.31, 0.30])
    ins.scatter(d["lat_g_signed"], d["accel_g"], s=2, alpha=0.15, color="#333")
    for r, c in [(0.15, "#2a9d8f"), (0.30, "#c0492f")]:
        th = np.linspace(0, 2*np.pi, 100)
        ins.plot(r*np.cos(th), r*np.sin(th), color=c, lw=1)
    ins.set_xlim(-0.5, 0.5); ins.set_ylim(-0.5, 0.5); ins.set_aspect("equal")
    ins.set_xlabel("lateral g", fontsize=7); ins.set_ylabel("accel/brake g", fontsize=7)
    ins.tick_params(labelsize=6); ins.set_title("g-g envelope", fontsize=8)
    ins.set_facecolor("white"); ins.patch.set_alpha(0.9)

    peak_c = np.nanpercentile(latg, 99.9)
    peak_b = -np.nanpercentile(d["accel_g"], 0.1)
    ax.text(0.03, 0.05,
            f"peak cornering {peak_c:.2f} g   ·   peak braking {peak_b:.2f} g   ·   "
            f"{100*np.nanmean(latg>0.15):.0f}% of drive > 0.15 g lateral",
            transform=ax.transAxes, fontsize=10, fontweight="bold", color="#2a2a2a",
            va="bottom", bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#999", alpha=0.9))
    fig.tight_layout()

    outdir = Path("figures_compare"); outdir.mkdir(exist_ok=True)
    out = outdir / "ride_comfort_map.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(f"[comfort] peak cornering {peak_c:.2f} g, peak braking {peak_b:.2f} g")
    print(f"[done] {out}")


if __name__ == "__main__":
    main()

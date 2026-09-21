#!/usr/bin/env python3
"""Speed-colored drive on a web-map basemap (Web-Mercator aligned).

    python map_speed_basemap.py                      # full route, OSM + flattened ocean
    python map_speed_basemap.py --tiles esri         # gray canvas
    python map_speed_basemap.py --bbox 33.6 34.3 -118.7 -118.2 --zoom 11   # LA inset
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from src import gpx_io
from src.basemap import fetch_basemap, flatten_ocean, pick_zoom, merc, TILES

# slow = purple, fast = green (stays saturated on a light basemap; avoids white mid)
PURPLE_GREEN = LinearSegmentedColormap.from_list(
    "purple_green", ["#40004b", "#762a83", "#9970ab", "#7fbf7b", "#1b7837", "#00441b"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--legs", nargs="+", default=["data/raw/civic_sat.gpx", "data/raw/civic_sun.gpx"])
    ap.add_argument("--bbox", nargs=4, type=float, default=None,
                    metavar=("LATMIN", "LATMAX", "LONMIN", "LONMAX"))
    ap.add_argument("--zoom", type=int, default=None)
    ap.add_argument("--tiles", choices=list(TILES), default="osm")
    ap.add_argument("--no-flatten-ocean", action="store_true")
    ap.add_argument("--out", default="figures_compare/route_speed_basemap.png")
    args = ap.parse_args()

    frames, tot_mi, tot_min = [], 0.0, 0.0
    for p in args.legs:
        t = gpx_io.load_trace(p)
        tot_mi += float(t["dist_mi"].iloc[-1])
        tot_min += (t["time"].iloc[-1] - t["time"].iloc[0]).total_seconds() / 60.0
        frames.append(t.loc[~t["long_gap"].fillna(False)])
    trip = pd.concat(frames, ignore_index=True)
    lat, lon = trip["lat"].to_numpy(), trip["lon"].to_numpy()
    spd = trip["speed_mph_s"].to_numpy()
    ok = np.isfinite(spd); lat, lon, spd = lat[ok], lon[ok], spd[ok]

    if args.bbox:
        la0, la1, lo0, lo1 = args.bbox
        sel = (lat >= la0) & (lat <= la1) & (lon >= lo0) & (lon <= lo1)
        lat, lon, spd = lat[sel], lon[sel], spd[sel]
    else:
        pad = 0.08
        la0, la1, lo0, lo1 = lat.min()-pad, lat.max()+pad, lon.min()-pad, lon.max()+pad

    zoom = args.zoom or pick_zoom(la0, la1, lo0, lo1)
    canvas, extent, ntiles = fetch_basemap(la0, la1, lo0, lo1, zoom, args.tiles)
    if args.tiles == "osm" and not args.no_flatten_ocean:
        canvas = flatten_ocean(canvas)
    print(f"[basemap] {ntiles} tiles, zoom {zoom} -> {canvas.shape}")

    mx, my = merc(lat, lon)
    order = np.argsort(-spd)
    aspect = (extent[3] - extent[2]) / (extent[1] - extent[0])
    fig, ax = plt.subplots(figsize=(9, 9 * aspect))
    ax.imshow(canvas, extent=extent, origin="upper", interpolation="bilinear")
    sc = ax.scatter(mx[order], my[order], c=spd[order], cmap=PURPLE_GREEN,
                    vmin=10, vmax=70, s=8, linewidths=0)
    cb = fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.01); cb.set_label("GPS speed (mph)")
    ax.set_xlim(merc(la0, lo0)[0], merc(la0, lo1)[0])
    ax.set_ylim(merc(la0, lo0)[1], merc(la1, lo0)[1])
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.03, 0.05, f"{tot_mi:.0f} mi  ·  {tot_min/60:.1f} h driving  ·  2 days",
            transform=ax.transAxes, fontsize=12, fontweight="bold", color="#2a2a2a",
            va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#999", alpha=0.9))
    ax.set_title(f"Road-trip speed  —  San Diego → Cupertino   ({TILES[args.tiles][1]})", fontsize=11)
    fig.tight_layout()

    out = Path(args.out); out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(f"[done] {out}")


if __name__ == "__main__":
    main()

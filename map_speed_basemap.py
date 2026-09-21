#!/usr/bin/env python3
"""Speed-colored drive on a real OpenStreetMap basemap (Web-Mercator aligned).

Fetches OSM tiles for the route's bounding box, stitches them, and scatters the
concatenated trip's GPS speed on top. Everything is projected to Web Mercator so
the points line up with the map.

    python map_speed_basemap.py                      # full route
    python map_speed_basemap.py --bbox 33.6 34.3 -118.7 -118.2 --zoom 11   # LA inset
"""
from __future__ import annotations

import argparse
import io
import math
import time
import urllib.request
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from src import gpx_io

R = 6378137.0
UA = {"User-Agent": "roadtrip-traffic-eval/1.0 (personal portfolio; contact via github)"}


def merc(lat, lon):
    x = R * np.radians(lon)
    y = R * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))
    return x, y


def deg2num(lat, lon, z):
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n
    y = (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n
    return x, y


def num2deg(x, y, z):
    n = 2 ** z
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lat, lon


def fetch_basemap(lat_min, lat_max, lon_min, lon_max, zoom):
    x0, y0 = deg2num(lat_max, lon_min, zoom)   # top-left
    x1, y1 = deg2num(lat_min, lon_max, zoom)   # bottom-right
    xt0, xt1 = int(math.floor(x0)), int(math.floor(x1))
    yt0, yt1 = int(math.floor(y0)), int(math.floor(y1))
    nx, ny = xt1 - xt0 + 1, yt1 - yt0 + 1
    canvas = np.ones(((ny) * 256, (nx) * 256, 3), float)
    for j, yt in enumerate(range(yt0, yt1 + 1)):
        for i, xt in enumerate(range(xt0, xt1 + 1)):
            url = f"https://tile.openstreetmap.org/{zoom}/{xt}/{yt}.png"
            try:
                data = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=15).read()
                tile = mpimg.imread(io.BytesIO(data), format="png")[..., :3]
                canvas[j*256:(j+1)*256, i*256:(i+1)*256] = tile
            except Exception as e:
                print(f"  tile {zoom}/{xt}/{yt} failed: {e}")
            time.sleep(0.05)
    lat_t, lon_l = num2deg(xt0, yt0, zoom)
    lat_b, lon_r = num2deg(xt1 + 1, yt1 + 1, zoom)
    xl, yt = merc(lat_t, lon_l)
    xr, yb = merc(lat_b, lon_r)
    return canvas, (xl, xr, yb, yt), (nx * ny)


def pick_zoom(lat_min, lat_max, lon_min, lon_max, max_tiles=80):
    for z in range(12, 4, -1):
        x0, y0 = deg2num(lat_max, lon_min, z)
        x1, y1 = deg2num(lat_min, lon_max, z)
        nt = (int(x1) - int(x0) + 1) * (int(y1) - int(y0) + 1)
        if nt <= max_tiles:
            return z
    return 6


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--legs", nargs="+",
                    default=["data/raw/civic_sat.gpx", "data/raw/civic_sun.gpx"])
    ap.add_argument("--bbox", nargs=4, type=float, default=None,
                    metavar=("LATMIN", "LATMAX", "LONMIN", "LONMAX"))
    ap.add_argument("--zoom", type=int, default=None)
    ap.add_argument("--out", default="figures_compare/route_speed_basemap.png")
    args = ap.parse_args()

    frames = []
    for p in args.legs:
        t = gpx_io.load_trace(p)
        frames.append(t.loc[~t["long_gap"].fillna(False)])
    trip = __import__("pandas").concat(frames, ignore_index=True)
    lat = trip["lat"].to_numpy(); lon = trip["lon"].to_numpy()
    spd = trip["speed_mph_s"].to_numpy()
    ok = np.isfinite(spd); lat, lon, spd = lat[ok], lon[ok], spd[ok]

    if args.bbox:
        la0, la1, lo0, lo1 = args.bbox
        sel = (lat >= la0) & (lat <= la1) & (lon >= lo0) & (lon <= lo1)
        lat, lon, spd = lat[sel], lon[sel], spd[sel]
    else:
        pad = 0.08
        la0, la1 = lat.min() - pad, lat.max() + pad
        lo0, lo1 = lon.min() - pad, lon.max() + pad

    zoom = args.zoom or pick_zoom(la0, la1, lo0, lo1)
    print(f"[basemap] bbox lat {la0:.2f}..{la1:.2f} lon {lo0:.2f}..{lo1:.2f}  zoom {zoom}")
    canvas, extent, ntiles = fetch_basemap(la0, la1, lo0, lo1, zoom)
    print(f"[basemap] {ntiles} tiles stitched -> {canvas.shape}")

    mx, my = merc(lat, lon)
    order = np.argsort(-spd)  # slow points on top
    fig_w = 9
    aspect = (extent[3] - extent[2]) / (extent[1] - extent[0])
    fig, ax = plt.subplots(figsize=(fig_w, fig_w * aspect))
    ax.imshow(canvas, extent=extent, origin="upper", interpolation="bilinear")
    sc = ax.scatter(mx[order], my[order], c=spd[order], cmap="RdYlGn",
                    vmin=10, vmax=70, s=7, linewidths=0)
    cb = fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.01); cb.set_label("GPS speed (mph)")
    ax.set_xlim(merc(la0, lo0)[0], merc(la0, lo1)[0])
    ax.set_ylim(merc(la0, lo0)[1], merc(la1, lo0)[1])
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Road-trip speed on OpenStreetMap  (© OpenStreetMap contributors)", fontsize=11)
    fig.tight_layout()

    out = Path(args.out); out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(f"[done] {out}")


if __name__ == "__main__":
    main()

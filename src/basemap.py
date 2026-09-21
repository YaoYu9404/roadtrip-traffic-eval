"""Fetch + stitch web-map tiles and project lat/lon to Web Mercator.

Shared by the map figures. Tiles come from OSM (colourful terrain) or Esri Light
Gray; ``flatten_ocean`` repaints the sea flat and strips over-water clutter from
OSM tiles. No API key required for either source.
"""
from __future__ import annotations

import io
import math
import time
import urllib.request

import numpy as np
import matplotlib.image as mpimg

R = 6378137.0
UA = {"User-Agent": "roadtrip-traffic-eval/1.0 (personal portfolio)"}
TILES = {  # name: (url template, attribution, image format)
    "osm": ("https://tile.openstreetmap.org/{z}/{x}/{y}.png",
            "© OpenStreetMap contributors", "png"),
    "esri": ("https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/"
             "World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
             "Tiles © Esri", "jpeg"),
}


def merc(lat, lon):
    x = R * np.radians(lon)
    y = R * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))
    return x, y


def deg2num(lat, lon, z):
    n = 2 ** z
    return (lon + 180.0) / 360.0 * n, (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n


def num2deg(x, y, z):
    n = 2 ** z
    return math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n)))), x / n * 360.0 - 180.0


def pick_zoom(lat_min, lat_max, lon_min, lon_max, max_tiles=80):
    for z in range(12, 4, -1):
        x0, y0 = deg2num(lat_max, lon_min, z)
        x1, y1 = deg2num(lat_min, lon_max, z)
        if (int(x1) - int(x0) + 1) * (int(y1) - int(y0) + 1) <= max_tiles:
            return z
    return 6


def fetch_basemap(lat_min, lat_max, lon_min, lon_max, zoom, tiles="osm"):
    """Return (canvas RGB float array, mercator extent (xl,xr,yb,yt), n_tiles)."""
    url_tmpl, _, fmt = TILES[tiles]
    x0, y0 = deg2num(lat_max, lon_min, zoom)
    x1, y1 = deg2num(lat_min, lon_max, zoom)
    xt0, xt1 = int(math.floor(x0)), int(math.floor(x1))
    yt0, yt1 = int(math.floor(y0)), int(math.floor(y1))
    nx, ny = xt1 - xt0 + 1, yt1 - yt0 + 1
    canvas = np.ones((ny * 256, nx * 256, 3), float)
    for j, yt in enumerate(range(yt0, yt1 + 1)):
        for i, xt in enumerate(range(xt0, xt1 + 1)):
            url = url_tmpl.format(z=zoom, x=xt, y=yt)
            try:
                data = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=15).read()
                tile = mpimg.imread(io.BytesIO(data), format=fmt)[..., :3].astype(float)
                if tile.max() > 1.0:
                    tile /= 255.0
                canvas[j*256:(j+1)*256, i*256:(i+1)*256] = tile
            except Exception as e:
                print(f"  tile {zoom}/{xt}/{yt} failed: {e}")
            time.sleep(0.05)
    lat_t, lon_l = num2deg(xt0, yt0, zoom)
    lat_b, lon_r = num2deg(xt1 + 1, yt1 + 1, zoom)
    xl, yt = merc(lat_t, lon_l)
    xr, yb = merc(lat_b, lon_r)
    return canvas, (xl, xr, yb, yt), nx * ny


def flatten_ocean(canvas, sea_rgb=(170/255, 211/255, 223/255), tol=0.08):
    """Repaint the sea flat and remove thin over-water clutter (depth/boundary
    lines, ocean labels) from OSM tiles, keeping solid landmasses (islands, coast)."""
    from scipy.ndimage import binary_closing, binary_opening, binary_fill_holes
    sea = np.array(sea_rgb)
    seed = np.abs(canvas - sea).sum(axis=2) < tol
    ocean = binary_fill_holes(binary_closing(seed, structure=np.ones((9, 9))))
    bluish = canvas[:, :, 2] >= canvas[:, :, 1] - 0.02
    nonsea = ~bluish
    thin = nonsea & ~binary_opening(nonsea, structure=np.ones((4, 4)))
    canvas[ocean & (bluish | thin)] = sea
    return canvas

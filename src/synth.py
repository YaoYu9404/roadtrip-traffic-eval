"""Generate synthetic GPX traces so the pipeline runs before real data exists.

Produces two correlated vehicle traces (a "Civic" and a "Crosstrek") along a
US-101-like polyline, with a congestion zone and a couple of hard-brake events,
independent GPS noise, and a small following offset between the cars.

Replace these with your real GPX files in data/raw/ and everything downstream
regenerates from the real drive.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .geo import haversine_m, MPS_TO_MPH

# Approximate US-101 corridor Templeton -> Cupertino (lat, lon), decimated.
WAYPOINTS = np.array([
    [35.5480, -120.7085],  # Templeton
    [35.7900, -120.6500],  # Paso Robles / San Miguel
    [36.2000, -121.1300],  # up the Salinas Valley
    [36.6777, -121.6555],  # Salinas
    [37.0058, -121.5683],  # Gilroy
    [37.2000, -121.7500],  # Morgan Hill
    [37.3350, -121.8900],  # San Jose
    [37.3230, -122.0322],  # Cupertino
])


def _densify(waypoints: np.ndarray, step_m: float = 25.0):
    """Interpolate the polyline to a dense set of (lat, lon, cum_dist_m) samples."""
    lats, lons, dists = [waypoints[0, 0]], [waypoints[0, 1]], [0.0]
    for (la0, lo0), (la1, lo1) in zip(waypoints[:-1], waypoints[1:]):
        seg = float(haversine_m(la0, lo0, la1, lo1))
        n = max(2, int(seg / step_m))
        for k in range(1, n + 1):
            f = k / n
            lats.append(la0 + f * (la1 - la0))
            lons.append(lo0 + f * (lo1 - lo0))
            dists.append(dists[-1] + seg / n)
    return np.array(lats), np.array(lons), np.array(dists)


def _speed_profile_mph(s_mi: np.ndarray, length_mi: float, brake_locs, seed=0):
    rng = np.random.default_rng(seed)
    v = np.full_like(s_mi, 66.0)
    # congestion zone with a defined edge (triggers congestion-onset: >55 -> <30)
    v -= 46.0 * np.exp(-((s_mi - 0.42 * length_mi) ** 2) / (2 * 2.5 ** 2))
    # sharp hard-brake events (deep, narrow -> strong deceleration ~ -4 m/s^2)
    for loc in brake_locs:
        v -= 50.0 * np.exp(-((s_mi - loc) ** 2) / (2 * 0.028 ** 2))
    v += rng.normal(0, 0.8, size=v.shape)
    return np.clip(v, 8.0, 78.0)


def simulate_trace(
    start_iso="2026-09-13T05:30:00Z",
    speed_scale: float = 1.0,
    start_offset_s: float = 0.0,
    gps_noise_m: float = 2.5,
    brake_locs=(0.25, 0.70),          # as fractions of corridor length
    seed: int = 0,
):
    """Return arrays (time_iso, lat, lon, ele) for one synthetic vehicle."""
    lat_d, lon_d, dist_d = _densify(WAYPOINTS)
    length_mi = dist_d[-1] / 1609.344
    s_mi = dist_d / 1609.344
    brake_abs = [f * length_mi for f in brake_locs]

    v_mph = _speed_profile_mph(s_mi, length_mi, brake_abs, seed=seed) * speed_scale
    v_mps = np.clip(v_mph / MPS_TO_MPH, 1.0, None)

    # integrate dt = ds / v along the densified path
    ds = np.diff(dist_d)
    dt = ds / v_mps[1:]
    t = np.concatenate([[0.0], np.cumsum(dt)]) + start_offset_s

    # resample to 1 Hz in time
    t_grid = np.arange(0, t[-1], 1.0)
    lat = np.interp(t_grid, t, lat_d)
    lon = np.interp(t_grid, t, lon_d)

    rng = np.random.default_rng(seed + 100)
    dlat = rng.normal(0, gps_noise_m / 111_000.0, size=lat.shape)
    dlon = rng.normal(0, gps_noise_m / (111_000.0 * np.cos(np.radians(35.5))), size=lon.shape)
    lat = lat + dlat
    lon = lon + dlon
    ele = 200 + 50 * np.sin(t_grid / 600.0) + rng.normal(0, 2, size=lat.shape)

    start = np.datetime64(start_iso.replace("Z", ""))
    times = start + (t_grid * 1e9).astype("timedelta64[ns]")
    time_iso = [np.datetime_as_string(x, unit="s") + "Z" for x in times]
    return time_iso, lat, lon, ele


def write_gpx(path, time_iso, lat, lon, ele, name="track"):
    path = Path(path)
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<gpx version="1.1" creator="roadtrip-traffic-eval/synth" '
        'xmlns="http://www.topografix.com/GPX/1/1">',
        f"  <trk><name>{name}</name><trkseg>",
    ]
    for t, la, lo, el in zip(time_iso, lat, lon, ele):
        parts.append(
            f'    <trkpt lat="{la:.6f}" lon="{lo:.6f}">'
            f"<ele>{el:.1f}</ele><time>{t}</time></trkpt>"
        )
    parts += ["  </trkseg></trk>", "</gpx>"]
    path.write_text("\n".join(parts))
    return path


def generate_pair(out_dir="data/synthetic"):
    """Write two synthetic GPX files (Civic ahead, Crosstrek following ~25 s)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    civic = simulate_trace(speed_scale=1.00, start_offset_s=0.0, seed=1)
    cross = simulate_trace(speed_scale=0.98, start_offset_s=25.0, seed=2)
    p1 = write_gpx(out / "civic.gpx", *civic, name="Civic")
    p2 = write_gpx(out / "crosstrek.gpx", *cross, name="Crosstrek")
    return p1, p2


if __name__ == "__main__":
    print("Wrote:", *generate_pair())

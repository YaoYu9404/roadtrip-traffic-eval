"""Small geospatial helpers: distances and a local planar projection.

Kept dependency-free (numpy only) so the rest of the pipeline stays portable.
"""
from __future__ import annotations

import numpy as np

EARTH_RADIUS_M = 6_371_000.0
MPS_TO_MPH = 2.2369362920544
METERS_PER_MILE = 1609.344


def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres between two (arrays of) lat/lon points."""
    lat1, lon1, lat2, lon2 = map(np.asarray, (lat1, lon1, lat2, lon2))
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlmb / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def to_local_xy(lat, lon, lat0, lon0):
    """Equirectangular projection to local metres about (lat0, lon0).

    Good enough (sub-metre over a few-hundred-km corridor at these latitudes)
    for nearest-segment linear referencing.
    """
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    x = np.radians(lon - lon0) * np.cos(np.radians(lat0)) * EARTH_RADIUS_M
    y = np.radians(lat - lat0) * EARTH_RADIUS_M
    return x, y

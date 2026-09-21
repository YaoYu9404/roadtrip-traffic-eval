"""Caltrans PeMS loader + space-time speed field for the corridor.

Real PeMS 5-minute station data (Data Clearinghouse -> Station 5-Minute) comes
as CSV/TXT with columns including timestamp, station, and aggregated speed/flow.
Export a per-station CSV for the corridor and point ``load_pems_5min`` at it.

For development (before you have real data) ``synthetic_speed_field`` produces a
plausible congestion wave so the plots and pipeline run end-to-end.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SpeedField:
    """A time x space grid of speeds (and optionally flows) for the corridor."""
    times: pd.DatetimeIndex     # length T
    postmiles: np.ndarray       # length S (miles along corridor)
    speed: np.ndarray           # shape (T, S), mph
    source: str = "pems"
    flow: np.ndarray | None = None   # shape (T, S), veh/h (total across lanes)


def load_pems_5min(csv_path: str, postmile_col="Postmile", speed_col="Speed",
                   time_col="Timestamp", flow_col="Flow") -> SpeedField:
    """Load a tidy PeMS export (one row per station-timestamp) into a SpeedField.

    Expects columns for timestamp, postmile (or absolute postmile), and speed;
    a flow column is used if present. Adjust the names to match your export.
    """
    df = pd.read_csv(csv_path)
    df[time_col] = pd.to_datetime(df[time_col])

    def grid(col):
        p = df.pivot_table(index=time_col, columns=postmile_col, values=col, aggfunc="mean")
        return p.sort_index().sort_index(axis=1)

    sp = grid(speed_col)
    flow = grid(flow_col).to_numpy(float) if flow_col in df.columns else None
    return SpeedField(times=sp.index, postmiles=sp.columns.to_numpy(float),
                      speed=sp.to_numpy(float), source=csv_path, flow=flow)


# PeMS Data Clearinghouse "Station 5-Minute" files are headerless CSV; these are
# the (0-based) column positions we need. Speed is col 11 (avg over lanes, mph).
_STATION5MIN_COLS = {0: "timestamp", 1: "station", 3: "freeway",
                     4: "direction", 5: "lane_type", 9: "flow", 11: "speed"}


def load_clearinghouse_5min(path: str, lane_type: str | None = "ML") -> pd.DataFrame:
    """Parse a raw PeMS clearinghouse station_5min .txt(.gz) into a tidy frame.

    Returns columns [timestamp, station, freeway, direction, lane_type, flow, speed],
    filtered to ``lane_type`` (default "ML" = mainline) and to rows with a speed.
    Station id is the join key to the metadata file (postmile / lat-lon).
    """
    cols = sorted(_STATION5MIN_COLS)
    df = pd.read_csv(path, header=None, usecols=cols,
                     names=[_STATION5MIN_COLS[i] for i in cols], compression="infer")
    if lane_type:
        df = df[df["lane_type"] == lane_type]
    df = df.dropna(subset=["speed"]).copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%m/%d/%Y %H:%M:%S")
    df["station"] = df["station"].astype("int64")
    return df.reset_index(drop=True)


def load_meta(path: str, lane_type: str | None = "ML") -> pd.DataFrame:
    """Load a PeMS station metadata file (tab-delimited) -> station geometry.

    Returns [station, freeway, direction, abs_pm, lat, lon] for the given lane
    type (default mainline). ``abs_pm`` is absolute postmile along the freeway.
    """
    m = pd.read_csv(path, sep="\t")
    if lane_type:
        m = m[m["Type"] == lane_type]
    m = m[m["Latitude"].notna() & m["Longitude"].notna()]
    out = m[["ID", "Fwy", "Dir", "Abs_PM", "Latitude", "Longitude"]].copy()
    out.columns = ["station", "freeway", "direction", "abs_pm", "lat", "lon"]
    return out.reset_index(drop=True)


def build_corridor_field(speeds: pd.DataFrame, meta: pd.DataFrame,
                         freeway: int, direction: str,
                         start: str | None = None, end: str | None = None) -> SpeedField:
    """Postmile x time speed field for one freeway+direction from clearinghouse data.

    ``speeds`` is the tidy frame from :func:`load_clearinghouse_5min`; ``meta`` is
    from :func:`load_meta`. Speeds are averaged per (postmile, 5-min) cell.
    """
    ms = meta[(meta["freeway"] == freeway) & (meta["direction"] == direction)]
    sp = speeds[speeds["station"].isin(ms["station"])].merge(
        ms[["station", "abs_pm"]], on="station")
    grid = sp.pivot_table(index="timestamp", columns="abs_pm", values="speed",
                          aggfunc="mean").sort_index().sort_index(axis=1)
    if start or end:
        grid = grid.loc[(grid.index >= (start or grid.index.min())) &
                        (grid.index <= (end or grid.index.max()))]
    return SpeedField(times=grid.index, postmiles=grid.columns.to_numpy(float),
                      speed=grid.to_numpy(float),
                      source=f"pems_d{int(meta['freeway'].iloc[0])//100 or ''}_fwy{freeway}{direction}")


def drive_postmiles_on(trace: pd.DataFrame, meta: pd.DataFrame,
                       freeway: int, direction: str, max_dist_m: float = 150.0) -> pd.DataFrame:
    """Locate the trace along one freeway: nearest same-freeway/direction mainline
    station gives each near-corridor point an absolute postmile.

    Returns [t_local, abs_pm, dist_m, speed_mph] for points within ``max_dist_m``.
    """
    from scipy.spatial import cKDTree
    ms = meta[(meta["freeway"] == freeway) & (meta["direction"] == direction)].reset_index(drop=True)
    lat0 = float(ms["lat"].mean()); mpd = 111320.0
    def xy(lat, lon):
        return np.c_[lon * mpd * np.cos(np.radians(lat0)), lat * mpd]
    tree = cKDTree(xy(ms["lat"].values, ms["lon"].values))
    t = trace.copy()
    t_local = t["time"].dt.tz_convert("America/Los_Angeles").dt.tz_localize(None)
    dist, idx = tree.query(xy(t["lat"].values, t["lon"].values))
    out = pd.DataFrame({
        "t_local": t_local.to_numpy(),
        "abs_pm": ms["abs_pm"].values[idx],
        "dist_m": dist,
        "speed_mph": t["speed_mph_s"].to_numpy(),
    })
    return out[out["dist_m"] <= max_dist_m].reset_index(drop=True)


def synthetic_speed_field(
    length_mi: float,
    start="2026-09-13T05:00:00",
    end="2026-09-13T11:00:00",
    freeflow_mph: float = 65.0,
    seed: int = 1,
) -> SpeedField:
    """A fake corridor speed field with one propagating congestion wave."""
    rng = np.random.default_rng(seed)
    times = pd.date_range(start, end, freq="5min")
    postmiles = np.arange(0, length_mi, 0.5)
    T, S = len(times), len(postmiles)
    t_hr = (times - times[0]).total_seconds().to_numpy() / 3600.0

    speed = np.full((T, S), freeflow_mph, float)
    # congestion nucleates at ~40% of corridor around +2.5 h and moves upstream
    center0 = 0.40 * length_mi
    for ti in range(T):
        center = center0 - 6.0 * (t_hr[ti] - 2.5)          # upstream propagation
        width = 8.0 + 3.0 * np.exp(-((t_hr[ti] - 3.0) ** 2))
        depth = 40.0 * np.exp(-((t_hr[ti] - 3.0) ** 2) / 0.8)  # deepest near +3 h
        speed[ti] -= depth * np.exp(-((postmiles - center) ** 2) / (2 * width ** 2))
    speed += rng.normal(0, 1.5, size=speed.shape)
    speed = np.clip(speed, 5, 80)

    # Realistic flow field (veh/h, ~4 lanes): demand-driven on the free-flow
    # branch, near-capacity queue discharge inside congestion.
    qcap_lane, n_lanes = 2000.0, 4
    # peak demand kept below saturation so a plain corridor doesn't gridlock in SUMO
    demand = np.clip(0.40 + 0.35 * np.exp(-((t_hr - 3.0) ** 2) / (2 * 1.2 ** 2)), 0, 0.80)
    w = np.clip((speed - 30.0) / 15.0, 0.0, 1.0)                 # 1 free-flow, 0 jammed
    q_lane = w * (qcap_lane * demand[:, None]) + (1 - w) * (0.88 * qcap_lane)
    flow = n_lanes * q_lane + rng.normal(0, 40, size=speed.shape)
    flow = np.clip(flow, 0, None)
    return SpeedField(times=times, postmiles=postmiles, speed=speed,
                      source="synthetic", flow=flow)

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

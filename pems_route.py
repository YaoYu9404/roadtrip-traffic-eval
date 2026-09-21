#!/usr/bin/env python3
"""Full-route drive vs. PeMS mainline speed, both legs, along the whole trip.

Matches every GPS point to the nearest same-direction mainline PeMS station across
all districts the route crosses (D11 San Diego, D12 Orange, D07 LA, D05 Central
Coast, D04 Bay Area), then plots your GPS speed and the live PeMS mainline speed
against distance along the route. Confirms visually where the corridor itself was
congested vs. where only you slowed (ramps/stops).

    python pems_route.py
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import gpx_io
from src.route import Route, add_route_distance
from src.pems import load_clearinghouse_5min, load_meta, match_trace_to_pems

_DRIVE = "#1f6f8b"
_PEMS = "#c0492f"
PEMS = "data/pems"


def load_all_meta() -> pd.DataFrame:
    return pd.concat([load_meta(p) for p in glob.glob(f"{PEMS}/d*_meta_*.txt")
                      if "_text_meta_" in p or "_meta_" in p], ignore_index=True)


def load_day_speeds(day: str, districts: list[str]) -> pd.DataFrame:
    frames = []
    for d in districts:
        hits = glob.glob(f"{PEMS}/{d}_*station_5min_{day}.txt.gz") + \
               glob.glob(f"{PEMS}/{d}_{day}.txt.gz")
        for p in set(hits):
            frames.append(load_clearinghouse_5min(p))
    return pd.concat(frames, ignore_index=True)


def leg_profile(gpx: str, day: str, districts: list[str], meta: pd.DataFrame,
                bin_mi: float = 1.0) -> pd.DataFrame:
    trace = gpx_io.load_trace(gpx)
    route = Route.from_trace(trace)
    trace = add_route_distance(trace, route)
    speeds = load_day_speeds(day, districts)
    m = match_trace_to_pems(trace, meta, speeds, max_dist_m=150)
    m["mi_bin"] = (m["route_mi"] / bin_mi).round() * bin_mi
    prof = m.groupby("mi_bin").agg(drive=("speed_mph_s", "median"),
                                   pems=("pems_speed", "median"),
                                   cover=("pems_speed", lambda s: s.notna().mean())).reset_index()
    return prof, float(route.length_mi)


def main():
    meta = load_all_meta()
    print(f"[meta] {len(meta)} mainline stations across {meta['freeway'].nunique()} freeways")

    sat, sat_len = leg_profile("data/raw/civic_sat.gpx", "2026_09_12",
                               ["d11", "d12", "d07", "d05"], meta)
    sun, sun_len = leg_profile("data/raw/civic_sun.gpx", "2026_09_13",
                               ["d05", "d04"], meta)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7.4))
    for ax, prof, name, length in [
        (axes[0], sat, "Saturday PM  —  San Diego → Templeton (I-5 · I-405 · US-101)", sat_len),
        (axes[1], sun, "Sunday AM  —  Templeton → Cupertino (US-101, dawn)", sun_len)]:
        ax.plot(prof["mi_bin"], prof["drive"], color=_DRIVE, lw=1.3, label="Your GPS speed")
        ax.plot(prof["mi_bin"], prof["pems"], color=_PEMS, lw=1.6, alpha=0.85,
                label="PeMS mainline speed")
        ax.fill_between(prof["mi_bin"], 0, 80, where=prof["pems"] < 45,
                        color=_PEMS, alpha=0.10, step="mid")
        ax.set_xlim(0, length); ax.set_ylim(0, 80)
        ax.set_ylabel("speed (mph)")
        ax.set_xlabel("distance along route (mi)")
        ax.set_title(name, fontsize=10)
        ax.legend(loc="lower right", fontsize=8.5, ncol=2)

    fig.suptitle("Your drive vs. the live PeMS mainline field — full route "
                 "(shaded = corridor congested, PeMS < 45 mph)", fontsize=12.5, y=0.99)
    fig.tight_layout()

    outdir = Path("figures_compare"); outdir.mkdir(exist_ok=True)
    out = outdir / "pems_route_overlay.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")

    for name, prof in [("Sat", sat), ("Sun", sun)]:
        cov = prof["pems"].notna().mean()
        cong = prof.loc[prof["pems"] < 45, "mi_bin"]
        span = f"{cong.min():.0f}-{cong.max():.0f} mi" if len(cong) else "none"
        print(f"[{name}] PeMS coverage {100*cov:.0f}% of miles | corridor-congested miles: "
              f"{len(cong)} ({span})")
    print(f"[done] {out}")


if __name__ == "__main__":
    main()

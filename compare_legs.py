#!/usr/bin/env python3
"""Two-regime comparison: congested Saturday-afternoon leg vs. free-flow Sunday-dawn leg.

Quantifies the difference between the two traffic regimes the road trip sampled:
  * Leg 1 (Saturday PM): San Diego -> Templeton, through LA -- congested.
  * Leg 2 (Sunday AM):   Templeton -> Cupertino on US-101 -- dawn, free-flow.

For each leg we report moving/stopped time, moving-average speed, time spent in
congestion, congestion-onset rate, and congestion *delay* relative to a free-flow
reference speed (minutes lost per 100 mi). Produces figures_compare/regime_comparison.png
and figures_compare/regime_metrics.json.

    python compare_legs.py                       # uses data/raw/civic_{sat,sun}.gpx
    python compare_legs.py --sat A.gpx --sun B.gpx --free-mph 65 --congested-mph 45

Caveat (stated in the figure): the two legs are different corridors as well as
different times of day, so the contrast reflects route + timing together, not a
clean controlled experiment on one road.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from src import gpx_io
from src.route import Route, add_route_distance
from src.events import add_accel_jerk, detect_congestion_onset
from src.metrics import stopped_time_min

_SAT = "#c0492f"   # congested regime
_SUN = "#1f6f8b"   # free-flow regime


def leg_metrics(path: str, name: str, free_mph: float, congested_mph: float) -> dict:
    """Load one leg (primary trace) and compute regime metrics + a time-weighted
    speed sample (dt-weighted, moving points only) for distributions."""
    t = gpx_io.load_trace(path)
    route = Route.from_trace(t)
    t = add_route_distance(t, route)
    t = add_accel_jerk(t)

    dist_mi = float(t["dist_mi"].iloc[-1])
    total_min = float((t["time"].iloc[-1] - t["time"].iloc[0]).total_seconds() / 60)
    stopped_min = stopped_time_min(t)
    moving_min = total_min - stopped_min

    # moving points only (exclude the long parking gaps) for speed statistics
    dt = t["dt_s"].to_numpy()
    moving = ~t["long_gap"].fillna(False).to_numpy()
    v = t["speed_mph_s"].to_numpy()
    w = np.where(moving & np.isfinite(v) & (dt > 0), dt, 0.0)
    w = np.clip(w, 0, np.nanmedian(dt[dt > 0]) * 5)  # cap gap weights
    vv, ww = v[w > 0], w[w > 0]

    def wq(q):  # dt-weighted quantile of speed
        order = np.argsort(vv)
        cw = np.cumsum(ww[order]) / ww.sum()
        return float(np.interp(q, cw, vv[order]))

    moving_avg_mph = 60 * dist_mi / moving_min if moving_min else np.nan
    pct_time_congested = 100 * ww[vv < congested_mph].sum() / ww.sum()

    onset = detect_congestion_onset(t)
    n_onset = int(len(onset))

    # congestion delay vs a free-flow reference: extra minutes beyond free-flow driving
    free_flow_min = 60 * dist_mi / free_mph
    delay_min = max(0.0, moving_min - free_flow_min)

    return {
        "name": name,
        "distance_mi": dist_mi,
        "total_min": total_min,
        "moving_min": moving_min,
        "stopped_min": stopped_min,
        "moving_avg_mph": moving_avg_mph,
        "median_speed_mph": wq(0.5),
        "p15_speed_mph": wq(0.15),
        "pct_time_congested": pct_time_congested,
        "congestion_onsets": n_onset,
        "onsets_per_100mi": 100 * n_onset / dist_mi if dist_mi else np.nan,
        "free_flow_ref_mph": free_mph,
        "delay_min": delay_min,
        "delay_min_per_100mi": 100 * delay_min / dist_mi if dist_mi else np.nan,
        "_x": t["route_mi"].to_numpy(),
        "_v": v,
        "_vv": vv, "_ww": ww,
    }


def make_figure(sat: dict, sun: dict, congested_mph: float):
    fig = plt.figure(figsize=(12, 8.2))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.15], hspace=0.55, wspace=0.28)

    # Row 1-2: speed-vs-distance for each leg
    for row, leg, color in [(0, sat, _SAT), (1, sun, _SUN)]:
        ax = fig.add_subplot(gs[row, :])
        ax.plot(leg["_x"], leg["_v"], color=color, lw=0.9)
        ax.axhline(congested_mph, color="0.5", ls=":", lw=1)
        ax.fill_between(leg["_x"], 0, leg["_v"], where=leg["_v"] < congested_mph,
                        color=color, alpha=0.18, interpolate=True)
        ax.set_xlim(0, leg["_x"].max())
        ax.set_ylim(0, 90)
        ax.set_ylabel("speed (mph)")
        ax.set_title(f"{leg['name']}  —  {leg['distance_mi']:.0f} mi, "
                     f"moving avg {leg['moving_avg_mph']:.0f} mph, "
                     f"{leg['pct_time_congested']:.0f}% of drive-time in congestion "
                     f"(<{congested_mph:.0f} mph)", fontsize=10)
        ax.set_xlabel("distance along leg (mi)")

    # Row 3 left: dt-weighted speed distributions
    axd = fig.add_subplot(gs[2, 0])
    grid = np.linspace(0, 90, 240)
    for leg, color in [(sat, _SAT), (sun, _SUN)]:
        kde = stats.gaussian_kde(leg["_vv"], weights=leg["_ww"])
        axd.plot(grid, kde(grid), color=color, lw=2, label=leg["name"].split(" —")[0])
        axd.fill_between(grid, kde(grid), color=color, alpha=0.12)
    axd.axvline(congested_mph, color="0.5", ls=":", lw=1)
    axd.set_xlabel("speed (mph, drive-time weighted)")
    axd.set_ylabel("density")
    axd.set_title("Speed distribution by regime", fontsize=10)
    axd.legend(fontsize=9)

    # Row 3 right: metric comparison bars
    axb = fig.add_subplot(gs[2, 1])
    labels = ["moving avg\n(mph)", "% time in\ncongestion", "onsets\n/100 mi", "delay\nmin/100 mi"]
    svals = [sat["moving_avg_mph"], sat["pct_time_congested"],
             sat["onsets_per_100mi"], sat["delay_min_per_100mi"]]
    uvals = [sun["moving_avg_mph"], sun["pct_time_congested"],
             sun["onsets_per_100mi"], sun["delay_min_per_100mi"]]
    x = np.arange(len(labels)); wbar = 0.38
    axb.bar(x - wbar/2, svals, wbar, color=_SAT, label="Sat PM (congested)")
    axb.bar(x + wbar/2, uvals, wbar, color=_SUN, label="Sun AM (free-flow)")
    for xi, (sv, uv) in enumerate(zip(svals, uvals)):
        axb.text(xi - wbar/2, sv, f"{sv:.0f}", ha="center", va="bottom", fontsize=8)
        axb.text(xi + wbar/2, uv, f"{uv:.0f}", ha="center", va="bottom", fontsize=8)
    axb.set_xticks(x); axb.set_xticklabels(labels, fontsize=8.5)
    axb.set_title("Regime metrics", fontsize=10)
    axb.legend(fontsize=8, loc="upper right")

    fig.suptitle("Two traffic regimes on one road trip: congested afternoon vs. free-flow dawn",
                 fontsize=13, y=0.995)
    fig.text(0.5, 0.005,
             "Note: the two legs are different corridors as well as different times of day, "
             "so the contrast reflects route + timing together.",
             ha="center", fontsize=8, color="0.4")
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sat", default="data/raw/civic_sat.gpx")
    ap.add_argument("--sun", default="data/raw/civic_sun.gpx")
    ap.add_argument("--free-mph", type=float, default=65.0)
    ap.add_argument("--congested-mph", type=float, default=45.0)
    ap.add_argument("--outdir", default="figures_compare")
    args = ap.parse_args()

    sat = leg_metrics(args.sat, "Saturday PM  (San Diego → Templeton, via LA)",
                      args.free_mph, args.congested_mph)
    sun = leg_metrics(args.sun, "Sunday AM  (Templeton → Cupertino, US-101 dawn)",
                      args.free_mph, args.congested_mph)

    outdir = Path(args.outdir); outdir.mkdir(exist_ok=True)
    fig = make_figure(sat, sun, args.congested_mph)
    fig.savefig(outdir / "regime_comparison.png", dpi=150, bbox_inches="tight")

    clean = {k: {kk: vv for kk, vv in leg.items() if not kk.startswith("_")}
             for k, leg in [("saturday_pm", sat), ("sunday_am", sun)]}
    clean["contrast"] = {
        "moving_avg_gain_mph": sun["moving_avg_mph"] - sat["moving_avg_mph"],
        "delay_saved_min_per_100mi": sat["delay_min_per_100mi"] - sun["delay_min_per_100mi"],
    }
    (outdir / "regime_metrics.json").write_text(json.dumps(clean, indent=2, default=float))

    print(f"[saturday] {clean['saturday_pm']}")
    print(f"[sunday]   {clean['sunday_am']}")
    print(f"[contrast] {clean['contrast']}")
    print(f"[done] {outdir}/regime_comparison.png + regime_metrics.json")


if __name__ == "__main__":
    main()

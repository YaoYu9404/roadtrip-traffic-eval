#!/usr/bin/env python3
"""ETA validation: naive free-flow vs. live-traffic (PeMS) prediction, on the LA corridor.

Predicts travel time through the congested LA segment two ways — a naive free-flow
assumption, and one that integrates the live PeMS mainline speed field — and validates
both against the actual GPS drive time. Shows the value of real-time traffic in ETA.

    python analyze_eta.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from src import gpx_io
from src.pems import load_clearinghouse_5min, load_meta

FF = 65.0            # naive free-flow speed (mph)
LA_BOX = dict(lat=(33.6, 34.4), lon=(-118.7, -117.6))
_ACT, _NAIVE, _PEMS = "#222222", "#e08a1e", "#1f6f8b"


def pems_speed_along(t, meta, key, max_dist=200):
    lat, lon = t["lat"].to_numpy(), t["lon"].to_numpy()
    tl = t["time"].dt.tz_convert("America/Los_Angeles").dt.tz_localize(None)
    dlat = pd.Series(lat).diff(8).to_numpy(); dlon = pd.Series(lon).diff(8).to_numpy(); c = np.cos(np.radians(34))
    ddir = np.where(np.abs(dlat) > np.abs(dlon*c), np.where(dlat > 0, "N", "S"), np.where(dlon > 0, "E", "W"))
    lat0 = 34.0; mpd = 111320.0
    xy = lambda la, lo: np.c_[lo*mpd*np.cos(np.radians(lat0)), la*mpd]
    st = np.full(len(t), -1, np.int64); di = np.full(len(t), 1e9)
    for dd in ["N", "S", "E", "W"]:
        sub = meta[meta.direction == dd]; m = ddir == dd
        if len(sub) == 0 or m.sum() == 0:
            continue
        d, idx = cKDTree(xy(sub.lat.values, sub.lon.values)).query(xy(lat[m], lon[m]))
        st[m] = sub["station"].values[idx]; di[m] = d
    ps = key.reindex(pd.MultiIndex.from_arrays([pd.Series(st), tl.dt.floor("5min")])).to_numpy()
    return np.where(di <= max_dist, ps, np.nan)


def main():
    meta = load_meta("data/pems/d07_text_meta_2026_08_25.txt")
    key = load_clearinghouse_5min("data/pems/d07_text_station_5min_2026_09_12.txt.gz") \
        .set_index(["station", "timestamp"])["speed"]

    t = gpx_io.load_trace("data/raw/civic_sat.gpx")
    lat, lon = t["lat"].to_numpy(), t["lon"].to_numpy()
    step, dt = t["step_m"].to_numpy(), t["dt_s"].to_numpy()
    box = ((lat >= LA_BOX["lat"][0]) & (lat <= LA_BOX["lat"][1]) &
           (lon >= LA_BOX["lon"][0]) & (lon <= LA_BOX["lon"][1]) & (dt > 0) & (dt < 120))

    ps = pems_speed_along(t, meta, key)
    ps_fill = pd.Series(np.where(box, ps, np.nan)).ffill().bfill().to_numpy()

    seg = t[box].index
    d_mi = np.cumsum(step[box]) / 1609.34
    cum_actual = np.cumsum(dt[box]) / 60.0
    cum_naive = d_mi / FF * 60.0
    cum_pems = np.cumsum(step[box] / (ps_fill[box] * 0.44704)) / 60.0

    dist_tot = d_mi[-1]
    act, naive, pems = cum_actual[-1], cum_naive[-1], cum_pems[-1]
    err_naive, err_pems = naive - act, pems - act

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.7, 1]})
    ax.plot(d_mi, cum_actual, color=_ACT, lw=2.6, label=f"actual drive ({act:.0f} min)")
    ax.plot(d_mi, cum_pems, color=_PEMS, lw=2.2, label=f"ETA w/ live PeMS traffic ({pems:.0f} min)")
    ax.plot(d_mi, cum_naive, color=_NAIVE, lw=2.2, ls="--", label=f"naive free-flow ETA @ {FF:.0f} mph ({naive:.0f} min)")
    ax.set_xlabel("distance along LA corridor (mi)"); ax.set_ylabel("cumulative travel time (min)")
    ax.set_xlim(0, dist_tot); ax.set_ylim(0, max(act, naive, pems) * 1.05)
    ax.set_title(f"ETA on the LA corridor ({dist_tot:.0f} mi, avg {60*dist_tot/act:.0f} mph)", fontsize=11)
    ax.legend(loc="upper left", fontsize=9)

    ax2.bar(["naive\nfree-flow", "live PeMS\ntraffic"], [abs(err_naive), abs(err_pems)],
            color=[_NAIVE, _PEMS], width=0.6)
    for i, e in enumerate([err_naive, err_pems]):
        ax2.text(i, abs(e), f"{e:+.0f} min", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax2.set_ylabel("ETA error vs actual (min)")
    ax2.set_ylim(0, abs(err_naive) * 1.25)
    ax2.set_title(f"Live traffic cuts ETA error {100*(1-abs(err_pems)/abs(err_naive)):.0f}%", fontsize=11)

    fig.suptitle("Predict-then-validate ETA: the value of real-time traffic", fontsize=13, y=1.0)
    fig.tight_layout()
    outdir = Path("figures_compare"); outdir.mkdir(exist_ok=True)
    out = outdir / "eta_validation.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[eta] actual {act:.0f} | naive {naive:.0f} ({err_naive:+.0f}) | pems {pems:.0f} ({err_pems:+.0f})")
    print(f"[done] {out}")


if __name__ == "__main__":
    main()

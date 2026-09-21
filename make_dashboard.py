#!/usr/bin/env python3
"""One-figure road-trip dashboard: speed map (left) + two-car speed & comfort (right).

Left  : GPS speed on an OSM basemap (Web-Mercator, flattened ocean).
Right-top   : Civic vs Crosstrek speed along the whole route (both days concatenated).
Right-bottom: driver smoothness scorecard (comfort comparison).

    python make_dashboard.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import gpx_io
from src.route import Route, add_route_distance
from src.traces import align_dual
from src.basemap import fetch_basemap, flatten_ocean, pick_zoom, merc, TILES
from map_speed_basemap import PURPLE_GREEN
from analyze_comfort_compare import driver, metrics

_A, _B = "#1f6f8b", "#e08a1e"


def main():
    # ---- data ----
    legs = ["data/raw/civic_sat.gpx", "data/raw/civic_sun.gpx"]
    frames, tot_mi, tot_min = [], 0.0, 0.0
    for p in legs:
        t = gpx_io.load_trace(p)
        tot_mi += float(t["dist_mi"].iloc[-1]); tot_min += (t["time"].iloc[-1]-t["time"].iloc[0]).total_seconds()/60
        frames.append(t.loc[~t["long_gap"].fillna(False)])
    trip = pd.concat(frames, ignore_index=True)
    lat, lon, spd = trip["lat"].to_numpy(), trip["lon"].to_numpy(), trip["speed_mph_s"].to_numpy()
    ok = np.isfinite(spd); lat, lon, spd = lat[ok], lon[ok], spd[ok]

    # two-car speed aligned on route distance, both days concatenated
    cs, ks = gpx_io.load_trace("data/raw/civic_sat.gpx"), gpx_io.load_trace("data/raw/crosstrek_sat.gpx")
    cu, ku = gpx_io.load_trace("data/raw/civic_sun.gpx"), gpx_io.load_trace("data/raw/crosstrek_sun.gpx")
    rsat, rsun = Route.from_trace(cs), Route.from_trace(cu)
    a_sat = align_dual(cs, ks, rsat)
    a_sun = align_dual(cu, ku, rsun); a_sun["route_mi"] = a_sun["route_mi"] + rsat.length_mi
    dual = pd.concat([a_sat, a_sun], ignore_index=True)
    boundary = rsat.length_mi

    # comfort metrics
    my, mf = metrics(driver(["civic_sat", "civic_sun"])), metrics(driver(["crosstrek_sat", "crosstrek_sun"]))

    # ---- basemap ----
    pad = 0.08
    la0, la1, lo0, lo1 = lat.min()-pad, lat.max()+pad, lon.min()-pad, lon.max()+pad
    zoom = pick_zoom(la0, la1, lo0, lo1)
    canvas, extent, _ = fetch_basemap(la0, la1, lo0, lo1, zoom, "osm")
    canvas = flatten_ocean(canvas)

    # ---- layout ----
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.2], height_ratios=[1, 1],
                          wspace=0.16, hspace=0.32)
    axm = fig.add_subplot(gs[:, 0])
    axs = fig.add_subplot(gs[0, 1])
    axc = fig.add_subplot(gs[1, 1])

    # map
    axm.imshow(canvas, extent=extent, origin="upper", interpolation="bilinear")
    mx, my_ = merc(lat, lon); order = np.argsort(-spd)
    sc = axm.scatter(mx[order], my_[order], c=spd[order], cmap=PURPLE_GREEN, vmin=10, vmax=70, s=7, linewidths=0)
    cb = fig.colorbar(sc, ax=axm, shrink=0.5, pad=0.01); cb.set_label("GPS speed (mph)")
    axm.set_xlim(merc(la0, lo0)[0], merc(la0, lo1)[0]); axm.set_ylim(merc(la0, lo0)[1], merc(la1, lo0)[1])
    axm.set_xticks([]); axm.set_yticks([])
    axm.text(0.03, 0.04, f"{tot_mi:.0f} mi  ·  {tot_min/60:.1f} h driving  ·  2 days",
             transform=axm.transAxes, fontsize=11, fontweight="bold", color="#2a2a2a", va="bottom",
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#999", alpha=0.9))
    axm.set_title(f"Road-trip speed — San Diego → Cupertino  ({TILES['osm'][1]})", fontsize=10.5)

    # speed comparison
    axs.plot(dual["route_mi"], dual["speed_a"], color=_A, lw=0.8, label="Civic (Yao)")
    axs.plot(dual["route_mi"], dual["speed_b"], color=_B, lw=0.8, alpha=0.8, label="Crosstrek (Fernando)")
    axs.axvline(boundary, color="0.5", ls="--", lw=1)
    axs.text(boundary, 6, " Templeton\n (overnight)", fontsize=7.5, color="0.4", va="bottom")
    axs.set_xlim(0, dual["route_mi"].max()); axs.set_ylim(0, 90)
    axs.set_xlabel("distance along route (mi)"); axs.set_ylabel("speed (mph)")
    axs.set_title("Two GPS sensors, same drive — speed along the whole route "
                  "(both days)", fontsize=10.5)
    axs.legend(loc="lower right", ncol=2, fontsize=8, framealpha=0.9)

    # comfort scorecard (colours match the speed panel, so no separate legend)
    keys = list(my)
    short = {"peak braking (g)": "peak braking (g)", "accel variability (g)": "accel var. (g)",
             "jerk RMS (g/s)": "jerk RMS (g/s)", "% harsh accel/brake": "harsh accel/brake (%)"}
    y = np.arange(len(keys))[::-1]
    for i, k in enumerate(keys):
        vmax = max(my[k], mf[k]); yi = y[i]
        axc.barh(yi+0.18, my[k]/vmax, 0.34, color=_A)
        axc.barh(yi-0.18, mf[k]/vmax, 0.34, color=_B)
        axc.text(my[k]/vmax+0.02, yi+0.18, f"{my[k]:.3g}", va="center", fontsize=8, color=_A)
        axc.text(mf[k]/vmax+0.02, yi-0.18, f"{mf[k]:.3g}", va="center", fontsize=8, color=_B)
    axc.set_yticks(y); axc.set_yticklabels([short[k] for k in keys], fontsize=9)
    axc.set_xlim(0, 1.32); axc.set_xticks([])
    wins_f = sum(mf[k] < my[k] for k in keys)
    axc.set_title("Ride comfort — smoothness scorecard (shorter = smoother):  "
                  f"Crosstrek (Fernando) wins {wins_f}/{len(keys)}", fontsize=10.5)

    fig.suptitle("San Diego → Cupertino: two cars, two independent GPS logs", fontsize=14, y=0.99)
    outdir = Path("figures_compare"); outdir.mkdir(exist_ok=True)
    out = outdir / "roadtrip_dashboard.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[done] {out}")


if __name__ == "__main__":
    main()

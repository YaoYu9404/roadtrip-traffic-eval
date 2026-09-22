#!/usr/bin/env python3
"""Comprehensive LinkedIn poster: one figure summarizing the whole road-trip analysis.

Speed map (left) + key-findings panel, with two-sensor agreement, ride comfort
(g-g + scorecard), and open-highway-vs-LA panels on the right.

    python make_poster.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from src import gpx_io
from src.comfort import add_dynamics
from src.geo import haversine_m
from src.route import Route
from src.traces import align_dual
from src.basemap import fetch_basemap, flatten_ocean, pick_zoom, merc, TILES
from map_speed_basemap import PURPLE_GREEN

_A, _B = "#1f6f8b", "#e08a1e"
BANDS = [(0, 5, "stopped"), (5, 45, "congested"), (45, 65, "moderate"),
         (65, 75, "cruise"), (75, 90, "fast")]
BAND_C = ["#6d6875", "#c0492f", "#caa233", "#2a9d8f", "#1f6f8b"]


def load_dyn(stems):
    d = pd.concat([add_dynamics(gpx_io.load_trace(f"data/raw/{s}.gpx")) for s in stems], ignore_index=True)
    d = d[~d["long_gap"].fillna(False)].copy(); d["w"] = d["dt_s"].clip(0, 3).fillna(0)
    return d[np.isfinite(d["speed_mph_s"]) & (d["w"] > 0)]


def metrics(d):
    a, j = d["accel_g"].to_numpy(), d["jerk_g_s"].to_numpy()
    return {"peak braking (g)": -np.percentile(a, 1), "accel var. (g)": np.std(a),
            "jerk RMS (g/s)": np.sqrt(np.nanmean(j ** 2)), "harsh accel/brake (%)": 100*np.mean(np.abs(a) > 0.15)}


def main():
    # --- data ---
    frames, tot_mi, tot_min = [], 0.0, 0.0
    for p in ["data/raw/civic_sat.gpx", "data/raw/civic_sun.gpx"]:
        t = gpx_io.load_trace(p); tot_mi += float(t["dist_mi"].iloc[-1])
        tot_min += (t["time"].iloc[-1]-t["time"].iloc[0]).total_seconds()/60
        frames.append(t.loc[~t["long_gap"].fillna(False)])
    trip = pd.concat(frames, ignore_index=True)
    lat, lon, spd = trip["lat"].to_numpy(), trip["lon"].to_numpy(), trip["speed_mph_s"].to_numpy()
    ok = np.isfinite(spd); lat, lon, spd = lat[ok], lon[ok], spd[ok]

    # for the public map: trim ~4 km off each end (hide home/destination); keep all
    # segments including LA.
    d_start = haversine_m(lat[0], lon[0], lat, lon)
    d_end = haversine_m(lat[-1], lon[-1], lat, lon)
    keep = (d_start > 4000) & (d_end > 4000)
    lat, lon, spd = lat[keep], lon[keep], spd[keep]

    cs, ks = gpx_io.load_trace("data/raw/civic_sat.gpx"), gpx_io.load_trace("data/raw/crosstrek_sat.gpx")
    cu, ku = gpx_io.load_trace("data/raw/civic_sun.gpx"), gpx_io.load_trace("data/raw/crosstrek_sun.gpx")
    rsat, rsun = Route.from_trace(cs), Route.from_trace(cu)
    a_sat, a_sun = align_dual(cs, ks, rsat), align_dual(cu, ku, rsun)
    a_sun = a_sun.assign(route_mi=a_sun["route_mi"] + rsat.length_mi)
    dual = pd.concat([a_sat, a_sun], ignore_index=True).dropna(subset=["speed_a", "speed_b"])
    r = np.corrcoef(dual["speed_a"], dual["speed_b"])[0, 1]
    rmse = np.sqrt(np.mean((dual["speed_a"] - dual["speed_b"]) ** 2))

    yao, fer = load_dyn(["civic_sat", "civic_sun"]), load_dyn(["crosstrek_sat", "crosstrek_sun"])
    my, mf = metrics(yao), metrics(fer)

    # --- basemap ---
    pad = 0.08
    la0, la1, lo0, lo1 = lat.min()-pad, lat.max()+pad, lon.min()-pad, lon.max()+pad
    zoom = pick_zoom(la0, la1, lo0, lo1)
    canvas, extent, _ = fetch_basemap(la0, la1, lo0, lo1, zoom, "osm"); canvas = flatten_ocean(canvas)

    # --- layout ---
    fig = plt.figure(figsize=(17, 12))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.0, 1.05], height_ratios=[1.0, 1.0, 1.0],
                          wspace=0.14, hspace=0.42)
    axm = fig.add_subplot(gs[:, 0])
    ax1 = fig.add_subplot(gs[0, 1]); ax2 = fig.add_subplot(gs[1, 1]); ax3 = fig.add_subplot(gs[2, 1])

    # map
    axm.imshow(canvas, extent=extent, origin="upper", interpolation="bilinear")
    mx, myy = merc(lat, lon); order = np.argsort(-spd)
    sc = axm.scatter(mx[order], myy[order], c=spd[order], cmap="PRGn", vmin=10, vmax=70, s=6, linewidths=0)
    cb = fig.colorbar(sc, ax=axm, shrink=0.5, pad=0.01); cb.set_label("GPS speed (mph)", fontsize=9)
    axm.set_xlim(merc(la0, lo0)[0], merc(la0, lo1)[0]); axm.set_ylim(merc(la0, lo0)[1], merc(la1, lo0)[1])
    axm.set_xticks([]); axm.set_yticks([])
    axm.text(0.03, 0.03, f"{tot_mi:.0f} mi · {tot_min/60:.1f} h driving · 2 days", transform=axm.transAxes,
             fontsize=11, fontweight="bold", va="bottom",
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#999", alpha=0.9))
    axm.set_title("GPS speed on the road  (purple = slow, green = fast)", fontsize=10.5)

    # key findings overlaid on the empty inland area of the map
    lines = [
        ("2 cars, 2 phones", "independent GPS, same drive"),
        (f"r = {r:.2f}, RMSE {rmse:.1f} mph", "the two sensors agree"),
        ("6,477 PeMS sensors", "validated vs Caltrans traffic field"),
        ("17 → 0 hard brakes", "naive detections were GPS spikes"),
        ("Fernando smoother 4/4", "same speed, ~20% less jerk"),
        ("3.9 h at 65–75 mph", "mostly highway cruise (see distribution)"),
    ]
    axm.text(0.46, 0.955, "KEY FINDINGS", transform=axm.transAxes, fontsize=12.5, fontweight="bold",
             va="top", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#888", alpha=0.92))
    for i, (a, b) in enumerate(lines):
        yv = 0.86 - i * 0.092
        axm.text(0.46, yv, "▸ " + a, transform=axm.transAxes, fontsize=10.5, fontweight="bold",
                 color="#111", va="top",
                 bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
        axm.text(0.49, yv - 0.043, b, transform=axm.transAxes, fontsize=8.5, color="0.35", va="top",
                 bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7))

    # dual-sensor agreement
    ax1.plot(dual["route_mi"], dual["speed_a"], color=_A, lw=0.7, label="Civic (Yao)")
    ax1.plot(dual["route_mi"], dual["speed_b"], color=_B, lw=0.7, alpha=0.8, label="Crosstrek (Fernando)")
    ax1.axvline(rsat.length_mi, color="0.6", ls="--", lw=1)
    ax1.set_xlim(0, dual["route_mi"].max()); ax1.set_ylim(0, 90)
    ax1.set_xlabel("distance along route (mi)", fontsize=9); ax1.set_ylabel("speed (mph)", fontsize=9)
    ax1.set_title(f"Two independent GPS sensors agree along 500 mi  (r={r:.2f}, RMSE {rmse:.1f} mph)", fontsize=10)
    ax1.legend(loc="lower center", ncol=2, fontsize=8, framealpha=0.9)

    # comfort g-g + scorecard result
    for d, c in [(yao, _A), (fer, _B)]:
        ax2.scatter(d["lat_g_signed"], d["accel_g"], s=1.5, alpha=0.10, color=c)
    for rr, c in [(0.15, "#2a9d8f"), (0.30, "#c0492f")]:
        th = np.linspace(0, 2*np.pi, 100); ax2.plot(rr*np.cos(th), rr*np.sin(th), color=c, lw=1)
    ax2.set_xlim(-0.45, 0.45); ax2.set_ylim(-0.45, 0.45); ax2.set_aspect("equal")
    ax2.set_xlabel("lateral g", fontsize=9); ax2.set_ylabel("accel / brake g", fontsize=9)
    ax2.set_title("Ride comfort — g-g envelope (rings 0.15 / 0.30 g)", fontsize=10)
    txt = "Fernando smoother on all 4:\n" + "\n".join(
        f"  {k}: Yao {100*my[k]/mf[k]:.0f}% of Fer" for k in my)
    ax2.text(1.03, 0.5, txt, transform=ax2.transAxes, fontsize=8, va="center",
             bbox=dict(boxstyle="round,pad=0.4", fc="#f7f7f7", ec="#ccc"))

    # speed distribution + hours per band
    grid = np.linspace(0, 90, 300)
    for d, c, name in [(yao, _A, "Yao (Civic)"), (fer, _B, "Fernando (Crosstrek)")]:
        k = stats.gaussian_kde(d["speed_mph_s"], weights=d["w"])
        ax3.plot(grid, k(grid), color=c, lw=1.8, label=name); ax3.fill_between(grid, k(grid), color=c, alpha=0.10)
    for (lo, hi, _), cc in zip(BANDS, BAND_C):
        ax3.axvspan(lo, hi, color=cc, alpha=0.06)
    ax3.set_ylim(0, 0.079); ax3.set_xlim(0, 90)
    for (lo, hi, nm), cc in zip(BANDS, BAND_C):
        hrs = yao.loc[(yao["speed_mph_s"] >= lo) & (yao["speed_mph_s"] < hi), "w"].sum()/3600
        ax3.text((lo+min(hi, 90))/2, 0.074, f"{nm}\n{hrs:.1f} h", ha="center", va="top",
                 fontsize=8, color=cc, fontweight="bold")
    ax3.set_xlabel("speed (mph, drive-time weighted)", fontsize=9); ax3.set_ylabel("density", fontsize=9)
    ax3.set_title("Speed distribution — hours spent in each band", fontsize=10)
    ax3.legend(fontsize=8, loc="center left")

    fig.suptitle("500 miles, two cars, two phones — what consumer GPS reveals about a road trip",
                 fontsize=16, fontweight="bold", y=0.995)
    fig.text(0.5, 0.005, "San Diego → Cupertino · GPS + Caltrans PeMS · an evaluation-framework portfolio project",
             ha="center", fontsize=9.5, color="0.4")
    outdir = Path("figures_compare"); outdir.mkdir(exist_ok=True)
    out = outdir / "roadtrip_poster.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[poster] r={r:.3f} rmse={rmse:.2f}")
    print(f"[done] {out}")


if __name__ == "__main__":
    main()

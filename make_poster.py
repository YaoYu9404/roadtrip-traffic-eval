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
from analyze_timeline import find_breaks, nearest, load as load_leg

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
    sd_pt, cup_pt = (lat[0], lon[0]), (lat[-1], lon[-1])   # trimmed route ends (near SD / Cupertino)

    yao, fer = load_dyn(["civic_sat", "civic_sun"]), load_dyn(["crosstrek_sat", "crosstrek_sun"])
    my, mf = metrics(yao), metrics(fer)

    # Saturday rest stops + overnight (Templeton = end of the Saturday leg)
    sat_raw = load_leg("civic_sat")
    stops = [b for b in find_breaks(sat_raw) if (sat_raw["tl"].iloc[-1] - b[1]).total_seconds() > 90]
    templeton = (float(sat_raw["lat"].iloc[-1]), float(sat_raw["lon"].iloc[-1]))

    # --- basemap ---
    pad = 0.08
    la0, la1, lo0, lo1 = lat.min()-pad, lat.max()+pad, lon.min()-pad, lon.max()+pad
    zoom = pick_zoom(la0, la1, lo0, lo1)
    canvas, extent, _ = fetch_basemap(la0, la1, lo0, lo1, zoom, "osm"); canvas = flatten_ocean(canvas)

    # --- layout: map + right panels on top, full-width speed plot along the bottom ---
    fig = plt.figure(figsize=(17, 13))
    outer = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.0], hspace=0.16)
    topgs = outer[0].subgridspec(1, 2, width_ratios=[1.24, 1.0], wspace=0.12)
    left = topgs[0, 0].subgridspec(2, 1, height_ratios=[40, 1], hspace=0.04)
    axm = fig.add_subplot(left[0]); axcb = fig.add_subplot(left[1])
    right = topgs[0, 1].subgridspec(2, 1, height_ratios=[1.6, 1.0], hspace=0.34)
    ax_top = fig.add_subplot(right[0]); ax_bot = fig.add_subplot(right[1])
    ax_tl = fig.add_subplot(outer[1])

    # map
    axm.imshow(canvas, extent=extent, origin="upper", interpolation="bilinear")
    mx, myy = merc(lat, lon); order = np.argsort(-spd)
    sc = axm.scatter(mx[order], myy[order], c=spd[order], cmap="PRGn", vmin=10, vmax=70, s=6, linewidths=0)
    cb = fig.colorbar(sc, cax=axcb, orientation="horizontal")
    cb.set_label("GPS speed (mph)", fontsize=9.5); cb.ax.tick_params(labelsize=8)
    axm.set_xlim(merc(la0, lo0)[0], merc(la0, lo1)[0]); axm.set_ylim(merc(la0, lo0)[1], merc(la1, lo0)[1])
    axm.set_xticks([]); axm.set_yticks([])
    # city labels at the (trimmed) route ends
    for nm, (cla, clo), off, ha in [("San Diego", sd_pt, (-9, 6), "right"),
                                    ("Cupertino", cup_pt, (9, -4), "left")]:
        cxx, cyy = merc(cla, clo)
        axm.plot(cxx, cyy, "o", ms=6, color="#111", zorder=6)
        axm.annotate(nm, (cxx, cyy), textcoords="offset points", xytext=off, ha=ha,
                     fontsize=10, fontweight="bold", zorder=6,
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
    # Saturday rest stops (numbered) + overnight star, with a key in the empty inland area
    key_lines = ["Rest stops (Saturday):"]
    for i, (s, e, la, lo) in enumerate(stops, 1):
        gx, gy = merc(la, lo)
        axm.plot(gx, gy, "o", ms=13, color="#e08a1e", mec="white", mew=1.2, zorder=7)
        axm.text(gx, gy, str(i), ha="center", va="center", fontsize=8, fontweight="bold",
                 color="white", zorder=8)
        key_lines.append(f"{i}  {nearest(la, lo)}  ({(e-s).total_seconds()/60:.0f} min)")
    tx, ty = merc(*templeton)
    axm.plot(tx, ty, "*", ms=17, color="#c0492f", mec="white", mew=0.8, zorder=7)
    axm.annotate("Templeton\n(overnight)", (tx, ty), textcoords="offset points", xytext=(9, 0),
                 ha="left", va="center", fontsize=8.5, fontweight="bold", color="#c0492f", zorder=7,
                 bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
    key_lines.append("★  Overnight: Templeton")
    axm.text(0.52, 0.60, "\n".join(key_lines), transform=axm.transAxes, fontsize=9, va="top",
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#999", alpha=0.92))
    axm.text(0.03, 0.03, f"{tot_mi:.0f} mi · {tot_min/60:.1f} h driving · 2 days", transform=axm.transAxes,
             fontsize=11, fontweight="bold", va="bottom",
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#999", alpha=0.9))
    axm.set_title("GPS speed on the road  (purple = slow, green = fast)", fontsize=11)

    # speed distribution + hours per band (top-right)
    grid = np.linspace(0, 90, 300)
    for d, c, name in [(yao, _A, "Yao (Civic)"), (fer, _B, "Fernando (Crosstrek)")]:
        k = stats.gaussian_kde(d["speed_mph_s"], weights=d["w"])
        ax_top.plot(grid, k(grid), color=c, lw=1.8, label=name); ax_top.fill_between(grid, k(grid), color=c, alpha=0.10)
    for (lo, hi, _), cc in zip(BANDS, BAND_C):
        ax_top.axvspan(lo, hi, color=cc, alpha=0.06)
    ax_top.set_ylim(0, 0.079); ax_top.set_xlim(0, 90)
    for (lo, hi, nm), cc in zip(BANDS, BAND_C):
        hrs = yao.loc[(yao["speed_mph_s"] >= lo) & (yao["speed_mph_s"] < hi), "w"].sum()/3600
        ax_top.text((lo+min(hi, 90))/2, 0.074, f"{nm}\n{hrs:.1f} h", ha="center", va="top",
                    fontsize=9, color=cc, fontweight="bold")
    ax_top.set_xlabel("speed (mph, drive-time weighted)", fontsize=9.5); ax_top.set_ylabel("density", fontsize=9.5)
    ax_top.set_title("Speed distribution — hours spent in each band", fontsize=11.5)
    ax_top.legend(fontsize=9, loc="center left")

    # ride comfort — smoothness scorecard (bottom-right); bars are far easier to read
    keys = list(my)
    ypos = np.arange(len(keys))[::-1]
    for i, k in enumerate(keys):
        yi = ypos[i]; vmax = max(my[k], mf[k])
        ax_bot.barh(yi + 0.2, my[k] / vmax, 0.36, color=_A)
        ax_bot.barh(yi - 0.2, mf[k] / vmax, 0.36, color=_B)
        ax_bot.text(my[k] / vmax + 0.02, yi + 0.2, f"{my[k]:.3g}", va="center", fontsize=9, color=_A)
        ax_bot.text(mf[k] / vmax + 0.02, yi - 0.2, f"{mf[k]:.3g}", va="center", fontsize=9, color=_B)
    ax_bot.set_yticks(ypos); ax_bot.set_yticklabels([k.replace(" (%)", " %") for k in keys], fontsize=10)
    ax_bot.set_xlim(0, 1.32); ax_bot.set_xticks([])
    ax_bot.set_title("Ride comfort — Fernando smoother on all 4  (blue = Yao, orange = Fernando; "
                     "shorter = smoother)", fontsize=10.5)

    # combined speed plot (both days, along route distance) — full width, bottom
    sat_f, sun_f = frames
    sat_len = float(sat_f["dist_mi"].iloc[-1])
    tx_mi = np.concatenate([sat_f["dist_mi"].to_numpy(), sun_f["dist_mi"].to_numpy() + sat_len])
    tv = np.concatenate([sat_f["speed_mph_s"].to_numpy(), sun_f["speed_mph_s"].to_numpy()])
    ax_tl.plot(tx_mi, tv, color=_A, lw=0.8)
    ax_tl.axvline(sat_len, color="0.5", ls="--", lw=1.2)
    ax_tl.text(sat_len, 90, "  overnight (Templeton)", fontsize=8.5, color="0.4", va="top")
    ax_tl.set_xlim(0, tx_mi[-1]); ax_tl.set_ylim(0, 95)
    ax_tl.set_xlabel("distance along route (mi)", fontsize=9.5); ax_tl.set_ylabel("speed (mph)", fontsize=9.5)
    ax_tl.set_title("Speed across the whole trip — Saturday + Sunday combined", fontsize=11.5)

    fig.suptitle("500 miles, two cars, two phones — what consumer GPS reveals about a road trip",
                 fontsize=16, fontweight="bold", y=0.995)
    fig.text(0.5, 0.005, "San Diego → Cupertino · GPS + Caltrans PeMS · an evaluation-framework portfolio project",
             ha="center", fontsize=9.5, color="0.4")
    outdir = Path("figures_compare"); outdir.mkdir(exist_ok=True)
    out = outdir / "roadtrip_poster.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[done] {out}")


if __name__ == "__main__":
    main()

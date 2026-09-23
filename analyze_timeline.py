#!/usr/bin/env python3
"""Trip timeline: speed across the day, with breaks (dwell) detected and labeled.

Separates a trip into driving and dwell straight from GPS. Saturday shows three
rest stops; Sunday's dawn run is nonstop. A "trip = drive + dwell" view of the
kind Maps / rideshare products model.

    python analyze_timeline.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from src import gpx_io

_LINE, _BREAK = "#1f6f8b", "#c0492f"
LANDMARKS = [("Long Beach", 33.79, -118.13), ("Woodland Hills", 34.17, -118.57),
             ("Thousand Oaks", 34.18, -118.92), ("San Diego", 32.9, -117.2),
             ("Cupertino", 37.32, -122.03)]


def nearest(lat, lon):
    return min(LANDMARKS, key=lambda L: (L[1]-lat)**2 + (L[2]-lon)**2)[0]


def load(stem):
    t = gpx_io.load_trace(f"data/raw/{stem}.gpx", clean=False).reset_index(drop=True)
    t["tl"] = t["time"].dt.tz_convert("America/Los_Angeles").dt.tz_localize(None)
    return t


def find_breaks(t, min_s=90):
    tl = t["tl"]; dt = t["dt_s"].to_numpy(); v = np.nan_to_num(t["speed_mph"].to_numpy())
    iv = []
    for i in np.where(dt > min_s)[0]:
        iv.append([tl.iloc[i-1], tl.iloc[i-1] + pd.Timedelta(seconds=dt[i]), t.lat.iloc[i-1], t.lon.iloc[i-1]])
    st = v < 2.5; i = 0
    while i < len(st):
        if st[i]:
            j = i
            while j < len(st) and st[j]:
                j += 1
            if (tl.iloc[min(j, len(tl)-1)] - tl.iloc[i]).total_seconds() > min_s:
                iv.append([tl.iloc[i], tl.iloc[min(j, len(tl)-1)], t.lat.iloc[i], t.lon.iloc[i]])
            i = j
        else:
            i += 1
    iv.sort()
    merged = []
    for s, e, la, lo in iv:
        if merged and s <= merged[-1][1] + pd.Timedelta(seconds=150):
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e, la, lo])
    return merged


def plot_leg(ax, t, title):
    v = t["speed_mph_s"].to_numpy().copy()
    v[t["dt_s"].to_numpy() > 60] = np.nan          # break the line across gaps
    x = mdates.date2num(t["tl"].dt.to_pydatetime())
    ax.plot(x, v, color=_LINE, lw=1.0)
    # exclude the final stop (arrival at the destination) from the break count
    brks = [b for b in find_breaks(t) if (t["tl"].iloc[-1] - b[1]).total_seconds() > 90]
    for s, e, la, lo in brks:
        xs, xe = mdates.date2num(s.to_pydatetime()), mdates.date2num(e.to_pydatetime())
        ax.axvspan(xs, xe, color=_BREAK, alpha=0.18)
        mins = (e - s).total_seconds() / 60
        ax.annotate(f"{nearest(la, lo)}\n{mins:.0f} min", ((xs+xe)/2, 86), ha="center", va="top",
                    fontsize=8.5, fontweight="bold", color=_BREAK)
    drive_min = (t["dt_s"].to_numpy()[(t["dt_s"] > 0) & (t["dt_s"] < 60)]).sum() / 60
    dwell_min = sum((e - s).total_seconds() for s, e, la, lo in brks) / 60
    ax.xaxis.set_major_locator(mdates.HourLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%-I %p"))
    ax.set_ylim(0, 95); ax.set_ylabel("speed (mph)")
    ax.set_title(f"{title}   —   {drive_min:.0f} min driving  +  "
                 f"{len(brks)} break{'s' if len(brks)!=1 else ''} ({dwell_min:.0f} min)", fontsize=11)


def main():
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 7))
    plot_leg(a1, load("civic_sat"), "Saturday  —  San Diego → Templeton")
    plot_leg(a2, load("civic_sun"), "Sunday  —  Templeton → Cupertino")
    a2.set_xlabel("time of day (Pacific)")
    a2.text(0.5, 0.5, "nonstop dawn run — 0 breaks", transform=a2.transAxes, ha="center",
            fontsize=11, color="#2a9d8f", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#2a9d8f", alpha=0.9))
    fig.suptitle("A trip is drive + dwell: speed across the day, with breaks detected from GPS",
                 fontsize=13, y=0.98)
    fig.tight_layout()
    outdir = Path("figures_compare"); outdir.mkdir(exist_ok=True)
    out = outdir / "trip_timeline.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[done] {out}")


if __name__ == "__main__":
    main()

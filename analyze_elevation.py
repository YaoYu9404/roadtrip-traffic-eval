#!/usr/bin/env python3
"""Elevation + road-grade profile for both legs (energy / speed-modeling context).

Elevation comes from the GPS altitude channel; grade (%) is smoothed rise/run.
The profile line is colored by grade (blue = downhill, red = climb). Total ascent
and peak elevation annotate each leg.

    python analyze_elevation.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from src import gpx_io

M2FT = 3.28084


def leg(path):
    t = gpx_io.load_trace(path)
    mi = t["dist_mi"].to_numpy()
    ele = t["ele"].to_numpy() * M2FT
    dele = pd.Series(t["ele"]).diff().to_numpy()      # metres
    dm = t["step_m"].to_numpy()
    grade = 100 * dele / np.where(dm > 1, dm, np.nan)
    grade = pd.Series(grade).rolling(15, center=True, min_periods=1).median().to_numpy()
    ascent = np.nansum(np.clip(dele, 0, None)) * M2FT
    return mi, ele, grade, ascent


def main():
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    legs = [("Saturday — San Diego → Templeton (via LA)", "data/raw/civic_sat.gpx"),
            ("Sunday — Templeton → Cupertino (US-101)", "data/raw/civic_sun.gpx")]
    lc_last = None
    for ax, (name, path) in zip(axes, legs):
        mi, ele, grade, ascent = leg(path)
        pts = np.array([mi, ele]).T.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap="RdBu_r", norm=plt.Normalize(-6, 6))
        lc.set_array(grade[:-1]); lc.set_linewidth(2.2)
        ax.add_collection(lc); lc_last = lc
        ax.fill_between(mi, 0, ele, color="0.92")
        ax.set_xlim(0, mi.max()); ax.set_ylim(0, np.nanmax(ele) * 1.18)
        ax.set_ylabel("elevation (ft)")
        ax.set_title(f"{name}  —  {ascent:,.0f} ft total ascent, peak {np.nanmax(ele):,.0f} ft",
                     fontsize=10.5)
    axes[1].set_xlabel("distance along route (mi)")
    fig.subplots_adjust(hspace=0.38)

    cb = fig.colorbar(lc_last, ax=axes, shrink=0.7, pad=0.02)
    cb.set_label("road grade (%)   ·   red = climb, blue = descent")
    fig.suptitle("Terrain profile of the drive — elevation & grade", fontsize=13, y=0.98)

    outdir = Path("figures_compare"); outdir.mkdir(exist_ok=True)
    out = outdir / "elevation_grade.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[done] {out}")


if __name__ == "__main__":
    main()

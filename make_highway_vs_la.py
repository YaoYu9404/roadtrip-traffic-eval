#!/usr/bin/env python3
"""Open highway vs. LA basin — how much rougher does driving get in traffic?

Segments the drive geographically (LA basin box vs. everything else) and compares
the ride-comfort metrics. Bars are normalized to the open-highway value (=1) so the
"×N rougher in LA" story is immediate; actual values are labeled.

    python make_highway_vs_la.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import gpx_io
from src.comfort import add_dynamics

_HW, _LA = "#6d6875", "#c0492f"
LA_BOX = dict(lat=(33.6, 34.4), lon=(-118.7, -117.6))


def load(stems):
    d = pd.concat([add_dynamics(gpx_io.load_trace(f"data/raw/{s}.gpx")) for s in stems], ignore_index=True)
    d = d[~d["long_gap"].fillna(False)].copy()
    d["w"] = d["dt_s"].clip(0, 3).fillna(0)
    return d[np.isfinite(d["speed_mph_s"]) & (d["w"] > 0)]


def stat(d):
    a, j = d["accel_g"].to_numpy(), d["jerk_g_s"].to_numpy()
    v, w = d["speed_mph_s"].to_numpy(), d["w"].to_numpy()
    return {"peak braking (g)": -np.percentile(a, 1), "accel var. (g)": np.std(a),
            "jerk RMS (g/s)": np.sqrt(np.nanmean(j ** 2)), "harsh accel/brake (%)": 100*np.mean(np.abs(a) > 0.15),
            "_h": w.sum()/3600/2, "_mph": np.average(v, weights=w)}   # _h = per-driver hours


def main():
    alld = pd.concat([load(["civic_sat", "civic_sun"]), load(["crosstrek_sat", "crosstrek_sun"])],
                     ignore_index=True)
    la = alld["lat"].between(*LA_BOX["lat"]) & alld["lon"].between(*LA_BOX["lon"])
    hw, laa = stat(alld[~la]), stat(alld[la])
    keys = [k for k in hw if not k.startswith("_")]

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    x = np.arange(len(keys)); wbar = 0.38
    ax.bar(x - wbar/2, [1.0]*len(keys), wbar, color=_HW, label="Open highway")
    ax.bar(x + wbar/2, [laa[k]/hw[k] for k in keys], wbar, color=_LA, label="LA basin")
    ax.axhline(1.0, color="0.6", lw=0.8, ls=":")
    for xi, k in enumerate(keys):
        ax.text(xi - wbar/2, 0.92, f"{hw[k]:.3g}", ha="center", va="top", fontsize=8, color="white")
        r = laa[k]/hw[k]
        ax.text(xi + wbar/2, r, f"{laa[k]:.3g}\n×{r:.1f}", ha="center", va="bottom", fontsize=9,
                fontweight="bold", color=_LA)
    ax.set_xticks(x); ax.set_xticklabels(keys, fontsize=9.5)
    ax.set_ylabel("relative to open highway (= 1)")
    ax.set_ylim(0, max(laa[k]/hw[k] for k in keys) * 1.25)
    ax.set_title("Same two drivers, same trip: driving gets rougher in LA traffic", fontsize=12)
    ax.legend(loc="upper left", fontsize=9)
    ax.text(0.99, 0.96,
            f"Open highway: {hw['_mph']:.0f} mph avg · {hw['_h']:.1f} h/driver\n"
            f"LA basin: {laa['_mph']:.0f} mph avg · {laa['_h']:.1f} h/driver",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#999", alpha=0.9))
    fig.tight_layout()

    outdir = Path("figures_compare"); outdir.mkdir(exist_ok=True)
    out = outdir / "highway_vs_la.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("[hw]", {k: round(hw[k], 4) for k in keys})
    print("[la]", {k: round(laa[k], 4) for k in keys})
    print(f"[done] {out}")


if __name__ == "__main__":
    main()

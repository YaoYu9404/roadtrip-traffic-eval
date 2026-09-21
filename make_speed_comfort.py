#!/usr/bin/env python3
"""Speed distribution (with band hours) + ride-comfort comparison (g-g + scores + %).

Left  : drive-time-weighted speed distribution, both drivers, with hours-per-band.
Right : g-g envelope (both drivers) and a smoothness scorecard showing each metric
        for both drivers plus Yao as a percentage of Fernando.

    python make_speed_comfort.py
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

_A, _B = "#1f6f8b", "#e08a1e"
BANDS = [(0, 5, "stopped"), (5, 45, "congested"), (45, 65, "moderate"),
         (65, 75, "cruise"), (75, 90, "fast")]
BAND_C = ["#6d6875", "#c0492f", "#caa233", "#2a9d8f", "#1f6f8b"]


def load(stems):
    d = pd.concat([add_dynamics(gpx_io.load_trace(f"data/raw/{s}.gpx")) for s in stems],
                  ignore_index=True)
    d = d[~d["long_gap"].fillna(False)].copy()
    d["w"] = d["dt_s"].clip(0, 3).fillna(0)
    return d[np.isfinite(d["speed_mph_s"]) & (d["w"] > 0)]


def metrics(d):
    a, la, j = d["accel_g"].to_numpy(), d["lat_g"].to_numpy(), d["jerk_g_s"].to_numpy()
    return {"peak braking (g)": -np.percentile(a, 1),
            "accel var. (g)": np.std(a),
            "jerk RMS (g/s)": np.sqrt(np.nanmean(j ** 2)),
            "harsh accel/brake (%)": 100 * np.mean(np.abs(a) > 0.15)}


def main():
    yao, fer = load(["civic_sat", "civic_sun"]), load(["crosstrek_sat", "crosstrek_sun"])
    my, mf = metrics(yao), metrics(fer)

    fig = plt.figure(figsize=(15, 7.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1.0], height_ratios=[1, 1],
                          wspace=0.32, hspace=0.45)
    axk = fig.add_subplot(gs[:, 0])
    axg = fig.add_subplot(gs[0, 1])
    axs = fig.add_subplot(gs[1, 1])

    # --- speed distribution + band hours ---
    grid = np.linspace(0, 90, 300)
    for d, c, name in [(yao, _A, "Yao (Civic)"), (fer, _B, "Fernando (Crosstrek)")]:
        k = stats.gaussian_kde(d["speed_mph_s"], weights=d["w"])
        axk.plot(grid, k(grid), color=c, lw=2, label=name); axk.fill_between(grid, k(grid), color=c, alpha=0.10)
    for (lo, hi, _), c in zip(BANDS, BAND_C):
        axk.axvspan(lo, hi, color=c, alpha=0.06)
    axk.set_ylim(0, 0.079); axk.set_xlim(0, 90)
    for (lo, hi, name), c in zip(BANDS, BAND_C):
        hrs = yao.loc[(yao["speed_mph_s"] >= lo) & (yao["speed_mph_s"] < hi), "w"].sum() / 3600
        axk.text((lo + min(hi, 90)) / 2, 0.073, f"{name}\n{hrs:.1f} h", ha="center", va="top",
                 fontsize=8.5, color=c, fontweight="bold")
    axk.set_xlabel("speed (mph, drive-time weighted)"); axk.set_ylabel("density")
    axk.set_title("Speed distribution — hours spent in each band", fontsize=11)
    axk.legend(fontsize=8, loc="center left")

    # --- g-g envelope (both drivers) ---
    for d, c in [(yao, _A), (fer, _B)]:
        axg.scatter(d["lat_g_signed"], d["accel_g"], s=2, alpha=0.10, color=c)
    for r, c in [(0.15, "#2a9d8f"), (0.30, "#c0492f")]:
        th = np.linspace(0, 2 * np.pi, 100); axg.plot(r*np.cos(th), r*np.sin(th), color=c, lw=1)
    axg.set_xlim(-0.45, 0.45); axg.set_ylim(-0.45, 0.45); axg.set_aspect("equal")
    axg.set_xlabel("lateral g", fontsize=9); axg.set_ylabel("accel / brake g", fontsize=9)
    axg.set_title("g-g envelope (rings: 0.15 / 0.30 g)", fontsize=10.5)

    # --- scorecard: scores for both + Yao % of Fernando ---
    keys = list(my); y = np.arange(len(keys))[::-1]
    for i, k in enumerate(keys):
        yi = y[i]; vmax = max(my[k], mf[k])
        axs.barh(yi + 0.18, my[k] / vmax, 0.34, color=_A)
        axs.barh(yi - 0.18, mf[k] / vmax, 0.34, color=_B)
        axs.text(my[k] / vmax + 0.02, yi + 0.18, f"{my[k]:.3g}", va="center", fontsize=7.5, color=_A)
        axs.text(mf[k] / vmax + 0.02, yi - 0.18, f"{mf[k]:.3g}", va="center", fontsize=7.5, color=_B)
        pct = 100 * my[k] / mf[k]
        axs.text(1.60, yi, f"Yao {pct:.0f}%\nof Fernando", va="center", ha="right",
                 fontsize=8, color="#b23", fontweight="bold")
    axs.set_yticks(y); axs.set_yticklabels(keys, fontsize=8.5)
    axs.set_xlim(0, 1.62); axs.set_xticks([])
    wins = sum(mf[k] < my[k] for k in keys)
    axs.set_title(f"Smoothness scorecard (shorter = smoother) — Fernando wins {wins}/{len(keys)}",
                  fontsize=10)  # colours match the speed panel (blue = Yao, orange = Fernando)

    fig.suptitle("Speed & ride comfort — Yao (Civic) vs Fernando (Crosstrek), same drive",
                 fontsize=13, y=1.0)
    outdir = Path("figures_compare"); outdir.mkdir(exist_ok=True)
    out = outdir / "speed_comfort.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("[pct] " + ", ".join(f"{k}: Yao {100*my[k]/mf[k]:.0f}% of Fer" for k in keys))
    print(f"[done] {out}")


if __name__ == "__main__":
    main()

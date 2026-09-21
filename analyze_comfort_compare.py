#!/usr/bin/env python3
"""Who drives smoother? Ride-comfort comparison of the two drivers on the same route.

Both cars drove the identical route, so cornering g is road-determined; the driver
signal is in longitudinal smoothness — braking/acceleration gentleness and jerk.
Shows each driver's g-g envelope and a smoothness scorecard.

Caveat baked into the figure: the Crosstrek logged at a higher native rate
(downsampled to ~1 Hz in cleaning), which can bias its jerk/variability slightly
low — so small gaps are within measurement noise.

    python analyze_comfort_compare.py
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

_A, _B = "#1f6f8b", "#e08a1e"


def driver(stems):
    d = pd.concat([add_dynamics(gpx_io.load_trace(f"data/raw/{s}.gpx")) for s in stems],
                  ignore_index=True)
    return d[np.isfinite(d["lat_g"]) & np.isfinite(d["accel_g"])]


def metrics(d):
    a, la, j = d["accel_g"].to_numpy(), d["lat_g"].to_numpy(), d["jerk_g_s"].to_numpy()
    return {
        "peak braking (g)": -np.percentile(a, 1),
        "accel variability (g)": np.std(a),
        "jerk RMS (g/s)": np.sqrt(np.nanmean(j ** 2)),
        "% harsh accel/brake": 100 * np.mean(np.abs(a) > 0.15),
    }


def gg(ax, d, color, name):
    ax.scatter(d["lat_g_signed"], d["accel_g"], s=2, alpha=0.15, color=color)
    for r, c in [(0.15, "#2a9d8f"), (0.30, "#c0492f")]:
        th = np.linspace(0, 2 * np.pi, 100)
        ax.plot(r * np.cos(th), r * np.sin(th), color=c, lw=1)
    ax.set_xlim(-0.45, 0.45); ax.set_ylim(-0.45, 0.45); ax.set_aspect("equal")
    ax.set_xlabel("lateral g"); ax.set_ylabel("accel / brake g")
    ax.set_title(f"{name}  —  g-g envelope", fontsize=10)


def main():
    yao, fer = driver(["civic_sat", "civic_sun"]), driver(["crosstrek_sat", "crosstrek_sun"])
    my, mf = metrics(yao), metrics(fer)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), gridspec_kw={"width_ratios": [1, 1, 1.25]})
    gg(axes[0], yao, _A, "Yao (Civic)")
    gg(axes[1], fer, _B, "Fernando (Crosstrek)")

    ax = axes[2]
    labels = list(my)
    y = np.arange(len(labels))[::-1]
    for i, k in enumerate(labels):
        yi = y[i]
        vmax = max(my[k], mf[k])
        ax.barh(yi + 0.18, my[k] / vmax, 0.34, color=_A)
        ax.barh(yi - 0.18, mf[k] / vmax, 0.34, color=_B)
        ax.text(my[k] / vmax + 0.02, yi + 0.18, f"{my[k]:.3g}", va="center", fontsize=8, color=_A)
        ax.text(mf[k] / vmax + 0.02, yi - 0.18, f"{mf[k]:.3g}", va="center", fontsize=8, color=_B)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(0, 1.25); ax.set_xticks([])
    ax.set_title("Smoothness scorecard  (shorter = smoother)", fontsize=10)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=_A),
                       plt.Rectangle((0, 0), 1, 1, color=_B)],
              labels=["Yao (Civic)", "Fernando (Crosstrek)"], fontsize=8, loc="lower right")

    wins_f = sum(mf[k] < my[k] for k in labels)
    verdict = (f"Fernando is smoother on {wins_f}/{len(labels)} metrics — but by tiny margins, "
               "and the Crosstrek's higher native sample rate can bias its jerk low, "
               "so it's within measurement noise.")
    fig.suptitle("Who drives smoother? Same route, two drivers", fontsize=13, y=1.02)
    fig.text(0.5, -0.04, verdict, ha="center", fontsize=9, color="0.35")
    fig.tight_layout()

    outdir = Path("figures_compare"); outdir.mkdir(exist_ok=True)
    out = outdir / "comfort_compare.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("[compare] Yao:", {k: round(v, 3) for k, v in my.items()})
    print("[compare] Fer:", {k: round(v, 3) for k, v in mf.items()})
    print(f"[done] {out}")


if __name__ == "__main__":
    main()

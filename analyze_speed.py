#!/usr/bin/env python3
"""Speed analysis of the road trip: distribution, time-in-bands, speed vs grade,
and a two-driver speed comparison (does smoother mean slower?).

    python analyze_speed.py
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

_A, _B = "#1f6f8b", "#e08a1e"
BANDS = [(0, 5, "stopped"), (5, 45, "congested"), (45, 65, "moderate"),
         (65, 75, "highway cruise"), (75, 200, "fast")]
BAND_C = ["#6d6875", "#c0492f", "#e9c46a", "#2a9d8f", "#1f6f8b"]


def load(stems):
    d = pd.concat([gpx_io.load_trace(f"data/raw/{s}.gpx") for s in stems], ignore_index=True)
    d = d[~d["long_gap"].fillna(False)].copy()
    d["w"] = d["dt_s"].clip(0, 3).fillna(0)              # drive-time weight
    dele = d["ele"].diff().to_numpy(); dm = d["step_m"].to_numpy()
    g = 100 * dele / np.where(dm > 1, dm, np.nan)
    d["grade"] = pd.Series(g).rolling(15, center=True, min_periods=1).median().to_numpy()
    return d[np.isfinite(d["speed_mph_s"]) & (d["w"] > 0)]


def wq(v, w, q):
    o = np.argsort(v); cw = np.cumsum(w[o]) / w[o].sum()
    return float(np.interp(q, cw, v[o]))


def main():
    yao = load(["civic_sat", "civic_sun"])
    fer = load(["crosstrek_sat", "crosstrek_sun"])
    trip = yao  # single-driver views use the Civic (primary) trace

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.6))
    (a1, a2), (a3, a4) = axes

    # 1 — speed distribution (drive-time weighted), both drivers
    grid = np.linspace(0, 90, 300)
    for d, c, name in [(yao, _A, "Yao (Civic)"), (fer, _B, "Fernando (Crosstrek)")]:
        k = stats.gaussian_kde(d["speed_mph_s"], weights=d["w"])
        a1.plot(grid, k(grid), color=c, lw=2, label=name); a1.fill_between(grid, k(grid), color=c, alpha=0.10)
    for (lo, hi, _), c in zip(BANDS, BAND_C):
        a1.axvspan(lo, min(hi, 90), color=c, alpha=0.06)
    a1.set_xlim(0, 90); a1.set_xlabel("speed (mph, drive-time weighted)"); a1.set_ylabel("density")
    a1.set_title("Speed distribution", fontsize=11); a1.legend(fontsize=8)

    # 2 — hours spent in each speed band (trip, both legs, Civic)
    hrs = []
    for lo, hi, name in BANDS:
        m = (trip["speed_mph_s"] >= lo) & (trip["speed_mph_s"] < hi)
        hrs.append(trip.loc[m, "w"].sum() / 3600)
    ypos = np.arange(len(BANDS))[::-1]
    a2.barh(ypos, hrs, color=BAND_C)
    for yp, h in zip(ypos, hrs):
        a2.text(h + 0.05, yp, f"{h:.1f} h", va="center", fontsize=9)
    a2.set_yticks(ypos); a2.set_yticklabels(
        [f"{n}\n{lo}–{hi} mph" if hi < 200 else f"{n}\n{lo}+ mph" for lo, hi, n in BANDS],
        fontsize=8)
    a2.set_xlim(0, max(hrs) * 1.2); a2.set_xlabel("hours")
    a2.set_title("Time spent in each speed band (driving)", fontsize=11)

    # 3 — speed vs road grade (binned, drive-time weighted; drop sparse extreme bins)
    edges = np.arange(-6, 6.1, 2.0)
    cen, mean_v = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (trip["grade"] >= lo) & (trip["grade"] < hi) & np.isfinite(trip["grade"])
        if trip.loc[m, "w"].sum() > 150:
            cen.append((lo + hi) / 2)
            mean_v.append(np.average(trip.loc[m, "speed_mph_s"], weights=trip.loc[m, "w"]))
    a3.plot(cen, mean_v, "o-", color="#264653", lw=2)
    a3.axvline(0, color="0.7", lw=1)
    a3.set_xlabel("road grade (%)   ← descent   climb →"); a3.set_ylabel("mean speed (mph)")
    a3.set_title("Speed vs. road grade  (climbs trim ~6 mph)", fontsize=11)

    # 4 — two-driver speed comparison
    def stat(d):
        v, w = d["speed_mph_s"].to_numpy(), d["w"].to_numpy()
        dist = d["dist_mi"].iloc[-1] if False else None
        return {"moving avg (mph)": np.average(v, weights=w),
                "top speed p99.5 (mph)": wq(v, w, 0.995),
                "% time > 65 mph": 100 * w[v > 65].sum() / w.sum()}
    sy, sf = stat(yao), stat(fer)
    keys = list(sy); x = np.arange(len(keys)); wbar = 0.38
    a4.bar(x - wbar/2, [sy[k] for k in keys], wbar, color=_A, label="Yao (Civic)")
    a4.bar(x + wbar/2, [sf[k] for k in keys], wbar, color=_B, label="Fernando (Crosstrek)")
    for xi, k in enumerate(keys):
        a4.text(xi - wbar/2, sy[k], f"{sy[k]:.0f}", ha="center", va="bottom", fontsize=8)
        a4.text(xi + wbar/2, sf[k], f"{sf[k]:.0f}", ha="center", va="bottom", fontsize=8)
    a4.set_xticks(x); a4.set_xticklabels(keys, fontsize=8.5)
    a4.set_title("Who drove faster? (they drove together, so it's close)", fontsize=11)
    a4.legend(fontsize=8)

    fig.suptitle("Road-trip speed analysis — San Diego → Cupertino", fontsize=14, y=1.0)
    fig.tight_layout()
    outdir = Path("figures_compare"); outdir.mkdir(exist_ok=True)
    out = outdir / "speed_analysis.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[speed] Yao {sy}")
    print(f"[speed] Fer {sf}")
    print(f"[done] {out}")


if __name__ == "__main__":
    main()

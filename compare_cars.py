#!/usr/bin/env python3
"""Two cars, same route, same start: how much do they diverge? (ETA uncertainty)

The Saturday leg had a synchronized start (24 s apart), so it's a controlled look
at irreducible trip-time variability: even driving "together", two vehicles trade
the lead, drift apart along the route, and arrive a couple of minutes apart. Also
reports how many congestion onsets were corroborated by both cars (a cheap
ground-truth check).

    python compare_cars.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import gpx_io
from src.route import Route, add_route_distance
from src.events import detect_congestion_onset

_CIV = "#1f6f8b"
_CRO = "#e08a1e"
_ACC = "#c0492f"


def prep(civ_path, cro_path):
    civ = gpx_io.load_trace(civ_path)
    cro = gpx_io.load_trace(cro_path)
    route = Route.from_trace(civ)
    civ = add_route_distance(civ, route)
    cro = add_route_distance(cro, route)
    t0 = min(civ["time"].iloc[0], cro["time"].iloc[0])
    for d in (civ, cro):
        d["s"] = (d["time"] - t0).dt.total_seconds()
        d["rm"] = np.maximum.accumulate(d["route_mi"].to_numpy())  # forward progress
    return civ, cro


def main():
    civ, cro = prep("data/raw/civic_sat.gpx", "data/raw/crosstrek_sat.gpx")

    # along-route separation on a common 10 s time grid (+ = Civic ahead)
    tg = np.arange(0, min(civ["s"].max(), cro["s"].max()), 10.0)
    mc = np.interp(tg, civ["s"], civ["rm"])
    mk = np.interp(tg, cro["s"], cro["rm"])
    sep = mc - mk

    # time gap to reach each milepost (+ = Civic later)
    mg = np.arange(0, min(mc.max(), mk.max()), 0.5)
    dt = (np.interp(mg, civ["rm"], civ["s"]) - np.interp(mg, cro["rm"], cro["s"])) / 60.0

    # congestion-onset corroboration
    oc = detect_congestion_onset(civ); ok = detect_congestion_onset(cro)
    def corroborated(a, b, tol=2.0):
        if len(a) == 0:
            return 0
        return int(sum(np.min(np.abs(b["route_mi"].to_numpy() - r)) <= tol
                       for r in a["route_mi"].to_numpy())) if len(b) else 0
    both = corroborated(oc, ok)

    stats = {
        "start_offset_s": float((cro["time"].iloc[0] - civ["time"].iloc[0]).total_seconds()),
        "civic_min": float(civ["s"].iloc[-1] / 60), "crosstrek_min": float(cro["s"].iloc[-1] / 60),
        "arrival_gap_min": float(abs(civ["s"].iloc[-1] - cro["s"].iloc[-1]) / 60),
        "max_separation_mi": float(np.max(np.abs(sep))),
        "max_time_gap_min": float(np.max(np.abs(dt))),
        "lead_changes": int((np.diff(np.sign(sep[sep != 0])) != 0).sum()),
        "civic_onsets": int(len(oc)), "crosstrek_onsets": int(len(ok)),
        "onsets_corroborated": both,
    }

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 7.2))
    a1.axhline(0, color="0.6", lw=1)
    a1.fill_between(tg / 60, 0, sep, where=sep >= 0, color=_CIV, alpha=0.25, interpolate=True)
    a1.fill_between(tg / 60, 0, sep, where=sep < 0, color=_CRO, alpha=0.25, interpolate=True)
    a1.plot(tg / 60, sep, color="0.2", lw=1)
    a1.set_ylabel("Civic ahead  (mi)  behind")
    a1.set_xlabel("elapsed time (min)")
    a1.set_title(f"Along-route separation — two cars, same start (24 s apart): "
                 f"{stats['lead_changes']} lead changes, up to {stats['max_separation_mi']:.1f} mi apart",
                 fontsize=10.5)

    a2.axhline(0, color="0.6", lw=1)
    a2.plot(mg, dt, color=_ACC, lw=1.3)
    a2.set_ylabel("Civic later  (min)  earlier")
    a2.set_xlabel("distance along route (mi)")
    a2.set_title(f"Time gap to reach each milepost — arrival differed by "
                 f"{stats['arrival_gap_min']:.1f} min over {mg.max():.0f} mi "
                 f"(an ETA is a distribution, not a number)", fontsize=10.5)

    fig.suptitle("Two cars, one route, same time: irreducible trip-time divergence", fontsize=13, y=0.99)
    fig.tight_layout()
    outdir = Path("figures_compare"); outdir.mkdir(exist_ok=True)
    fig.savefig(outdir / "two_car_divergence.png", dpi=150, bbox_inches="tight")
    (outdir / "two_car_divergence_metrics.json").write_text(json.dumps(stats, indent=2, default=float))
    print(f"[stats] {stats}")
    print(f"[done] {outdir}/two_car_divergence.png")


if __name__ == "__main__":
    main()

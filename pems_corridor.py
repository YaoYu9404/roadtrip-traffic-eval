#!/usr/bin/env python3
"""Real PeMS space-time speed contour for one LA freeway, with the drive overlaid.

Builds the postmile x time mainline speed field from PeMS clearinghouse data and
draws the drive's trajectory through it: where the trajectory crosses a slow (red)
region, you were in mainline congestion the sensors also saw.

    python pems_corridor.py --freeway 405 --dir N \
        --pems ~/Downloads/d07_text_station_5min_2026_09_12.txt.gz \
        --meta ~/Downloads/d07_text_meta_2026_08_25.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from src import gpx_io
from src.pems import load_clearinghouse_5min, load_meta, build_corridor_field, drive_postmiles_on


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpx", default="data/raw/civic_sat.gpx")
    ap.add_argument("--pems", default="data/pems/d07_text_station_5min_2026_09_12.txt.gz")
    ap.add_argument("--meta", default="data/pems/d07_text_meta_2026_08_25.txt")
    ap.add_argument("--freeway", type=int, default=405)
    ap.add_argument("--dir", default="N")
    ap.add_argument("--start", default="2026-09-12 12:30")
    ap.add_argument("--end", default="2026-09-12 17:30")
    ap.add_argument("--outdir", default="figures_compare")
    args = ap.parse_args()

    speeds = load_clearinghouse_5min(args.pems)
    meta = load_meta(args.meta)
    field = build_corridor_field(speeds, meta, args.freeway, args.dir, args.start, args.end)
    trace = gpx_io.load_trace(args.gpx)
    traj = drive_postmiles_on(trace, meta, args.freeway, args.dir, max_dist_m=150)
    traj = traj[(traj["t_local"] >= pd.Timestamp(args.start)) & (traj["t_local"] <= pd.Timestamp(args.end))]

    fig, ax = plt.subplots(figsize=(12, 6.2))
    T = mdates.date2num(field.times.to_pydatetime())
    mesh = ax.pcolormesh(T, field.postmiles, np.ma.masked_invalid(field.speed.T),
                         cmap="RdYlGn", vmin=10, vmax=70, shading="auto")
    cb = fig.colorbar(mesh, ax=ax, pad=0.01); cb.set_label("PeMS mainline speed (mph)")

    # overlay the drive trajectory (postmile vs time)
    ax.plot(mdates.date2num(pd.to_datetime(traj["t_local"]).dt.to_pydatetime()),
            traj["abs_pm"], color="k", lw=2.4, label="Your drive")
    # mark where the drive itself was slow (<35 mph)
    slow = traj[traj["speed_mph"] < 35]
    ax.scatter(mdates.date2num(pd.to_datetime(slow["t_local"]).dt.to_pydatetime()),
               slow["abs_pm"], s=14, color="k", zorder=5, label="You < 35 mph")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_xlabel("time of day (Sat, Pacific)")
    ax.set_ylabel(f"absolute postmile on I-{args.freeway} {args.dir}")
    ax.set_ylim(traj["abs_pm"].min() - 2, traj["abs_pm"].max() + 2)
    ax.set_title(f"I-{args.freeway} {args.dir} speed field (PeMS) with your drive overlaid — "
                 f"Sat {field.times[0]:%b %d}")
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()

    outdir = Path(args.outdir); outdir.mkdir(exist_ok=True)
    out = outdir / f"pems_corridor_{args.freeway}{args.dir}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[field] {field.speed.shape} cells, {len(traj)} drive points on I-{args.freeway}{args.dir}")
    print(f"[done] {out}")


if __name__ == "__main__":
    main()

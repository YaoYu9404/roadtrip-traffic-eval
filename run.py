#!/usr/bin/env python3
"""End-to-end pipeline: GPX traces + PeMS -> figures + metrics.

Usage
-----
    python run.py                                  # synthetic demo data
    python run.py --gpx data/raw/civic.gpx data/raw/crosstrek.gpx
    python run.py --gpx data/raw/civic.gpx --pems data/pems/us101.csv

Outputs figures/*.png and figures/metrics.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src import gpx_io, synth, plots
from src.route import Route, add_route_distance
from src.traces import align_dual, agreement_stats
from src.events import add_accel_jerk, detect_hard_brakes, detect_congestion_onset, summarize_events
from src.pems import load_pems_5min, synthetic_speed_field
from src.metrics import travel_time_report

DEFAULTS = dict(
    hard_brake_decel_mps2=-3.0,
    congestion_slow_mph=30.0,
    congestion_free_mph=55.0,
    leg2_assumed_speed_mph=62.0,
)


def load_config(path="config.yaml") -> dict:
    cfg = dict(DEFAULTS)
    p = Path(path)
    if p.exists():
        try:
            import yaml  # optional
            user = yaml.safe_load(p.read_text()) or {}
            for k in DEFAULTS:
                for section in user.values() if isinstance(user, dict) else []:
                    if isinstance(section, dict) and k in section:
                        cfg[k] = section[k]
        except Exception as e:  # pragma: no cover - config is optional
            print(f"[config] using defaults ({e})")
    return cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpx", nargs="*", default=None, help="1 or 2 GPX files")
    ap.add_argument("--pems", default=None, help="PeMS 5-min CSV export")
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)

    # ---- 1. load traces (real if given, else synthetic) -------------------
    if args.gpx:
        gpx_paths = [Path(p) for p in args.gpx]
        print(f"[data] using real GPX: {[p.name for p in gpx_paths]}")
    else:
        p1, p2 = synth.generate_pair()
        gpx_paths = [p1, p2]
        print(f"[data] no --gpx given; generated synthetic traces in {p1.parent}")

    traces = [gpx_io.load_trace(p) for p in gpx_paths]
    for p, t in zip(gpx_paths, traces):
        print(f"[qc] {p.name}: {t.attrs['qc']}")

    primary = traces[0]
    route = Route.from_trace(primary)
    primary = add_route_distance(primary, route)
    print(f"[route] length {route.length_mi:.1f} mi from {len(route.lat)} vertices")

    # ---- 2. rare-event detection + rate estimation ------------------------
    primary = add_accel_jerk(primary)
    brakes = detect_hard_brakes(primary, cfg["hard_brake_decel_mps2"])
    congest = detect_congestion_onset(primary, cfg["congestion_slow_mph"], cfg["congestion_free_mph"])
    import pandas as pd
    events = pd.concat([brakes, congest], ignore_index=True) if len(brakes) or len(congest) else pd.DataFrame(columns=["type", "route_mi"])
    exposure_mi = float(primary["dist_mi"].iloc[-1])
    rate_summary = summarize_events(events, exposure_mi)
    print(f"[events] {len(brakes)} hard-brake, {len(congest)} congestion-onset over {exposure_mi:.1f} mi")

    # ---- 3. dual-sensor agreement (if two traces) -------------------------
    agree = {}
    if len(traces) == 2:
        aligned = align_dual(traces[0], traces[1], route)
        agree = agreement_stats(aligned)
        fig = plots.dual_agreement(aligned, agree)
        fig.savefig(outdir / "02_dual_agreement.png", dpi=150)
        print(f"[agreement] {agree}")

    # ---- 4. PeMS speed field + overlay ------------------------------------
    if args.pems:
        field = load_pems_5min(args.pems)
    else:
        field = synthetic_speed_field(route.length_mi)
        print("[pems] no --pems given; using synthetic corridor speed field")
    plots.speed_contour(field, primary).savefig(outdir / "01_speed_contour.png", dpi=150)

    # ---- 5. event map + travel-time evaluation ----------------------------
    plots.event_map(primary, events).savefig(outdir / "03_event_map.png", dpi=150)
    tt = travel_time_report(primary, cfg["leg2_assumed_speed_mph"])
    plots.travel_time(tt).savefig(outdir / "04_travel_time.png", dpi=150)
    print(f"[travel-time] {tt}")

    # ---- 6. write metrics.json --------------------------------------------
    metrics = {
        "route_length_mi": route.length_mi,
        "exposure_mi": exposure_mi,
        "n_hard_brake": int(len(brakes)),
        "n_congestion_onset": int(len(congest)),
        "rare_event_rates_per_100mi": rate_summary,
        "dual_sensor_agreement": agree,
        "travel_time": tt,
        "pems_source": field.source,
    }
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    print(f"[done] figures + metrics written to {outdir}/")


if __name__ == "__main__":
    main()

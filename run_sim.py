#!/usr/bin/env python3
"""Tier 2 entry point: SUMO corridor microsimulation + fidelity scoring.

    python run_sim.py                 # synthetic corridor field; SUMO if installed, else emulated
    python run_sim.py --pems data/pems/us101.csv

Writes:
    sim/out/*.xml            valid SUMO inputs (net, routes, detectors, cfg)
    figures/05_sim_fidelity.png
    figures/sim_metrics.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sim.corridor import Corridor
from sim.run import run_corridor, sumo_available
from sim.validate import validate, plot_fidelity
from src.pems import synthetic_speed_field, load_pems_5min


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pems", default=None, help="PeMS 5-min CSV; else synthetic field")
    ap.add_argument("--length-mi", type=float, default=10.0)
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()

    corr = Corridor(length_mi=args.length_mi)
    corr.detector_postmiles = [float(i) for i in range(1, int(args.length_mi))]

    field = load_pems_5min(args.pems) if args.pems else synthetic_speed_field(args.length_mi)
    print(f"[sim] SUMO available: {sumo_available()}")

    result = run_corridor(field, corr)
    print(f"[sim] engine = {result['engine']}; SUMO inputs written to sim/out/")

    report = validate(result, corr.length_mi)
    outdir = Path(args.outdir); outdir.mkdir(exist_ok=True)
    plot_fidelity(report).savefig(outdir / "05_sim_fidelity.png", dpi=150)

    summary = {k: v for k, v in report.items() if not k.startswith("_")}
    (outdir / "sim_metrics.json").write_text(json.dumps(summary, indent=2, default=float))
    print("[sim] summary:", json.dumps(summary, indent=2, default=float))
    if result["engine"] == "emulated":
        print("\n[note] SUMO not installed -> fidelity computed against an EMULATED run.")
        print("       Install SUMO to score a real simulation:")
        print("         brew tap dlr-ts/sumo && brew install sumo   # then re-run: python run_sim.py")


if __name__ == "__main__":
    main()

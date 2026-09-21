"""Write SUMO demand (.rou.xml) with mainline inflow calibrated from PeMS.

The upstream boundary inflow is set to the measured flow at the first detector
(so the simulation is *driven by real data*); on-ramp flows use a constant
default unless ramp counts are provided. Off-ramps are left as sinks.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def entry_flow_series(observed: pd.DataFrame, t0) -> list[tuple[int, int, float]]:
    """(begin_s, end_s, veh/h) intervals from the most-upstream detector."""
    up_pm = observed["postmile"].min()
    up = observed[observed["postmile"] == up_pm].sort_values("t")
    times = pd.to_datetime(up["t"]).to_numpy()
    secs = ((times - np.datetime64(t0)) / np.timedelta64(1, "s")).astype(float)
    flows = up["flow_vph"].to_numpy()
    out = []
    for i in range(len(secs)):
        end = secs[i + 1] if i + 1 < len(secs) else secs[i] + 300
        out.append((int(secs[i]), int(end), float(flows[i])))
    return out


def write_demand(paths: dict, entry: list[tuple[int, int, float]],
                 outdir: Path, ramp_vph: float = 600.0) -> Path:
    outdir = Path(outdir)
    mainline = paths["mainline_edges"]
    nodes = paths["node_postmiles"]
    through_edges = " ".join(mainline)

    lines = ['<routes>',
             '  <vType id="car" vClass="passenger" maxSpeed="40"/>',
             f'  <route id="through" edges="{through_edges}"/>']
    # on-ramp routes: ramp edge + downstream mainline edges
    onramps = [e for e in _iter_onramp_edges(paths)]
    for edge_id, start_idx in onramps:
        downstream = " ".join(mainline[start_idx:])
        lines.append(f'  <route id="rt_{edge_id}" edges="{edge_id} {downstream}"/>')

    # time-varying mainline inflow (calibrated from PeMS)
    for k, (b, e, vph) in enumerate(entry):
        lines.append(f'  <flow id="mainflow_{k}" type="car" route="through" '
                     f'begin="{b}" end="{e}" vehsPerHour="{vph:.0f}" '
                     f'departLane="best" departSpeed="max"/>')
    # constant on-ramp inflow across the whole horizon
    if entry:
        b0, e_last = entry[0][0], entry[-1][1]
        for edge_id, _ in onramps:
            lines.append(f'  <flow id="ramp_{edge_id}" type="car" route="rt_{edge_id}" '
                         f'begin="{b0}" end="{e_last}" vehsPerHour="{ramp_vph:.0f}" '
                         f'departLane="free" departSpeed="max"/>')
    lines.append('</routes>')

    p = outdir / "corridor.rou.xml"
    p.write_text("\n".join(lines))
    return p


def _iter_onramp_edges(paths: dict):
    """Yield (on_edge_id, mainline_start_index) for each on-ramp."""
    nodes = paths["node_postmiles"]
    # on-ramp edge ids look like "on_<pm>"; recover pm from the corridor via nodes
    # (network.py wrote on-ramps only where kind == 'on')
    for edge_id in paths.get("onramp_edges", []):
        pm = float(edge_id.split("_", 1)[1])
        start_idx = nodes.index(pm)
        yield edge_id, start_idx

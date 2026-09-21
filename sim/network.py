"""Write SUMO node/edge XML for the corridor and compile with netconvert.

The mainline is split into edges at every node postmile so that a detector or
ramp can attach at a clean edge boundary. Ramps connect to external
source/sink nodes offset from the mainline.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .corridor import Corridor, METERS_PER_MILE


def _mainline_edge_id(i: int) -> str:
    return f"mL_{i}"


def write_network_inputs(corr: Corridor, outdir: Path) -> dict:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    nodes = corr.node_postmiles()
    x = {pm: pm * METERS_PER_MILE for pm in nodes}

    # --- nodes ---
    nod = ['<nodes>']
    for pm in nodes:
        nod.append(f'  <node id="n_{pm:g}" x="{x[pm]:.1f}" y="0.0"/>')
    # ramp source/sink nodes, offset in y so geometry is valid
    for pm, kind in corr.ramps:
        y = -120.0 if kind == "on" else 120.0
        nod.append(f'  <node id="r_{kind}_{pm:g}" x="{x[pm]:.1f}" y="{y:.1f}"/>')
    nod.append('</nodes>')
    (outdir / "corridor.nod.xml").write_text("\n".join(nod))

    # --- edges ---
    edg = ['<edges>']
    for i in range(len(nodes) - 1):
        a, b = nodes[i], nodes[i + 1]
        edg.append(
            f'  <edge id="{_mainline_edge_id(i)}" from="n_{a:g}" to="n_{b:g}" '
            f'numLanes="{corr.n_lanes}" speed="{corr.speed_limit_mps:.2f}"/>'
        )
    for pm, kind in corr.ramps:
        if kind == "on":
            edg.append(f'  <edge id="on_{pm:g}" from="r_on_{pm:g}" to="n_{pm:g}" '
                       f'numLanes="1" speed="{0.7 * corr.speed_limit_mps:.2f}"/>')
        else:
            edg.append(f'  <edge id="off_{pm:g}" from="n_{pm:g}" to="r_off_{pm:g}" '
                       f'numLanes="1" speed="{0.7 * corr.speed_limit_mps:.2f}"/>')
    edg.append('</edges>')
    (outdir / "corridor.edg.xml").write_text("\n".join(edg))

    return {
        "nodes": outdir / "corridor.nod.xml",
        "edges": outdir / "corridor.edg.xml",
        "net": outdir / "corridor.net.xml",
        "mainline_edges": [_mainline_edge_id(i) for i in range(len(nodes) - 1)],
        "node_postmiles": nodes,
        "onramp_edges": [f"on_{pm:g}" for pm, kind in corr.ramps if kind == "on"],
        "offramp_edges": [f"off_{pm:g}" for pm, kind in corr.ramps if kind == "off"],
    }


def compile_net(paths: dict) -> bool:
    """Run netconvert if available. Returns True if net.xml was produced."""
    if not shutil.which("netconvert"):
        return False
    cmd = [
        "netconvert",
        "--node-files", str(paths["nodes"]),
        "--edge-files", str(paths["edges"]),
        "-o", str(paths["net"]),
        "--no-turnarounds", "true",
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return paths["net"].exists()

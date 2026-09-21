"""Write E1 induction-loop detectors (one per corridor detector postmile).

Each loop sits on lane 0 of the mainline edge that starts at the detector node,
aggregating every 300 s to mirror PeMS Station-5-Minute data.
"""
from __future__ import annotations

from pathlib import Path

from .corridor import Corridor


def write_detectors(corr: Corridor, paths: dict, outdir: Path, freq_s: int = 300) -> Path:
    outdir = Path(outdir)
    nodes = paths["node_postmiles"]
    lines = ['<additional>']
    for pm in corr.detector_postmiles:
        idx = nodes.index(pm)                      # edge mL_idx starts at this node
        edge = paths["mainline_edges"][idx]
        # one loop per lane; flows are summed and speeds flow-weighted at parse time
        for lane in range(corr.n_lanes):
            lines.append(
                f'  <inductionLoop id="det_{pm:g}_{lane}" lane="{edge}_{lane}" pos="5.0" '
                f'freq="{freq_s}" file="det_{pm:g}_{lane}.out.xml"/>'
            )
    lines.append('</additional>')
    p = outdir / "corridor.det.add.xml"
    p.write_text("\n".join(lines))
    return p

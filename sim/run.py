"""Build inputs, run SUMO if installed (else emulate), and return observed vs simulated."""
from __future__ import annotations

import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

from .corridor import Corridor
from . import network, demand, detectors, synth_sim

MPS_TO_MPH = 2.2369362920544


def sumo_available() -> bool:
    return bool(shutil.which("sumo") and shutil.which("netconvert"))


def _write_cfg(paths, outdir: Path, begin: int, end: int) -> Path:
    cfg = f"""<configuration>
  <input>
    <net-file value="{paths['net'].name}"/>
    <route-files value="corridor.rou.xml"/>
    <additional-files value="corridor.det.add.xml"/>
  </input>
  <time><begin>{begin}</begin><end>{end}</end><step-length>1.0</step-length></time>
  <report><no-step-log value="true"/><no-warnings value="true"/></report>
</configuration>"""
    p = outdir / "corridor.sumocfg"
    p.write_text(cfg)
    return p


def _parse_detectors(corr: Corridor, outdir: Path, t0) -> pd.DataFrame:
    """Aggregate per-lane E1 loops to a corridor cross-section per (postmile, interval).

    Flow is summed across lanes; speed is flow-weighted (lanes carrying no vehicle
    report speed = -1 and are ignored, defaulting to the free-flow limit).
    """
    rows = []
    for pm in corr.detector_postmiles:
        agg: dict[int, dict] = {}  # begin_s -> {flow, wspeed, wsum}
        for lane in range(corr.n_lanes):
            f = outdir / f"det_{pm:g}_{lane}.out.xml"
            if not f.exists():
                continue
            for iv in ET.parse(f).getroot().findall("interval"):
                b = int(float(iv.get("begin")))
                flow = float(iv.get("flow"))        # veh/h on this lane
                speed = float(iv.get("speed"))      # m/s, -1 when no vehicle
                a = agg.setdefault(b, {"flow": 0.0, "wspeed": 0.0, "wsum": 0.0})
                a["flow"] += flow
                if speed >= 0 and flow > 0:
                    a["wspeed"] += speed * flow
                    a["wsum"] += flow
        for b, a in agg.items():
            speed_mph = (a["wspeed"] / a["wsum"] * MPS_TO_MPH) if a["wsum"] > 0 \
                else corr.speed_limit_mph
            t = pd.Timestamp(t0) + pd.to_timedelta(b, "s")
            rows.append({"postmile": pm, "t": t.round("5min"),
                         "speed_mph": speed_mph, "flow_vph": a["flow"]})
    return pd.DataFrame(rows)


_STABILITY_ARGS = ["--time-to-teleport", "60", "--max-depart-delay", "120",
                   "--ignore-route-errors", "true", "--no-warnings", "true"]


def _run_sumo(corr: Corridor, outdir: Path, t0, extra: list[str]) -> pd.DataFrame:
    """Run SUMO with the given extra args and parse the detector cross-sections."""
    subprocess.run(["sumo", "-c", "corridor.sumocfg", *_STABILITY_ARGS, *extra],
                   cwd=outdir, check=True, capture_output=True)
    return _parse_detectors(corr, outdir, t0)


def run_corridor(field, corr: Corridor | None = None, outdir="sim/out") -> dict:
    """Returns {observed, simulated, engine, paths}. Writes SUMO inputs either way.

    * real PeMS field  -> observed = PeMS, simulated = one SUMO run
    * synthetic + SUMO -> twin validation: observed = SUMO baseline (ground truth),
      simulated = SUMO with a perturbed model (a small demand perturbation)
    * no SUMO          -> observed = field, simulated = emulated run
    """
    corr = corr or Corridor()
    outdir = Path(outdir)
    field_obs = synth_sim.sample_observed(field, corr)
    t0 = pd.to_datetime(field_obs["t"]).min()
    is_real = getattr(field, "source", "synthetic") != "synthetic"

    # Always generate the (real) SUMO input files, demand calibrated from the field.
    paths = network.write_network_inputs(corr, outdir)
    entry = demand.entry_flow_series(field_obs, t0)
    demand.write_demand(paths, entry, outdir, ramp_vph=400.0)
    detectors.write_detectors(corr, paths, outdir)
    _write_cfg(paths, outdir, entry[0][0], entry[-1][1])

    if sumo_available() and network.compile_net(paths):
        baseline = _run_sumo(corr, outdir, t0, [])
        if is_real:
            observed, simulated, engine = field_obs, baseline, "sumo"
        else:
            # twin: baseline = ground truth; perturbed model = a 5% demand change
            perturbed = _run_sumo(corr, outdir, t0, ["--scale", "0.95"])
            observed, simulated, engine = baseline, perturbed, "sumo-twin"
    else:
        observed = field_obs
        simulated = synth_sim.emulate_simulation(field_obs, corr)
        engine = "emulated"

    return {"observed": observed, "simulated": simulated, "engine": engine, "paths": paths}

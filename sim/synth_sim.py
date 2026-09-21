"""Observed sampling + an emulated-simulation fallback (used when SUMO is absent).

`sample_observed` reads the corridor's detector locations off a PeMS speed field
and derives flow via a Greenshields fundamental diagram. `emulate_simulation`
produces a plausible *imperfect* simulated run (systematic congestion-speed bias
plus noise) so the GEH/RMSE validation is demonstrable before SUMO is installed.
Everything it returns is explicitly labeled emulated.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .corridor import Corridor

KJ_PER_LANE = 123.0  # jam density (veh/mi/lane) -> Greenshields capacity ~2000 veh/h/lane


def greenshields_flow_vph(speed_mph, n_lanes: int, vf_mph: float) -> np.ndarray:
    v = np.clip(np.asarray(speed_mph, float), 0, vf_mph)
    k_per_lane = KJ_PER_LANE * (1 - v / vf_mph)   # density veh/mi/lane
    return n_lanes * k_per_lane * v               # q = k * v, summed over lanes


def sample_observed(field, corr: Corridor) -> pd.DataFrame:
    """Interpolate the PeMS speed field at each detector postmile & interval.

    Returns long-form [postmile, t, speed_mph, flow_vph]. If the field carries a
    real flow array it is used; otherwise flow is derived via Greenshields.
    """
    rows = []
    has_flow = getattr(field, "flow", None) is not None
    for pm in corr.detector_postmiles:
        speeds = np.array([np.interp(pm, field.postmiles, field.speed[ti])
                           for ti in range(len(field.times))])
        if has_flow:
            flows = np.array([np.interp(pm, field.postmiles, field.flow[ti])
                              for ti in range(len(field.times))])
        else:  # speed-only field -> impute via Greenshields
            flows = greenshields_flow_vph(speeds, corr.n_lanes, corr.speed_limit_mph)
        for t, s, q in zip(field.times, speeds, flows):
            rows.append({"postmile": pm, "t": t, "speed_mph": float(s), "flow_vph": float(q)})
    return pd.DataFrame(rows)


def emulate_simulation(observed: pd.DataFrame, corr: Corridor, seed: int = 7) -> pd.DataFrame:
    """Emulate an imperfect calibrated SUMO run from the observed field.

    NOT a real simulation - a stand-in until SUMO is installed. Adds a congestion
    speed bias (sims tend to over-queue) and realistic noise, then recomputes flow.
    """
    rng = np.random.default_rng(seed)
    v_obs = observed["speed_mph"].to_numpy()
    q_obs = observed["flow_vph"].to_numpy()
    congested = np.clip((45.0 - v_obs) / 45.0, 0, 1)          # 0 free-flow -> 1 jammed
    # a decent-but-imperfect calibration: mild congestion bias + measurement-scale noise
    v_sim = np.clip(v_obs - 2.0 * congested + rng.normal(0, 1.5, v_obs.shape),
                    3, corr.speed_limit_mph)
    q_sim = np.clip(q_obs - 80.0 * congested + rng.normal(0, 120, q_obs.shape), 0, None)
    out = observed.copy()
    out["speed_mph"] = v_sim
    out["flow_vph"] = q_sim
    out.attrs["emulated"] = True
    return out

"""Score simulated vs. observed: GEH on flows, RMSE on speeds, travel-time error."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.metrics import geh, geh_report  # noqa: E402

_OK = "#1f8a4c"
_BAD = "#c0492f"


def merge_obs_sim(observed: pd.DataFrame, simulated: pd.DataFrame) -> pd.DataFrame:
    o = observed.rename(columns={"speed_mph": "obs_speed", "flow_vph": "obs_flow"})
    s = simulated.rename(columns={"speed_mph": "sim_speed", "flow_vph": "sim_flow"})
    m = o.merge(s, on=["postmile", "t"], how="inner")
    return m.dropna(subset=["obs_flow", "sim_flow", "obs_speed", "sim_speed"])


def validate(result: dict, corridor_len_mi: float) -> dict:
    m = merge_obs_sim(result["observed"], result["simulated"])
    g = geh(m["obs_flow"], m["sim_flow"])
    rep = geh_report(m["obs_flow"], m["sim_flow"])
    speed_rmse = float(np.sqrt(np.mean((m["sim_speed"] - m["obs_speed"]) ** 2)))
    obs_tt = _corridor_tt(m, "obs_speed", corridor_len_mi)
    sim_tt = _corridor_tt(m, "sim_speed", corridor_len_mi)
    return {
        "engine": result["engine"],
        "n_matched": int(len(m)),
        "median_geh": rep["median_geh"],
        "pct_geh_below_5": rep["pct_below_5"],
        "speed_rmse_mph": speed_rmse,
        "obs_mean_travel_min": obs_tt,
        "sim_mean_travel_min": sim_tt,
        "travel_time_error_min": sim_tt - obs_tt,
        "_merged": m, "_geh": g,
    }


def _corridor_tt(m: pd.DataFrame, speed_col: str, corridor_len_mi: float) -> float:
    """Mean corridor travel time (min): average per interval over detector segments.

    Vectorized (no groupby.apply) so it works across pandas versions.
    """
    n_det = m["postmile"].nunique()
    seg = corridor_len_mi / n_det
    contrib = seg * 60.0 / np.clip(m[speed_col].to_numpy(), 3, None)  # min per segment
    per_interval = pd.Series(contrib, index=m["t"].to_numpy()).groupby(level=0).sum()
    return float(per_interval.mean())


def plot_fidelity(report: dict):
    m, g = report["_merged"], report["_geh"]
    ok = g < 5
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.scatter(m["obs_flow"][ok], m["sim_flow"][ok], s=12, c=_OK, alpha=0.6, label="GEH < 5")
    ax1.scatter(m["obs_flow"][~ok], m["sim_flow"][~ok], s=14, c=_BAD, alpha=0.7, label="GEH ≥ 5")
    lim = [0, max(m["obs_flow"].max(), m["sim_flow"].max()) * 1.05]
    ax1.plot(lim, lim, "k--", lw=1)
    ax1.set_xlim(lim); ax1.set_ylim(lim)
    ax1.set_xlabel("observed flow (veh/h)"); ax1.set_ylabel("simulated flow (veh/h)")
    ax1.set_title(f"Flow: median GEH {report['median_geh']:.2f}, "
                  f"{report['pct_geh_below_5']:.0f}% below 5")
    ax1.legend(loc="upper left")

    ax2.scatter(m["obs_speed"], m["sim_speed"], s=12, c="#1f6f8b", alpha=0.6)
    lim2 = [0, max(m["obs_speed"].max(), m["sim_speed"].max()) * 1.05]
    ax2.plot(lim2, lim2, "k--", lw=1)
    ax2.set_xlim(lim2); ax2.set_ylim(lim2)
    ax2.set_xlabel("observed speed (mph)"); ax2.set_ylabel("simulated speed (mph)")
    ax2.set_title(f"Speed: RMSE {report['speed_rmse_mph']:.1f} mph")

    tag = {
        "sumo": "SUMO vs PeMS",
        "sumo-twin": "SUMO twin: baseline vs perturbed model",
        "emulated": "EMULATED (SUMO not installed)",
    }.get(report["engine"], report["engine"])
    fig.suptitle(f"Simulation fidelity  —  {tag}", fontsize=13)
    fig.tight_layout()
    return fig

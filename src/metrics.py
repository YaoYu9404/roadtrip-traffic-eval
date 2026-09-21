"""Evaluation metrics: travel-time prediction and simulation fidelity (GEH).

These make the project an *evaluation framework*, not just an analysis:
  * travel_time_prediction: predict Leg-2 duration, then validate vs. the real drive.
  * geh: the standard traffic-engineering statistic for observed-vs-modeled flow,
    used to score a SUMO (or any) simulation against real PeMS counts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def predict_travel_time_min(length_mi: float, assumed_speed_mph: float,
                            fixed_delay_min: float = 0.0) -> float:
    """Naive free-flow travel-time prediction (minutes)."""
    return 60.0 * length_mi / assumed_speed_mph + fixed_delay_min


def observed_travel_time_min(trace: pd.DataFrame) -> float:
    return float((trace["time"].iloc[-1] - trace["time"].iloc[0]).total_seconds() / 60)


def stopped_time_min(trace: pd.DataFrame) -> float:
    """Minutes spent in flagged long gaps (parking/rest stops), not driving."""
    if "long_gap" not in trace or "dt_s" not in trace:
        return 0.0
    gap_s = trace.loc[trace["long_gap"].fillna(False), "dt_s"].sum()
    return float(gap_s / 60.0)


def travel_time_report(trace: pd.DataFrame, assumed_speed_mph: float,
                       fixed_delay_min: float = 0.0) -> dict:
    """Predict-then-validate: compare a free-flow prediction to the real drive.

    Reports both the full door-to-door time and the moving time (stops removed),
    since the free-flow prediction is about *driving*, not lunch breaks.
    """
    length_mi = float(trace["dist_mi"].iloc[-1])
    pred = predict_travel_time_min(length_mi, assumed_speed_mph, fixed_delay_min)
    obs = observed_travel_time_min(trace)
    stopped = stopped_time_min(trace)
    moving = obs - stopped
    return {
        "length_mi": length_mi,
        "assumed_speed_mph": assumed_speed_mph,
        "predicted_min": pred,
        "observed_min": obs,
        "stopped_min": stopped,
        "observed_moving_min": moving,
        "error_min": pred - obs,
        "error_vs_moving_min": pred - moving,
        "abs_pct_error": 100 * abs(pred - obs) / obs if obs else np.nan,
        "abs_pct_error_moving": 100 * abs(pred - moving) / moving if moving else np.nan,
        "observed_avg_mph": 60 * length_mi / obs if obs else np.nan,
        "moving_avg_mph": 60 * length_mi / moving if moving else np.nan,
    }


def geh(observed, modeled):
    """GEH statistic (per matched flow pair). GEH < 5 is a good match by convention."""
    observed, modeled = np.asarray(observed, float), np.asarray(modeled, float)
    denom = (observed + modeled)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = np.sqrt(2 * (modeled - observed) ** 2 / denom)
    return np.where(denom == 0, 0.0, g)


def geh_report(observed, modeled) -> dict:
    g = geh(observed, modeled)
    g = g[np.isfinite(g)]
    return {
        "n": int(g.size),
        "median_geh": float(np.median(g)) if g.size else np.nan,
        "pct_below_5": float(100 * np.mean(g < 5)) if g.size else np.nan,
    }

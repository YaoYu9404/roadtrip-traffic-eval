"""Figure builders. Each returns a matplotlib Figure and is theme-agnostic.

The four core figures:
  1. speed_contour  - PeMS space-time speed field with the drive overlaid (hero)
  2. dual_agreement - Civic vs Crosstrek speed profiles + Bland-Altman
  3. event_map      - speed vs distance with rare events marked
  4. travel_time    - predicted vs observed travel time
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

_ACCENT = "#c0492f"
_A = "#1f6f8b"
_B = "#e08a1e"


def _naive(series: pd.Series) -> pd.Series:
    """Return a timezone-naive datetime Series (matplotlib date axes want naive)."""
    s = pd.to_datetime(series)
    if getattr(s.dt, "tz", None) is not None:
        s = s.dt.tz_convert("UTC").dt.tz_localize(None)
    return s


def speed_contour(field, trace: pd.DataFrame | None = None, title="Corridor speed field"):
    fig, ax = plt.subplots(figsize=(11, 5.2))
    T = mdates.date2num(field.times.to_pydatetime())
    mesh = ax.pcolormesh(T, field.postmiles, field.speed.T, cmap="RdYlGn",
                         vmin=10, vmax=70, shading="auto")
    cb = fig.colorbar(mesh, ax=ax, pad=0.01)
    cb.set_label("speed (mph)")
    if trace is not None and "route_mi" in trace:
        ax.plot(mdates.date2num(_naive(trace["time"]).dt.to_pydatetime()),
                trace["route_mi"], color="k", lw=2.2, label="your drive")
        ax.legend(loc="upper right", framealpha=0.9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_xlabel("time of day")
    ax.set_ylabel("distance along corridor (mi)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def dual_agreement(aligned: pd.DataFrame, stats: dict, title="Dual-sensor agreement"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    ax1.plot(aligned["route_mi"], aligned["speed_a"], color=_A, lw=1.3, label="Civic")
    ax1.plot(aligned["route_mi"], aligned["speed_b"], color=_B, lw=1.3, label="Crosstrek", alpha=0.85)
    ax1.set_xlabel("distance along route (mi)")
    ax1.set_ylabel("speed (mph)")
    ax1.set_title("Speed profiles")
    ax1.legend()

    m = (aligned["speed_a"] + aligned["speed_b"]) / 2
    d = aligned["diff"]
    ax2.scatter(m, d, s=6, alpha=0.4, color=_A)
    for y, ls in [(stats.get("mean_bias_mph", 0), "-"),
                  (stats.get("loa_low_mph", 0), "--"),
                  (stats.get("loa_high_mph", 0), "--")]:
        ax2.axhline(y, color=_ACCENT, ls=ls, lw=1.2)
    ax2.set_xlabel("mean of two speeds (mph)")
    ax2.set_ylabel("Civic - Crosstrek (mph)")
    ax2.set_title(f"Bland-Altman  (bias {stats.get('mean_bias_mph', float('nan')):.2f}, "
                  f"RMSE {stats.get('rmse_mph', float('nan')):.2f} mph)")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def event_map(trace: pd.DataFrame, events: pd.DataFrame, title="On-road events"):
    fig, ax = plt.subplots(figsize=(11, 4.4))
    x = trace["route_mi"] if "route_mi" in trace else trace["dist_mi"]
    scol = "speed_mph_s" if "speed_mph_s" in trace else "speed_mph"
    ax.plot(x, trace[scol], color="#3a3a3a", lw=1.0)
    colors = {"hard_brake": _ACCENT, "congestion_onset": _A}
    if len(events):
        for etype, grp in events.groupby("type"):
            ax.scatter(grp["route_mi"], np.interp(grp["route_mi"], x, trace[scol]),
                       s=55, color=colors.get(etype, "k"), label=etype.replace("_", " "),
                       zorder=5, edgecolor="white")
        ax.legend()
    ax.set_xlabel("distance along route (mi)")
    ax.set_ylabel("speed (mph)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def travel_time(report: dict, title="Travel-time prediction vs. observed"):
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    vals = [report["predicted_min"], report["observed_min"]]
    bars = ax.bar(["predicted", "observed"], vals, color=[_B, _A], width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f} min", ha="center")
    ax.set_ylabel("travel time (min)")
    ax.set_title(f"{title}\nabs error {report['abs_pct_error']:.1f}%")
    fig.tight_layout()
    return fig

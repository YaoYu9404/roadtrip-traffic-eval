"""Smoke tests: the pipeline runs on synthetic data and produces sane numbers."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src import synth, gpx_io
from src.route import Route, add_route_distance
from src.traces import align_dual, agreement_stats
from src.events import add_accel_jerk, detect_hard_brakes, detect_congestion_onset, poisson_rate_ci


def _traces(tmp_path):
    synth.generate_pair(out_dir=tmp_path)
    a = gpx_io.load_trace(Path(tmp_path) / "civic.gpx")
    b = gpx_io.load_trace(Path(tmp_path) / "crosstrek.gpx")
    return a, b


def test_parse_and_kinematics(tmp_path):
    a, _ = _traces(tmp_path)
    assert len(a) > 100
    assert a["dist_mi"].iloc[-1] > 50            # corridor is long
    assert 0 < np.nanmedian(a["speed_mph"]) < 90


def test_events_detected(tmp_path):
    a, _ = _traces(tmp_path)
    route = Route.from_trace(a)
    a = add_route_distance(add_accel_jerk(a), route)
    brakes = detect_hard_brakes(a, -3.0)
    congest = detect_congestion_onset(a)
    assert len(brakes) >= 1                        # synth injects hard brakes
    assert len(congest) >= 1                        # synth injects a congestion zone


def test_poisson_ci_monotonic():
    ci = poisson_rate_ci(3, 100.0)
    assert ci["low"] < ci["rate"] < ci["high"]


def test_dual_agreement(tmp_path):
    a, b = _traces(tmp_path)
    route = Route.from_trace(a)
    aligned = align_dual(a, b, route)
    stats = agreement_stats(aligned)
    assert stats["n"] > 100
    assert abs(stats["mean_bias_mph"]) < 15        # two cars, same road -> close
    assert stats["corr"] > 0.8

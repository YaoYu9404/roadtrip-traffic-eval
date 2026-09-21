"""Smoke tests for the Tier-2 SUMO corridor pipeline (runs in emulated mode)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pems import synthetic_speed_field
from sim.corridor import Corridor
from sim.run import run_corridor
from sim.validate import validate


def _run(tmp_path):
    corr = Corridor(length_mi=10.0)
    field = synthetic_speed_field(corr.length_mi)
    return corr, run_corridor(field, corr, outdir=tmp_path)


def test_sumo_inputs_written(tmp_path):
    _, res = _run(tmp_path)
    for name in ["corridor.nod.xml", "corridor.edg.xml", "corridor.rou.xml",
                 "corridor.det.add.xml", "corridor.sumocfg"]:
        assert (Path(tmp_path) / name).exists(), name


def test_validation_reasonable(tmp_path):
    corr, res = _run(tmp_path)
    rep = validate(res, corr.length_mi)
    assert rep["n_matched"] > 100
    assert rep["speed_rmse_mph"] < 6.0
    if res["engine"] == "emulated":
        # emulated calibration is deliberately close -> most points GEH < 5
        assert rep["pct_geh_below_5"] >= 80.0
    else:
        # real SUMO twin: a small perturbation should stay well-matched overall
        assert rep["median_geh"] < 8.0


def test_flow_field_present(tmp_path):
    field = synthetic_speed_field(10.0)
    assert field.flow is not None
    assert field.flow.shape == field.speed.shape

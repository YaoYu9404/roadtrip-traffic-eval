# roadtrip-traffic-eval

**Evaluating driving quality, rare events, and traffic-simulation fidelity from a 500-mile road trip (San Diego → Cupertino).**

Two cars (a Honda Civic and a Subaru Crosstrek) drive the same route at the same
time, logging GPS. This repo turns those traces — plus public Caltrans PeMS
freeway data — into an **evaluation framework**: it measures how the drive
compares to the live traffic field, cross-validates the two vehicles as
independent sensors, estimates the rate of rare on-road events with confidence
intervals, and (optionally) scores a traffic *simulation* against reality.

> Built as a portfolio project mirroring the kind of work a data-science team does
> when it "measures and improves the quality of the software that drives the car":
> evaluation frameworks, metrics for complex systems, rare-event rate estimation,
> and combining real with synthetic data.

<p align="center"><img src="figures/01_speed_contour.png" width="760"></p>

## What it produces

| Figure / output | What it shows |
|---|---|
| `figures/01_speed_contour.png` | PeMS space-time speed field with **your drive overlaid** — where you hit (or missed) congestion |
| `figures/02_dual_agreement.png` | **Dual-sensor agreement**: Civic vs. Crosstrek speed profiles + Bland-Altman limits of agreement |
| `figures/03_event_map.png` | Speed vs. distance with **hard-brake** and **congestion-onset** events marked |
| `figures/04_travel_time.png` | **Predicted vs. observed** travel time (a predict-then-validate check) |
| `figures/metrics.json` | All numbers: rare-event rates per 100 mi (Poisson + bootstrap CIs), agreement stats, travel-time error |

## Quickstart

```bash
python -m pip install -r requirements.txt

# 1) Runs immediately on synthetic demo data (no real data needed):
python run.py

# 2) With your real GPX traces:
python run.py --gpx data/raw/civic.gpx data/raw/crosstrek.gpx

# 3) Add the real corridor traffic field:
python run.py --gpx data/raw/civic.gpx data/raw/crosstrek.gpx --pems data/pems/us101.csv
```

Everything downstream regenerates from whatever data you pass. With no arguments,
`run.py` writes two synthetic GPX files and uses a synthetic PeMS field so the
whole pipeline is reproducible before the trip.

## Getting the real data

- **GPS traces** — log both phones during the drive (≈1 Hz, high accuracy, keep
  the phone charging). Export **GPX** and drop the files in `data/raw/`.
- **Caltrans PeMS** — free account at [pems.dot.ca.gov](https://pems.dot.ca.gov).
  Export **Station 5-Minute** data for the US-101 corridor stations over your
  travel window, save the CSV to `data/pems/`, and match the column names in
  `src/pems.py::load_pems_5min` (timestamp / postmile / speed).

> **Privacy:** raw traces are `.gitignore`d. Trim the endpoints before publishing
> so you don't post your exact home/hotel coordinates — start the public trace at
> a freeway on-ramp.

## Method notes

- **Speed** is derived from consecutive GPS positions (works even if the logger
  stores no speed tag). A median-then-mean smoothed channel is used for agreement
  and congestion detection; acceleration uses a lighter smoothing so real braking
  peaks survive.
- **Linear referencing** (`src/route.py`) projects every point onto a route
  polyline, giving a "distance along route" axis so the two cars — and the PeMS
  field — share one coordinate.
- **Rare-event rates** are reported per 100 miles with an **exact Poisson CI** and
  a **block-bootstrap CI** (robust to spatial clustering of events).
- **Simulation fidelity** (Tier 2) uses the **GEH statistic** (`src/metrics.py`),
  the standard observed-vs-modeled flow metric, plus speed RMSE.

## Layout

```
run.py               end-to-end pipeline (synthetic or real)
config.yaml          thresholds & priors
src/
  geo.py             distances + local projection
  gpx_io.py          GPX parsing, kinematics, QC
  route.py           linear referencing onto a route polyline
  traces.py          dual-trace alignment + agreement stats
  events.py          hard-brake / congestion-onset detection + rate CIs
  pems.py            Caltrans PeMS loader + synthetic speed field
  metrics.py         travel-time prediction, GEH
  plots.py           the four figures
  synth.py           synthetic GPX + PeMS generators (dev/demo)
run_sim.py           Tier-2 entry point (SUMO corridor sim + fidelity)
sim/
  corridor.py        corridor geometry (lanes, ramps, detectors)
  network.py         SUMO node/edge XML + netconvert
  demand.py          PeMS-calibrated demand (.rou.xml)
  detectors.py       E1 induction-loop detectors (.add.xml)
  run.py             build + run SUMO (or emulate) -> observed vs simulated
  validate.py        GEH / speed RMSE / travel-time + fidelity figure
  synth_sim.py       observed sampling + emulated-sim fallback
notebooks/01_analysis.ipynb   narrative walkthrough
tests/test_smoke.py           Tier-1 smoke tests
tests/test_sim.py             Tier-2 smoke tests
```

## Tier 2 — SUMO corridor simulation + fidelity scoring

Builds a SUMO microsimulation of a congested corridor segment, drives it with an
inflow **calibrated from PeMS**, and scores simulated vs. observed traffic with
the **GEH statistic** (flows), **speed RMSE**, and **travel-time error** — the
"evaluate the quality of simulation" and "combine real and synthetic data"
pieces of the job.

```bash
python run_sim.py                       # synthetic corridor; SUMO if installed, else emulated
python run_sim.py --pems data/pems/us101.csv --length-mi 10
```

Outputs `figures/05_sim_fidelity.png` (observed-vs-simulated flow + speed with GEH)
and `figures/sim_metrics.json`. Valid SUMO inputs are always written to `sim/out/`
(`corridor.net.xml` via netconvert, `corridor.rou.xml`, detectors, `corridor.sumocfg`).

**Installing SUMO** — the simplest, most portable route is the PyPI wheel
(the Homebrew `dlr-ts/sumo` tap is broken on recent Homebrew):

```bash
python3.12 -m venv .venv
./.venv/bin/pip install -r requirements.txt eclipse-sumo sumolib traci
# run with the venv on PATH so run_sim.py finds the sumo/netconvert binaries:
PATH="$PWD/.venv/bin:$PATH" SUMO_HOME="$PWD/.venv/lib/python3.12/site-packages/sumo" \
  ./.venv/bin/python run_sim.py
```

Behaviour by mode:
- **real PeMS field** → observed = PeMS, simulated = one calibrated SUMO run.
- **synthetic + SUMO** → *twin validation*: observed = SUMO baseline (ground
  truth), simulated = SUMO with a perturbed model. Coherent and honest — you're
  scoring a simulation against a simulation, not against a fabricated field.
- **no SUMO** → scores against a clearly-labeled emulated run so the metric
  machinery stays reproducible. Once SUMO is installed it runs automatically, no
  code change.

### Roadmap
- [ ] Import the real corridor geometry from OpenStreetMap instead of the straight-line stub.
- [ ] Calibrate on-ramp flows from PeMS ramp detectors (currently a constant default).
- [ ] Compare the two legs (Sat through LA vs. Sun dawn on US-101) as two traffic
      regimes, and quantify what leaving early bought in avoided delay.

## License

MIT.

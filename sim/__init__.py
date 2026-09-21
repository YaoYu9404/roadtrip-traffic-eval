"""Tier 2: SUMO microsimulation of a US-101 corridor segment + fidelity scoring.

Generates valid SUMO input (network, demand, detectors, config), runs SUMO when
it is installed, and otherwise produces a clearly-labeled *emulated* run so the
observed-vs-simulated validation (GEH statistic, speed RMSE) is reproducible
before SUMO is installed.
"""

"""Corridor geometry: a straight multi-lane freeway segment with ramps + detectors.

A deliberately simple, reproducible stand-in for a real congested US-101 stretch.
Postmiles are in miles from the segment start; everything else is derived.
"""
from __future__ import annotations

from dataclasses import dataclass, field

METERS_PER_MILE = 1609.344


@dataclass
class Corridor:
    length_mi: float = 10.0
    n_lanes: int = 4
    speed_limit_mph: float = 65.0
    # ramps as (postmile, kind) with kind in {"on", "off"}
    ramps: list = field(default_factory=lambda: [(2.0, "on"), (4.0, "off"),
                                                 (6.0, "on"), (8.0, "off")])
    # induction-loop detector locations (miles). Default: every mile.
    detector_postmiles: list = field(default_factory=lambda: [float(i) for i in range(1, 10)])

    @property
    def speed_limit_mps(self) -> float:
        return self.speed_limit_mph / 2.2369362920544

    def node_postmiles(self) -> list[float]:
        """Mainline node positions: endpoints + every ramp + every detector."""
        pts = {0.0, self.length_mi}
        pts |= {p for p, _ in self.ramps}
        pts |= set(self.detector_postmiles)
        return sorted(pts)


DEFAULT = Corridor()

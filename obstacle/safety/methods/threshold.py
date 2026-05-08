"""Method B: distance threshold + emergency stop.

Naive baseline. If any obstacle is within `r_safe` of the end-effector, command
zero joint velocity. Otherwise pass the nominal command through unchanged.

Used as the floor in the benchmark — every other method must beat this on the
"mission disruption" axis while matching it on the "min separation" axis.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ..types import ControlOutput, Obstacle, RobotState
from .base import SafetyMethod


class DistanceThreshold(SafetyMethod):
    name = "threshold"

    def __init__(self, r_safe: float = 0.4) -> None:
        self.r_safe = r_safe

    def step(
        self,
        state: RobotState,
        obstacles: Sequence[Obstacle],
        qdot_nominal: np.ndarray,
    ) -> ControlOutput:
        min_d = float("inf")
        for o in obstacles:
            d = float(np.linalg.norm(state.ee_pos - o.p)) - o.radius
            min_d = min(min_d, d)

        if min_d < self.r_safe:
            qdot = np.zeros_like(qdot_nominal)
            return ControlOutput(qdot, safe=False, info={"min_d": min_d})

        return ControlOutput(qdot_nominal, safe=True, info={"min_d": min_d})

"""Abstract base for safety methods.

Every method (threshold, APF, NEO, HOCBF) implements the same interface so the
benchmark harness can swap them without code changes.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from ..types import ControlOutput, Obstacle, RobotState


class SafetyMethod(ABC):
    """One-tick reactive safety policy."""

    name: str = "abstract"

    @abstractmethod
    def step(
        self,
        state: RobotState,
        obstacles: Sequence[Obstacle],
        qdot_nominal,
    ) -> ControlOutput:
        """Return the safety-modified joint velocity for this tick.

        Parameters
        ----------
        state : RobotState
            Current arm snapshot.
        obstacles : sequence of Obstacle
            Detected dynamic obstacles (from camera/ground-truth).
        qdot_nominal : (6,) ndarray
            What the refueling pipeline wanted to command in the absence of
            obstacles. Methods minimise deviation from this.
        """

    def reset(self) -> None:
        """Clear any internal state between scenarios. Override if needed."""

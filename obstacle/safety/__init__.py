"""Reactive obstacle-avoidance safety system for the KR6 R700.

Sub-packages:
    methods/   reactive policies (threshold, APF, NEO, HOCBF)
    harness/   benchmark scenarios + metrics

Top-level types in `types.py` define the RobotState/Obstacle/ControlOutput
interface every method speaks.
"""
from .types import ControlOutput, Obstacle, RobotState

__all__ = ["ControlOutput", "Obstacle", "RobotState"]

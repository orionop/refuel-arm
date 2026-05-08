"""Shared dataclasses for the safety pipeline.

All methods consume a RobotState + list of Obstacles and return a ControlOutput.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class RobotState:
    """Snapshot of the arm at one control tick."""
    q: np.ndarray              # (6,) joint positions [rad]
    qdot: np.ndarray           # (6,) joint velocities [rad/s]
    ee_pos: np.ndarray         # (3,) end-effector position [m, world frame]
    ee_R: np.ndarray           # (3,3) end-effector rotation
    jacobian: np.ndarray       # (6,6) geometric Jacobian at ee


@dataclass
class Obstacle:
    """Spherical obstacle with linear velocity (constant-velocity model)."""
    p: np.ndarray              # (3,) center position [m, world frame]
    v: np.ndarray              # (3,) linear velocity [m/s]
    radius: float              # safety bubble radius [m]
    label: str = "obstacle"    # e.g. "child", "dog"


@dataclass
class ControlOutput:
    """What every safety method returns each tick."""
    qdot_cmd: np.ndarray                   # (6,) commanded joint velocity
    safe: bool                             # method's verdict
    info: dict = field(default_factory=dict)  # method-specific telemetry

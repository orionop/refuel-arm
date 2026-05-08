"""Test scenarios for the safety method benchmark.

Each scenario is a deterministic obstacle trajectory expressed in the world
frame, parameterised by (start_pos, end_pos, speed, radius). The arm is
assumed to be running its nominal refueling motion in the background.

Scenarios are kept simple and reproducible so we can quote concrete numbers
in the paper without hidden randomness.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

from ..types import Obstacle


ObstacleTraj = Callable[[float], Obstacle]   # t [s] -> obstacle at time t


@dataclass
class Scenario:
    name: str
    duration: float                        # seconds
    obstacle_traj: ObstacleTraj
    nominal_qdot: np.ndarray = field(default_factory=lambda: np.zeros(6))
    description: str = ""


def linear_obstacle(start: np.ndarray, end: np.ndarray, duration: float,
                    radius: float = 0.10, label: str = "child") -> ObstacleTraj:
    """Constant-velocity straight-line obstacle from start to end over duration."""
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    v = (end - start) / duration

    def at(t: float) -> Obstacle:
        t = max(0.0, min(t, duration))
        p = start + v * t
        return Obstacle(p=p.copy(), v=v.copy(), radius=radius, label=label)

    return at


def stationary_obstacle(p: np.ndarray, radius: float = 0.10,
                        label: str = "drum") -> ObstacleTraj:
    p = np.asarray(p, dtype=float)
    zero = np.zeros(3)

    def at(t: float) -> Obstacle:
        return Obstacle(p=p.copy(), v=zero.copy(), radius=radius, label=label)

    return at


# Default arm nominal velocity: small forward motion in joint 1 to mimic the
# refueling pipeline's coarse approach segment. Replace with a recorded
# nominal trajectory once we have one from the actual UR5 mission.
_DEFAULT_NOM = np.array([0.0, 0.1, 0.0, 0.0, 0.0, 0.0])


def head_on(speed: float = 0.8) -> Scenario:
    """Obstacle approaches the EE along +x → -x at constant speed."""
    return Scenario(
        name=f"head_on_{speed:.1f}",
        duration=3.0,
        obstacle_traj=linear_obstacle(
            start=np.array([1.5, 0.0, 0.5]),
            end=np.array([1.5 - speed * 3.0, 0.0, 0.5]),
            duration=3.0,
        ),
        nominal_qdot=_DEFAULT_NOM.copy(),
        description="Head-on approach, axis-aligned",
    )


def oblique(speed: float = 0.8, angle_deg: float = 45.0) -> Scenario:
    """Obstacle approaches at an angle to the arm's frontal plane."""
    a = np.deg2rad(angle_deg)
    direction = np.array([-np.cos(a), -np.sin(a), 0.0])
    start = np.array([1.5, -1.0, 0.5])
    end = start + direction * speed * 3.0
    return Scenario(
        name=f"oblique_{int(angle_deg)}_{speed:.1f}",
        duration=3.0,
        obstacle_traj=linear_obstacle(start=start, end=end, duration=3.0),
        nominal_qdot=_DEFAULT_NOM.copy(),
        description=f"Oblique approach {angle_deg}° from frontal axis",
    )


def passing(speed: float = 1.0) -> Scenario:
    """Obstacle passes by (does NOT collide) — tests false-positive rate."""
    return Scenario(
        name=f"passing_{speed:.1f}",
        duration=3.0,
        obstacle_traj=linear_obstacle(
            start=np.array([1.0, -2.0, 0.5]),
            end=np.array([1.0, +1.0, 0.5]),
            duration=3.0,
        ),
        nominal_qdot=_DEFAULT_NOM.copy(),
        description="Passes by 1m in front of arm without colliding",
    )


def fast_dash(speed: float = 1.5) -> Scenario:
    """Worst-case fast head-on approach (e.g. a child running)."""
    return Scenario(
        name=f"fast_dash_{speed:.1f}",
        duration=2.0,
        obstacle_traj=linear_obstacle(
            start=np.array([2.0, 0.0, 0.5]),
            end=np.array([2.0 - speed * 2.0, 0.0, 0.5]),
            duration=2.0,
            radius=0.15,
            label="child_running",
        ),
        nominal_qdot=_DEFAULT_NOM.copy(),
        description="Fast head-on dash, larger safety bubble",
    )


def static_obstacle_in_path(p: Optional[Sequence[float]] = None) -> Scenario:
    """Static obstacle the nominal path would hit. Sanity check for false stops."""
    p = np.array([0.5, 0.0, 0.5]) if p is None else np.asarray(p, dtype=float)
    return Scenario(
        name=f"static_obstacle",
        duration=3.0,
        obstacle_traj=stationary_obstacle(p),
        nominal_qdot=_DEFAULT_NOM.copy(),
        description="Stationary drum in the way",
    )


def all_scenarios() -> list[Scenario]:
    """Default benchmark suite."""
    return [
        head_on(speed=0.8),
        head_on(speed=1.5),
        oblique(angle_deg=45.0, speed=0.8),
        oblique(angle_deg=30.0, speed=1.0),
        passing(speed=1.0),
        fast_dash(speed=1.5),
        static_obstacle_in_path(),
    ]

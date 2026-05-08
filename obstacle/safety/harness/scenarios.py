"""Test scenarios for the safety method benchmark.

All scenarios are anchored to the UR5 home pose end-effector position
(`p_ee = [-0.487, -0.109, 0.432]`, computed from `kinematics.fk(q_home)`).
Obstacle trajectories are constructed in EE-relative offsets so the obstacle
actually approaches the arm rather than drifting through empty space.

Each scenario starts the obstacle OUTSIDE the typical influence radius
(0.5–0.6 m for NEO/HOCBF) so methods get a real "approach phase" to react to.

Scenarios are kept deterministic — fixed start/end, fixed speed, fixed nominal
joint motion — so we can quote concrete numbers in the paper.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..types import Obstacle


# UR5 home pose EE position, precomputed from kinematics.fk(q_home).
# Hardcoded so scenarios.py has no kinematics import (avoids a circular
# dependency at module import time).
P_EE_HOME = np.array([-0.487, -0.109, 0.432])

ObstacleTraj = Callable[[float], Obstacle]


@dataclass
class Scenario:
    name: str
    duration: float
    obstacle_traj: ObstacleTraj
    nominal_qdot: np.ndarray = field(default_factory=lambda: np.zeros(6))
    description: str = ""


# ── obstacle trajectory builders ──────────────────────────────────────

def linear_obstacle(start: np.ndarray, end: np.ndarray, duration: float,
                    radius: float = 0.10, label: str = "obstacle") -> ObstacleTraj:
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    v = (end - start) / duration

    def at(t: float) -> Obstacle:
        t_clamped = max(0.0, min(t, duration))
        p = start + v * t_clamped
        # report zero velocity once the obstacle has stopped at end pose
        v_now = v if 0.0 <= t <= duration else np.zeros(3)
        return Obstacle(p=p.copy(), v=v_now.copy(), radius=radius, label=label)

    return at


def stationary_obstacle(p: np.ndarray, radius: float = 0.10,
                        label: str = "drum") -> ObstacleTraj:
    p = np.asarray(p, dtype=float)
    zero = np.zeros(3)

    def at(t: float) -> Obstacle:
        return Obstacle(p=p.copy(), v=zero.copy(), radius=radius, label=label)

    return at


# ── nominal arm motions ───────────────────────────────────────────────
# Richer than "shoulder-lift drift" — exercises shoulder, elbow and wrist so
# the EE actually traces a meaningful arc and the Jacobian rows interact.
_NOM_REFUEL_LIKE = np.array([0.10, 0.15, -0.20, 0.10, 0.05, 0.0])


# ── concrete scenarios ────────────────────────────────────────────────

def _along_x(offset_x: float, side_y: float = 0.0, dz: float = 0.0) -> np.ndarray:
    return P_EE_HOME + np.array([offset_x, side_y, dz])


def head_on(speed: float = 0.8) -> Scenario:
    """Head-on approach toward EE home along +x → −x.

    Starts 1.5 m away (well outside influence radius), aims at EE_home, ends
    0.3 m past EE_home so a passive arm would be hit.
    """
    duration = 1.8 / max(speed, 0.1) + 0.5  # slight slack so it overruns past EE
    return Scenario(
        name=f"head_on_{speed:.1f}",
        duration=duration,
        obstacle_traj=linear_obstacle(
            start=_along_x(+1.5),
            end=_along_x(-0.3),
            duration=duration,
            radius=0.10,
            label="child",
        ),
        nominal_qdot=_NOM_REFUEL_LIKE.copy(),
        description="Head-on approach, axis-aligned, real refueling-like nominal motion.",
    )


def oblique(speed: float = 0.8, angle_deg: float = 45.0) -> Scenario:
    """Diagonal approach toward EE home from +x/+y quadrant."""
    a = np.deg2rad(angle_deg)
    direction = np.array([-np.cos(a), -np.sin(a), 0.0])
    duration = 1.8 / max(speed, 0.1) + 0.5
    start = P_EE_HOME - direction * 1.5
    end = P_EE_HOME + direction * 0.3
    return Scenario(
        name=f"oblique_{int(angle_deg)}_{speed:.1f}",
        duration=duration,
        obstacle_traj=linear_obstacle(start=start, end=end, duration=duration,
                                      radius=0.10),
        nominal_qdot=_NOM_REFUEL_LIKE.copy(),
        description=f"Oblique approach {angle_deg}° from frontal axis.",
    )


def passing(speed: float = 1.0) -> Scenario:
    """Obstacle passes by ~0.8 m to the side — should NOT trigger any method.

    This is the false-positive test: a person walking past the arm at safe
    distance should leave the nominal motion unchanged.
    """
    duration = 3.0
    side = 0.8
    return Scenario(
        name=f"passing_{speed:.1f}",
        duration=duration,
        obstacle_traj=linear_obstacle(
            start=P_EE_HOME + np.array([+0.1, +side, 0.0]) - np.array([0, speed * duration / 2, 0]),
            end=P_EE_HOME + np.array([+0.1, +side, 0.0]) + np.array([0, speed * duration / 2, 0]),
            duration=duration,
            radius=0.10,
            label="passerby",
        ),
        nominal_qdot=_NOM_REFUEL_LIKE.copy(),
        description="Walks past 0.8 m to the side without entering danger zone.",
    )


def fast_dash(speed: float = 1.5) -> Scenario:
    """Worst-case dynamic: child sprints toward EE.

    Starts 2 m away (gives ~0.5 s approach phase even at 1.5 m/s), larger
    safety bubble (0.15 m).
    """
    duration = 2.0 / max(speed, 0.1) + 0.3
    return Scenario(
        name=f"fast_dash_{speed:.1f}",
        duration=duration,
        obstacle_traj=linear_obstacle(
            start=_along_x(+2.0),
            end=_along_x(-0.5),
            duration=duration,
            radius=0.15,
            label="child_running",
        ),
        nominal_qdot=_NOM_REFUEL_LIKE.copy(),
        description="Fast head-on sprint, larger bubble.",
    )


def static_in_path() -> Scenario:
    """Stationary obstacle 0.3 m in front of EE — must brake for it."""
    p = P_EE_HOME + np.array([+0.3, 0.0, 0.0])
    return Scenario(
        name="static_in_path",
        duration=2.5,
        obstacle_traj=stationary_obstacle(p, radius=0.10, label="drum"),
        nominal_qdot=_NOM_REFUEL_LIKE.copy(),
        description="Drum 0.3 m in front of EE, intersects nominal motion.",
    )


def vertical_drop(speed: float = 1.0) -> Scenario:
    """Object falls toward EE from above (e.g. dropped tool). Tests +z axis."""
    duration = 1.6 / max(speed, 0.1) + 0.2
    return Scenario(
        name=f"vertical_drop_{speed:.1f}",
        duration=duration,
        obstacle_traj=linear_obstacle(
            start=P_EE_HOME + np.array([0.0, 0.0, +1.5]),
            end=P_EE_HOME + np.array([0.0, 0.0, -0.3]),
            duration=duration,
            radius=0.08,
            label="dropped_tool",
        ),
        nominal_qdot=_NOM_REFUEL_LIKE.copy(),
        description="Falling object, tests +z axis avoidance.",
    )


def all_scenarios() -> list[Scenario]:
    """Default benchmark suite — diverse coverage of speed, angle, axis."""
    return [
        head_on(speed=0.5),
        head_on(speed=1.0),
        head_on(speed=1.5),
        oblique(angle_deg=30.0, speed=1.0),
        oblique(angle_deg=60.0, speed=1.0),
        passing(speed=1.0),
        fast_dash(speed=1.8),
        static_in_path(),
        vertical_drop(speed=1.2),
    ]

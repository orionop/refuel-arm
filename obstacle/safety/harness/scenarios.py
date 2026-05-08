"""Test scenarios for the safety method benchmark.

All scenarios are anchored to the myCobot 280 home pose end-effector position
(`p_ee = [0.252, -0.1338, 0.13156]`, computed from `kinematics.fk(q_home)`).

Design principle (revised): scenarios use **zero nominal motion** by default
so the robot is stationary and methods MUST react to any threat. This avoids
the case where nominal task motion accidentally avoids the obstacle and the
safety method appears to do nothing.

For task-disruption tests we provide an `adversarial` variant where nominal
qdot points EE toward the obstacle — the safety method must override task.

Each scenario starts the obstacle OUTSIDE the typical influence radius
(~0.25 m for myCobot scaled NEO/HOCBF) so methods get a real "approach phase".

Scenarios are deterministic — fixed start/end, fixed speed, fixed nominal
joint motion — so we can quote concrete numbers in the paper.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..types import Obstacle


# myCobot 280 home pose EE position at q=0.
P_EE_HOME = np.array([0.252, -0.1338, 0.13156])

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
                    radius: float = 0.08, label: str = "obstacle") -> ObstacleTraj:
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    v = (end - start) / duration

    def at(t: float) -> Obstacle:
        t_clamped = max(0.0, min(t, duration))
        p = start + v * t_clamped
        v_now = v if 0.0 <= t <= duration else np.zeros(3)
        return Obstacle(p=p.copy(), v=v_now.copy(), radius=radius, label=label)

    return at


def stationary_obstacle(p: np.ndarray, radius: float = 0.08,
                        label: str = "drum") -> ObstacleTraj:
    p = np.asarray(p, dtype=float)
    zero = np.zeros(3)

    def at(t: float) -> Obstacle:
        return Obstacle(p=p.copy(), v=zero.copy(), radius=radius, label=label)

    return at


# Adversarial nominal motion: small EE +x drift (toward obstacle in head-on).
# Used for "task pulls toward danger" variants. ~0.05 m/s EE motion via shoulder.
_NOM_DRIFT_X = np.array([0.0, 0.0, 0.0, -0.5, 0.0, 0.0])  # wrist1 ≈ EE +x


# ── concrete scenarios ────────────────────────────────────────────────

def _along_x(offset_x: float, side_y: float = 0.0, dz: float = 0.0) -> np.ndarray:
    return P_EE_HOME + np.array([offset_x, side_y, dz])


def head_on(speed: float = 0.5) -> Scenario:
    """Head-on approach toward stationary EE.

    Robot at rest; obstacle aimed straight at EE_home from +x. End point is
    0.10 m past EE so a passive arm WILL be hit (sphere radius 0.08 m).
    """
    duration = 0.45 / max(speed, 0.1) + 0.5
    return Scenario(
        name=f"head_on_{speed:.1f}",
        duration=duration,
        obstacle_traj=linear_obstacle(
            start=_along_x(+0.35),
            end=_along_x(-0.10),
            duration=duration,
            radius=0.08,
            label="head_on",
        ),
        nominal_qdot=_NOM_DRIFT_X.copy(),
        description="Robot at rest; obstacle approaches head-on along +x.",
    )


def oblique(speed: float = 0.5, angle_deg: float = 45.0) -> Scenario:
    """Diagonal approach toward stationary EE."""
    a = np.deg2rad(angle_deg)
    direction = np.array([-np.cos(a), -np.sin(a), 0.0])
    duration = 0.45 / max(speed, 0.1) + 0.5
    start = P_EE_HOME - direction * 0.35
    end = P_EE_HOME + direction * 0.10
    return Scenario(
        name=f"oblique_{int(angle_deg)}_{speed:.1f}",
        duration=duration,
        obstacle_traj=linear_obstacle(start=start, end=end, duration=duration,
                                      radius=0.08, label=f"oblique_{int(angle_deg)}"),
        nominal_qdot=_NOM_DRIFT_X.copy(),
        description=f"Robot at rest; obstacle approaches at {angle_deg}° in xy plane.",
    )


def passing(speed: float = 0.5) -> Scenario:
    """Obstacle passes ~0.30 m to the side. Methods should NOT trigger.

    False-positive test. With 0.08 m radius and 0.30 m offset, separation
    is always > 0.22 m which exceeds danger_distance (0.20 m).
    """
    duration = 3.0
    side = 0.30
    travel = speed * duration
    return Scenario(
        name=f"passing_{speed:.1f}",
        duration=duration,
        obstacle_traj=linear_obstacle(
            start=P_EE_HOME + np.array([+0.05, +side, 0.0]) - np.array([0, travel / 2, 0]),
            end=P_EE_HOME + np.array([+0.05, +side, 0.0]) + np.array([0, travel / 2, 0]),
            duration=duration,
            radius=0.08,
            label="passerby",
        ),
        nominal_qdot=_NOM_DRIFT_X.copy(),
        description="Obstacle passes 0.30 m to the side — must NOT false-trigger.",
    )


def fast_dash(speed: float = 1.5) -> Scenario:
    """Worst-case fast head-on dash. Larger bubble."""
    duration = 0.55 / max(speed, 0.1) + 0.3
    return Scenario(
        name=f"fast_dash_{speed:.1f}",
        duration=duration,
        obstacle_traj=linear_obstacle(
            start=_along_x(+0.45),
            end=_along_x(-0.10),
            duration=duration,
            radius=0.10,
            label="dash",
        ),
        nominal_qdot=_NOM_DRIFT_X.copy(),
        description="Fast head-on dash, 0.10 m bubble; tight reaction window.",
    )


def static_in_path() -> Scenario:
    """Stationary obstacle in workspace. With nominal=0 the EE wouldn't hit it,
    so we use adversarial qdot pulling EE toward the obstacle."""
    p = P_EE_HOME + np.array([+0.10, 0.0, 0.0])
    return Scenario(
        name="static_in_path",
        duration=2.0,
        obstacle_traj=stationary_obstacle(p, radius=0.08, label="drum"),
        nominal_qdot=_NOM_DRIFT_X.copy(),  # task moves EE toward drum
        description="Drum 0.10 m ahead; nominal task pulls EE toward it — must override.",
    )


def vertical_drop(speed: float = 0.5) -> Scenario:
    """Object drops onto stationary EE from above."""
    duration = 0.5 / max(speed, 0.1) + 0.2
    return Scenario(
        name=f"vertical_drop_{speed:.1f}",
        duration=duration,
        obstacle_traj=linear_obstacle(
            start=P_EE_HOME + np.array([0.0, 0.0, +0.40]),
            end=P_EE_HOME + np.array([0.0, 0.0, -0.10]),
            duration=duration,
            radius=0.08,
            label="dropped_tool",
        ),
        nominal_qdot=_NOM_DRIFT_X.copy(),
        description="Object falls onto stationary EE from above.",
    )


def adversarial_head_on(speed: float = 0.5) -> Scenario:
    """Worst case: nominal task moves EE toward incoming obstacle.

    Tests whether the safety method can override active task motion.
    """
    duration = 0.45 / max(speed, 0.1) + 0.5
    return Scenario(
        name=f"adversarial_head_on_{speed:.1f}",
        duration=duration,
        obstacle_traj=linear_obstacle(
            start=_along_x(+0.35),
            end=_along_x(-0.10),
            duration=duration,
            radius=0.08,
            label="adversarial",
        ),
        nominal_qdot=_NOM_DRIFT_X.copy(),
        description="Obstacle head-on AND nominal task pulls EE forward.",
    )


def all_scenarios() -> list[Scenario]:
    """Default benchmark suite — diverse, deterministic, all force a reaction."""
    return [
        head_on(speed=0.3),
        head_on(speed=0.6),
        head_on(speed=1.0),
        oblique(angle_deg=30.0, speed=0.5),
        oblique(angle_deg=60.0, speed=0.5),
        passing(speed=0.5),
        fast_dash(speed=1.5),
        static_in_path(),
        vertical_drop(speed=0.5),
        adversarial_head_on(speed=0.5),
    ]

"""Test scenarios for the safety method benchmark.

All scenarios are anchored to the KUKA KR6 R700 candle-pose end-effector
position (`p_ee = [0.48, 0, 0.715]`), computed from
`kinematics.fk(q_candle)` where `q_candle = [0, -π/2, π/2, 0, 0, 0]`.

Design principles:
  1. Every scenario MUST force at least one method to deviate from nominal.
     No null tests.
  2. Obstacles start OUTSIDE the influence radius and approach, so methods
     get a real "detection → reaction" window.
  3. Nominal joint velocity is nonzero (multi-joint refueling arc), so the
     benchmark tests "moving arm vs moving obstacle" rather than the trivial
     "stationary arm" case.
  4. One explicit false-positive test (passing) where methods must NOT brake.
  5. All trajectories are deterministic — fixed start, end, speed.

KUKA KR6 R700 workspace reference:
  - Max reach: ~0.70 m (extended)
  - Candle EE:  [0.48, 0, 0.715]
  - Obstacle radii: 0.10 m (adult) or 0.15 m (child running)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..types import Obstacle


# KUKA KR6 R700 candle-pose EE position at q = [0, -π/2, π/2, 0, 0, 0]
P_EE_HOME = np.array([0.48, 0.0, 0.715])

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
    """Constant-velocity straight-line obstacle from start to end."""
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    v = (end - start) / duration

    def at(t: float) -> Obstacle:
        t_clamped = max(0.0, min(t, duration))
        p = start + v * t_clamped
        v_now = v if 0.0 <= t <= duration else np.zeros(3)
        return Obstacle(p=p.copy(), v=v_now.copy(), radius=radius, label=label)

    return at


def stationary_obstacle(p: np.ndarray, radius: float = 0.10,
                        label: str = "drum") -> ObstacleTraj:
    """Fixed obstacle at position p."""
    p = np.asarray(p, dtype=float)
    zero = np.zeros(3)

    def at(t: float) -> Obstacle:
        return Obstacle(p=p.copy(), v=zero.copy(), radius=radius, label=label)

    return at


# Nominal refueling motion: multi-joint arc that moves the EE forward and
# slightly downward — mimics the coarse approach segment of a refueling mission.
# ~0.05 m/s EE velocity via joints 2 and 4.
_NOM_REFUEL = np.array([0.0, 0.1, -0.05, 0.0, 0.1, 0.0])


# ── concrete scenarios ────────────────────────────────────────────────

def _offset(dx: float = 0, dy: float = 0, dz: float = 0) -> np.ndarray:
    """Helper: offset from home EE position."""
    return P_EE_HOME + np.array([dx, dy, dz])


def head_on(speed: float = 0.5) -> Scenario:
    """Obstacle approaches the EE head-on along +x.

    Start: 0.60 m ahead of EE (well outside all influence radii).
    End: 0.10 m behind EE (guaranteed collision if arm stays put).
    """
    duration = 0.70 / max(speed, 0.1)
    return Scenario(
        name=f"head_on_{speed:.1f}",
        duration=duration,
        obstacle_traj=linear_obstacle(
            start=_offset(dx=+0.60),
            end=_offset(dx=-0.10),
            duration=duration,
            radius=0.10,
            label="head_on",
        ),
        nominal_qdot=_NOM_REFUEL.copy(),
        description=f"Head-on +x approach at {speed} m/s; arm executing refuel arc.",
    )


def oblique(speed: float = 0.5, angle_deg: float = 45.0) -> Scenario:
    """Obstacle approaches at an angle in the xy-plane."""
    a = np.deg2rad(angle_deg)
    direction = np.array([-np.cos(a), -np.sin(a), 0.0])
    duration = 0.70 / max(speed, 0.1)
    start = P_EE_HOME - direction * 0.60   # 0.60 m away along approach axis
    end = P_EE_HOME + direction * 0.10     # 0.10 m past EE
    return Scenario(
        name=f"oblique_{int(angle_deg)}_{speed:.1f}",
        duration=duration,
        obstacle_traj=linear_obstacle(start=start, end=end, duration=duration,
                                      radius=0.10, label=f"oblique_{int(angle_deg)}"),
        nominal_qdot=_NOM_REFUEL.copy(),
        description=f"Oblique {angle_deg}° approach at {speed} m/s.",
    )


def passing(speed: float = 0.5) -> Scenario:
    """Obstacle passes 0.50 m to the side. Methods should NOT trigger.

    False-positive test. Obstacle travels along x-axis at y = +0.50 m offset.
    With 0.10 m radius the minimum clearance is ~0.40 m.
    """
    duration = 2.0
    travel = speed * duration
    return Scenario(
        name=f"passing_{speed:.1f}",
        duration=duration,
        obstacle_traj=linear_obstacle(
            start=P_EE_HOME + np.array([-travel/2, +0.50, 0.0]),
            end=P_EE_HOME + np.array([+travel/2, +0.50, 0.0]),
            duration=duration,
            radius=0.10,
            label="passerby",
        ),
        nominal_qdot=_NOM_REFUEL.copy(),
        description="Obstacle passes 0.50 m laterally — must NOT trigger.",
    )


def fast_dash(speed: float = 1.5) -> Scenario:
    """Worst-case fast head-on dash (e.g. a child running). Larger bubble."""
    duration = 0.80 / max(speed, 0.1)
    return Scenario(
        name=f"fast_dash_{speed:.1f}",
        duration=duration,
        obstacle_traj=linear_obstacle(
            start=_offset(dx=+0.70),
            end=_offset(dx=-0.10),
            duration=duration,
            radius=0.15,
            label="child_running",
        ),
        nominal_qdot=_NOM_REFUEL.copy(),
        description=f"Fast dash at {speed} m/s; larger 0.15 m bubble.",
    )


def static_in_path() -> Scenario:
    """Stationary drum placed 0.25 m ahead of EE in the refueling path.

    Nominal motion drives the EE toward it — method MUST override task.
    """
    p = _offset(dx=+0.25)
    return Scenario(
        name="static_in_path",
        duration=3.0,
        obstacle_traj=stationary_obstacle(p, radius=0.10, label="drum"),
        nominal_qdot=_NOM_REFUEL.copy(),
        description="Drum 0.25 m ahead; nominal task drives EE into it.",
    )


def vertical_drop(speed: float = 0.8) -> Scenario:
    """Object drops onto EE from above (e.g. a tool falling from a shelf)."""
    duration = 0.70 / max(speed, 0.1)
    return Scenario(
        name=f"vertical_drop_{speed:.1f}",
        duration=duration,
        obstacle_traj=linear_obstacle(
            start=_offset(dz=+0.60),
            end=_offset(dz=-0.10),
            duration=duration,
            radius=0.10,
            label="dropped_tool",
        ),
        nominal_qdot=_NOM_REFUEL.copy(),
        description="Object drops onto EE from 0.60 m above.",
    )


def adversarial_head_on(speed: float = 0.5) -> Scenario:
    """Worst case: obstacle head-on AND nominal task pulls EE toward it.

    Uses a stronger forward-driving nominal qdot to stress the safety filter.
    """
    duration = 0.70 / max(speed, 0.1)
    # Stronger forward motion (joint 2 pushing EE in +x)
    adversarial_nom = np.array([0.0, 0.3, -0.15, 0.0, 0.2, 0.0])
    return Scenario(
        name=f"adversarial_head_on_{speed:.1f}",
        duration=duration,
        obstacle_traj=linear_obstacle(
            start=_offset(dx=+0.60),
            end=_offset(dx=-0.10),
            duration=duration,
            radius=0.10,
            label="adversarial",
        ),
        nominal_qdot=adversarial_nom.copy(),
        description="Head-on + nominal task drives EE forward — hardest scenario.",
    )


def all_scenarios() -> list[Scenario]:
    """Default benchmark suite — 10 scenarios, all deterministic."""
    return [
        head_on(speed=0.3),
        head_on(speed=0.6),
        head_on(speed=1.0),
        oblique(angle_deg=30.0, speed=0.5),
        oblique(angle_deg=60.0, speed=0.5),
        passing(speed=0.5),
        fast_dash(speed=1.5),
        static_in_path(),
        vertical_drop(speed=0.8),
        adversarial_head_on(speed=0.5),
    ]

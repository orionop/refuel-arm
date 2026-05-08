"""Metrics for evaluating safety methods on a recorded trajectory.

A run produces a list of (t, state, obstacle, qdot_cmd, info) tuples. From
that we compute:

- min_separation         minimum point-to-obstacle clearance over the run [m]
- collision              True if min_separation went negative
- reaction_time          time from "obstacle entered danger zone" to
                         "method commanded a deviation from nominal qdot" [s]
- deviation_l2           total ‖qdot_cmd − qdot_nominal‖ over the run [rad/s]
                         (proxy for mission disruption)
- total_vel_variation    total ‖Δqdot_cmd‖ summed over the run [rad/s]
                         (smoothness penalty — lower = smoother)
- peak_jerk              max ‖d²(qdot)/dt²‖ across the run [rad/s³]
                         (true physical jerk — third time derivative of position)
- mean_acceleration      mean ‖Δqdot/dt‖ across the run [rad/s²]
                         (proper acceleration magnitude)
- false_positive         True if the method braked when obstacle never
                         actually entered the danger zone

All metrics are scalar floats / bools.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RunMetrics:
    scenario: str
    method: str
    min_separation: float        # m
    collision: bool
    reaction_time: float         # s, NaN if method never deviated
    deviation_l2: float          # rad/s, summed
    total_vel_variation: float   # rad/s, summed |Δqdot|
    peak_jerk: float             # rad/s³, true jerk
    mean_acceleration: float     # rad/s², |Δqdot/dt|
    false_positive: bool


def compute_metrics(
    *,
    scenario_name: str,
    method_name: str,
    times: np.ndarray,
    ee_positions: np.ndarray,        # (T,3)
    obstacle_positions: np.ndarray,  # (T,3)
    obstacle_radii: np.ndarray,      # (T,)
    qdot_cmds: np.ndarray,           # (T,n_joints)
    qdot_nominals: np.ndarray,       # (T,n_joints)
    danger_distance: float,
    expect_collision: bool = True,
) -> RunMetrics:
    delta = ee_positions - obstacle_positions
    separations = np.linalg.norm(delta, axis=1) - obstacle_radii
    min_sep = float(np.min(separations))
    collision = bool(min_sep < 0.0)

    # reaction time: first index where qdot deviates from nominal AFTER the
    # obstacle entered the danger zone
    in_danger = separations < danger_distance
    deviates = np.linalg.norm(qdot_cmds - qdot_nominals, axis=1) > 1e-3
    reaction_time = float("nan")
    if np.any(in_danger):
        first_danger = int(np.argmax(in_danger))
        post = deviates[first_danger:]
        if np.any(post):
            first_react = first_danger + int(np.argmax(post))
            reaction_time = float(times[first_react] - times[first_danger])

    deviation_l2 = float(np.sum(np.linalg.norm(qdot_cmds - qdot_nominals, axis=1)))

    # smoothness metrics
    if qdot_cmds.shape[0] > 2 and len(times) > 2:
        dt = float(np.median(np.diff(times)))
        if dt <= 0:
            dt = 1e-3
        # 1st diff: velocity changes (rad/s)
        dq = np.diff(qdot_cmds, axis=0)
        total_vel_variation = float(np.sum(np.linalg.norm(dq, axis=1)))
        # acceleration (rad/s²) = dq/dt
        accel = dq / dt
        mean_acceleration = float(np.mean(np.linalg.norm(accel, axis=1)))
        # jerk (rad/s³) = d(accel)/dt = d²(qdot)/dt²
        jerk_arr = np.diff(accel, axis=0) / dt
        peak_jerk = float(np.max(np.linalg.norm(jerk_arr, axis=1))) if len(jerk_arr) else 0.0
    else:
        total_vel_variation = 0.0
        peak_jerk = 0.0
        mean_acceleration = 0.0

    # false positive: braked when obstacle never entered danger
    braked = np.linalg.norm(qdot_cmds, axis=1) < 1e-3
    false_positive = bool((not expect_collision) and np.any(braked))

    return RunMetrics(
        scenario=scenario_name,
        method=method_name,
        min_separation=min_sep,
        collision=collision,
        reaction_time=reaction_time,
        deviation_l2=deviation_l2,
        total_vel_variation=total_vel_variation,
        peak_jerk=peak_jerk,
        mean_acceleration=mean_acceleration,
        false_positive=false_positive,
    )

"""Metrics for evaluating safety methods on a recorded trajectory.

A run produces a list of (t, state, obstacle, qdot_cmd, info) tuples. From
that we compute:

- min_separation         minimum point-to-obstacle clearance over the run
- collision              True if min_separation went negative
- reaction_time          time from "obstacle entered danger zone" to
                         "method commanded a deviation from nominal qdot"
- deviation_l2           total ‖qdot_cmd − qdot_nominal‖ over the run
                         (proxy for mission disruption)
- smoothness_jerk        proxy via ‖Δqdot_cmd‖ summed over the run
- false_positive         True if the method braked when obstacle never
                         actually entered the danger zone (used with the
                         "passing" scenario)

All metrics are scalar floats / bools. Logging the full trajectory tape stays
the responsibility of the runner; metrics are post-processed once.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RunMetrics:
    scenario: str
    method: str
    min_separation: float
    collision: bool
    reaction_time: float        # NaN if method never deviated
    deviation_l2: float
    smoothness_jerk: float
    false_positive: bool


def compute_metrics(
    *,
    scenario_name: str,
    method_name: str,
    times: np.ndarray,
    ee_positions: np.ndarray,        # (T,3)
    obstacle_positions: np.ndarray,  # (T,3)
    obstacle_radii: np.ndarray,      # (T,) — usually constant
    qdot_cmds: np.ndarray,           # (T,n_joints)
    qdot_nominals: np.ndarray,       # (T,n_joints)
    danger_distance: float,
    expect_collision: bool = True,   # True for "must brake", False for passing
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

    if qdot_cmds.shape[0] > 1:
        dq = np.diff(qdot_cmds, axis=0)
        smoothness_jerk = float(np.sum(np.linalg.norm(dq, axis=1)))
    else:
        smoothness_jerk = 0.0

    # false positive: braked (qdot dropped to ~0) when obstacle never entered danger
    braked = np.linalg.norm(qdot_cmds, axis=1) < 1e-3
    false_positive = bool((not expect_collision) and np.any(braked))

    return RunMetrics(
        scenario=scenario_name,
        method=method_name,
        min_separation=min_sep,
        collision=collision,
        reaction_time=reaction_time,
        deviation_l2=deviation_l2,
        smoothness_jerk=smoothness_jerk,
        false_positive=false_positive,
    )

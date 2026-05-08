"""Pure-python benchmark runner.

Simulates the arm + obstacle in Python (Euler integration of qdot at fixed dt).
No Gazebo, no ROS2 — useful for fast iteration and CI. The same SafetyMethod
instances will later be driven by the ROS2 node against the real Gazebo
simulation; metrics computed here should match qualitatively.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from ..kinematics import UR5Kinematics
from ..methods.base import SafetyMethod
from ..types import Obstacle, RobotState
from .metrics import RunMetrics, compute_metrics
from .scenarios import Scenario


@dataclass
class RunLog:
    times: np.ndarray
    qs: np.ndarray
    qdots_cmd: np.ndarray
    qdots_nom: np.ndarray
    ee_positions: np.ndarray
    obstacle_positions: np.ndarray
    obstacle_radii: np.ndarray


def run_scenario(
    method: SafetyMethod,
    scenario: Scenario,
    q0: np.ndarray,
    *,
    dt: float = 0.02,
    danger_distance: float = 0.5,
    kinematics: UR5Kinematics | None = None,
) -> tuple[RunLog, RunMetrics]:
    kin = kinematics or UR5Kinematics()

    n_steps = int(round(scenario.duration / dt)) + 1
    n_joints = q0.shape[0]

    times = np.linspace(0.0, scenario.duration, n_steps)
    qs = np.zeros((n_steps, n_joints))
    qdots_cmd = np.zeros((n_steps, n_joints))
    qdots_nom = np.zeros((n_steps, n_joints))
    ees = np.zeros((n_steps, 3))
    obs_p = np.zeros((n_steps, 3))
    obs_r = np.zeros(n_steps)

    q = q0.copy()
    qs[0] = q

    for i, t in enumerate(times):
        R, p_ee = kin.fk(q)
        J = kin.jacobian(q)
        ees[i] = p_ee

        obs = scenario.obstacle_traj(t)
        obs_p[i] = obs.p
        obs_r[i] = obs.radius

        state = RobotState(q=q.copy(), qdot=np.zeros(n_joints),
                           ee_pos=p_ee, ee_R=R, jacobian=J)

        out = method.step(state, [obs], scenario.nominal_qdot)
        qdots_cmd[i] = out.qdot_cmd
        qdots_nom[i] = scenario.nominal_qdot

        if i < n_steps - 1:
            q = q + out.qdot_cmd * dt
            qs[i + 1] = q

    log = RunLog(
        times=times, qs=qs,
        qdots_cmd=qdots_cmd, qdots_nom=qdots_nom,
        ee_positions=ees,
        obstacle_positions=obs_p, obstacle_radii=obs_r,
    )

    # only "passing" scenarios are false-positive tests; static_in_path is a real threat.
    expect_collision = "passing" not in scenario.name
    metrics = compute_metrics(
        scenario_name=scenario.name,
        method_name=method.name,
        times=times,
        ee_positions=ees,
        obstacle_positions=obs_p,
        obstacle_radii=obs_r,
        qdot_cmds=qdots_cmd,
        qdot_nominals=qdots_nom,
        danger_distance=danger_distance,
        expect_collision=expect_collision,
    )
    return log, metrics


def run_benchmark(
    methods: Sequence[SafetyMethod],
    scenarios: Sequence[Scenario],
    q0: np.ndarray,
    **kwargs,
) -> List[RunMetrics]:
    """Run every method against every scenario and return a flat metrics list."""
    out: List[RunMetrics] = []
    for s in scenarios:
        for m in methods:
            m.reset()
            _, mtr = run_scenario(m, s, q0, **kwargs)
            out.append(mtr)
    return out


def print_metrics_table(rows: Sequence[RunMetrics]) -> None:
    header = (f"{'scenario':<20s} {'method':<14s} {'min_sep':>8s} {'coll':>5s} "
              f"{'reac_t':>8s} {'dev_l2':>8s} {'tvv':>8s} {'pk_jerk':>10s} "
              f"{'mean_a':>8s} {'fp':>4s}")
    print(header)
    print("-" * len(header))
    for r in rows:
        rt = f"{r.reaction_time:.3f}" if not np.isnan(r.reaction_time) else "  -  "
        print(f"{r.scenario:<20s} {r.method:<14s} "
              f"{r.min_separation:>8.3f} {str(r.collision):>5s} "
              f"{rt:>8s} {r.deviation_l2:>8.2f} "
              f"{r.total_vel_variation:>8.2f} {r.peak_jerk:>10.1f} "
              f"{r.mean_acceleration:>8.2f} {str(r.false_positive):>4s}")

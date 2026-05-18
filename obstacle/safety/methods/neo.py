"""Method 2: NEO velocity damper.

Reactive QP that minimally modifies a nominal joint velocity to keep the arm a
safe distance from obstacles. The collision constraint is the velocity damper
inequality from Faverjon & Tournassoud (1987), adapted by Haviland & Corke into
a unified QP for redundancy resolution + obstacle avoidance.

Per active obstacle we add the linear constraint

    nᵀ J_o (qdot − qdot_obs_eq) ≤ ξ · (d − d_s) / (d_i − d_s)     when  d ≤ d_i,

where:
    n              unit vector from obstacle center to nearest arm point,
    J_o            translational Jacobian at that nearest arm point,
    qdot_obs_eq    joint-rate equivalent of the obstacle's linear velocity
                   (only the n-aligned component matters: nᵀ v_obs),
    d              current clearance (point-to-sphere),
    d_s            stop distance (hard barrier),
    d_i            influence distance (when the damper activates),
    ξ              damper gain (positive scalar).

The QP solved each tick:
    min_qdot  ½ ‖qdot − qdot_nominal‖² + ½ λ ‖qdot‖²
    s.t.      damper inequalities (one per active obstacle)
              −qdot_max ≤ qdot ≤ qdot_max

References
----------
Haviland & Corke, "NEO: A Novel Expeditious Optimisation Algorithm for Reactive
    Motion Control of Manipulators," RA-L 2021.
Faverjon & Tournassoud, "A local based approach for path planning of
    manipulators with a high number of degrees of freedom," ICRA 1987.

Dynamic-obstacle extension: the constraint subtracts the obstacle's closure-rate
component (n·v_obs) so the bound on the *relative* approach speed is what the
damper enforces, not the arm's absolute approach speed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..types import ControlOutput, Obstacle, RobotState
from .base import SafetyMethod
from ._qp import solve_qp_box


@dataclass
class NEOParams:
    d_s: float = 0.05        # hard stop distance [m] (scaled for KR6 R700)
    d_i: float = 0.40        # influence distance [m] (~57% of reach)
    xi: float = 1.0          # damper gain
    lam: float = 1e-3        # QP regularisation
    qdot_max: float = 5.59   # per-joint velocity bound [rad/s] (KR6 R700 slowest axis)


class NEOVelocityDamper(SafetyMethod):
    name = "neo"

    def __init__(self, params: NEOParams | None = None) -> None:
        self.p = params or NEOParams()

    def step(
        self,
        state: RobotState,
        obstacles: Sequence[Obstacle],
        qdot_nominal: np.ndarray,
    ) -> ControlOutput:
        n_joints = qdot_nominal.shape[0]
        J_pos = state.jacobian[:3]   # (3, n_joints)

        A_rows: list[np.ndarray] = []
        b_rows: list[float] = []
        min_d = float("inf")

        for o in obstacles:
            d_vec = state.ee_pos - o.p
            d_norm = float(np.linalg.norm(d_vec))
            d = d_norm - o.radius
            min_d = min(min_d, d)
            if d > self.p.d_i:
                continue

            n_hat = d_vec / max(d_norm, 1e-6)

            damper_rhs = self.p.xi * (d - self.p.d_s) / (self.p.d_i - self.p.d_s)
            obstacle_closure = float(n_hat @ o.v)

            # closure rate of arm point along -n must satisfy:
            #     −nᵀ J_pos qdot − (−nᵀ v_obs) ≤ damper_rhs
            #     ⇒  −(nᵀ J_pos) qdot ≤ damper_rhs − nᵀ v_obs
            A_rows.append(-(n_hat @ J_pos))
            b_rows.append(damper_rhs - obstacle_closure)

        # Hessian: I + λI, gradient: −qdot_nominal
        H = (1.0 + self.p.lam) * np.eye(n_joints)
        g = -qdot_nominal

        A = np.vstack(A_rows) if A_rows else None
        b = np.array(b_rows) if b_rows else None
        lb = -self.p.qdot_max * np.ones(n_joints)
        ub = +self.p.qdot_max * np.ones(n_joints)

        qdot, status = solve_qp_box(H, g, A, b, lb, ub)
        if qdot is None:
            qdot = np.zeros(n_joints)
            return ControlOutput(
                qdot, safe=False,
                info={"min_d": min_d, "active": len(A_rows), "qp": status},
            )

        return ControlOutput(
            qdot,
            safe=min_d > self.p.d_s,
            info={"min_d": min_d, "active": len(A_rows), "qp": status},
        )

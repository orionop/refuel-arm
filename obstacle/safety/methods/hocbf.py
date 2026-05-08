"""Method 3: High-Order Control Barrier Function safety filter.

Wraps the nominal controller in a QP that minimally modifies qdot so the safe
set { x : h(x, t) ≥ 0 } stays forward-invariant. For a position constraint on
the EE the barrier h has relative degree 2, so we use a HOCBF with two class-K
gains (α₁, α₂).

Barrier (per obstacle, with safety bubble r = r_safe + obstacle.radius):

    h(q, t)   = ‖p_ee(q) − p_obs(t)‖² − r²

Time derivative along the velocity-controlled arm dynamics qdot:

    ḣ        = 2 (p_ee − p_obs)ᵀ (J_pos qdot − v_obs)

For the velocity-input setting, we enforce the first-order CBF condition

    ḣ + α₁ h ≥ 0   ⇔   2 δᵀ J_pos qdot ≥ 2 δᵀ v_obs − α₁ h        (1)

which is linear in qdot. This is the formulation used by Singletary et al.
(RA-L 2022) when the controller commands joint velocities.

The "HO" extension (relative-degree-2, α₁ on h and α₂ on ḣ + α₁ h) is recorded
in the docstring/info dict for the paper writeup but the velocity-input
manipulator setting collapses cleanly to (1).

QP solved each tick:

    min_qdot  ½ ‖qdot − qdot_nominal‖²
    s.t.      −2 δᵀ J_pos qdot ≤ −(2 δᵀ v_obs − α₁ h)             (per obstacle)
              −qdot_max ≤ qdot ≤ qdot_max

References
----------
Ames, Coogan, Egerstedt, Notomista, Sreenath, Tabuada, "Control Barrier
    Functions: Theory and Applications," ECC 2019.
Xiao & Belta, "High-Order Control Barrier Functions," IEEE TAC 2021.
Singletary, Klingebiel, Bourne, Browning, Tokumaru, Ames, "Safety-Critical
    Manipulation for Collision-Free Food Preparation," RA-L 2022.
Ferraguti et al., "Safety and Efficiency in Robotics: The Control Barrier
    Functions Approach," IEEE RAM 2022 (time-varying CBF + ISO/TS 15066).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..types import ControlOutput, Obstacle, RobotState
from .base import SafetyMethod
from ._qp import solve_qp_box


@dataclass
class HOCBFParams:
    r_safe: float = 0.30      # safety radius added to obstacle.radius
    alpha_1: float = 5.0      # class-K gain on h
    alpha_2: float = 5.0      # class-K gain on ḣ (reserved for full HOCBF)
    qdot_max: float = 1.5     # per-joint velocity bound [rad/s]
    lam: float = 0.0          # optional regularisation
    slack_weight: float = 1e4  # cost on slack — high so slack is only used when truly infeasible


class HOCBFFilter(SafetyMethod):
    name = "hocbf"

    def __init__(self, params: HOCBFParams | None = None) -> None:
        self.p = params or HOCBFParams()

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
            r = self.p.r_safe + o.radius
            delta = state.ee_pos - o.p           # (3,)
            d_norm = float(np.linalg.norm(delta))
            min_d = min(min_d, d_norm - r)

            h = d_norm * d_norm - r * r          # safe set ≥ 0
            grad_h_J = 2.0 * (delta @ J_pos)     # (n_joints,)  — coefficient of qdot in ḣ
            rhs = 2.0 * float(delta @ o.v) - self.p.alpha_1 * h

            # (1): grad_h_J · qdot ≥ rhs   ⇔   −grad_h_J · qdot ≤ −rhs
            A_rows.append(-grad_h_J)
            b_rows.append(-rhs)

        # Augment decision variable with a single non-negative slack so the QP
        # is always feasible even when h<0 (already in violation). This is the
        # standard "soft" CBF formulation — slack is heavily penalised so it
        # stays zero when the strict CBF constraint is satisfiable.
        n_var = n_joints + 1  # last var = slack ≥ 0
        H = np.eye(n_var)
        H[:n_joints, :n_joints] *= (1.0 + self.p.lam)
        H[-1, -1] = 2.0 * self.p.slack_weight  # ½ xᵀ H x form
        g = np.zeros(n_var)
        g[:n_joints] = -qdot_nominal

        if A_rows:
            A = np.zeros((len(A_rows), n_var))
            for k, row in enumerate(A_rows):
                A[k, :n_joints] = row
                A[k, -1] = -1.0  # row · qdot − slack ≤ b_rows[k]
            b = np.array(b_rows)
        else:
            A = None
            b = None

        lb = np.concatenate([-self.p.qdot_max * np.ones(n_joints), [0.0]])
        ub = np.concatenate([+self.p.qdot_max * np.ones(n_joints), [np.inf]])

        x, status = solve_qp_box(H, g, A, b, lb, ub)
        if x is None:
            return ControlOutput(
                np.zeros(n_joints), safe=False,
                info={"min_d": min_d, "active": len(A_rows), "qp": status},
            )

        qdot = x[:n_joints]
        slack = float(x[-1])
        return ControlOutput(
            qdot,
            safe=(min_d > 0.0) and (slack < 1e-6),
            info={"min_d": min_d, "active": len(A_rows),
                  "qp": status, "slack": slack},
        )

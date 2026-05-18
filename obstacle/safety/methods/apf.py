"""Method 1: Artificial Potential Field with Informed Circular Fields.

Reactive vector-field controller. Combines:
  - attractive field toward the nominal goal (drives nominal motion),
  - repulsive field around each obstacle,
  - circular (swirl) field that biases motion *around* the obstacle along the
    direction the global plan was already going, killing the local-minimum
    failure mode of pure APF.

References
----------
Khatib, "Real-time obstacle avoidance for manipulators and mobile robots,"
    ICRA 1985 — original APF.
Haddadin et al., "Real-time reactive motion generation based on variable
    attractor dynamics and shaped obstacles," IROS 2011 — circular fields.
Becker, Caspers, Haddadin, "Informed Circular Fields for Global Reactive
    Obstacle Avoidance of Robotic Manipulators," Frontiers in Robotics & AI
    2024 — global-plan-informed swirl direction.

Dynamic-obstacle extension: include obstacle velocity in the relative-velocity
term so the repulsive field anticipates moving obstacles.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..types import ControlOutput, Obstacle, RobotState
from .base import SafetyMethod


@dataclass
class APFParams:
    k_att: float = 1.0       # attractive gain (toward nominal motion)
    k_rep: float = 0.15      # repulsive gain (scaled for KR6 R700, 0.70m reach)
    k_swirl: float = 0.5     # circular-field gain
    rho_0: float = 0.50      # influence radius [m] (~70% of reach)
    eta_v: float = 0.3       # weight on obstacle relative velocity
    qdot_max: float = 5.59   # per-joint velocity bound [rad/s] (KR6 R700 slowest axis)


class APFCircularFields(SafetyMethod):
    name = "apf_circular"

    def __init__(self, params: APFParams | None = None) -> None:
        self.p = params or APFParams()

    def step(
        self,
        state: RobotState,
        obstacles: Sequence[Obstacle],
        qdot_nominal: np.ndarray,
    ) -> ControlOutput:
        # Forward map nominal joint velocity to a Cartesian "intent" velocity at
        # the EE; this also defines the global-plan direction for the swirl bias.
        v_intent = (state.jacobian[:3] @ qdot_nominal).astype(float)

        f_total = self.p.k_att * v_intent
        min_d = float("inf")

        for o in obstacles:
            d_vec = state.ee_pos - o.p
            d = float(np.linalg.norm(d_vec))
            clearance = d - o.radius
            min_d = min(min_d, clearance)
            if clearance > self.p.rho_0:
                continue

            n_hat = d_vec / max(d, 1e-6)

            # Standard repulsive term (with relative-velocity boost for dynamic obstacles)
            v_rel = -o.v  # obstacle's velocity component closing the gap
            rho = max(clearance, 1e-3)
            f_rep = self.p.k_rep * (1.0 / rho - 1.0 / self.p.rho_0) * (1.0 / rho ** 2) * n_hat
            f_rep = f_rep + self.p.eta_v * v_rel

            # Circular field: rotate v_intent around the obstacle-EE axis so the
            # arm flows tangentially instead of head-on. "Informed" means we pick
            # the rotation sign that aligns with v_intent.
            # TODO: implement informed-swirl projection per Becker 2024 §III-B.
            tangent = np.cross(n_hat, v_intent)
            if np.linalg.norm(tangent) < 1e-6:
                # degenerate when v_intent ‖ n_hat → pick an arbitrary perpendicular
                tangent = np.cross(n_hat, np.array([0.0, 0.0, 1.0]))
            tangent = tangent / max(np.linalg.norm(tangent), 1e-6)
            f_swirl = self.p.k_swirl * tangent / max(rho, 1e-3)

            f_total = f_total + f_rep + f_swirl

        # Map Cartesian force to joint velocity via Jacobian pseudo-inverse,
        # then clip to per-joint velocity limits (APF makes no admissibility
        # guarantees, so clipping is the standard safety net).
        J_pos = state.jacobian[:3]
        qdot = np.linalg.pinv(J_pos) @ f_total
        qdot = np.clip(qdot, -self.p.qdot_max, self.p.qdot_max)

        return ControlOutput(
            qdot,
            safe=min_d > 0.0,
            info={"min_d": min_d, "f_norm": float(np.linalg.norm(f_total))},
        )

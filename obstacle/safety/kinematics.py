"""Forward kinematics + Jacobian for the Elephant Robotics myCobot 280.

DH parameters extracted from the official myCobot 280 URDF (Elephant Robotics).
All 6 joints are revolute around the Z-axis. Max reach ~0.28 m.

    i |   a (m)    |  d (m)   | alpha (rad) | theta (rad)
    --+------------+----------+-------------+-------------
    1 |  0.0       | 0.13156  |  +π/2       | q1
    2 |  0.1104    | 0.0      |   0         | q2
    3 |  0.096     | 0.0      |   0         | q3
    4 |  0.0       | 0.06062  |  +π/2       | q4
    5 |  0.0       | 0.0      |  −π/2       | q5
    6 |  0.0456    | 0.07318  |   0         | q6

The Jacobian is computed by finite difference over FK (more robust than
hand-derived form for a smaller arm where numerical precision matters).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# myCobot 280 DH parameters (Elephant Robotics URDF)
_DH_A = np.array([0.0, 0.1104, 0.096, 0.0, 0.0, 0.0456])
_DH_D = np.array([0.13156, 0.0, 0.0, 0.06062, 0.0, 0.07318])
_DH_ALPHA = np.array([np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0])

# myCobot 280 joint velocity / acceleration limits (rad/s, rad/s²)
JOINT_VEL_LIMITS = np.array([2.793, 2.793, 2.793, 2.793, 2.793, 2.793])
JOINT_ACC_LIMITS = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0])


def _dh_transform(a: float, d: float, alpha: float, theta: float) -> np.ndarray:
    """Standard DH homogeneous transform."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0,      sa,       ca,      d],
        [0.0,     0.0,      0.0,    1.0],
    ])


def fk_chain(q: np.ndarray):
    """Return list of 7 cumulative transforms T_0_0 … T_0_6."""
    q = np.asarray(q, dtype=float)
    T = np.eye(4)
    chain = [T.copy()]
    for i in range(6):
        T = T @ _dh_transform(_DH_A[i], _DH_D[i], _DH_ALPHA[i], q[i])
        chain.append(T.copy())
    return chain


def fk(q: np.ndarray):
    """Forward kinematics for UR5.

    Returns
    -------
    R_0e : (3,3) ndarray
    p_0e : (3,) ndarray
    """
    chain = fk_chain(q)
    T = chain[-1]
    return T[:3, :3].copy(), T[:3, 3].copy()


def jacobian(q: np.ndarray) -> np.ndarray:
    """Analytical geometric Jacobian about the EE in world frame.

    Rows 0:3 are translational, rows 3:6 are angular.
    Computed in closed form from the DH chain (faster + more accurate than
    finite differences, no perturbation tuning).
    """
    chain = fk_chain(q)
    p_e = chain[-1][:3, 3]
    J = np.zeros((6, 6))
    for i in range(6):
        T_i = chain[i]
        z_i = T_i[:3, 2]
        p_i = T_i[:3, 3]
        J[:3, i] = np.cross(z_i, p_e - p_i)
        J[3:, i] = z_i
    return J


@dataclass
class MyCobotKinematics:
    name: str = "mycobot_280"

    def fk(self, q):
        return fk(q)

    def jacobian(self, q):
        return jacobian(q)

    @property
    def vel_limits(self):
        return JOINT_VEL_LIMITS.copy()

    @property
    def acc_limits(self):
        return JOINT_ACC_LIMITS.copy()


UR5Kinematics = MyCobotKinematics  # alias for backwards compat

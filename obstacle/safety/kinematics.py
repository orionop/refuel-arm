"""Forward kinematics + Jacobian for the KUKA KR6 R700.

Kinematic parameters extracted from the verified IK-Geo solver used in the
main refueling pipeline (`refueling/kuka_refuel_ws/src/.../ik_geometric.py`).

The KR6 R700 belongs to the KUKA Agilus family — a compact 6-DOF industrial
manipulator with a spherical wrist and parallel axes 2/3. Max reach ~0.70 m.

Kinematic representation uses the product-of-exponentials (PoE) convention
from the IK-Geo library (Elias et al.): H columns are joint axes, P columns
are inter-joint offsets. We convert to standard DH-style cumulative
homogeneous transforms for FK and Jacobian computation.

Joint velocity limits from KUKA KR6 R700-2 datasheet.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ── KR6 R700 kinematic parameters (PoE form, from ik_geometric.py) ────
# H: (3, 6) — each column is the joint axis in the predecessor frame
# P: (3, 7) — inter-joint offsets P[:,0]=base→j1, ..., P[:,6]=j6→tool
_H = np.array([
    [0, 0, -1],    # H1: -Z
    [0, 1, 0],     # H2:  Y
    [0, 1, 0],     # H3:  Y
    [-1, 0, 0],    # H4: -X
    [0, 1, 0],     # H5:  Y
    [-1, 0, 0],    # H6: -X
]).T  # (3, 6)

_P = np.array([
    [0, 0, 0.208],       # P0: base → j1
    [0.025, 0, 0.192],   # P1: j1 → j2
    [0.315, 0, 0],       # P2: j2 → j3
    [0.365, 0, 0],       # P3: j3 → j4
    [0, 0, 0],           # P4: j4 → j5
    [0, 0, 0],           # P5: j5 → j6
    [0.09, 0, 0],        # P6: j6 → tool
]).T  # (3, 7)

# KR6 R700 joint velocity limits (rad/s) — from KUKA datasheet
JOINT_VEL_LIMITS = np.array([6.98, 5.59, 6.98, 7.85, 7.07, 10.47])

# KR6 R700 joint acceleration limits (rad/s²) — conservative estimates
JOINT_ACC_LIMITS = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0])


def _rot(k: np.ndarray, theta: float) -> np.ndarray:
    """Rodrigues' rotation: rotate around unit vector k by angle theta."""
    k = np.asarray(k, dtype=float).ravel()
    K = np.array([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0],
    ])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def fk_chain(q: np.ndarray):
    """Return list of 7 cumulative transforms T_0_0 … T_0_6 (base to each joint).

    Uses the PoE convention from the IK-Geo library.
    """
    q = np.asarray(q, dtype=float)
    chain = []
    R = np.eye(3)
    p = _P[:, 0].copy()  # base offset
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    chain.append(T.copy())

    for i in range(6):
        R = R @ _rot(_H[:, i], q[i])
        p = p + R @ _P[:, i + 1]
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = p
        chain.append(T.copy())

    return chain


def fk(q: np.ndarray):
    """Forward kinematics for the KUKA KR6 R700.

    Returns
    -------
    R_0e : (3,3) ndarray — end-effector rotation in world frame
    p_0e : (3,) ndarray  — end-effector position in world frame [m]
    """
    chain = fk_chain(q)
    T = chain[-1]
    return T[:3, :3].copy(), T[:3, 3].copy()


def jacobian(q: np.ndarray) -> np.ndarray:
    """Geometric Jacobian at the EE in world frame.

    Rows 0:3 are translational (m/rad), rows 3:6 are angular (rad/rad).
    Computed from the PoE chain: J_i = [z_i × (p_e − p_i); z_i].
    """
    chain = fk_chain(q)
    p_e = chain[-1][:3, 3]
    J = np.zeros((6, 6))
    for i in range(6):
        R_i = chain[i][:3, :3]
        p_i = chain[i][:3, 3]
        z_i = R_i @ _H[:, i]  # joint axis in world frame
        J[:3, i] = np.cross(z_i, p_e - p_i)
        J[3:, i] = z_i
    return J


@dataclass
class KR6Kinematics:
    """KUKA KR6 R700 kinematics wrapper for the safety benchmark."""
    name: str = "kuka_kr6_r700"

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


# Backwards compat alias
UR5Kinematics = KR6Kinematics

#!/usr/bin/env python3
"""
IK-Geo Exact Algebraic Solver for the KUKA KR6 R700.

Kinematic family: IK_spherical_2_parallel (spherical wrist + axes 2,3 parallel)
Translated from MATLAB: ik-geo/matlab/+IK/IK_spherical_2_parallel.m
Uses the `linearSubproblemSltns` PyPI package for canonical subproblems.

Kinematic parameters from: matlab/+hardcoded_IK_setups/KR6_R700.m
"""
import numpy as np
from linearSubproblemSltns import sp1_lib as sp1
from linearSubproblemSltns import sp3_lib as sp3
from linearSubproblemSltns import sp4_lib as sp4


def _ensure_iterable(x):
    """Ensure subproblem result is always iterable (handles scalar vs array)."""
    x = np.atleast_1d(x)
    return x[~np.isnan(x)]  # filter out NaN solutions


def rot(k, theta):
    """Rodrigues' rotation: rotate around unit vector k by angle theta."""
    k = np.asarray(k, dtype=float).flatten()
    K = np.array([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0]
    ])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


# ── Verified KUKA KR6 R700 Kinematic Parameters ─────────────────
# Source: matlab/+hardcoded_IK_setups/KR6_R700.m (URDF-derived POE)
KIN_KR6_R700 = {
    'H': np.array([
        [0, 0, -1],    # H1: -ez
        [0, 1, 0],     # H2:  ey
        [0, 1, 0],     # H3:  ey  (parallel to H2)
        [-1, 0, 0],    # H4: -ex
        [0, 1, 0],     # H5:  ey
        [-1, 0, 0],    # H6: -ex
    ]).T,              # shape (3, 6)
    'P': np.array([
        [0, 0, 0.208],               # P1
        [0.025, 0.0907, 0.192],       # P2
        [0.335, -0.0042, 0],          # P3
        [0.365, -0.0865, 0.025],      # P4
        [0, 0, 0],                    # P5 (zero = spherical wrist)
        [0, 0, 0],                    # P6 (zero = spherical wrist)
        [0.09, 0, 0],                 # P7
    ]).T,              # shape (3, 7)
}


def fwd_kinematics(q, kin=None):
    """Forward kinematics: returns (R_06, p_0T) for joint vector q."""
    if kin is None:
        kin = KIN_KR6_R700
    H = kin['H']
    P = kin['P']
    R = np.eye(3)
    p = P[:, 0].copy()
    for i in range(6):
        R = R @ rot(H[:, i], q[i])
        p = p + R @ P[:, i + 1]
    return R, p


def IK_spherical_2_parallel(R_06, p_0T, kin=None):
    """
    Exact algebraic IK for the KUKA KR6 R700.

    Parameters
    ----------
    R_06 : (3,3) ndarray – desired end-effector rotation
    p_0T : (3,) ndarray  – desired end-effector position
    kin  : dict with 'H' (3×6) and 'P' (3×7), defaults to KR6 R700

    Returns
    -------
    Q : (6, N) ndarray – each column is a valid joint solution
    """
    if kin is None:
        kin = KIN_KR6_R700

    Q = []
    H = kin['H']
    P = kin['P']
    p_0T = np.asarray(p_0T, dtype=float).flatten()

    # MATLAB sp_4(h, p, k, d) = Python sp4_run(p, k, h, d)
    # MATLAB: sp_4(H(:,2), p_shifted, -H(:,1), d)
    #   -> h=H2, p=p_shifted, k=-H1
    #   -> Python: sp4_run(p=p_shifted, k=-H1, h=H2, d)
    p_sp4 = p_0T - R_06 @ P[:, 6] - P[:, 0]
    d_sp4 = float(H[:, 1] @ (P[:, 1] + P[:, 2] + P[:, 3]))
    t1_arr, _ = sp4.sp4_run(p_sp4, -H[:, 0], H[:, 1], d_sp4)
    t1_arr = _ensure_iterable(t1_arr)

    for q1 in t1_arr:
        # ── Subproblem 3 for q3 ──────────────────────────────────
        v = rot(-H[:, 0], q1) @ (-p_0T + R_06 @ P[:, 6] + P[:, 0]) + P[:, 1]
        d_sp3 = float(np.linalg.norm(v))
        t3_arr, _ = sp3.sp3_run(-P[:, 3], P[:, 2], H[:, 2], d_sp3)
        t3_arr = _ensure_iterable(t3_arr)

        for q3 in t3_arr:
            # ── Subproblem 1 for q2 ──────────────────────────────
            p1_q2 = -P[:, 2] - rot(H[:, 2], q3) @ P[:, 3]
            p2_q2 = v
            q2, _ = sp1.sp1_run(p1_q2, p2_q2, H[:, 1])

            R_36 = rot(-H[:, 2], q3) @ rot(-H[:, 1], q2) @ rot(-H[:, 0], q1) @ R_06

            # MATLAB: sp_4(H(:,4), H(:,6), H(:,5), H(:,4)'*R_36*H(:,6))
            #   -> h=H4, p=H6, k=H5
            #   -> Python: sp4_run(p=H6, k=H5, h=H4, d)
            d_sp4_q5 = float(H[:, 3] @ R_36 @ H[:, 5])
            t5_arr, _ = sp4.sp4_run(H[:, 5], H[:, 4], H[:, 3], d_sp4_q5)
            t5_arr = _ensure_iterable(t5_arr)

            for q5 in t5_arr:
                # ── Subproblem 1 for q4 ──────────────────────────
                p1_q4 = rot(H[:, 4], q5) @ H[:, 5]
                p2_q4 = R_36 @ H[:, 5]
                q4, _ = sp1.sp1_run(p1_q4, p2_q4, H[:, 3])

                # ── Subproblem 1 for q6 ──────────────────────────
                p1_q6 = rot(-H[:, 4], q5) @ H[:, 3]
                p2_q6 = R_36.T @ H[:, 3]
                q6, _ = sp1.sp1_run(p1_q6, p2_q6, -H[:, 5])

                Q.append([q1, q2, q3, q4, q5, q6])

    if len(Q) == 0:
        return np.empty((6, 0))
    return np.array(Q).T

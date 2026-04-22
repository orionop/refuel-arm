#!/usr/bin/env python3
"""
IK-Geo Exact Algebraic Solver for the KUKA KR6 R700.

Kinematic family: IK_spherical_2_parallel (spherical wrist + axes 2,3 parallel)
Translated from MATLAB: ik-geo/matlab/+IK/IK_spherical_2_parallel.m
Uses the `linearSubproblemSltns` PyPI package for canonical subproblems.

Kinematic parameters from: matlab/+hardcoded_IK_setups/KR6_R700.m
"""
import numpy as np # type: ignore
from linearSubproblemSltns import sp1_lib as sp1 # type: ignore
from linearSubproblemSltns import sp3_lib as sp3 # type: ignore
from linearSubproblemSltns import sp4_lib as sp4 # type: ignore
from scipy.optimize import least_squares # type: ignore


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
KIN_KR6_R700 = {
    'H': np.array([
        [0, 0, -1],    # H1
        [0, 1, 0],     # H2
        [0, 1, 0],     # H3
        [-1, 0, 0],    # H4
        [0, 1, 0],     # H5
        [-1, 0, 0],    # H6
    ]).T,
    'P': np.array([
        [0, 0, 0.208],
        [0.025, 0, 0.192],
        [0.315, 0, 0],
        [0.365, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0.09, 0, 0],
    ]).T,
}

# ── KUKA KR8 R2100 (Cybertech nano) Kinematic Parameters ────────
KIN_KR8_R2100 = {
    'H': np.array([
        [0, 0, -1],    # H1
        [0, 1, 0],     # H2
        [0, 1, 0],     # H3
        [-1, 0, 0],    # H4
        [0, 1, 0],     # H5
        [-1, 0, 0],    # H6
    ]).T,
    'P': np.array([
        [0, 0, 0.3127],                # P1: base to j1
        [0.16, 0.0643, -0.2073],       # P2: j1 to j2
        [0.98, 0, -0.0123],            # P3: j2 to j3
        [0.4166, -0.22, 0.0766],       # P4: j3 to j4
        [0, -0.0415, -0.5184],         # P5: j4 to j5
        [0.0665, 0, 0.0415],           # P6: j5 to j6
        [0, 0, -0.0135],               # P7: j6 to tool0
    ]).T,
    'joint_limits': np.array([
        [-185, 185], [-275, -25], [-138, 175],
        [-165, 165], [-25, 230], [-350, 350]
    ])
}

# ── KUKA KR210 R3100 Prime Kinematic Parameters ─────────────────
KIN_KR210_R3100 = {
    'H': np.array([
        [0, 0, 1],    # H1
        [0, 1, 0],    # H2
        [0, 1, 0],    # H3
        [1, 0, 0],    # H4
        [0, 1, 0],    # H5
        [1, 0, 0],    # H6
    ]).T,
    'P': np.array([
        [-2.6200e-03,  9.7586e-04,  3.3099e-01],
        [ 3.5277e-01, -3.7476e-02,  4.1920e-01],
        [-9.8483e-05, -1.4750e-01,  1.2499e+00],
        [ 9.5795e-01,  1.8400e-01, -5.5059e-02],
        [ 5.4200e-01,  0.0000e+00,  0.0000e+00],
        [ 1.9250e-01,  0.0000e+00,  0.0000e+00],
        [ 3.7500e-02,  0.0000e+00, -2.3900e-04]
    ]).T,
}

# ── Official UR5 Kinematic Parameters (from rpiRobotics/ik-geo) ──────
KIN_UR5 = {
    'H': np.array([
        [0, 0, 1],    # H1
        [0, 1, 0],    # H2
        [0, 1, 0],    # H3
        [0, 1, 0],    # H4
        [0, 0, -1],   # H5 (Official specifies -ez)
        [0, 1, 0],    # H6
    ]).T,
    'P': np.array([
        [0, 0, 0.089159],               # P1
        [0, 0.1358, 0],                  # P2
        [0.425, -0.1197, 0],             # P3
        [0.3922, 0, 0],                  # P4
        [0, 0.093, -0.0946],             # P5
        [0, 0, 0],                       # P6
        [0, 0.0823, 0],                  # P7
    ]).T,
    'joint_limits': np.array([
        [-360, 360], [-360, 360], [-360, 360],
        [-360, 360], [-360, 360], [-360, 360]
    ])
}


def fwd_kinematics(q, kin=None):
    """Forward kinematics: returns (R_06, p_0T) for joint vector q."""
    if kin is None:
        kin = KIN_KR210_R3100
    H = kin['H']
    P = kin['P']
    R = np.eye(3, dtype=float)
    p = np.array(P[:, 0], dtype=float).copy()
    for i in range(6):
        R = R @ rot(H[:, i], q[i])
        offset = np.array(P[:, i + 1], dtype=float)
        p = p + (R @ offset)
    return R, p


def IK_spherical_2_parallel(R_06, p_0T, kin=None):
    """
    Exact algebraic IK for Spherical-2-Parallel family arm.
    """
    if kin is None:
        kin = KIN_KR210_R3100

    Q = []
    H_arr = np.array(kin['H'], dtype=float)
    P_arr = np.array(kin['P'], dtype=float)
    p_target = np.asarray(p_0T, dtype=float).flatten()

    h0 = np.array(H_arr[:, 0])
    h1 = np.array(H_arr[:, 1])
    h2 = np.array(H_arr[:, 2])
    h3 = np.array(H_arr[:, 3])
    h4 = np.array(H_arr[:, 4])
    h5 = np.array(H_arr[:, 5])

    p0 = np.array(P_arr[:, 0])
    p1 = np.array(P_arr[:, 1])
    p2 = np.array(P_arr[:, 2])
    p3 = np.array(P_arr[:, 3])
    p6 = np.array(P_arr[:, 6])

    # Subproblem 4 for q1
    p_ee_shifted = R_06 @ p6
    p_sp4 = p_target - p_ee_shifted - p0
    sum_p123 = p1 + p2 + p3
    d_sp4 = float(np.dot(h1, sum_p123))
    h0_neg = np.negative(h0)
    t1_arr, _ = sp4.sp4_run(p_sp4, h0_neg, h1, d_sp4)
    t1_arr = _ensure_iterable(t1_arr)

    for q1 in t1_arr:
        # Subproblem 3 for q3
        h0_neg_local = np.negative(h0)
        p_ee_shifted_local = R_06 @ p6
        v = rot(h0_neg_local, q1) @ (np.negative(p_target) + p_ee_shifted_local + p0) + p1
        d_sp3 = float(np.linalg.norm(v))
        p3_neg = np.negative(p3)
        t3_arr, _ = sp3.sp3_run(p3_neg, p2, h2, d_sp3)
        t3_arr = _ensure_iterable(t3_arr)

        for q3 in t3_arr:
            # Subproblem 1 for q2
            p2_neg = np.negative(p2)
            p1_q2 = p2_neg - rot(h2, q3) @ p3
            p2_q2 = v
            q2, _ = sp1.sp1_run(p1_q2, p2_q2, h1)

            # Orientation
            h2_neg = np.negative(h2)
            h1_neg = np.negative(h1)
            h0_neg_orient = np.negative(h0)
            r3_mat = rot(h2_neg, q3)
            r2_mat = rot(h1_neg, q2)
            r1_mat = rot(h0_neg_orient, q1)
            R_36 = r3_mat @ r2_mat @ r1_mat @ R_06
            
            d_sp4_q5 = float(np.dot(h3, (R_36 @ h5)))
            t5_arr, _ = sp4.sp4_run(h5, h4, h3, d_sp4_q5)
            t5_arr = _ensure_iterable(t5_arr)

            for q5 in t5_arr:
                # Subproblem 1 for q4
                p1_q4 = rot(h4, q5) @ h5
                p2_q4 = R_36 @ h5
                q4, _ = sp1.sp1_run(p1_q4, p2_q4, h3)
                
                # Subproblem 1 for q6
                h4_neg = np.negative(h4)
                p1_q6 = rot(h4_neg, q5) @ h3
                p2_q6 = R_36.T @ h3
                h5_neg = np.negative(h5)
                q6, _ = sp1.sp1_run(p1_q6, p2_q6, h5_neg)
                
                Q.append([q1, q2, q3, q4, q5, q6])

    Q = np.array(Q).T

    # --- LM POLISHER FOR ASYMMETRICAL CAD ---
    if Q.size > 0:
        Q_polished = []
        def residuals(q_opt):
            R_cur, p_cur = fwd_kinematics(q_opt, kin=kin)
            err_pos = p_cur - p_0T
            R_err = R_cur @ R_06.T
            err_rot = np.array([R_err[2,1]-R_err[1,2], 
                                R_err[0,2]-R_err[2,0], 
                                R_err[1,0]-R_err[0,1]])
            return np.concatenate((err_pos, err_rot))

        for i in range(Q.shape[1]):
            res = least_squares(residuals, Q[:, i], method='lm', ftol=1e-6, xtol=1e-6, max_nfev=50)
            if np.linalg.norm(res.fun) < 1e-3:
                Q_polished.append(res.x)
        if Q_polished:
            Q = np.array(Q_polished).T
        else:
            Q = np.array([[]])

    return Q


def IK_ur5(R_06, p_0T, kin=None):
    """
    Exact algebraic IK for the UR5 robot (3-Parallel-2-Intersecting family).
    Uses the official subproblem decomposition from rpiRobotics/ik-geo.
    """
    if kin is None:
        kin = KIN_UR5
    H_arr = np.array(kin['H'], dtype=float)
    P_arr = np.array(kin['P'], dtype=float)
    p_target = np.asarray(p_0T, dtype=float).flatten()
    Q = []

    # Frame-specific axes from official H matrix
    h1 = H_arr[:, 0]
    h2 = H_arr[:, 1]
    h5 = H_arr[:, 4]
    h6 = H_arr[:, 5]

    # Frame-specific offsets from official P matrix
    p1 = P_arr[:, 0]
    p2 = P_arr[:, 1]
    p3 = P_arr[:, 2]
    p4 = P_arr[:, 3]
    p5 = P_arr[:, 4]
    p7 = P_arr[:, 6]

    # p_06 = p_0T - P(:,1) - R_06*P(:,7)
    p_06 = p_target - p1 - (R_06 @ p7)
    
    # solve for q1 using SP4
    sum_p2345 = p2 + p3 + p4 + p5
    d_q1 = float(np.dot(h2, sum_p2345))
    h1_neg = np.negative(h1)
    
    t1_arr, _ = sp4.sp4_run(p_06, h1_neg, h2, d_q1)
    t1_arr = _ensure_iterable(t1_arr)

    for q1 in t1_arr:
        R_01 = rot(h1, q1)
        # solve for q5 using SP4
        # d_q5 = h2' * R_01' * R_06 * h6
        d_q5 = float(np.dot(h2, R_01.T @ R_06 @ h6))
        t5_arr, _ = sp4.sp4_run(h6, h5, h2, d_q5)
        t5_arr = _ensure_iterable(t5_arr)

        for q5 in t5_arr:
            R_45 = rot(h5, q5)
            # solve for q_14 using SP1
            # sp1(R_45*h6, R_01'*R_06*h6, h2)
            p1_q14 = R_45 @ h6
            p2_q14 = R_01.T @ R_06 @ h6
            q_14, _ = sp1.sp1_run(p1_q14, p2_q14, h2)
            
            # solve for q6 using SP1
            # sp1(R_45'*h2, R_06'*R_01*h2, -h6)
            p1_q6 = R_45.T @ h2
            p2_q6 = R_06.T @ R_01 @ h2
            h6_neg = np.negative(h6)
            q6, _ = sp1.sp1_run(p1_q6, p2_q6, h6_neg)
            
            # solve for q3 using SP3
            # d_inner = R_01' * p_06 - p2 - rot(h2, q_14) * p5
            d_inner = (R_01.T @ p_06) - p2 - (rot(h2, q_14) @ p5)
            d = float(np.linalg.norm(d_inner))
            p4_neg = np.negative(p4)
            t3_arr, _ = sp3.sp3_run(p4_neg, p3, h2, d)
            t3_arr = _ensure_iterable(t3_arr)

            for q3 in t3_arr:
                # solve for q2 using SP1
                # p_sum = p3 + rot(h2, q3) * p4
                p_sum_q2 = p3 + (rot(h2, q3) @ p4)
                q2, _ = sp1.sp1_run(p_sum_q2, d_inner, h2)
                
                # q4 by subtraction (wrap to Pi)
                q4 = (q_14 - q2 - q3 + np.pi) % (2 * np.pi) - np.pi
                Q.append([q1, q2, q3, q4, q5, q6])

    if not Q:
        return np.empty((6, 0))
    return np.array(Q).T


def IK_solve(R, p, robot="kuka"):
    """Unified solver interface."""
    r_lower = robot.lower()
    if r_lower == "ur5":
        return IK_ur5(R, p, KIN_UR5)
    if r_lower == "kr8":
        return IK_spherical_2_parallel(R, p, KIN_KR8_R2100)
    return IK_spherical_2_parallel(R, p, KIN_KR210_R3100)

import numpy as np
import sys
import os

# Mock IK components
def rot(k, theta):
    k = np.asarray(k, dtype=float).flatten()
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

KIN_KR6_R700 = {
    'H': np.array([[0, 0, -1], [0, 1, 0], [0, 1, 0], [-1, 0, 0], [0, 1, 0], [-1, 0, 0]]).T,
    'P': np.array([[0, 0, 0.208], [0.025, 0.0907, 0.192], [0.335, -0.0042, 0], [0.365, -0.0865, 0.025], [0, 0, 0], [0, 0, 0], [0.09, 0, 0]]).T,
}

def fwd_kinematics(q, kin):
    H = kin['H']
    P = kin['P']
    R = np.eye(3)
    p = np.array(P[:, 0]).copy()
    for i in range(6):
        R = R @ rot(H[:, i], q[i])
        p = p + (R @ P[:, i + 1])
    return R, p

# The joint angles from the user's log (degrees)
q_deg = np.array([-35.2, -63.3, 75.5, 81.5, -55.7, -75.2])
q_rad = np.radians(q_deg)

R, p = fwd_kinematics(q_rad, KIN_KR6_R700)
print(f"Goal was: [-0.25, 0.3, 0.5]")
print(f"FK Result p: {p}")
print(f"Distance error: {np.linalg.norm(p - np.array([-0.25, 0.3, 0.5]))}")

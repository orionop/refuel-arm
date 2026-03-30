#!/usr/bin/env python3
import numpy as np  # type: ignore

# UR5 Kinematic Parameters (Standard DH)
# Ref: UR5 documentation and extracted from ur5.xml
# a: link length, d: link offset, alpha: link twist
UR5_PARAMS = {
    'd': np.array([0.089159, 0.0, 0.0, 0.10915, 0.09465, 0.0823]),
    'a': np.array([0.0, -0.425, -0.39225, 0.0, 0.0, 0.0]),
    'alpha': np.array([np.pi/2, 0.0, 0.0, np.pi/2, -np.pi/2, 0.0]),
}

# IK-Geo format (H for axes, P for translation vectors)
# This mapping is used for the KUKA; we need to adapt it for UR5.
# For UR5, the axes are simpler (0,0,1 or 0,1,0).
H_UR5 = np.array([
    [0, 0, 1], # Joint 1
    [0, 1, 0], # Joint 2
    [0, 1, 0], # Joint 3
    [0, 1, 0], # Joint 4
    [0, 0, 1], # Joint 5
    [0, 1, 0], # Joint 6
]).T

P_UR5 = np.array([
    [0, 0, 0.089159],   # P01
    [0, 0, 0],          # P12
    [0, -0.425, 0],     # P23
    [0, -0.39225, 0],   # P34
    [0, 0, 0.10915],    # P45
    [0, 0, 0.09465],    # P56
    [0, 0, 0.0823],     # P6T
]).T

UR5_KIN = {
    'H': H_UR5,
    'P': P_UR5,
    'joint_limits': np.array([
        [-2*np.pi, 2*np.pi],
        [-2*np.pi, 2*np.pi],
        [-2*np.pi, 2*np.pi],
        [-2*np.pi, 2*np.pi],
        [-2*np.pi, 2*np.pi],
        [-2*np.pi, 2*np.pi],
    ])
}

if __name__ == "__main__":
    print("UR5 Kinematics Defined.")
    print("H axes:\n", H_UR5)
    print("P vectors:\n", P_UR5)

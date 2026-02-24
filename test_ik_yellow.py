import sys, os
sys.path.append(os.path.abspath('kuka_refuel_ws/src/kuka_kr6_gazebo/scripts'))
import numpy as np
import ik_geometric as ik

target_pos = np.array([0.0, -0.5, 0.3])

# Try a few orientations to find a successful IK solution
def test_R(R):
    Q = ik.IK_spherical_2_parallel(R, target_pos)
    if Q.size > 0:
        for i in range(Q.shape[1]):
            q = Q[:, i]
            q = (q + np.pi) % (2 * np.pi) - np.pi
            print(f"Valid Q: {np.round(q, 3)}")
            return q
    return None

import math
# Orientation 1: pointing down (-Z)
R1 = np.array([
    [1, 0, 0],
    [0, -1, 0],
    [0, 0, -1]
])
# Orientation 2: pointing right (-Y)
R2 = np.array([
    [0, -1, 0],
    [1, 0, 0],
    [0, 0, 1]
])
# Orientation 3: pointing forward (X)
R3 = np.array([
    [0, 0, 1],
    [0, 1, 0],
    [-1, 0, 0]
])

print("Testing R1 (Down)...")
q_best = test_R(R1)
if q_best is None:
    print("Testing R2 (Right)...")
    q_best = test_R(R2)
if q_best is None:
    print("Testing R3 (Forward)...")
    q_best = test_R(R3)

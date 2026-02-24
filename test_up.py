import sys, os
sys.path.append(os.path.abspath('kuka_refuel_ws/src/kuka_kr6_gazebo/scripts'))
import numpy as np
import ik_geometric as ik

best_z = 0
best_q = None

for j2 in np.linspace(-np.pi, 0, 20):
    for j3 in np.linspace(-np.pi, np.pi, 20):
        q = np.array([0.0, j2, j3, 0.0, 0.0, 0.0])
        R, p = ik.fwd_kinematics(q)
        # We want X and Y to be close to 0 relative to base, and Z to be max
        if p[2] > best_z and abs(p[0]-0.025) < 0.1 and abs(p[1]) < 0.1:
            best_z = p[2]
            best_q = q

print(f"Best upward pose: Joints = {np.round(best_q, 3)}, XYZ = {np.round(ik.fwd_kinematics(best_q)[1], 3)}")

# Let's test standard -pi/2
q_std = np.array([0.0, -np.pi/2, np.pi/2, 0.0, 0.0, 0.0])
print(f"Std upright 1: {np.round(ik.fwd_kinematics(q_std)[1], 3)}")

q_std2 = np.array([0.0, -np.pi/2, 0.0, 0.0, 0.0, 0.0])
print(f"Std upright 2: {np.round(ik.fwd_kinematics(q_std2)[1], 3)}")

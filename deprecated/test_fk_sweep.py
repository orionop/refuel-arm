import sys, os
sys.path.append(os.path.abspath('kuka_refuel_ws/src/kuka_kr6_gazebo/scripts'))
import numpy as np
import ik_geometric as ik

# Let's test standard rotations of Joint 1
for j1 in [1.57, 0.785, 0.0, -0.785, -1.57]:
    q = np.array([j1, 0.5, 1.0, 0.0, 0.5, 0.0])
    R, p = ik.fwd_kinematics(q)
    print(f"Joint 1 = {j1:>6} -> XYZ: {np.round(p, 3)}")

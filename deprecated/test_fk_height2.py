import sys, os
sys.path.append(os.path.abspath('kuka_refuel_ws/src/kuka_kr6_gazebo/scripts'))
import numpy as np
import ik_geometric as ik

for j2 in np.linspace(-1.5, -0.5, 10):
    for j3 in np.linspace(0.5, 1.5, 10):
        q = np.array([0.785, j2, j3, 0.0, 0.0, 0.0])
        R, p = ik.fwd_kinematics(q)
        if 0.65 < p[2] < 0.75 and p[0] > 0.3 and p[1] < -0.3:
            print(f"J2={j2:.2f}, J3={j3:.2f} -> XYZ: {np.round(p, 3)}")

import sys, os
sys.path.append(os.path.abspath('kuka_refuel_ws/src/kuka_kr6_gazebo/scripts'))
import numpy as np
import ik_geometric as ik

best_z = -10
best_q = None
best_p = None

for j2 in np.linspace(-1.5, 0, 10):
    for j3 in np.linspace(0, 2.0, 10):
        q = np.array([0.785, j2, j3, 0.0, 0.5, 0.0])
        R, p = ik.fwd_kinematics(q)
        # We want to grab a nozzle at least 0.3m off the ground
        # And it should be relatively far out radially (X > 0.1, Y < -0.1)
        if p[2] > 0.3 and p[2] < 0.6 and p[0] > 0.2 and p[1] < -0.2:
            print(f"J2={j2:.2f}, J3={j3:.2f} -> XYZ: {np.round(p, 3)}")

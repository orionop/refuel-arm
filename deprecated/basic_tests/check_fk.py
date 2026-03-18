import sys, os
sys.path.append(os.path.abspath('kuka_refuel_ws/src/kuka_kr6_gazebo/scripts'))
import numpy as np
import ik_geometric as ik

q = np.array([0.785, -0.5, 0.89, 0.0, 0.0, 0.0])
R, p = ik.fwd_kinematics(q)
print("q1:", q)
print("p1:", p)

R2, p2 = ik.fwd_kinematics(q2)
print("q2:", q2)
print("p2:", p2)

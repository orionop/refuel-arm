import sys, os
sys.path.append(os.path.abspath('kuka_refuel_ws/src/kuka_kr6_gazebo/scripts'))
import numpy as np
import ik_geometric as ik

q = np.array([0.0, -np.pi/2, np.pi/2, 0.0, 0.0, 0.0])
p = ik.fwd_kinematics(q)[1]
print("Pos 1 [0, -pi/2, pi/2, 0, 0, 0]:", np.round(p, 3))

q2 = np.array([0.0, -np.pi/2, 0.0, 0.0, 0.0, 0.0])
p2 = ik.fwd_kinematics(q2)[1]
print("Pos 2 [0, -pi/2, 0, 0, 0, 0]:", np.round(p2, 3))

q3 = np.array([0.0, -np.pi/2, np.pi/2, 0.0, np.pi/2, 0.0])
p3 = ik.fwd_kinematics(q3)[1]
print("Pos 3 [0, -pi/2, pi/2, 0, pi/2, 0]:", np.round(p3, 3))

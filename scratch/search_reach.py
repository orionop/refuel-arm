import numpy as np
import sys
import os

# Import IK-Geo
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'kuka_refuel_ws', 'src', 'kuka_kr6_gazebo', 'scripts')))
import ik_geometric as ik

def find_reachable_R(p_target):
    # Try different yaws and pitches to see if anything is reachable
    disp_yaw = np.arctan2(p_target[1], p_target[0])
    
    # Try a range of pitches and rolls
    for pitch in np.linspace(-np.pi/2, np.pi/2, 10):
        for roll in np.linspace(-np.pi, np.pi, 10):
            # Try to construct an R
            # The tool axis H6 is [-1, 0, 0] in local frame
            # Let's try some standard orientations
            R_test = ik.rot([0,0,1], disp_yaw) @ ik.rot([0,1,0], pitch) @ ik.rot([1,0,0], roll)
            Q = ik.IK_solve(R_test, p_target, robot='kr6')
            if Q.size > 0:
                print(f"Found solution at pitch={pitch:.2f}, roll={roll:.2f}")
                return R_test, Q
    return None, None

p_disp = np.array([-0.25, 0.3, 0.5])
R, Q = find_reachable_R(p_disp)
if R is not None:
    print("Success")
    print(f"Joints: {np.degrees(Q[:, 0])}")
else:
    print("FAILED to find reachable R")

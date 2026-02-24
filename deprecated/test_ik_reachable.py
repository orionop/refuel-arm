import sys, os
sys.path.append(os.path.abspath('kuka_refuel_ws/src/kuka_kr6_gazebo/scripts'))
import numpy as np
import ik_geometric as ik

def check_ik(pos):
    # Try forward orientation first, refuel inlet needs the nozzle pointing towards it
    R = np.array([
        [0, 0, 1],
        [0, 1, 0],
        [-1, 0, 0]
    ])
    Q = ik.IK_spherical_2_parallel(R, pos)
    if Q.size > 0:
        for i in range(Q.shape[1]):
            q = Q[:, i]
            q = (q + np.pi) % (2 * np.pi) - np.pi
            
            # Check limits
            limits = np.array([
                [-2.96, 2.96], [-3.31, 0.78], [-2.09, 2.72],
                [-6.10, 6.10], [-2.09, 2.09], [-6.10, 6.10]
            ])
            valid = True
            for j in range(6):
                if q[j] < limits[j, 0] or q[j] > limits[j, 1]:
                    valid = False
            
            if valid:
                print(f"Success for {pos}: {np.round(q, 3)}")
                return q
    print(f"Failed for {pos}")
    return None

check_ik(np.array([0.5, 0.2, 0.3]))
check_ik(np.array([0.55, 0.3, 0.3]))
check_ik(np.array([0.6, 0.2, 0.3]))
check_ik(np.array([0.55, 0.4, 0.3]))

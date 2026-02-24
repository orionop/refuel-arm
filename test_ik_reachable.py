import sys, os
sys.path.append(os.path.abspath('kuka_refuel_ws/src/kuka_kr6_gazebo/scripts'))
import numpy as np
import ik_geometric as ik

def check_ik(pos):
    # Try multiple random orientations to find any valid IK solution
    np.random.seed(42)
    for _ in range(100):
        # generate random rotation matrix
        theta = np.random.uniform(0, 2*np.pi)
        phi = np.random.uniform(0, np.pi)
        r = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [            0,              0, 1]
        ])
        
        Q = ik.IK_spherical_2_parallel(r, pos)
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

q_y = check_ik(np.array([0.3, -0.4, 0.4]))
q_y2 = check_ik(np.array([0.2, -0.5, 0.4]))
q_y3 = check_ik(np.array([0.0, -0.5, 0.5]))

# also check our red refuel inlet if needed
q_r = check_ik(np.array([0.45, 0.0, 0.3]))

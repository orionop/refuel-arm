import numpy as np
from linearSubproblemSltns import sp1_lib as sp1
from linearSubproblemSltns import sp3_lib as sp3
from linearSubproblemSltns import sp4_lib as sp4

def rot(k, theta):
    """Rodrigues' rotation formula for rotating around unit vector k by theta."""
    k = np.array(k, dtype=float).flatten()
    K = np.array([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0]
    ])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

def IK_spherical_2_parallel(R_06, p_0T, kin):
    """
    Inverse Kinematics solver for the KUKA KR6 R700 structure
    (Spherical wrist, axes 2 & 3 parallel).
    """
    Q = []
    is_LS_vec = []
    
    H = kin['H']
    P = kin['P']

    # Subproblem 4 for q1
    t1 = sp4.sp4_run(
        p_0T - R_06 @ P[:, 6] - P[:, 0],
        -H[:, 0],
        H[:, 1],
        H[:, 1].T @ (P[:, 1] + P[:, 2] + P[:, 3])
    )
    
    for q1 in t1:
        # Subproblem 3 for q3
        t3 = sp3.sp3_run(
            -P[:, 3],
            P[:, 2],
            H[:, 2],
            np.linalg.norm(rot(-H[:, 0], q1) @ (-p_0T + R_06 @ P[:, 6] + P[:, 0]) + P[:, 1])
        )
        
        for q3 in t3:
            # Subproblem 1 for q2
            p1_sp1 = -P[:, 2] - rot(H[:, 2], q3) @ P[:, 3]
            p2_sp1 = rot(-H[:, 0], q1) @ (-p_0T + R_06 @ P[:, 6] + P[:, 0]) + P[:, 1]
            q2, _ = sp1.sp1_run(p1_sp1, p2_sp1, H[:, 1])
            
            R_36 = rot(-H[:, 2], q3) @ rot(-H[:, 1], q2) @ rot(-H[:, 0], q1) @ R_06
            
            # Subproblem 4 for q5
            t5 = sp4.sp4_run(
                H[:, 5],
                H[:, 3],
                H[:, 4],
                H[:, 3].T @ R_36 @ H[:, 5]
            )
            
            for q5 in t5:
                # Subproblem 1 for q4
                p1_sp1_q4 = rot(H[:, 4], q5) @ H[:, 5]
                p2_sp1_q4 = R_36 @ H[:, 5]
                q4, _ = sp1.sp1_run(p1_sp1_q4, p2_sp1_q4, H[:, 3])
                
                # Subproblem 1 for q6
                p1_sp1_q6 = rot(-H[:, 4], q5) @ H[:, 3]
                p2_sp1_q6 = R_36.T @ H[:, 3]
                q6, _ = sp1.sp1_run(p1_sp1_q6, p2_sp1_q6, -H[:, 5])
                
                Q.append([q1, q2[0], q3, q4[0], q5, q6[0]])
                
    return np.array(Q).T

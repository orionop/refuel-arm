import sys
import numpy as np

sys.path.insert(0, "kuka_refuel_ws/src/kuka_kr6_gazebo/scripts")
from ik_geometric import IK_spherical_2_parallel, fwd_kinematics

YELLOW_TARGET_XYZ = np.array([0.3, -0.4, 0.4])
YELLOW_TARGET_R = np.eye(3)

Q_HOME = np.zeros(6)

JOINT_LIMITS = np.array([
    [-2.967059725,  2.967059725],
    [-3.316125575,  0.785398163],
    [-2.094395100,  2.722713630],
    [-3.228859113,  3.228859113],
    [-2.094395100,  2.094395100],
    [-6.108652375,  6.108652375],
])

def within_joint_limits(q):
    return all(JOINT_LIMITS[i, 0] <= q[i] <= JOINT_LIMITS[i, 1] for i in range(6))

def wrap_to_limits(q):
    q_wrapped = np.copy(q)
    for i in range(6):
        while q_wrapped[i] > np.pi:
            q_wrapped[i] -= 2 * np.pi
        while q_wrapped[i] < -np.pi:
            q_wrapped[i] += 2 * np.pi
    return q_wrapped

def filter_solutions(Q, q_prev=None):
    if Q.size == 0: return np.empty((6, 0))
    valid = [wrap_to_limits(Q[:, i]) for i in range(Q.shape[1]) if within_joint_limits(wrap_to_limits(Q[:, i]))]
    if not valid: return np.empty((6, 0))
    valid = np.array(valid).T
    if q_prev is not None:
        valid = valid[:, np.argsort(np.linalg.norm(valid.T - q_prev, axis=1))]
    return valid

Q_yellow = IK_spherical_2_parallel(YELLOW_TARGET_R, YELLOW_TARGET_XYZ)
Q_yellow_valid = filter_solutions(Q_yellow, Q_HOME)
if Q_yellow_valid.size > 0:
    print("Q_NOZZLE =", repr(Q_yellow_valid[:, 0]))
else:
    print("NO VALID IK")

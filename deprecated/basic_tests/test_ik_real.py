import sys
sys.path.append('/Users/anuragx/Desktop/random/projects/refuel-arm')
import numpy as np
from ik_geometric import IK_spherical_2_parallel, fwd_kinematics

def test_ik(pos, label):
    R = np.eye(3)
    Q = IK_spherical_2_parallel(R, np.array(pos))
    print(f"{label}: pos={pos}")
    if Q.size > 0:
        print(f"  Found {Q.shape[1]} solutions.")
        for i in range(Q.shape[1]):
            q = Q[:, i]
            # Wrap to limits
            for j in range(6):
                while q[j] > np.pi: q[j] -= 2*np.pi
                while q[j] < -np.pi: q[j] += 2*np.pi
            print(f"  Sol {i}: {np.round(q, 3)}")
    else:
        print("  ❌ No solutions.")

test_ik([0.55, 0.1, 0.3], "RED (Car Inlet)")
test_ik([0.3, -0.4, 0.4], "YELLOW (Nozzle Station)")

# Let's try another orientation if np.eye(3) fails for yellow
R_side = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
print("\nTesting YELLOW with R_side:")
Q = IK_spherical_2_parallel(R_side, np.array([0.3, -0.4, 0.4]))
print("Solutions:", Q.shape[1] if Q.size > 0 else 0)

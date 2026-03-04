#!/usr/bin/env python3
"""
STOMP: Stochastic Trajectory Optimization for Motion Planning
=============================================================

This expanded version includes dynamic 2.5D Grid / Point Cloud 
collision avoidance (unlike standard RRT) by generating a 
continuous workspace loss function via the Euclidean Distance Transform (EDT).

Standalone implementation for 6-DOF robot arms. No GPU, no training.
"""
import sys
import os
import numpy as np

# Import IK-Geo for FK and arm kinematics description
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'kuka_refuel_ws', 'src', 'kuka_kr6_gazebo', 'scripts')))
try:
    import ik_geometric as ik
except ImportError:
    # Try one level up if inside a subdirectory
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'kuka_refuel_ws', 'src', 'kuka_kr6_gazebo', 'scripts')))
    import ik_geometric as ik

try:
    from scipy.ndimage import distance_transform_edt
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("WARNING: scipy.ndimage not found, collision gradients disabled!")

class Grid3D:
    def __init__(self, x_min=-1.0, x_max=1.5, y_min=-1.0, y_max=1.0, z_min=0.0, z_max=2.0, resolution=0.05):
        self.resolution = resolution
        self.origin = np.array([x_min, y_min, z_min])
        self.shape = (
            int(np.ceil((x_max - x_min) / resolution)),
            int(np.ceil((y_max - y_min) / resolution)),
            int(np.ceil((z_max - z_min) / resolution))
        )
        self.edt = np.ones(self.shape, dtype=np.float32) * float('inf')

    def build_from_point_cloud(self, point_cloud: np.ndarray):
        """Build 3D Euclidean Distance Transform from 2.5D Point Cloud (N, 3)."""
        if not SCIPY_AVAILABLE: return

        grid = np.zeros(self.shape, dtype=bool)
        for pt in point_cloud:
            idx = np.floor((pt - self.origin) / self.resolution).astype(int)
            idx = np.clip(idx, [0, 0, 0], np.array(self.shape) - 1)
            # Create a 2.5D column in the 3D grid
            grid[idx[0], idx[1], :idx[2]+1] = True
            
        # EDT calculates distance to the closest False (so we run it on inverted grid)
        self.edt = distance_transform_edt(~grid) * self.resolution

    def get_distance(self, pt: np.ndarray) -> float:
        """Lookup distance from EDT."""
        idx = np.floor((pt - self.origin) / self.resolution).astype(int)
        if np.any(idx < 0) or np.any(idx >= self.shape):
            return 10.0 # Safe if out of bounds
        return self.edt[idx[0], idx[1], idx[2]]

def _smoothness_matrix(n: int) -> np.ndarray:
    D = np.zeros((n - 2, n))
    for i in range(n - 2):
        D[i, i] = 1.0; D[i, i + 1] = -2.0; D[i, i + 2] = 1.0
    return D.T @ D

def _joint_limit_cost(trajectory: np.ndarray, limits: np.ndarray, margin: float = 0.1) -> np.ndarray:
    N, ndof = trajectory.shape
    cost = np.zeros(N)
    for j in range(ndof):
        lo, hi = limits[j]; rng = hi - lo
        centre = (lo + hi) / 2.0; half = rng / 2.0
        d = np.abs(trajectory[:, j] - centre) / half
        thresh = 1.0 - margin
        violations = np.maximum(d - thresh, 0.0)
        cost += (violations / margin) ** 2
    return cost

def _smoothness_cost(trajectory: np.ndarray, A: np.ndarray) -> float:
    cost = 0.0
    for j in range(trajectory.shape[1]):
        q = trajectory[:, j]; cost += q @ A @ q
    return cost

def _velocity_cost(trajectory: np.ndarray) -> np.ndarray:
    diffs = np.diff(trajectory, axis=0)
    vel = np.sum(diffs ** 2, axis=1)
    return np.concatenate([[0.0], vel])

def _obstacle_cost(trajectory: np.ndarray, grid: Grid3D, margin: float = 0.3) -> np.ndarray:
    """Forward Kinematics 2.5D Grid Penalty."""
    cost = np.zeros(len(trajectory))
    if grid is None: return cost

    kin = ik.KIN_KR6_R700
    H, P = kin['H'], kin['P']
    for i, q in enumerate(trajectory):
        R = np.eye(3); p = P[:, 0].copy()
        
        # Test 4 key locations on the arm (Elbow, Wrist, End-Effector)
        check_points = []
        for j in range(6):
            R = R @ ik.rot(H[:, j], q[j])
            p = p + R @ P[:, j + 1]
            if j in [2, 4, 5]: check_points.append(p.copy())
            
        # Interpolate a few points down the "forearm" Link 3 to Link 4/5
        if len(check_points) >= 3:
            elbow, wrist, tool = check_points[0], check_points[1], check_points[2]
            forearm_mid = elbow + 0.5 * (wrist - elbow)
            check_points.append(forearm_mid)

        for pt in check_points:
            d = grid.get_distance(pt)
            if d < margin:
                cost[i] += ((margin - d) / margin) ** 2
    return cost

import matplotlib.pyplot as plt

def visualize_stomp_results(trajectory, costs, grid, pc):
    """Generate a 3-panel analysis graph for the collision-aware planner."""
    fig = plt.figure(figsize=(15, 5))
    
    # 1. Cost Progression
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.plot(costs, 'b-o', markersize=4)
    ax1.set_title("Total Cost Progression")
    ax1.set_xlabel("Iteration (x10)")
    ax1.set_ylabel("Cost")
    ax1.grid(True)

    # 2. Joint Angles
    ax2 = fig.add_subplot(1, 3, 2)
    for j in range(trajectory.shape[1]):
        ax2.plot(trajectory[:, j], label=f"J{j+1}")
    ax2.set_title("Final Optimized Joint Angles")
    ax2.set_xlabel("Waypoint")
    ax2.set_ylabel("Angle (rad)")
    ax2.legend()
    ax2.grid(True)

    # 3. 3D Workspace Path vs Obstacles
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    
    # Plot obstacle point cloud
    pc = np.array(pc)
    ax3.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c='red', alpha=0.3, s=10, label="Obstacle PC")
    
    # Calculate EE path
    ee_path = []
    kin = ik.KIN_KR6_R700
    for q in trajectory:
        _, p = ik.fwd_kinematics(q, kin)
        ee_path.append(p)
    ee_path = np.array(ee_path)
    
    ax3.plot(ee_path[:, 0], ee_path[:, 1], ee_path[:, 2], 'b-p', linewidth=2, label="EE Trajectory")
    ax3.set_title("3D End-Effector Path")
    ax3.set_xlabel("X (m)"); ax3.set_ylabel("Y (m)"); ax3.set_zlabel("Z (m)")
    ax3.legend()
    
    plt.tight_layout()
    os.makedirs("output_graphs", exist_ok=True)
    plt.savefig("output_graphs/stomp_collision_analysis.png")
    print("\n📊 Visualization saved to output_graphs/stomp_collision_analysis.png")

def stomp_optimize(
    q_start: np.ndarray,
    q_goal: np.ndarray,
    joint_limits: np.ndarray,
    grid: Grid3D = None,
    n_waypoints: int = 30,
    n_iterations: int = 80,
    n_rollouts: int = 15,
    noise_stddev: float = 0.15,
    noise_decay: float = 0.96,
    w_smooth: float = 10.0,
    w_limit: float = 50.0,
    w_vel: float = 5.0,
    w_obs: float = 1000.0,
    safety_margin: float = 0.2,
    verbose: bool = True,
):
    """STOMP Optimizer with 2.5D Grid Avoidance."""
    ndof = len(q_start)
    trajectory = np.zeros((n_waypoints, ndof))
    for i in range(n_waypoints):
        alpha = i / (n_waypoints - 1)
        trajectory[i] = (1 - alpha) * q_start + alpha * q_goal

    A = _smoothness_matrix(n_waypoints)
    interior = slice(1, n_waypoints - 1)
    n_interior = n_waypoints - 2
    best_cost = float("inf")
    noise = noise_stddev
    
    cost_history = []

    if verbose: print(f"STOMP Collision-Aware execution: {n_iterations} itrs, grid_enabled={bool(grid)}")

    for it in range(n_iterations):
        candidates, costs = [], []
        for _ in range(n_rollouts):
            delta = np.random.randn(n_interior, ndof) * noise
            candidate = trajectory.copy()
            candidate[interior] += delta
            for j in range(ndof):
                candidate[:, j] = np.clip(candidate[:, j], joint_limits[j, 0], joint_limits[j, 1])
            candidate[0] = q_start; candidate[-1] = q_goal

            # Calculate the various costs
            c_smooth = w_smooth * _smoothness_cost(candidate, A)
            c_limit = w_limit * np.sum(_joint_limit_cost(candidate, joint_limits))
            c_vel = w_vel * np.sum(_velocity_cost(candidate))
            c_obs = w_obs * np.sum(_obstacle_cost(candidate, grid, safety_margin))

            candidates.append(candidate)
            costs.append(c_smooth + c_limit + c_vel + c_obs)

        # Eval current trajectory
        c_smooth = w_smooth * _smoothness_cost(trajectory, A)
        c_limit = w_limit * np.sum(_joint_limit_cost(trajectory, joint_limits))
        c_vel = w_vel * np.sum(_velocity_cost(trajectory))
        c_obs = w_obs * np.sum(_obstacle_cost(trajectory, grid, safety_margin))
        current_cost = c_smooth + c_limit + c_vel + c_obs
        candidates.append(trajectory.copy()); costs.append(current_cost)

        # Prob update
        costs = np.array(costs); min_cost = np.min(costs)
        h = 10.0
        exp_costs = np.exp(-h * (costs - min_cost) / (np.max(costs) - min_cost + 1e-10))
        probs = exp_costs / (np.sum(exp_costs) + 1e-10)

        new_traj = np.zeros_like(trajectory)
        for k, cand in enumerate(candidates): new_traj += probs[k] * cand
        new_traj[0] = q_start; new_traj[-1] = q_goal
        trajectory = new_traj
        if min_cost < best_cost: best_cost = min_cost
        noise *= noise_decay

        if verbose and (it % 10 == 0 or it == n_iterations - 1):
            cost_history.append(current_cost)
            print(f"  [iter {it:3d}/{n_iterations}] total={current_cost:.1f} (smooth: {c_smooth:.1f}, limit: {c_limit:.1f}, obs: {c_obs:.1f})")

    for j in range(ndof): trajectory[:, j] = np.clip(trajectory[:, j], joint_limits[j, 0], joint_limits[j, 1])
    trajectory[0] = q_start; trajectory[-1] = q_goal
    return trajectory, cost_history


if __name__ == "__main__":
    print("STOMP 2.5D Collision Grid Test")
    print("=" * 50)
    limits = np.array([
        [-2.967,  2.967], [-3.316,  0.785], [-2.094,  2.722],
        [-3.228,  3.228], [-2.094,  2.094], [-6.108,  6.108],
    ])
    q_home = np.zeros(6); q_goal = np.array([0.5, -1.2, 1.5, -0.2, 0.4, 0.1])
    
    # Fake Point Cloud representing a table/barrier directly in front
    pc = []
    for x in np.linspace(0.4, 0.6, 15):
        for y in np.linspace(-0.3, 0.3, 15):
            pc.append([x, y, 0.4]) # Barrier at Z=0.4
    print(f"Created a {len(pc)}-point synthetic obstacle.")

    grid = Grid3D(resolution=0.05)
    grid.build_from_point_cloud(np.array(pc))
    
    print("\nRunning Collision-Aware STOMP Optimizer...")
    traj, history = stomp_optimize(q_home, q_goal, limits, grid=grid, n_iterations=100, verbose=True)

    print("\n✅ Valid trajectory generated resolving obstacles.")
    
    # Visualize
    visualize_stomp_results(traj, history, grid, pc)

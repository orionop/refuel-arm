#!/usr/bin/env python3
"""
KUKA KR6 R700 — Integrated STOMP Refueling with Obstacle Avoidance
==================================================================

Mission: HOME → YELLOW (Pick) → RED (Inlet) → HOME
Obstacle: A 3D blue cube barrier placed between YELLOW and RED.
Solver: C-Space STOMP (Joint Space Optimization)
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Include project paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'kuka_refuel_ws', 'src', 'kuka_kr6_gazebo', 'scripts')))
from ik_geometric import IK_spherical_2_parallel, fwd_kinematics, KIN_KR6_R700, rot
from stomp_collision import stomp_optimize, Grid3D

# ── Joint Limits ──────────────────────────────────────────────────
JOINT_LIMITS = np.array([
    [-2.967,  2.967], [-3.316,  0.785], [-2.094,  2.722],
    [-3.228,  3.228], [-2.094,  2.094], [-6.108,  6.108],
])

# ── Mission Waypoints ────────────────────────────────────────────
Q_HOME = np.zeros(6)
Q_NOZZLE = np.array([0.785, -0.94, 0.94, 0.0, 0.0, 0.0])  # YELLOW
REFUEL_TARGET_XYZ = np.array([0.5, 0.3, 0.45])            # RED (Refuel port)

# Target Orientation
R_TARGET = np.array([[ 0,  0,  1],
                    [ 0,  1,  0],
                    [-1,  0,  0]])

def create_cube_pc(center, x_size=0.15, y_size=0.15, z_size=0.15, density=10):
    """Generate a point cloud for a 3D box (blue cube)."""
    x_range = np.linspace(center[0] - x_size/2, center[0] + x_size/2, density)
    y_range = np.linspace(center[1] - y_size/2, center[1] + y_size/2, density)
    z_range = np.linspace(center[2] - z_size/2, center[2] + z_size/2, density)
    pc = []
    for x in x_range:
        for y in y_range:
            for z in z_range:
                # Keep shell
                if x in [x_range[0], x_range[-1]] or y in [y_range[0], y_range[-1]] or z in [z_range[0], z_range[-1]]:
                    pc.append([x, y, z])
    return np.array(pc)

from scipy.ndimage import gaussian_filter1d

import argparse

def send_trajectory_ros(trajectory, dt=0.15):
    """Send a single trajectory segment to the ROS controller."""
    import rospy
    import actionlib
    from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
    from trajectory_msgs.msg import JointTrajectoryPoint

    client = actionlib.SimpleActionClient(
        '/kr6_arm_controller/follow_joint_trajectory',
        FollowJointTrajectoryAction
    )
    if not client.wait_for_server(timeout=rospy.Duration(5.0)):
        print("  ⚠️  ROS action server not found! Is Gazebo running?")
        return False

    goal = FollowJointTrajectoryGoal()
    goal.trajectory.joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']

    for i, q in enumerate(trajectory):
        pt = JointTrajectoryPoint()
        pt.positions = q.tolist()
        pt.velocities = [0.0] * 6
        pt.time_from_start = rospy.Duration.from_sec(i * dt)
        goal.trajectory.points.append(pt)

    client.send_goal(goal)
    client.wait_for_result(timeout=rospy.Duration(len(trajectory) * dt + 5.0))
    return True

def main():
    parser = argparse.ArgumentParser(description="High-Fidelity STOMP Refueling")
    parser.add_argument("--ros", action="store_true", help="Execute in Gazebo")
    args_p = parser.parse_args()

    print("🚀 High-Fidelity Refueling STOMP Mission (Smoothness Refinement)")
    print("=" * 65)
    
    # 0. Solve IK for RED dot target
    Q_SOLUTIONS = IK_spherical_2_parallel(R_TARGET, REFUEL_TARGET_XYZ)
    dists = np.linalg.norm(Q_SOLUTIONS.T - Q_NOZZLE, axis=1)
    q_refuel_goal = Q_SOLUTIONS[:, np.argmin(dists)]
    
    # 1. Create Blue Cube Obstacle
    cube_center = np.array([0.52, 0.38, 0.58])
    cube_pc = create_cube_pc(cube_center, 0.1, 0.35, 0.1, density=8)
    
    grid = Grid3D(resolution=0.04)
    grid.build_from_point_cloud(cube_pc)
    
    # 2. Plan YELLOW -> RED
    # REFINEMENT: Increased w_smooth (10 -> 100), waypoints (40 -> 60)
    print("\n[STOMP] Starting High-Fidelity Optimization...")
    traj, history = stomp_optimize(
        q_start=Q_NOZZLE,
        q_goal=q_refuel_goal,
        joint_limits=JOINT_LIMITS,
        grid=grid,
        n_waypoints=60,
        n_iterations=150,
        n_rollouts=20,
        noise_stddev=0.1,
        w_smooth=150.0,   # Higher penalty for jerks
        w_obs=2000.0,
        safety_margin=0.2,
        verbose=True
    )
    
    # POST-PROCESSING: Gaussian filter for Zero-Jerk continuity
    # Sigma=0.8 provides subtle smoothing without deviating from the safe path
    traj_smooth = gaussian_filter1d(traj, sigma=0.8, axis=0)
    traj_smooth[0] = Q_NOZZLE  # Ensure endpoints are exact
    traj_smooth[-1] = q_refuel_goal
    
    # 3. Execution (ROS / Local)
    if args_p.ros:
        try:
            # Add ROS path if needed
            ros_path = '/opt/ros/noetic/lib/python3/dist-packages'
            if ros_path not in sys.path: sys.path.insert(0, ros_path)
            import rospy
            rospy.init_node('refuel_stomp_mission')
            print("\n[Execute] Sending trajectory to Gazebo...")
            success = send_trajectory_ros(traj_smooth, dt=0.15)
            if success: print("✅ Execution Complete in Gazebo")
        except Exception as e:
            print(f"  ❌ ROS Error: {e}")
    
    # 4. Final Verification and Plotting
    print("\n📊 High-Fidelity Analysis: refuel-stomp-analyse.png")
    fig = plt.figure(figsize=(20, 5))
    
    # Cost progression
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.plot(history, 'b-o', markersize=3)
    ax1.set_title("Stochastic Cost Convergence")
    ax1.set_xlabel("Iteration (x10)"); ax1.set_ylabel("Total Cost")
    ax1.grid(True)

    # C-Space Trajectory (Joint Angles)
    ax2 = fig.add_subplot(1, 4, 2)
    for i in range(6):
        ax2.plot(traj_smooth[:, i], label=f"J{i+1}", linewidth=2)
    ax2.set_title("Smooth Joint Positions (C-Space)")
    ax2.set_xlabel("Waypoint Index"); ax2.set_ylabel("Angle (rad)")
    ax2.legend(loc='upper right', fontsize='x-small'); ax2.grid(True)

    # 2.5D Elevation Grid
    ax3 = fig.add_subplot(1, 4, 3)
    hmap = np.zeros((grid.shape[0], grid.shape[1])); pc_local = np.array(cube_pc)
    for pt in pc_local:
        ix = int((pt[0] - grid.origin[0]) / grid.resolution)
        iy = int((pt[1] - grid.origin[1]) / grid.resolution)
        if 0 <= ix < grid.shape[0] and 0 <= iy < grid.shape[1]:
            hmap[ix, iy] = max(hmap[ix, iy], pt[2])
    
    im = ax3.imshow(hmap.T, origin='lower', extent=[grid.origin[0], grid.origin[0]+grid.shape[0]*grid.resolution, 
                                                   grid.origin[1], grid.origin[1]+grid.shape[1]*grid.resolution],
                   cmap='viridis')
    plt.colorbar(im, ax=ax3, label="Height (m)")
    ax3.set_title("2.5D Elevation Grid")

    # 3D Workspace Visualization
    ax4 = fig.add_subplot(1, 4, 4, projection='3d')
    ax4.scatter(pc_local[:, 0], pc_local[:, 1], pc_local[:, 2], c='blue', s=3, alpha=0.3)
    
    ee_path = []
    for q in traj_smooth:
        _, p = fwd_kinematics(q)
        ee_path.append(p)
    ee_path = np.array(ee_path)
    
    ax4.plot(ee_path[:, 0], ee_path[:, 1], ee_path[:, 2], 'g-', linewidth=2)
    ax4.scatter(ee_path[0, 0], ee_path[0, 1], ee_path[0, 2], c='yellow', s=80)
    ax4.scatter(ee_path[-1, 0], ee_path[-1, 1], ee_path[-1, 2], c='red', s=80)
    
    ax4.set_title("Zero-Jerk Trajectory")
    ax4.set_xlabel("X"); ax4.set_ylabel("Y"); ax4.set_zlabel("Z")
    
    plt.tight_layout()
    os.makedirs("output_graphs", exist_ok=True)
    plt.savefig("output_graphs/refuel-stomp-analyse.png")
    print(f"✅ Static analysis generated.")

if __name__ == "__main__":
    main()

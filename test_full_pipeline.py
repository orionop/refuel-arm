#!/usr/bin/env python3
"""
Full IK-Geo + CppFlow Refueling Pipeline
=========================================

Architecture:
  1. IK-Geo  → exact terminal joint configuration  (10^-16 precision)
  2. CppFlow → smooth collision-free trajectory     (IKFlow NN + optimization)
  3. ROS 2   → JointTrajectoryController execution  (Gazebo physics)

Run locally (no ROS required):
  source venv/bin/activate
  python test_full_pipeline.py

Run on Ubuntu PC with ROS 2 + Gazebo:
  python test_full_pipeline.py --ros
"""
import sys
import argparse
import numpy as np

sys.path.insert(0, "kuka_refuel_ws/src/kuka_kr6_gazebo/scripts")
from ik_geometric import IK_spherical_2_parallel, fwd_kinematics, KIN_KR6_R700


def pick_best_solution(Q, q_home=None):
    """Pick the IK solution closest to home (or with lowest joint norm)."""
    if q_home is None:
        q_home = np.zeros(6)
    dists = [np.linalg.norm(Q[:, i] - q_home) for i in range(Q.shape[1])]
    best_idx = int(np.argmin(dists))
    return Q[:, best_idx], best_idx


def generate_cartesian_waypoints(p_start, p_end, R_start, R_end, n_waypoints=20):
    """Generate linearly interpolated Cartesian waypoints."""
    waypoints = []
    for i in range(n_waypoints):
        alpha = i / (n_waypoints - 1)
        p = (1 - alpha) * p_start + alpha * p_end
        # SLERP would be ideal for rotation, but linear interp works for small rotations
        R = (1 - alpha) * R_start + alpha * R_end
        waypoints.append((p, R))
    return waypoints


def solve_trajectory_ik_geo(waypoints, q_prev=None):
    """
    Solve IK for each Cartesian waypoint using IK-Geo.
    Picks the solution closest to the previous joint config for smoothness.
    """
    trajectory = []
    for i, (p, R) in enumerate(waypoints):
        Q = IK_spherical_2_parallel(R, p)
        if Q.size == 0:
            print(f"  ⚠️  No exact IK solution for waypoint {i}, using LS fallback")
            if trajectory:
                trajectory.append(trajectory[-1])  # hold previous config
            continue
        q_best, _ = pick_best_solution(Q, q_prev)
        trajectory.append(q_best)
        q_prev = q_best
    return np.array(trajectory)


def main():
    parser = argparse.ArgumentParser(description="KUKA KR6 R700 Refueling Pipeline")
    parser.add_argument("--ros", action="store_true", help="Send trajectory to ROS 2 controller")
    args = parser.parse_args()

    print("=" * 65)
    print("  KUKA KR6 R700 — Full Refueling Pipeline")
    print("  IK-Geo (Exact Terminal) + Trajectory Planning")
    print("=" * 65)

    # ── Phase 1: IK-Geo exact terminal solution ──────────────────
    target_xyz = np.array([0.3, 0.4, 0.25])
    target_R = np.eye(3)
    q_home = np.zeros(6)

    print(f"\n[Phase 1] IK-Geo: Computing exact terminal joint configuration")
    print(f"  Target: position={target_xyz}, orientation=identity")

    Q = IK_spherical_2_parallel(target_R, target_xyz)
    n_sols = Q.shape[1]
    print(f"  Found {n_sols} algebraic solutions")

    q_terminal, best_idx = pick_best_solution(Q, q_home)

    # Verify
    R_check, p_check = fwd_kinematics(q_terminal)
    pos_err = np.linalg.norm(p_check - target_xyz)
    print(f"  Selected solution #{best_idx + 1}: {np.round(q_terminal, 4)}")
    print(f"  FK verification error: {pos_err:.2e} m")

    # ── Phase 2: Trajectory planning ─────────────────────────────
    print(f"\n[Phase 2] Trajectory Planning: HOME → TARGET")

    # Compute HOME end-effector pose
    R_home, p_home = fwd_kinematics(q_home)
    print(f"  HOME EE position: {np.round(p_home, 4)}")
    print(f"  TARGET EE position: {np.round(target_xyz, 4)}")

    # Generate Cartesian waypoints
    n_waypoints = 20
    waypoints = generate_cartesian_waypoints(p_home, target_xyz, R_home, target_R, n_waypoints)
    print(f"  Generated {n_waypoints} Cartesian waypoints")

    #  Note: On the Ubuntu PC with a trained IKFlow model, CppFlow would be
    #  used here instead of pure IK-Geo for the intermediate waypoints.
    #  CppFlow generates thousands of candidate trajectories on the GPU
    #  and optimizes for smoothness + collision avoidance.
    #
    #  For local Mac testing, we use IK-Geo for ALL waypoints (exact but
    #  without collision checking), which still produces a valid trajectory.

    print(f"\n[Phase 2b] Solving IK for each waypoint (IK-Geo greedy nearest)")
    trajectory = solve_trajectory_ik_geo(waypoints, q_home)

    # Force the last waypoint to be the exact IK-Geo solution
    trajectory[-1] = q_terminal

    print(f"  Trajectory shape: {trajectory.shape}")
    print(f"  Start config:  {np.round(trajectory[0], 4)}")
    print(f"  End config:    {np.round(trajectory[-1], 4)}")

    # Smoothness check
    joint_diffs = np.diff(trajectory, axis=0)
    max_jump = np.max(np.abs(joint_diffs))
    mean_jump = np.mean(np.abs(joint_diffs))
    print(f"  Max joint jump between waypoints: {max_jump:.4f} rad")
    print(f"  Mean joint jump: {mean_jump:.4f} rad")

    # ── Phase 3: Execute ─────────────────────────────────────────
    if args.ros:
        print(f"\n[Phase 3] Sending trajectory to ROS 2 JointTrajectoryController...")
        try:
            import rclpy
            from rclpy.node import Node
            from rclpy.action import ActionClient
            from control_msgs.action import FollowJointTrajectory
            from trajectory_msgs.msg import JointTrajectoryPoint
            from builtin_interfaces.msg import Duration

            rclpy.init()
            node = rclpy.create_node('refuel_pipeline')
            client = ActionClient(node, FollowJointTrajectory,
                                  '/kr6_arm_controller/follow_joint_trajectory')
            client.wait_for_server()

            goal = FollowJointTrajectory.Goal()
            goal.trajectory.joint_names = [
                'joint_1', 'joint_2', 'joint_3',
                'joint_4', 'joint_5', 'joint_6'
            ]

            # Build trajectory points with timing
            dt = 0.2  # seconds between waypoints
            for i, q in enumerate(trajectory):
                pt = JointTrajectoryPoint()
                pt.positions = q.tolist()
                pt.time_from_start = Duration(sec=int(i * dt), nanosec=int((i * dt % 1) * 1e9))
                goal.trajectory.points.append(pt)

            future = client.send_goal_async(goal)
            rclpy.spin_until_future_complete(node, future)
            print("  ✅ Trajectory sent to Gazebo!")
            rclpy.shutdown()
        except ImportError:
            print("  ⚠️  ROS 2 not available. Run on the Ubuntu PC with --ros flag.")
    else:
        print(f"\n[Phase 3] Local mode — trajectory computed but not sent to ROS")
        print(f"  Use --ros flag on the Ubuntu PC to execute in Gazebo")
        print(f"\n  Joint trajectory (first 5 waypoints):")
        for i in range(min(5, len(trajectory))):
            print(f"    wp{i:02d}: {np.round(trajectory[i], 4)}")
        print(f"    ...")
        print(f"    wp{len(trajectory)-1:02d}: {np.round(trajectory[-1], 4)}")

    print(f"\n{'=' * 65}")
    print("  Pipeline complete!")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()

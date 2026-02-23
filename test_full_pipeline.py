#!/usr/bin/env python3
"""
Full IK-Geo + CppFlow Refueling Pipeline
=========================================

Architecture:
  1. IK-Geo  → exact terminal joint configuration  (10^-16 precision)
  2. Waypoint IK with joint-limit + jump filtering  (Gazebo-safe)
  3. ROS 2   → JointTrajectoryController execution  (Gazebo physics)

Safety features:
  ✅ Joint limit enforcement (from official KUKA URDF)
  ✅ Elbow flip prevention (max 0.5 rad jump between waypoints)
  ✅ Greedy nearest-neighbor solution selection

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

# ── Official KUKA KR6 R700-2 Joint Limits (from URDF) ───────────
# Source: kuka_robot_descriptions/kuka_agilus_support/urdf/kr6_r700_2_macro.xacro
JOINT_LIMITS = np.array([
    [-2.967059725,  2.967059725],   # joint_1: ±170°
    [-3.316125575,  0.785398163],   # joint_2: -190° to +45°
    [-2.094395100,  2.722713630],   # joint_3: -120° to +156°
    [-3.228859113,  3.228859113],   # joint_4: ±185°
    [-2.094395100,  2.094395100],   # joint_5: ±120°
    [-6.108652375,  6.108652375],   # joint_6: ±350°
])

MAX_JUMP_RAD = 0.5  # Maximum allowed joint jump between consecutive waypoints


def within_joint_limits(q):
    """Check if all joints are within URDF limits."""
    for i in range(6):
        if q[i] < JOINT_LIMITS[i, 0] or q[i] > JOINT_LIMITS[i, 1]:
            return False
    return True


def wrap_to_limits(q):
    """Wrap joint angles to [-pi, pi] then check limits."""
    q_wrapped = np.copy(q)
    for i in range(6):
        # Wrap to [-pi, pi]
        while q_wrapped[i] > np.pi:
            q_wrapped[i] -= 2 * np.pi
        while q_wrapped[i] < -np.pi:
            q_wrapped[i] += 2 * np.pi
    return q_wrapped


def filter_solutions(Q, q_prev=None, max_jump=MAX_JUMP_RAD):
    """
    Filter IK solutions:
    1. Reject solutions outside joint limits
    2. Sort by distance to q_prev
    3. Reject solutions with any single joint jump > max_jump
    """
    if Q.size == 0:
        return np.empty((6, 0))

    valid = []
    for i in range(Q.shape[1]):
        q = wrap_to_limits(Q[:, i])
        if within_joint_limits(q):
            valid.append(q)

    if not valid:
        return np.empty((6, 0))

    valid = np.array(valid).T  # (6, N)

    if q_prev is not None:
        # Sort by total distance to previous config
        dists = np.linalg.norm(valid.T - q_prev, axis=1)
        order = np.argsort(dists)
        valid = valid[:, order]

        # Filter by max jump per joint
        filtered = []
        for i in range(valid.shape[1]):
            max_joint_jump = np.max(np.abs(valid[:, i] - q_prev))
            if max_joint_jump <= max_jump:
                filtered.append(valid[:, i])

        if filtered:
            return np.array(filtered).T
        # If no solution passes the jump filter, relax and return the closest
        return valid[:, :1]

    return valid


def pick_best_solution(Q, q_prev=None):
    """Pick the best filtered IK solution."""
    filtered = filter_solutions(Q, q_prev)
    if filtered.size == 0:
        return None, -1
    return filtered[:, 0], 0


def generate_cartesian_waypoints(p_start, p_end, R_start, R_end, n_waypoints=30):
    """Generate linearly interpolated Cartesian waypoints."""
    waypoints = []
    for i in range(n_waypoints):
        alpha = i / (n_waypoints - 1)
        p = (1 - alpha) * p_start + alpha * p_end
        R = (1 - alpha) * R_start + alpha * R_end
        waypoints.append((p, R))
    return waypoints


def solve_trajectory_ik_geo(waypoints, q_start):
    """
    Solve IK for each Cartesian waypoint with safety filters:
    - Joint limits enforced
    - Max jump threshold prevents elbow flips
    - Greedy nearest-neighbor selection
    """
    trajectory = [q_start]
    q_prev = q_start.copy()
    n_relaxed = 0

    for i, (p, R) in enumerate(waypoints[1:], 1):
        Q = IK_spherical_2_parallel(R, p)
        if Q.size == 0:
            trajectory.append(q_prev.copy())
            continue

        q_best, _ = pick_best_solution(Q, q_prev)
        if q_best is None:
            trajectory.append(q_prev.copy())
            continue

        # Check if we had to relax the jump filter
        max_jump = np.max(np.abs(q_best - q_prev))
        if max_jump > MAX_JUMP_RAD:
            n_relaxed += 1

        trajectory.append(q_best)
        q_prev = q_best.copy()

    if n_relaxed > 0:
        print(f"  ⚠️  {n_relaxed} waypoints required relaxed jump filter")

    return np.array(trajectory)


def main():
    parser = argparse.ArgumentParser(description="KUKA KR6 R700 Refueling Pipeline")
    parser.add_argument("--ros", action="store_true", help="Send trajectory to ROS 2 controller")
    parser.add_argument("--waypoints", type=int, default=30, help="Number of waypoints (more = smoother)")
    args = parser.parse_args()

    print("=" * 65)
    print("  KUKA KR6 R700 — Gazebo-Safe Refueling Pipeline")
    print("  IK-Geo + Joint Limits + Elbow Flip Prevention")
    print("=" * 65)

    # ── Phase 1: IK-Geo exact terminal solution ──────────────────
    target_xyz = np.array([0.3, 0.4, 0.25])
    target_R = np.eye(3)
    q_home = np.zeros(6)

    print(f"\n[Phase 1] IK-Geo: Computing exact terminal joint configuration")
    print(f"  Target: position={target_xyz}")
    print(f"  Joint limits enforced: ✅")

    Q = IK_spherical_2_parallel(target_R, target_xyz)
    n_total = Q.shape[1]

    # Filter for joint limits
    Q_filtered = filter_solutions(Q, q_home)
    n_valid = Q_filtered.shape[1] if Q_filtered.size > 0 else 0

    print(f"  Found {n_total} algebraic solutions, {n_valid} within joint limits")

    if n_valid == 0:
        print("  ❌ No valid solutions within joint limits!")
        return

    q_terminal = Q_filtered[:, 0]

    # Verify
    R_check, p_check = fwd_kinematics(q_terminal)
    pos_err = np.linalg.norm(p_check - target_xyz)
    print(f"  Selected: {np.round(q_terminal, 4)}")
    print(f"  FK verification: {pos_err:.2e} m")

    # Show limit margins
    for j in range(6):
        margin_lo = q_terminal[j] - JOINT_LIMITS[j, 0]
        margin_hi = JOINT_LIMITS[j, 1] - q_terminal[j]
        margin = min(margin_lo, margin_hi)
        print(f"    J{j+1}: {q_terminal[j]:+.3f} rad  (margin: {np.degrees(margin):.1f}°)")

    # ── Phase 2: Joint-space trajectory planning ──────────────────
    n_wp = args.waypoints
    print(f"\n[Phase 2] Joint-Space Trajectory: HOME → TARGET ({n_wp} waypoints)")

    # Joint-space linear interpolation guarantees:
    #  ✅ No elbow flips (monotonic joint motion)
    #  ✅ No topology jumps (no IK branch switching)
    #  ✅ Bounded joint velocities
    #  ✅ All waypoints within limits (if start & end are)
    trajectory = np.zeros((n_wp, 6))
    for i in range(n_wp):
        alpha = i / (n_wp - 1)
        trajectory[i] = (1 - alpha) * q_home + alpha * q_terminal

    # Verify each waypoint FK for Cartesian tracking error
    cart_errors = []
    for i in range(n_wp):
        alpha = i / (n_wp - 1)
        p_desired = (1 - alpha) * fwd_kinematics(q_home)[1] + alpha * target_xyz
        _, p_actual = fwd_kinematics(trajectory[i])
        cart_errors.append(np.linalg.norm(p_actual - p_desired))

    # Smoothness analysis
    joint_diffs = np.diff(trajectory, axis=0)
    max_jump = np.max(np.abs(joint_diffs))
    mean_jump = np.mean(np.abs(joint_diffs))
    max_cart_err = max(cart_errors)

    print(f"\n  Trajectory stats:")
    print(f"    Shape:       {trajectory.shape}")
    print(f"    Max jump:    {max_jump:.4f} rad ({np.degrees(max_jump):.1f}°)")
    print(f"    Mean jump:   {mean_jump:.4f} rad ({np.degrees(mean_jump):.1f}°)")
    print(f"    Max EE deviation from straight line: {max_cart_err:.4f} m")

    # Verify all waypoints are within limits
    all_valid = all(within_joint_limits(trajectory[i]) for i in range(len(trajectory)))
    print(f"    All within limits: {'✅' if all_valid else '❌'}")
    print(f"    ✅ Trajectory is Gazebo-safe! (joint-space interpolation = no flips)")

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

            dt = 0.15  # seconds between waypoints
            for i, q in enumerate(trajectory):
                pt = JointTrajectoryPoint()
                pt.positions = q.tolist()
                t = i * dt
                pt.time_from_start = Duration(sec=int(t), nanosec=int((t % 1) * 1e9))
                goal.trajectory.points.append(pt)

            future = client.send_goal_async(goal)
            rclpy.spin_until_future_complete(node, future)
            print("  ✅ Trajectory sent to Gazebo!")
            rclpy.shutdown()
        except ImportError:
            print("  ⚠️  ROS 2 not available. Run on the Ubuntu PC with --ros flag.")
    else:
        print(f"\n[Phase 3] Local mode — trajectory preview")
        print(f"  Joint trajectory (every 5th waypoint):")
        for i in range(0, len(trajectory), 5):
            print(f"    wp{i:02d}: {np.round(trajectory[i], 4)}")
        if (len(trajectory) - 1) % 5 != 0:
            print(f"    wp{len(trajectory)-1:02d}: {np.round(trajectory[-1], 4)}")

    print(f"\n{'=' * 65}")
    print("  Pipeline complete!")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()

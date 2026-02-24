#!/usr/bin/env python3
"""
KUKA KR6 R700 — Autonomous Refueling Mission
=============================================

Mission sequence:
  REST → YELLOW (pick nozzle) → REST → RED (refuel, 5s dwell) → REST → YELLOW (return nozzle) → REST

Components:
  1. IK-Geo  → exact terminal joint configuration  (10^-16 precision)
  2. STOMP   → smooth trajectory optimization       (Kalakrishnan, ICRA 2011)
  3. ROS Noetic → JointTrajectoryController execution (Gazebo physics)

Run locally:   python3 test_full_pipeline.py
Run in Gazebo: python3 test_full_pipeline.py --ros
"""
import sys
import time
import argparse
import numpy as np

sys.path.insert(0, "kuka_refuel_ws/src/kuka_kr6_gazebo/scripts")
from ik_geometric import IK_spherical_2_parallel, fwd_kinematics, KIN_KR6_R700
from stomp_planner import stomp_optimize

# ── Official KUKA KR6 R700-2 Joint Limits (from URDF) ───────────
JOINT_LIMITS = np.array([
    [-2.967059725,  2.967059725],   # joint_1: ±170°
    [-3.316125575,  0.785398163],   # joint_2: -190° to +45°
    [-2.094395100,  2.722713630],   # joint_3: -120° to +156°
    [-3.228859113,  3.228859113],   # joint_4: ±185°
    [-2.094395100,  2.094395100],   # joint_5: ±120°
    [-6.108652375,  6.108652375],   # joint_6: ±350°
])

# ── Mission Waypoints ────────────────────────────────────────────
Q_HOME = np.zeros(6)                                         # REST position
Q_NOZZLE = np.array([-0.785, 0.5, 1.0, 0.0, 0.5, 0.0])     # YELLOW dot (realistic dock to the right)
REFUEL_TARGET_XYZ = np.array([0.45, 0.0, 0.3])              # RED dot (front, slightly lower)
REFUEL_TARGET_R = np.eye(3)                                  # Tool orientation
DWELL_TIME = 10.0                                             # Seconds to hold at refuel position


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
        while q_wrapped[i] > np.pi:
            q_wrapped[i] -= 2 * np.pi
        while q_wrapped[i] < -np.pi:
            q_wrapped[i] += 2 * np.pi
    return q_wrapped


def filter_solutions(Q, q_prev=None, max_jump=0.5):
    """Filter IK solutions by joint limits and proximity."""
    if Q.size == 0:
        return np.empty((6, 0))

    valid = []
    for i in range(Q.shape[1]):
        q = wrap_to_limits(Q[:, i])
        if within_joint_limits(q):
            valid.append(q)

    if not valid:
        return np.empty((6, 0))

    valid = np.array(valid).T

    if q_prev is not None:
        dists = np.linalg.norm(valid.T - q_prev, axis=1)
        order = np.argsort(dists)
        valid = valid[:, order]

    return valid


def plan_segment(q_start, q_goal, name, n_waypoints=30):
    """Plan a STOMP-optimized trajectory between two joint configs."""
    print(f"\n  📍 Planning: {name}")
    trajectory = stomp_optimize(
        q_start=q_start,
        q_goal=q_goal,
        joint_limits=JOINT_LIMITS,
        n_waypoints=n_waypoints,
        n_iterations=80,
        n_rollouts=10,
        noise_stddev=0.08,
        verbose=False,
    )
    diffs = np.diff(trajectory, axis=0)
    max_jump = np.max(np.abs(diffs))
    all_valid = all(within_joint_limits(trajectory[i]) for i in range(len(trajectory)))
    print(f"     → {n_waypoints} waypoints, max_jump={np.degrees(max_jump):.1f}°, "
          f"limits OK: {'✅' if all_valid else '❌'}")
    return trajectory


def send_trajectory_ros(trajectory, dt=0.15):
    """Send a single trajectory segment to the ROS controller."""
    # Auto-add ROS Noetic Python path (needed when running inside a venv)
    import os
    ros_python = '/opt/ros/noetic/lib/python3/dist-packages'
    if ros_python not in sys.path and os.path.isdir(ros_python):
        sys.path.insert(0, ros_python)

    import rospy
    import actionlib
    from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
    from trajectory_msgs.msg import JointTrajectoryPoint

    client = actionlib.SimpleActionClient(
        '/kr6_arm_controller/follow_joint_trajectory',
        FollowJointTrajectoryAction
    )
    client.wait_for_server(timeout=rospy.Duration(5.0))

    goal = FollowJointTrajectoryGoal()
    goal.trajectory.joint_names = [
        'joint_1', 'joint_2', 'joint_3',
        'joint_4', 'joint_5', 'joint_6'
    ]

    for i, q in enumerate(trajectory):
        pt = JointTrajectoryPoint()
        pt.positions = q.tolist()
        pt.velocities = [0.0] * 6
        pt.time_from_start = rospy.Duration.from_sec(i * dt)
        goal.trajectory.points.append(pt)

    client.send_goal(goal)
    client.wait_for_result(timeout=rospy.Duration(len(trajectory) * dt + 10.0))
    return client.get_result()


def send_trajectory_rviz(trajectory, dt=0.15):
    """Publish a trajectory directly to RViz via /joint_states."""
    # Auto-add ROS Noetic Python path
    import os
    import sys
    ros_python = '/opt/ros/noetic/lib/python3/dist-packages'
    if ros_python not in sys.path and os.path.isdir(ros_python):
        sys.path.insert(0, ros_python)

    import rospy
    from sensor_msgs.msg import JointState

    # Create publisher if it doesn't exist yet
    if not hasattr(send_trajectory_rviz, "pub"):
        send_trajectory_rviz.pub = rospy.Publisher('/joint_states', JointState, queue_size=10)
        rospy.sleep(0.5)  # Wait for publisher connection

    msg = JointState()
    msg.name = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']

    rate = rospy.Rate(1.0 / dt)
    for q in trajectory:
        msg.header.stamp = rospy.Time.now()
        msg.position = q.tolist()
        send_trajectory_rviz.pub.publish(msg)
        rate.sleep()
    return True


def main():
    parser = argparse.ArgumentParser(description="KUKA KR6 R700 Refueling Mission")
    parser.add_argument("--ros", action="store_true", help="Execute on ROS Noetic + Gazebo")
    parser.add_argument("--rviz", action="store_true", help="Visualize trajectory purely in RViz (no Gazebo physics)")
    parser.add_argument("--waypoints", type=int, default=30, help="Waypoints per segment")
    args = parser.parse_args()

    print("=" * 65)
    print("  KUKA KR6 R700 — Autonomous Refueling Mission")
    print("  IK-Geo + STOMP + ROS Noetic")
    print("=" * 65)

    # ── Step 0: Verify waypoints ─────────────────────────────────
    print("\n[Setup] Verifying mission waypoints...")

    # Verify nozzle station (yellow dot) FK
    R_nozzle, p_nozzle = fwd_kinematics(Q_NOZZLE)
    print(f"  🟡 YELLOW (nozzle station): {np.round(p_nozzle, 4)} m")
    print(f"     Joint config: {np.round(Q_NOZZLE, 4)}")
    assert within_joint_limits(Q_NOZZLE), "Nozzle config violates joint limits!"

    # Solve IK for refueling target (red dot)
    print(f"  🔴 RED (refuel inlet):      {REFUEL_TARGET_XYZ} m")
    Q = IK_spherical_2_parallel(REFUEL_TARGET_R, REFUEL_TARGET_XYZ)
    Q_valid = filter_solutions(Q, Q_HOME)

    if Q_valid.size == 0:
        print("  ❌ No valid IK solution for refuel target!")
        return

    q_refuel = Q_valid[:, 0]
    R_check, p_check = fwd_kinematics(q_refuel)
    fk_err = np.linalg.norm(p_check - REFUEL_TARGET_XYZ)
    print(f"     IK-Geo: {Q.shape[1]} solutions, {Q_valid.shape[1]} valid")
    print(f"     Selected: {np.round(q_refuel, 4)}")
    print(f"     FK error: {fk_err:.2e} m")

    # ── Plan all 6 trajectory segments ───────────────────────────
    print(f"\n[Planning] STOMP trajectory optimization for 6 mission segments")
    n_wp = args.waypoints

    seg1 = plan_segment(Q_HOME,    Q_NOZZLE, "REST → YELLOW (pick up nozzle)",  n_wp)
    seg2 = plan_segment(Q_NOZZLE,  Q_HOME,   "YELLOW → REST (nozzle acquired)", n_wp)
    seg3 = plan_segment(Q_HOME,    q_refuel, "REST → RED (approach refuel)",     n_wp)
    seg4 = plan_segment(q_refuel,  Q_HOME,   "RED → REST (refueling done)",     n_wp)
    seg5 = plan_segment(Q_HOME,    Q_NOZZLE, "REST → YELLOW (return nozzle)",   n_wp)
    seg6 = plan_segment(Q_NOZZLE,  Q_HOME,   "YELLOW → REST (mission complete)", n_wp)

    all_segments = [
        ("REST → YELLOW (pick up nozzle)", seg1, None, 0.12),
        ("YELLOW → REST (nozzle acquired)", seg2, None, 0.20),
        ("REST → RED (approach refuel)", seg3, None, 0.25),
        ("🔴 REFUELING — holding position", None, DWELL_TIME, None),
        ("RED → REST (refueling done)", seg4, None, 0.15),
        ("REST → YELLOW (return nozzle)", seg5, None, 0.12),
        ("YELLOW → REST (mission complete)", seg6, None, 0.10),
    ]

    # ── Execute ──────────────────────────────────────────────────
    if args.ros or args.rviz:
        mode_str = "Gazebo" if args.ros else "RViz"
        print(f"\n[Execute] Running mission visualization in {mode_str}")
        try:
            import os
            ros_python = '/opt/ros/noetic/lib/python3/dist-packages'
            if ros_python not in sys.path and os.path.isdir(ros_python):
                sys.path.insert(0, ros_python)
            import rospy
            from visualization_msgs.msg import Marker, MarkerArray
            rospy.init_node('refuel_mission', anonymous=True)

            if args.rviz:
                # Publish static markers for RViz
                marker_pub = rospy.Publisher('/visualization_marker_array', MarkerArray, queue_size=10)
                rospy.sleep(0.5)  # wait for connection
                
                ma = MarkerArray()
                
                # Yellow Nozzle Station (tall thin cylinder)
                m_y = Marker()
                m_y.header.frame_id = "world"  # same as robot base anchor
                m_y.ns = "stations"
                m_y.id = 0
                m_y.type = Marker.CYLINDER
                m_y.action = Marker.ADD
                m_y.pose.position.x = p_nozzle[0]
                m_y.pose.position.y = p_nozzle[1]
                m_y.pose.position.z = p_nozzle[2] / 2.0  # rest on ground
                m_y.pose.orientation.w = 1.0
                m_y.scale.x = 0.05
                m_y.scale.y = 0.05
                m_y.scale.z = p_nozzle[2]
                m_y.color.r = 1.0; m_y.color.g = 1.0; m_y.color.b = 0.0; m_y.color.a = 0.8
                ma.markers.append(m_y)

                # Red Refuel Inlet (sphere)
                m_r = Marker()
                m_r.header.frame_id = "world"
                m_r.ns = "stations"
                m_r.id = 1
                m_r.type = Marker.SPHERE
                m_r.action = Marker.ADD
                m_r.pose.position.x = REFUEL_TARGET_XYZ[0]
                m_r.pose.position.y = REFUEL_TARGET_XYZ[1]
                m_r.pose.position.z = REFUEL_TARGET_XYZ[2]
                m_r.pose.orientation.w = 1.0
                m_r.scale.x = 0.1
                m_r.scale.y = 0.1
                m_r.scale.z = 0.1
                m_r.color.r = 1.0; m_r.color.g = 0.0; m_r.color.b = 0.0; m_r.color.a = 0.8
                ma.markers.append(m_r)

                marker_pub.publish(ma)

            for i, (name, traj, dwell, dt) in enumerate(all_segments, 1):
                print(f"\n  Step {i}/7: {name}")
                if dwell is not None:
                    print(f"     ⏱️  Holding for {dwell:.0f} seconds...")
                    rospy.sleep(dwell)
                    print(f"     ✅ Dwell complete")
                else:
                    if args.ros:
                        result = send_trajectory_ros(traj, dt=dt)
                    else:
                        result = send_trajectory_rviz(traj, dt=dt)

                    if result:
                        print(f"     ✅ Segment executed")
                    else:
                        print(f"     ⚠️  Segment failed/timed out")

            print(f"\n  🎉 Mission complete!")
        except ImportError:
            print("  ⚠️  ROS Noetic not available. Source /opt/ros/noetic/setup.bash first.")
    else:
        print(f"\n[Preview] Mission trajectory summary")
        total_waypoints = 0
        for i, (name, traj, dwell, dt) in enumerate(all_segments, 1):
            if dwell is not None:
                print(f"  Step {i}/7: {name} ({dwell:.0f}s dwell)")
            else:
                total_waypoints += len(traj)
                start = np.round(traj[0], 3)
                end = np.round(traj[-1], 3)
                print(f"  Step {i}/7: {name}")
                print(f"           start={start}")
                print(f"           end  ={end}")
        print(f"\n  Total waypoints: {total_waypoints}")
        print(f"  Total segments: 6 motion + 1 dwell ({DWELL_TIME:.0f}s)")

    print(f"\n{'=' * 65}")
    print("  Mission complete!")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()

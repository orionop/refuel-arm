#!/usr/bin/env python3
"""
KUKA KR6 R700 — Autonomous Refueling Mission (Consolidated)
=============================================================

Mission sequence:
  HOME → Pre-approach → Inlet (slow insert, 5 s dwell) → Pre-approach → HOME

Components:
  1. IK-Geo          → exact closed-form IK  (10^-16 precision)
  2. STOMP            → smooth trajectory optimisation  (Kalakrishnan, ICRA 2011)
  3. Elastic Strips   → reactive obstacle deformation   (Brock & Khatib, 2002)
  4. Car Model        → elevated platform + car SDF with fuel inlet
  5. Obstacle Detector→ simulated sensor via /gazebo/model_states

Run locally:   python3 refuel_mission.py
Run in Gazebo: python3 refuel_mission.py --ros
Run in RViz:   python3 refuel_mission.py --rviz
"""
import sys
import os
import time
import argparse
import numpy as np

# ── Path setup ────────────────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), 'kuka_refuel_ws', 'src',
    'kuka_kr6_gazebo', 'scripts')))

from ik_geometric import IK_spherical_2_parallel, fwd_kinematics, rot
from stomp_collision import stomp_optimize
from elastic_strips import elastic_strip_deform, plot_elastic_comparison
from car_model import (
    get_inlet_pose, get_preapproach_pose,
    spawn_car_gazebo, get_car_rviz_markers,
    CAR_POSITION_DEFAULT, CAR_YAW_DEFAULT,
)
from obstacle_detector import (
    ObstacleDetector, DummyDetector,
    spawn_default_obstacle,
)

# ── Official KUKA KR6 R700-2 Joint Limits (from URDF) ────────────
JOINT_LIMITS = np.array([
    [-2.967059725,  2.967059725],   # joint_1: +/-170 deg
    [-3.316125575,  0.785398163],   # joint_2: -190 to +45 deg
    [-2.094395100,  2.722713630],   # joint_3: -120 to +156 deg
    [-3.228859113,  3.228859113],   # joint_4: +/-185 deg
    [-2.094395100,  2.094395100],   # joint_5: +/-120 deg
    [-6.108652375,  6.108652375],   # joint_6: +/-350 deg
])

Q_HOME     = np.array([0.0, -np.pi / 2, 0.0, 0.0, 0.0, 0.0])
DWELL_TIME = 5.0   # seconds at refuel position


# ── Utility (from test_full_pipeline.py, unchanged) ──────────────

def within_joint_limits(q):
    for i in range(6):
        if q[i] < JOINT_LIMITS[i, 0] or q[i] > JOINT_LIMITS[i, 1]:
            return False
    return True


def wrap_to_limits(q):
    q_w = np.copy(q)
    for i in range(6):
        while q_w[i] > np.pi:
            q_w[i] -= 2 * np.pi
        while q_w[i] < -np.pi:
            q_w[i] += 2 * np.pi
    return q_w


def filter_solutions(Q, q_prev=None):
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
        valid = valid[:, np.argsort(dists)]
    return valid


# ── Trajectory helpers ────────────────────────────────────────────

def plan_coarse(q_start, q_goal, obstacles, name, n_wp=30):
    """STOMP plan (informed with obstacles) + Elastic Strips refinement."""
    print(f"\n  Planning: {name}")
    traj = stomp_optimize(
        q_start=q_start,
        q_goal=q_goal,
        joint_limits=JOINT_LIMITS,
        simple_obstacles=obstacles or None,
        n_waypoints=n_wp,
        n_iterations=80,
        n_rollouts=10,
        noise_stddev=0.08,
        verbose=False,
    )
    diffs = np.diff(traj, axis=0)
    max_jump = np.max(np.abs(diffs))
    ok = all(within_joint_limits(traj[i]) for i in range(len(traj)))
    print(f"     STOMP: {n_wp} wp, max_jump={np.degrees(max_jump):.1f} deg, "
          f"limits {'OK' if ok else 'VIOLATED'}")

    if obstacles:
        traj, _ = elastic_strip_deform(
            traj, obstacles,
            n_iterations=200, alpha=0.02,
            k_contraction=1.5, k_repulsion=10.0,
            safety_margin=0.20, damping=0.90,
            verbose=False,
        )
        print(f"     Elastic Strips: deformed around {len(obstacles)} obstacle(s)")

    return traj


def plan_fine(q_start, q_goal, n_wp=20):
    """C-space linear interpolation for short precise motions."""
    traj = np.zeros((n_wp, 6))
    for i in range(n_wp):
        alpha = i / (n_wp - 1)
        traj[i] = (1 - alpha) * q_start + alpha * q_goal
    return traj


# ── ROS trajectory execution ─────────────────────────────────────

def _ensure_ros_path():
    ros_python = '/opt/ros/noetic/lib/python3/dist-packages'
    if ros_python not in sys.path and os.path.isdir(ros_python):
        sys.path.insert(0, ros_python)


def send_trajectory_ros(trajectory, dt=0.15):
    _ensure_ros_path()
    import rospy
    import actionlib
    from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
    from trajectory_msgs.msg import JointTrajectoryPoint

    client = actionlib.SimpleActionClient(
        '/kr6_arm_controller/follow_joint_trajectory',
        FollowJointTrajectoryAction)
    client.wait_for_server(timeout=rospy.Duration(5.0))

    goal = FollowJointTrajectoryGoal()
    goal.trajectory.joint_names = [
        'joint_1', 'joint_2', 'joint_3',
        'joint_4', 'joint_5', 'joint_6']
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
    _ensure_ros_path()
    import rospy
    from sensor_msgs.msg import JointState

    if not hasattr(send_trajectory_rviz, "pub"):
        send_trajectory_rviz.pub = rospy.Publisher(
            '/joint_states', JointState, queue_size=10)
        rospy.sleep(0.5)

    msg = JointState()
    msg.name = ['joint_1', 'joint_2', 'joint_3',
                'joint_4', 'joint_5', 'joint_6']
    rate = rospy.Rate(1.0 / dt)
    for q in trajectory:
        msg.header.stamp = rospy.Time.now()
        msg.position = q.tolist()
        send_trajectory_rviz.pub.publish(msg)
        rate.sleep()
    return True


# ── Analysis graphs (--analyze) ───────────────────────────────────

def run_analysis(q_start, q_goal, stomp_traj, elastic_traj, obstacles,
                 elastic_history):
    """Generate comparison graphs to output_graphs/."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from ik_geometric import KIN_KR6_R700

    kin = KIN_KR6_R700
    H, P = kin['H'], kin['P']

    def min_dist(q):
        R = np.eye(3); p = P[:, 0].copy()
        pts = []
        for j in range(6):
            R = R @ rot(H[:, j], q[j])
            p = p + R @ P[:, j + 1]
            if j in [2, 4, 5]:
                pts.append(p.copy())
        if len(pts) >= 3:
            pts.append(pts[0] + 0.5 * (pts[1] - pts[0]))
        md = float('inf')
        for pt in pts:
            for c, r in obstacles:
                d = np.linalg.norm(pt - c) - r
                if d < md:
                    md = d
        return md

    n_wp = len(stomp_traj)
    cspace = np.array([q_start + t / (n_wp - 1) * (q_goal - q_start)
                        for t in range(n_wp)])

    stomp_d = [min_dist(stomp_traj[i]) for i in range(n_wp)]
    cspace_d = [min_dist(cspace[i]) for i in range(n_wp)]

    fig, ax = plt.subplots(figsize=(10, 6))
    steps = np.arange(1, n_wp + 1)
    ax.plot(steps, cspace_d, 'r--', lw=2, label='Pure C-Space LERP')
    ax.plot(steps, stomp_d, 'b-', lw=2, label='STOMP Optimised')
    ax.axhline(0, color='k', ls=':', label='Collision Surface')
    ax.fill_between(steps, cspace_d, 0,
                    where=(np.array(cspace_d) < 0),
                    color='red', alpha=0.3, label='Collision Zone')
    ax.set_title("Obstacle Avoidance: C-Space LERP vs STOMP", fontweight='bold')
    ax.set_xlabel("Waypoint")
    ax.set_ylabel("Min Distance to Obstacle (m)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    os.makedirs("output_graphs", exist_ok=True)
    plt.savefig("output_graphs/stomp_vs_cspace_avoidance.png", dpi=200)
    plt.close()
    print("\n  Graph saved: output_graphs/stomp_vs_cspace_avoidance.png")

    if elastic_traj is not None and elastic_history is not None:
        plot_elastic_comparison(
            stomp_traj, elastic_traj, obstacles, elastic_history,
            save_path="output_graphs/elastic_strips_analysis.png")
        print("  Graph saved: output_graphs/elastic_strips_analysis.png")


# ── RViz marker publishing ────────────────────────────────────────

def publish_markers(car_pos, car_yaw, obstacles, segments):
    """Publish car, obstacles, and trajectory markers to RViz."""
    _ensure_ros_path()
    import rospy
    from visualization_msgs.msg import Marker, MarkerArray
    from geometry_msgs.msg import Point

    pub = rospy.Publisher('/visualization_marker_array', MarkerArray, queue_size=10)
    rospy.sleep(0.5)

    ma = MarkerArray()

    # Car + platform + inlet markers
    ma.markers.extend(get_car_rviz_markers(car_pos, car_yaw))

    # Obstacle spheres (blue, semi-transparent)
    for idx, (center, radius) in enumerate(obstacles):
        m = Marker()
        m.header.frame_id = "world"
        m.ns = "obstacles"
        m.id = 50 + idx
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = center[0]
        m.pose.position.y = center[1]
        m.pose.position.z = center[2]
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = radius * 2
        m.color.r = 0.0; m.color.g = 0.0; m.color.b = 1.0; m.color.a = 0.6
        ma.markers.append(m)

    # Full trajectory trace (white LINE_STRIP)
    m_path = Marker()
    m_path.header.frame_id = "world"
    m_path.ns = "trajectory"
    m_path.id = 100
    m_path.type = Marker.LINE_STRIP
    m_path.action = Marker.ADD
    m_path.pose.orientation.w = 1.0
    m_path.scale.x = 0.008
    m_path.color.r = 1.0; m_path.color.g = 1.0; m_path.color.b = 1.0; m_path.color.a = 0.8

    for label, traj, _ in segments:
        if traj is not None:
            for q in traj:
                _, p = fwd_kinematics(q)
                m_path.points.append(Point(x=p[0], y=p[1], z=p[2]))
    ma.markers.append(m_path)

    # Fine insertion segment (green LINE_STRIP)
    for label, traj, _ in segments:
        if 'Insert' in label and traj is not None:
            m_ins = Marker()
            m_ins.header.frame_id = "world"
            m_ins.ns = "trajectory"
            m_ins.id = 101
            m_ins.type = Marker.LINE_STRIP
            m_ins.action = Marker.ADD
            m_ins.pose.orientation.w = 1.0
            m_ins.scale.x = 0.012
            m_ins.color.r = 0.0; m_ins.color.g = 1.0; m_ins.color.b = 0.3; m_ins.color.a = 0.9
            for q in traj:
                _, p = fwd_kinematics(q)
                m_ins.points.append(Point(x=p[0], y=p[1], z=p[2]))
            ma.markers.append(m_ins)

    pub.publish(ma)
    print("  RViz markers published")


# ── Main Mission ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="KUKA KR6 R700 Autonomous Refueling Mission")
    parser.add_argument("--ros", action="store_true",
                        help="Execute on ROS Noetic + Gazebo")
    parser.add_argument("--rviz", action="store_true",
                        help="Visualise in RViz only (no Gazebo physics)")
    parser.add_argument("--analyze", action="store_true",
                        help="Generate comparison graphs to output_graphs/")
    parser.add_argument("--waypoints", type=int, default=30,
                        help="Waypoints per coarse segment")
    parser.add_argument("--car-x", type=float, default=CAR_POSITION_DEFAULT[0])
    parser.add_argument("--car-y", type=float, default=CAR_POSITION_DEFAULT[1])
    parser.add_argument("--car-yaw", type=float, default=CAR_YAW_DEFAULT)
    args = parser.parse_args()

    car_pos = np.array([args.car_x, args.car_y, CAR_POSITION_DEFAULT[2]])
    car_yaw = args.car_yaw
    n_wp = args.waypoints

    print("=" * 65)
    print("  KUKA KR6 R700 — Autonomous Refueling Mission")
    print("  IK-Geo + STOMP (Informed) + Elastic Strips (Reactive)")
    print("=" * 65)

    # ── Step 0: Compute inlet target from car pose ────────────────
    inlet_xyz, inlet_R = get_inlet_pose(car_pos, car_yaw)
    preapproach_xyz, _ = get_preapproach_pose(inlet_xyz, inlet_R)

    print(f"\n[Car]  Position: [{car_pos[0]:.2f}, {car_pos[1]:.2f}, {car_pos[2]:.2f}]")
    print(f"[Inlet] Position: [{inlet_xyz[0]:.3f}, {inlet_xyz[1]:.3f}, {inlet_xyz[2]:.3f}]")
    print(f"[Pre-approach]  : [{preapproach_xyz[0]:.3f}, {preapproach_xyz[1]:.3f}, "
          f"{preapproach_xyz[2]:.3f}]")

    # ── Step 1: IK-Geo for pre-approach and insertion ─────────────
    print("\n[IK-Geo] Solving for pre-approach pose...")
    Q_pre = IK_spherical_2_parallel(inlet_R, preapproach_xyz)
    Q_pre_valid = filter_solutions(Q_pre, Q_HOME)
    if Q_pre_valid.size == 0:
        print("  No valid IK solution for pre-approach!")
        return
    q_preapproach = Q_pre_valid[:, 0]

    R_chk, p_chk = fwd_kinematics(q_preapproach)
    print(f"     {Q_pre.shape[1]} solutions, {Q_pre_valid.shape[1]} valid")
    print(f"     Selected: {np.round(q_preapproach, 4)}")
    print(f"     FK error: {np.linalg.norm(p_chk - preapproach_xyz):.2e} m")

    print("\n[IK-Geo] Solving for insertion pose...")
    Q_ins = IK_spherical_2_parallel(inlet_R, inlet_xyz)
    Q_ins_valid = filter_solutions(Q_ins, q_preapproach)   # branch-consistent
    if Q_ins_valid.size == 0:
        print("  No valid IK solution for insertion!")
        return
    q_insertion = Q_ins_valid[:, 0]

    R_chk, p_chk = fwd_kinematics(q_insertion)
    print(f"     {Q_ins.shape[1]} solutions, {Q_ins_valid.shape[1]} valid")
    print(f"     Selected: {np.round(q_insertion, 4)}")
    print(f"     FK error: {np.linalg.norm(p_chk - inlet_xyz):.2e} m")

    # ── Step 2: Obstacle detector setup ───────────────────────────
    use_ros = args.ros or args.rviz
    if use_ros:
        _ensure_ros_path()
        import rospy
        rospy.init_node('refuel_mission', anonymous=True)

    if args.ros:
        spawn_car_gazebo(car_pos, car_yaw)
        spawn_default_obstacle()
        detector = ObstacleDetector()
        rospy.sleep(1.0)  # let model_states populate
    else:
        detector = DummyDetector()

    # Initial detection pass from HOME
    detector.update(Q_HOME)

    # ── Step 3: Plan trajectories ─────────────────────────────────
    # Seg A: HOME → Pre-approach (coarse, obstacle-informed)
    obs = detector.get_obstacles()
    print(f"\n[Plan] {len(obs)} obstacle(s) known at planning time")

    seg_approach = plan_coarse(Q_HOME, q_preapproach, obs,
                               "HOME -> Pre-approach", n_wp)

    # Seg B: Pre-approach → Insertion (fine, slow)
    seg_insert = plan_fine(q_preapproach, q_insertion)
    print(f"\n  Fine insertion: {len(seg_insert)} wp, dt=0.40s (slow approach)")

    # Seg C: Withdrawal (reverse of insertion)
    seg_withdraw = plan_fine(q_insertion, q_preapproach)

    # Update detections from pre-approach position
    detector.update(q_preapproach)
    obs = detector.get_obstacles()

    # Seg D: Pre-approach → HOME (coarse, obstacle-informed)
    seg_return = plan_coarse(q_preapproach, Q_HOME, obs,
                              "Pre-approach -> HOME", n_wp)

    # ── Analysis graphs (after all planning, obstacles now known) ──
    if args.analyze and obs:
        # Run STOMP blind for comparison
        seg_blind = stomp_optimize(
            q_start=Q_HOME, q_goal=q_preapproach,
            joint_limits=JOINT_LIMITS, simple_obstacles=None,
            n_waypoints=n_wp, n_iterations=80, n_rollouts=10,
            noise_stddev=0.08, verbose=False)
        seg_deformed, elastic_history = elastic_strip_deform(
            seg_blind, obs,
            n_iterations=200, alpha=0.02,
            k_contraction=1.5, k_repulsion=10.0,
            safety_margin=0.20, damping=0.90, verbose=False)
        run_analysis(Q_HOME, q_preapproach,
                     seg_blind, seg_deformed, obs, elastic_history)

    # ── Step 4: Execute ───────────────────────────────────────────
    segments = [
        ("HOME -> Pre-approach",  seg_approach,  0.15),
        ("Insert (slow)",         seg_insert,    0.40),
        ("Dwell",                 None,          DWELL_TIME),
        ("Withdraw (slow)",       seg_withdraw,  0.40),
        ("Pre-approach -> HOME",  seg_return,    0.15),
    ]

    if use_ros:
        # Publish RViz markers
        publish_markers(car_pos, car_yaw, obs, segments)

        for i, (label, traj, dt) in enumerate(segments, 1):
            print(f"\n  Step {i}/{len(segments)}: {label}")
            if traj is None:
                print(f"     Refueling: holding for {dt:.0f}s...")
                if args.ros:
                    import rospy
                    rospy.sleep(dt)
                else:
                    time.sleep(dt)
                print(f"     Dwell complete")
            else:
                if args.ros:
                    result = send_trajectory_ros(traj, dt=dt)
                else:
                    result = send_trajectory_rviz(traj, dt=dt)
                status = "done" if result else "timeout/fail"
                print(f"     Segment {status}")
    else:
        # Dry-run preview
        print(f"\n[Preview] Mission trajectory summary")
        total_wp = 0
        for i, (label, traj, dt) in enumerate(segments, 1):
            if traj is None:
                print(f"  Step {i}: {label} ({dt:.0f}s dwell)")
            else:
                total_wp += len(traj)
                print(f"  Step {i}: {label}  ({len(traj)} wp, dt={dt}s)")
                print(f"           start={np.round(traj[0], 3)}")
                print(f"           end  ={np.round(traj[-1], 3)}")
        print(f"\n  Total waypoints: {total_wp}")

    print(f"\n{'=' * 65}")
    print("  Mission complete!")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()

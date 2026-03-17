#!/usr/bin/env python3
"""
KUKA KR6 R700 — Autonomous Refueling Mission
==============================================

For a given EE target pose (green marker), a random obstacle (blue sphere)
is spawned somewhere along the arm's planned path. The arm must:
  1. IK-Geo  → solve exact joint angles for the target
  2. STOMP   → plan a smooth trajectory
  3. Elastic Strips → reactively avoid the obstacle
  4. Refuel (dwell 5 s) → return to HOME

Outputs two graphs:
  - EE workspace trajectory (3D)
  - Joint angle trajectories (degrees) over the full mission

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
from elastic_strips import elastic_strip_deform
from car_model import (
    get_inlet_pose, get_preapproach_pose, spawn_target_marker,
    TARGET_XYZ_DEFAULT,
)

# ── KUKA KR6 R700-2 Joint Limits (URDF) ──────────────────────────
JOINT_LIMITS = np.array([
    [-2.967059725,  2.967059725],   # joint_1: +/-170 deg
    [-3.316125575,  0.785398163],   # joint_2: -190 to +45 deg
    [-2.094395100,  2.722713630],   # joint_3: -120 to +156 deg
    [-3.228859113,  3.228859113],   # joint_4: +/-185 deg
    [-2.094395100,  2.094395100],   # joint_5: +/-120 deg
    [-6.108652375,  6.108652375],   # joint_6: +/-350 deg
])

Q_HOME     = np.array([0.0, -np.pi / 2, 0.0, 0.0, 0.0, 0.0])
DWELL_TIME = 5.0
OBS_RADIUS = 0.05


# ── Utility ───────────────────────────────────────────────────────

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


def ee_positions(trajectory):
    """Compute EE XYZ for every waypoint."""
    pts = np.zeros((len(trajectory), 3))
    for i, q in enumerate(trajectory):
        _, p = fwd_kinematics(q)
        pts[i] = p
    return pts


# ── Random obstacle along the path ───────────────────────────────

def random_obstacle_on_path(trajectory, rng=None):
    """Pick a random waypoint (30-70% along the path) and place an obstacle near it."""
    if rng is None:
        rng = np.random.default_rng()
    n = len(trajectory)
    idx = rng.integers(int(n * 0.3), int(n * 0.7))
    _, ee_pos = fwd_kinematics(trajectory[idx])
    # Offset slightly so it's in the path but not exactly on the waypoint
    offset = rng.uniform(-0.04, 0.04, size=3)
    offset[2] = -abs(offset[2])  # nudge below the EE path
    center = ee_pos + offset
    center[2] = max(center[2], 0.05)  # keep above ground
    print(f"  Obstacle placed near waypoint {idx}/{n} at "
          f"[{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]")
    return (center, OBS_RADIUS)


def spawn_obstacle_gazebo(center, radius):
    """Spawn a blue sphere obstacle in Gazebo."""
    import rospy
    from gazebo_msgs.srv import SpawnModel
    from geometry_msgs.msg import Pose

    sdf = f"""<?xml version="1.0" ?>
    <sdf version="1.5">
      <model name="random_obstacle">
        <static>true</static>
        <link name="link">
          <visual name="vis">
            <geometry><sphere><radius>{radius}</radius></sphere></geometry>
            <material><ambient>0 0 1 1</ambient><diffuse>0 0.1 1 1</diffuse></material>
          </visual>
          <collision name="col">
            <geometry><sphere><radius>{radius}</radius></sphere></geometry>
          </collision>
        </link>
      </model>
    </sdf>"""

    rospy.wait_for_service('/gazebo/spawn_sdf_model', timeout=5.0)
    spawn = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
    p = Pose()
    p.position.x, p.position.y, p.position.z = center
    p.orientation.w = 1.0
    try:
        spawn("random_obstacle", sdf, "/", p, "world")
    except Exception:
        pass


# ── Trajectory planning ──────────────────────────────────────────

def plan_stomp(q_start, q_goal, obstacles, name, n_wp=30):
    """STOMP + Elastic Strips."""
    print(f"\n  Planning: {name}")
    traj = stomp_optimize(
        q_start=q_start, q_goal=q_goal,
        joint_limits=JOINT_LIMITS,
        simple_obstacles=obstacles or None,
        n_waypoints=n_wp, n_iterations=80, n_rollouts=10,
        noise_stddev=0.08, verbose=False,
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
            safety_margin=0.20, damping=0.90, verbose=False,
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


# ── ROS execution ─────────────────────────────────────────────────

def _ensure_ros_path():
    ros_python = '/opt/ros/noetic/lib/python3/dist-packages'
    if ros_python not in sys.path and os.path.isdir(ros_python):
        sys.path.insert(0, ros_python)


def send_trajectory_ros(trajectory, dt=0.15):
    _ensure_ros_path()
    import rospy, actionlib
    from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
    from trajectory_msgs.msg import JointTrajectoryPoint

    client = actionlib.SimpleActionClient(
        '/kr6_arm_controller/follow_joint_trajectory',
        FollowJointTrajectoryAction)
    client.wait_for_server(timeout=rospy.Duration(5.0))

    goal = FollowJointTrajectoryGoal()
    goal.trajectory.joint_names = [f'joint_{i}' for i in range(1, 7)]
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
        send_trajectory_rviz.pub = rospy.Publisher('/joint_states', JointState, queue_size=10)
        rospy.sleep(0.5)
    msg = JointState()
    msg.name = [f'joint_{i}' for i in range(1, 7)]
    rate = rospy.Rate(1.0 / dt)
    for q in trajectory:
        msg.header.stamp = rospy.Time.now()
        msg.position = q.tolist()
        send_trajectory_rviz.pub.publish(msg)
        rate.sleep()
    return True


# ── RViz markers ──────────────────────────────────────────────────

def publish_markers(target_xyz, obstacle, segments):
    _ensure_ros_path()
    import rospy
    from visualization_msgs.msg import Marker, MarkerArray
    from geometry_msgs.msg import Point

    pub = rospy.Publisher('/visualization_marker_array', MarkerArray, queue_size=10)
    rospy.sleep(0.5)
    ma = MarkerArray()

    # Green target
    m = Marker()
    m.header.frame_id = "world"; m.ns = "mission"; m.id = 1
    m.type = Marker.CUBE; m.action = Marker.ADD
    m.pose.position.x, m.pose.position.y, m.pose.position.z = target_xyz
    m.pose.orientation.w = 1.0
    m.scale.x = 0.06; m.scale.y = 0.06; m.scale.z = 0.06
    m.color.r = 0; m.color.g = 0.9; m.color.b = 0; m.color.a = 1.0
    ma.markers.append(m)

    # Blue obstacle
    if obstacle:
        c, r = obstacle
        m2 = Marker()
        m2.header.frame_id = "world"; m2.ns = "mission"; m2.id = 2
        m2.type = Marker.SPHERE; m2.action = Marker.ADD
        m2.pose.position.x, m2.pose.position.y, m2.pose.position.z = c
        m2.pose.orientation.w = 1.0
        m2.scale.x = m2.scale.y = m2.scale.z = r * 2
        m2.color.r = 0; m2.color.g = 0; m2.color.b = 1.0; m2.color.a = 0.7
        ma.markers.append(m2)

    # Trajectory trace (white)
    m_path = Marker()
    m_path.header.frame_id = "world"; m_path.ns = "trajectory"; m_path.id = 100
    m_path.type = Marker.LINE_STRIP; m_path.action = Marker.ADD
    m_path.pose.orientation.w = 1.0
    m_path.scale.x = 0.008
    m_path.color.r = 1; m_path.color.g = 1; m_path.color.b = 1; m_path.color.a = 0.8
    for label, traj, _ in segments:
        if traj is not None:
            for q in traj:
                _, p = fwd_kinematics(q)
                m_path.points.append(Point(x=p[0], y=p[1], z=p[2]))
    ma.markers.append(m_path)

    pub.publish(ma)


# ── Graphs ────────────────────────────────────────────────────────

def plot_trajectory_3d(all_traj, target_xyz, obstacle, save_path):
    """Plot 3D EE workspace trajectory with target and obstacle."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers '3d' projection

    pts = ee_positions(all_traj)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], 'k-', lw=1.5, label='EE Trajectory')
    ax.scatter(*pts[0], color='blue', s=80, marker='^', label='HOME (start)', zorder=5)
    ax.scatter(*target_xyz, color='green', s=100, marker='s', label='Target (refuel)', zorder=5)
    ax.scatter(*pts[-1], color='blue', s=80, marker='v', label='HOME (return)', zorder=5)

    if obstacle:
        c, r = obstacle
        # Draw obstacle sphere wireframe
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 15)
        xs = c[0] + r * np.outer(np.cos(u), np.sin(v))
        ys = c[1] + r * np.outer(np.sin(u), np.sin(v))
        zs = c[2] + r * np.outer(np.ones_like(u), np.cos(v))
        ax.plot_surface(xs, ys, zs, alpha=0.25, color='blue')
        ax.scatter(*c, color='red', s=40, marker='x', label='Obstacle center', zorder=5)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('End-Effector Trajectory — IK-Geo + STOMP + Elastic Strips',
                 fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_joint_angles(all_traj, save_path):
    """Plot joint angles (degrees) over the full mission trajectory."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n = len(all_traj)
    angles_deg = np.degrees(all_traj)
    waypoints = np.arange(n)

    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628']
    labels = [f'Joint {i+1}' for i in range(6)]

    fig, ax = plt.subplots(figsize=(12, 6))
    for j in range(6):
        ax.plot(waypoints, angles_deg[:, j], color=colors[j], lw=1.5,
                label=labels[j])

    ax.set_xlabel('Waypoint Index')
    ax.set_ylabel('Joint Angle (degrees)')
    ax.set_title('Joint Angle Trajectories — Full Mission', fontweight='bold')
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="KUKA KR6 R700 Autonomous Refueling Mission")
    parser.add_argument("--ros", action="store_true",
                        help="Execute on ROS Noetic + Gazebo")
    parser.add_argument("--rviz", action="store_true",
                        help="Visualise in RViz only (no Gazebo physics)")
    parser.add_argument("--waypoints", type=int, default=30,
                        help="Waypoints per coarse segment")
    parser.add_argument("--target-x", type=float, default=TARGET_XYZ_DEFAULT[0])
    parser.add_argument("--target-y", type=float, default=TARGET_XYZ_DEFAULT[1])
    parser.add_argument("--target-z", type=float, default=TARGET_XYZ_DEFAULT[2])
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for obstacle placement (default: random)")
    args = parser.parse_args()

    target_xyz = np.array([args.target_x, args.target_y, args.target_z])
    n_wp = args.waypoints
    rng = np.random.default_rng(args.seed)

    print("=" * 65)
    print("  KUKA KR6 R700 — Autonomous Refueling Mission")
    print("  IK-Geo + STOMP + Elastic Strips")
    print("=" * 65)

    # ── Step 1: IK-Geo ────────────────────────────────────────────
    inlet_xyz, inlet_R = get_inlet_pose(target_xyz)
    preapproach_xyz, _ = get_preapproach_pose(inlet_xyz, inlet_R)

    print(f"\n[Target]       [{target_xyz[0]:.3f}, {target_xyz[1]:.3f}, {target_xyz[2]:.3f}]")
    print(f"[Pre-approach] [{preapproach_xyz[0]:.3f}, {preapproach_xyz[1]:.3f}, "
          f"{preapproach_xyz[2]:.3f}]")

    print("\n[IK-Geo] Solving for target pose...")
    Q = IK_spherical_2_parallel(inlet_R, inlet_xyz)
    Q_valid = filter_solutions(Q, Q_HOME)
    if Q_valid.size == 0:
        print("  No valid IK solution!")
        return
    q_target = Q_valid[:, 0]
    _, p_chk = fwd_kinematics(q_target)
    print(f"     {Q.shape[1]} solutions, {Q_valid.shape[1]} valid")
    print(f"     Selected: {np.round(np.degrees(q_target), 1)} deg")
    print(f"     FK error: {np.linalg.norm(p_chk - inlet_xyz):.2e} m")

    # ── Step 2: Blind STOMP to find the path, then place obstacle ─
    print("\n[STOMP] Blind plan HOME -> Target (to determine path)...")
    seg_blind = stomp_optimize(
        q_start=Q_HOME, q_goal=q_target,
        joint_limits=JOINT_LIMITS, simple_obstacles=None,
        n_waypoints=n_wp, n_iterations=80, n_rollouts=10,
        noise_stddev=0.08, verbose=False)

    print("\n[Obstacle] Spawning random obstacle on the path...")
    obstacle = random_obstacle_on_path(seg_blind, rng)
    obs_list = [obstacle]

    # ── Step 3: Re-plan with obstacle knowledge ───────────────────
    seg_go = plan_stomp(Q_HOME, q_target, obs_list,
                        "HOME -> Target (obstacle-aware)", n_wp)

    seg_insert = plan_fine(q_target, q_target, n_wp=5)  # hold at target

    seg_return = plan_stomp(q_target, Q_HOME, obs_list,
                            "Target -> HOME (obstacle-aware)", n_wp)

    # ── Step 4: Concatenate full trajectory for graphs ────────────
    full_traj = np.vstack([seg_go, seg_insert, seg_return])

    segments = [
        ("HOME -> Target",  seg_go,      0.15),
        ("Refueling",       None,        DWELL_TIME),
        ("Target -> HOME",  seg_return,  0.15),
    ]

    # ── Step 5: Spawn in Gazebo / RViz FIRST ──────────────────────
    use_ros = args.ros or args.rviz
    if use_ros:
        _ensure_ros_path()
        import rospy
        rospy.init_node('refuel_mission', anonymous=True)

        if args.ros:
            spawn_target_marker(target_xyz)
            spawn_obstacle_gazebo(obstacle[0], obstacle[1])

        publish_markers(target_xyz, obstacle, segments)

    # ── Step 6: Generate graphs ───────────────────────────────────
    print("\n[Graphs]")
    plot_trajectory_3d(full_traj, target_xyz, obstacle,
                       "output_graphs/ee_trajectory_3d.png")
    plot_joint_angles(full_traj, "output_graphs/joint_angle_trajectories.png")

    # ── Step 7: Execute motion ────────────────────────────────────
    if use_ros:
        for i, (label, traj, dt) in enumerate(segments, 1):
            print(f"\n  Step {i}/{len(segments)}: {label}")
            if traj is None:
                print(f"     Refueling: holding for {dt:.0f}s...")
                rospy.sleep(dt)
                print(f"     Dwell complete")
            else:
                if args.ros:
                    result = send_trajectory_ros(traj, dt=dt)
                else:
                    result = send_trajectory_rviz(traj, dt=dt)
                print(f"     {'done' if result else 'timeout/fail'}")
    else:
        print(f"\n[Preview]")
        total_wp = 0
        for i, (label, traj, dt) in enumerate(segments, 1):
            if traj is None:
                print(f"  Step {i}: {label} ({dt:.0f}s dwell)")
            else:
                total_wp += len(traj)
                print(f"  Step {i}: {label}  ({len(traj)} wp, dt={dt}s)")
                print(f"           start={np.round(np.degrees(traj[0]), 1)} deg")
                print(f"           end  ={np.round(np.degrees(traj[-1]), 1)} deg")
        print(f"\n  Total waypoints: {total_wp}")

    print(f"\n{'=' * 65}")
    print("  Mission complete!")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()

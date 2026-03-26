#!/usr/bin/env python3
"""
KUKA KR6 R700 — Autonomous Refueling Mission
==============================================

Hybrid C-Space / W-Space 4-phase mission architecture:

  Phase 1  HOME → Pre-approach     (C-Space: STOMP + Elastic Strips)
  Phase 2  Pre-approach → Target   (W-Space: Cartesian straight-line + IK)
  Phase 3  Target → Pre-approach   (W-Space: Cartesian straight-line + IK)
  Phase 4  Pre-approach → HOME     (C-Space: STOMP + Elastic Strips)

Random obstacles (blue spheres) are placed along the gross-motion path.
STOMP + Elastic Strips avoid them. Fine insertion/extraction use Cartesian
interpolation with fixed orientation to guarantee a straight-line nozzle path.

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
from bubble_strips import bubble_strip_deform
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


def delta_wrap(d):
    """Wrap angular delta to [-pi, pi]."""
    return (d + np.pi) % (2 * np.pi) - np.pi


def filter_solutions(Q, history=None):
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
    
    # Graceful fallback for single q_prev vs history list
    if history is not None:
        if isinstance(history, np.ndarray):
            history = [history]
            
        if len(history) > 0:
            scores = np.zeros(valid.shape[1])
            W_v, W_a, W_j = 1.0, 5.0, 10.0
            
            for i in range(valid.shape[1]):
                q = valid[:, i]
                score = 0.0
                
                # Explicitly cast for linter if needed, though logic is same
                prev_q = history[-1]
                
                # Velocity (1st Derivative)
                v_k = delta_wrap(q - prev_q)
                score += W_v * np.linalg.norm(v_k)
                
                # Acceleration (2nd Derivative)
                if len(history) >= 2:
                    v_k_1 = delta_wrap(history[-1] - history[-2])
                    a_k = v_k - v_k_1
                    score += W_a * np.linalg.norm(a_k)
                    
                    # Jerk (3rd Derivative)
                    if len(history) >= 3:
                        v_k_2 = delta_wrap(history[-2] - history[-3])
                        a_k_1 = v_k_1 - v_k_2
                        j_k = a_k - a_k_1
                        score += W_j * np.linalg.norm(j_k)
                        
                scores[i] = score
                
            valid = valid[:, np.argsort(scores)]
    return valid


def ee_positions(trajectory):
    """Compute EE XYZ for every waypoint."""
    pts = np.zeros((len(trajectory), 3))
    for i, q in enumerate(trajectory):
        _, p = fwd_kinematics(q)
        pts[i] = p
    return pts


# ── Random obstacles along the path ──────────────────────────────
NUM_OBSTACLES = 2

def random_obstacles_on_path(trajectory, n_obs=NUM_OBSTACLES, rng=None):
    """Place n_obs obstacles in non-overlapping zones along the path."""
    if rng is None:
        rng = np.random.default_rng()
    n = len(trajectory)
    # Split the 25-75% range into n_obs equal zones
    zone_lo = 0.25
    zone_hi = 0.75
    zone_width = (zone_hi - zone_lo) / n_obs
    obstacles = []
    for k in range(n_obs):
        lo = int(n * (zone_lo + k * zone_width))
        hi = int(n * (zone_lo + (k + 1) * zone_width))
        hi = max(hi, lo + 1)
        idx = rng.integers(lo, hi)
        _, ee_pos = fwd_kinematics(trajectory[idx])
        offset = np.array(rng.uniform(-0.04, 0.04, size=3))
        # Use explicit float logic to satisfy linter
        z_val = float(offset[2])
        offset[2] = -abs(z_val)
        center = ee_pos + offset
        center[2] = max(center[2], 0.05)
        obstacles.append((center, OBS_RADIUS))
        print(f"  Obstacle {k+1}/{n_obs} near waypoint {idx}/{n} at "
              f"[{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]")
    return obstacles


def spawn_obstacles_gazebo(obs_list):
    """Spawn blue sphere obstacles in Gazebo."""
    import rospy
    from gazebo_msgs.srv import SpawnModel
    from geometry_msgs.msg import Pose

    rospy.wait_for_service('/gazebo/spawn_sdf_model', timeout=5.0)
    spawn = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)

    for k, (center, radius) in enumerate(obs_list):
        sdf = f"""<?xml version="1.0" ?>
        <sdf version="1.5">
          <model name="random_obstacle_{k}">
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
        p = Pose()
        p.position.x, p.position.y, p.position.z = center
        p.orientation.w = 1.0
        try:
            spawn(f"random_obstacle_{k}", sdf, "/", p, "world")
        except Exception:
            pass


# ── Trajectory planning ──────────────────────────────────────────

def smooth_trajectory(traj, window=5, passes=2):
    """Moving-average smoothing with pinned endpoints and joint-limit clamping.

    Applies a centered moving average ``passes`` times. Start and end
    waypoints are never modified so the trajectory still hits its goals.
    """
    smoothed = traj.copy()
    half = window // 2
    for _ in range(passes):
        buf = smoothed.copy()
        for i in range(1, len(smoothed) - 1):
            lo = max(0, i - half)
            hi = min(len(smoothed), i + half + 1)
            buf[i] = smoothed[lo:hi].mean(axis=0)
        # Clamp to joint limits
        for j in range(6):
            buf[:, j] = np.clip(buf[:, j], JOINT_LIMITS[j, 0], JOINT_LIMITS[j, 1])
        buf[0] = traj[0]
        buf[-1] = traj[-1]
        smoothed = buf
    return smoothed


def plan_stomp(q_start, q_goal, obstacles, name, n_wp=30):
    """STOMP + Elastic Strips + post-smoothing."""
    print(f"\n  Planning: {name}")
    traj = stomp_optimize(
        q_start=q_start, q_goal=q_goal,
        joint_limits=JOINT_LIMITS,
        simple_obstacles=obstacles or None,
        n_waypoints=n_wp, n_iterations=100, n_rollouts=12,
        noise_stddev=0.08, w_smooth=20.0, w_vel=15.0,
        verbose=False,
    )
    diffs = np.diff(traj, axis=0)
    max_jump = np.max(np.abs(diffs))
    ok = all(within_joint_limits(traj[i]) for i in range(len(traj)))
    print(f"     STOMP: {n_wp} wp, max_jump={np.degrees(max_jump):.1f} deg, "
          f"limits {'OK' if ok else 'VIOLATED'}")

    if obstacles:
        traj, _, stats = bubble_strip_deform(
            traj, obstacles,
            n_iterations=150,
            k_contraction=0.5, k_repulsion=30.0,
            rho_0=0.20, damping=0.85, verbose=False,
        )
        print(f"     Bubble Strips: {stats['final_waypoints']} wp, "
              f"min_rho={stats['final_min_clearance']:.4f}m")

    traj = smooth_trajectory(traj, window=5, passes=2)
    diffs = np.diff(traj, axis=0)
    max_jump_post = np.max(np.abs(diffs))
    print(f"     Smoothed: max_jump {np.degrees(max_jump):.1f} -> "
          f"{np.degrees(max_jump_post):.1f} deg")
    return traj


def plan_cartesian(pos_start, pos_goal, R_fixed, q_seed, name, n_wp=20):
    """Straight-line Cartesian interpolation with IK at every point.

    Interpolates linearly in XYZ while keeping orientation fixed.
    Each waypoint is solved with IK-Geo, selecting the solution with the
    smoothest velocity/acceleration/jerk profile via historical derivatives.
    """
    print(f"\n  Planning: {name} (Cartesian, {n_wp} wp)")
    traj = np.zeros((n_wp, 6))
    history = [q_seed.copy()]
    for i in range(n_wp):
        alpha = i / (n_wp - 1)
        p_i = (1 - alpha) * pos_start + alpha * pos_goal
        Q = IK_spherical_2_parallel(R_fixed, p_i)
        Q_valid = filter_solutions(Q, history)
        if Q_valid.size == 0:
            # Fallback: C-space lerp from last good config toward seed goal
            print(f"     WARNING: no IK at wp {i}, falling back to C-space lerp")
            traj[i] = history[-1]
        else:
            traj[i] = Q_valid[:, 0]
            history.append(traj[i])
            if len(history) > 3:
                history.pop(0)
    _, p_end = fwd_kinematics(traj[-1])
    err = np.linalg.norm(p_end - pos_goal)
    print(f"     Cartesian: {n_wp} wp, endpoint FK error: {err:.2e} m")
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


_RVIZ_PUB = None

def send_trajectory_rviz(trajectory, dt=0.15):
    global _RVIZ_PUB
    _ensure_ros_path()
    import rospy
    from sensor_msgs.msg import JointState
    if _RVIZ_PUB is None:
        _RVIZ_PUB = rospy.Publisher('/joint_states', JointState, queue_size=10)
        rospy.sleep(0.5)
    msg = JointState()
    msg.name = [f'joint_{i}' for i in range(1, 7)]
    rate = rospy.Rate(1.0 / dt)
    for q in trajectory:
        msg.header.stamp = rospy.Time.now()
        msg.position = q.tolist()
        _RVIZ_PUB.publish(msg)
        rate.sleep()
    return True


# ── RViz markers ──────────────────────────────────────────────────

def publish_markers(target_xyz, obs_list, segments):
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

    # Blue obstacles
    for k, (c, r) in enumerate(obs_list):
        m2 = Marker()
        m2.header.frame_id = "world"; m2.ns = "mission"; m2.id = 10 + k
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

def plot_trajectory_3d(all_traj, target_xyz, obs_list, save_path):
    """Plot 3D EE workspace trajectory with target and obstacles."""
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

    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 15)
    for k, (c, r) in enumerate(obs_list):
        xs = c[0] + r * np.outer(np.cos(u), np.sin(v))
        ys = c[1] + r * np.outer(np.sin(u), np.sin(v))
        zs = c[2] + r * np.outer(np.ones_like(u), np.cos(v))
        ax.plot_surface(xs, ys, zs, alpha=0.25, color='blue')
        lbl = f'Obstacle {k+1}' if k == 0 else f'Obstacle {k+1}'
        ax.scatter(*c, color='red', s=40, marker='x', label=lbl, zorder=5)

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
    parser.add_argument("--mirror-return", action="store_true",
                        help="Return along the reversed approach path instead of re-planning")
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
    Q_target = IK_spherical_2_parallel(inlet_R, inlet_xyz)
    Q_valid_target = filter_solutions(Q_target, Q_HOME)
    if Q_valid_target.size == 0:
        print("  No valid IK solution for Target!")
        return
    q_target = Q_valid_target[:, 0]
    
    print("\n[IK-Geo] Solving for pre-approach pose...")
    Q_pre = IK_spherical_2_parallel(inlet_R, preapproach_xyz)
    Q_valid_pre = filter_solutions(Q_pre, Q_HOME)
    if Q_valid_pre.size == 0:
        print("  No valid IK solution for Pre-approach!")
        return
    q_pre = Q_valid_pre[:, 0]

    _, p_chk = fwd_kinematics(q_target)
    print(f"     Target Selected: {np.round(np.degrees(q_target), 1)} deg")
    print(f"     Target FK error: {np.linalg.norm(p_chk - inlet_xyz):.2e} m")

    # ── Step 2: Blind STOMP to find the path, then place obstacle ─
    print("\n[STOMP] Blind plan HOME -> Pre-approach (to determine path)...")
    seg_blind = stomp_optimize(
        q_start=Q_HOME, q_goal=q_pre,
        joint_limits=JOINT_LIMITS, simple_obstacles=None,
        n_waypoints=n_wp, n_iterations=80, n_rollouts=10,
        noise_stddev=0.08, verbose=False)

    print("\n[Obstacles] Spawning 2 random obstacles on the path...")
    obs_list = random_obstacles_on_path(seg_blind, n_obs=NUM_OBSTACLES, rng=rng)

    # ── Step 3: 4-Phase Hybrid Mission Architecture ─────────────────
    
    # Phase 1: Gross Approach (C-Space)
    seg_approach = plan_stomp(Q_HOME, q_pre, obs_list,
                              "Phase 1: HOME -> Pre-approach", n_wp)

    # Phase 2: Fine Insertion (W-Space)
    seg_insert = plan_cartesian(preapproach_xyz, inlet_xyz, inlet_R, 
                                q_pre, "Phase 2: Pre-approach -> Target", n_wp=20)
                                
    # Dwell at target
    seg_dwell = plan_fine(q_target, q_target, n_wp=5)

    # Phase 3: Fine Extraction (W-Space)
    seg_extract = plan_cartesian(inlet_xyz, preapproach_xyz, inlet_R,
                                 q_target, "Phase 3: Target -> Pre-approach", n_wp=20)

    # Phase 4: Gross Return (C-Space)
    if args.mirror_return:
        seg_return = seg_approach[::-1].copy()
        print(f"\n  Planning: Phase 4: Pre-approach -> HOME (mirrored approach)")
        print(f"     Reversed approach trajectory ({len(seg_return)} wp)")
    else:
        seg_return = plan_stomp(q_pre, Q_HOME, obs_list,
                                "Phase 4: Pre-approach -> HOME", n_wp)

    # ── Step 4: Concatenate full trajectory for graphs ────────────
    full_traj = np.vstack([seg_approach, seg_insert, seg_dwell, seg_extract, seg_return])

    segments = [
        ("Gross Approach",  seg_approach, 0.15),
        ("Fine Insertion",  seg_insert,   0.05),
        ("Refueling",       None,         DWELL_TIME),
        ("Fine Extraction", seg_extract,  0.05),
        ("Gross Return",    seg_return,   0.15),
    ]

    # ── Step 5: Spawn in Gazebo / RViz FIRST ──────────────────────
    use_ros = args.ros or args.rviz
    if use_ros:
        _ensure_ros_path()
        import rospy
        rospy.init_node('refuel_mission', anonymous=True)

        if args.ros:
            spawn_target_marker(target_xyz)
            spawn_obstacles_gazebo(obs_list)

        publish_markers(target_xyz, obs_list, segments)

    # ── Step 6: Generate graphs ───────────────────────────────────
    print("\n[Graphs]")
    plot_trajectory_3d(full_traj, target_xyz, obs_list,
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
        total_wp_list = []
        for i, (label, traj, dt) in enumerate(segments, 1):
            if traj is None:
                print(f"  Step {i}: {label} ({dt:.0f}s dwell)")
            else:
                # Local shadow for linter type-inference safety
                t = traj 
                if t is not None:
                    seg_len = int(len(t))
                    total_wp_list.append(seg_len)
                    print(f"  Step {i}: {label}  ({seg_len} wp, dt={dt}s)")
                    if seg_len > 0:
                        print(f"           start={np.round(np.degrees(t[0]), 1)} deg")
                        print(f"           end  ={np.round(np.degrees(t[-1]), 1)} deg")
        print(f"\n  Total waypoints: {sum(total_wp_list)}")

    print(f"\n{'=' * 65}")
    print("  Mission complete!")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()

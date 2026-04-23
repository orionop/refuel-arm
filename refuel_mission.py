#!/usr/bin/env python3
"""
KUKA KR 8 R2100 (Cybertech nano) — Autonomous Refueling Mission
============================================================

Hybrid C-Space / W-Space 8-phase mission architecture:

  Phase 1  REST → Dispenser        (C-Space: STOMP + Bubble Strips)
  Phase 2  Dwell: Nozzle Fetch     (Wait 3s)
  Phase 3  Dispenser → Pre-app     (C-Space: STOMP + Bubble Strips)
  Phase 4  Pre-app → Fuel Inlet    (W-Space: Cartesian straight-line + IK)
  Phase 5  Dwell: Active Refuel    (Wait 6s)
  Phase 6  Fuel Inlet → Pre-app    (W-Space: Cartesian straight-line + IK)
  Phase 7  Pre-app → Dispenser     (C-Space: STOMP + Bubble Strips)
  Phase 8  Dispenser → REST        (C-Space: STOMP + Bubble Strips)

Random obstacles are avoided via STOMP. Fine insertion/extraction uses 
Cartesian interpolation with fixed orientation for straight-line path.

Run locally:   python3 refuel_mission.py --robot kr8
Run in Gazebo: python3 refuel_mission.py --robot kr8 --ros
"""
import sys
import os
import time
import argparse
import tempfile
import numpy as np # type: ignore

# ── Path setup ────────────────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), 'kuka_refuel_ws', 'src',
    'kuka_kr6_gazebo', 'scripts')))

from ik_geometric import ( # type: ignore
    IK_spherical_2_parallel, fwd_kinematics, rot,
    IK_solve, KIN_UR5, KIN_KR6_R700, KIN_KR210_R3100, KIN_KR8_R2100, KIN_UR20
)

from stomp_collision import stomp_optimize # type: ignore
from bubble_strips import bubble_strip_deform, set_kinematics as bs_set_kinematics # type: ignore
from car_model import ( # type: ignore
    get_inlet_pose, get_preapproach_pose, spawn_target_marker,
    TARGET_XYZ_DEFAULT,
)

# ── KUKA KR210 R3100 joint limits ──────────────────────────
JOINT_LIMITS_DEFAULT = np.array([
    [-185.0, 185.0],
    [-45.0, 85.0],
    [-210.0, 65.0],
    [-350.0, 350.0],
    [-125.0, 125.0],
    [-350.0, 350.0]
])
# Legacy alias for internal functions
JOINT_LIMITS = JOINT_LIMITS_DEFAULT

def limits_deg_to_rad(limits_deg: np.ndarray) -> np.ndarray:
    """Convert joint limits specified in degrees to radians (for clipping/cost)."""
    return np.radians(np.asarray(limits_deg, dtype=float))

# Upright "Candle" home pose for KUKA
Q_HOME     = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
# "Bonkers" Spawn Pose (Rest) for KR8 (J5 shifted by +2pi to stay in limits)
Q_REST     = np.array([1.607, -0.5, 0.5, 0.0, 3.203, 0.0])
DWELL_TIME = 5.0
OBS_RADIUS = 0.15

# --- Distance Hallucination Configuration ---
# Helps the 0.7m reach KUKA work in a 3.0m scale "Perfect World" 
HALLUCINATION_FACTOR = 1.0
HALLUCINATION_ENABLED = False

def hallucinate_point(real_xyz):
    """Snaps a remote world point to a reachable 0.45m point in the same direction."""
    if not HALLUCINATION_ENABLED:
        return real_xyz
    
    # Calculate horizontal distance to target
    dist_2d = np.linalg.norm(real_xyz[:2])
    if dist_2d < 1e-3:
        return real_xyz
    
    # Reach exactly 0.25m in that direction (Ensures J5 stays within +/- 120deg)
    target_reach = 0.25
    scale = target_reach / dist_2d
    
    hallucinated = np.copy(real_xyz)
    hallucinated[0] = real_xyz[0] * scale
    hallucinated[1] = real_xyz[1] * scale
    
    # Set height to 0.35m (Proven safe Z)
    hallucinated[2] = 0.35 
    return hallucinated


# ── Utility ───────────────────────────────────────────────────────

def within_joint_limits(q):
    # Joint states are in radians; the stored limits are in degrees.
    q_deg = np.degrees(q)
    for i in range(6):
        if q_deg[i] < JOINT_LIMITS[i, 0] or q_deg[i] > JOINT_LIMITS[i, 1]:
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


def filter_solutions(Q, q_current, limits=None):
    """Filter IK solutions by joint limits and proximity to current configuration."""
    if Q.size == 0:
        return Q
    if limits is None:
        limits = JOINT_LIMITS
    
    # Cast to ndarray for linter
    lims = np.asarray(limits)
    Q_deg = np.degrees(Q)
    valid_idx = []
    for i in range(Q.shape[1]):
        q = Q_deg[:, i]
        in_limits = True
        for j in range(6):
            if q[j] < lims[j][0] or q[j] > lims[j][1]: # type: ignore
                in_limits = False
                break
        if in_limits: # type: ignore
            valid_idx.append(i) # type: ignore
    
    valid = Q[:, valid_idx] # type: ignore
    
    if valid.size > 0:
        # Sort by distance to q_current
        diff = valid - q_current[:, np.newaxis]
        dist = np.linalg.norm(diff, axis=0) # type: ignore
        valid = valid[:, np.argsort(dist)] # type: ignore
    return valid


def ee_positions(trajectory, kin=None):
    """Compute EE XYZ for every waypoint."""
    pts = np.zeros((len(trajectory), 3))
    for i, q in enumerate(trajectory):
        _, p = fwd_kinematics(q, kin=kin) # type: ignore
        pts[i] = p
    return pts


# ── Random obstacles along the path ──────────────────────────────
NUM_OBSTACLES = 0

def random_obstacles_on_path(trajectory, n_obs=NUM_OBSTACLES, rng=None, kin=None):
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
        _, ee_pos = fwd_kinematics(trajectory[idx], kin=kin) # type: ignore
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


def spawn_obstacles_gazebo(obs_list, ros2_node):
    """Spawn blue sphere obstacles in Gz Sim (ROS2 Jazzy).

    Uses ros_gz_sim create command (replaces gazebo_msgs SpawnEntity).
    Deletes previous obstacles first so re-runs show fresh positions.
    """
    import subprocess

    # Delete previous obstacles (silent fail on first run)
    for k in range(len(obs_list)):
        name = f"random_obstacle_{k}"
        subprocess.run(
            ['gz', 'service', '-s', '/world/refuel_world/remove',
             '--reqtype', 'gz.msgs.Entity',
             '--reptype', 'gz.msgs.Boolean',
             '--timeout', '1000',
             '--req', f'name: "{name}" type: 2'],
            capture_output=True, text=True, timeout=5,
        )

    for k, (center, radius) in enumerate(obs_list):
        x, y = float(center[0]), float(center[1])
        # Keep obstacle collision body above ground-plane contact threshold.
        z = max(float(center[2]), float(radius) + 1e-3)
        sdf = f"""<?xml version="1.0" ?>
<sdf version="1.9">
  <model name="random_obstacle_{k}">
    <static>true</static>
    <pose>{x} {y} {z} 0 0 0</pose>
    <link name="link">
      <visual name="vis">
        <geometry><sphere><radius>{radius}</radius></sphere></geometry>
        <material>
          <ambient>0 0 1 1</ambient>
          <diffuse>0 0.1 1 1</diffuse>
        </material>
      </visual>
      <collision name="col">
        <geometry><sphere><radius>{radius}</radius></sphere></geometry>
      </collision>
    </link>
  </model>
</sdf>
"""
        sdf_path = os.path.join(tempfile.gettempdir(), f"obstacle_{k}.sdf")
        try:
            with open(sdf_path, "w", encoding="utf-8") as f:
                f.write(sdf)
            subprocess.run(
                ['ros2', 'run', 'ros_gz_sim', 'create',
                 '-file', sdf_path,
                 '-name', f'random_obstacle_{k}',
                 '-x', str(x), '-y', str(y), '-z', str(z)],
                capture_output=True, text=True, timeout=10,
            )
        except Exception:
            pass


# ── Trajectory planning ──────────────────────────────────────────

def smooth_trajectory(traj, window=5, passes=2, limits=None):
    """Moving-average smoothing with pinned endpoints and joint-limit clamping.

    Applies a centered moving average ``passes`` times. Start and end
    waypoints are never modified so the trajectory still hits its goals.
    """
    if limits is None:
        limits = JOINT_LIMITS
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
            buf[:, j] = np.clip(buf[:, j], limits[j][0], limits[j][1])
        buf[0] = traj[0]
        buf[-1] = traj[-1]
        smoothed = buf
    return smoothed


def plan_stomp(q_start, q_goal, obstacles, name, n_wp=30, limits=None, kin=None):
    """STOMP + Elastic Strips + post-smoothing with platform-specific limits."""
    print(f"\n  Planning: {name}")
    limits_deg = JOINT_LIMITS if limits is None else np.asarray(limits, dtype=float)
    limits_rad = limits_deg_to_rad(limits_deg)
        
    traj = stomp_optimize(
        q_start=q_start, q_goal=q_goal,
        joint_limits=limits_rad,
        simple_obstacles=obstacles or None,
        n_waypoints=n_wp, n_iterations=100, n_rollouts=12,
        noise_stddev=0.08, w_smooth=20.0, w_vel=15.0,
        verbose=False, kin=kin,
    )
    diffs = np.diff(traj, axis=0)
    max_jump = np.max(np.abs(diffs))
    
    # Check limits using dynamic limits
    ok = True
    for wp in traj:
        wp_deg = np.degrees(wp)
        for j in range(6):
            if wp_deg[j] < limits_deg[j][0] or wp_deg[j] > limits_deg[j][1]: # type: ignore
                ok = False
                break
        if not ok: break
        
    print(f"     STOMP: {n_wp} wp, max_jump={np.degrees(max_jump):.1f} deg, "
          f"limits {'OK' if ok else 'VIOLATED'}")

    if obstacles:
        traj, _, stats = bubble_strip_deform(
            traj, obstacles,
            joint_limits=limits_rad,
            n_iterations=150,
            k_contraction=0.5, k_repulsion=30.0,
            rho_0=0.20, damping=0.85, verbose=False,
        )
        print(f"     Bubble Strips: {stats['final_waypoints']} wp, "
              f"min_rho={stats['final_min_clearance']:.4f}m")

    traj = smooth_trajectory(traj, window=5, passes=2, limits=limits_rad)
    diffs_post = np.diff(traj, axis=0)
    max_jump_post = np.max(np.abs(diffs_post)) # type: ignore
    print(f"     Smoothed: max_jump {np.degrees(max_jump):.1f} -> "
          f"{np.degrees(max_jump_post):.1f} deg")
    return traj


def plan_cartesian(pos_start, pos_goal, R_fixed, q_seed, name, n_wp=20, kin=None):
    """Straight-line Cartesian interpolation with IK at every point.
    
    Uses platform-specific IK_solve for UR5 or KUKA.
    """
    print(f"\n  Planning: {name} (Cartesian, {n_wp} wp)")
    active_bot = "kuka"
    if kin is not None and hasattr(kin, 'get'):
        # Check H1 axis orientation (UR5 is [0,0,1], KUKA is [0,0,-1])
        h_matrix = np.asarray(kin.get('H'))
        if h_matrix[2, 0] > 0:
            active_bot = "ur5"
        elif kin.get('joint_limits', [])[1, 0] < -200: # Heuristic for KR8
            active_bot = "kr8"
    
    traj = np.zeros((n_wp, 6))
    history = [q_seed.copy()]
    
    for i in range(n_wp):
        alpha = i / (n_wp - 1)
        p_i = (1 - alpha) * pos_start + alpha * pos_goal
        Q = IK_solve(R_fixed, p_i, robot=active_bot)
        
        # Safeguard limits
        l_dict = kin.get('joint_limits', JOINT_LIMITS) if (kin is not None and hasattr(kin, 'get')) else JOINT_LIMITS # type: ignore
        Q_valid = filter_solutions(Q, history[-1], limits=l_dict)
        
        if Q_valid.size == 0:
            print(f"     WARNING: no IK at wp {i}, falling back to prev")
            traj[i] = history[-1]
        else:
            traj[i] = Q_valid[:, 0]
            history.append(traj[i])
            if len(history) > 3:
                history.pop(0)
    
    _, p_end = fwd_kinematics(traj[-1], kin=kin)
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
    ros_python = '/opt/ros/jazzy/lib/python3/dist-packages'
    if ros_python not in sys.path and os.path.isdir(ros_python):
        sys.path.insert(0, ros_python)


# ── Robot-specific ROS configuration ──────────────────────────────
ROS_CONFIG = {
    'kuka': {
        'controller': '/kuka_kr210_controller/follow_joint_trajectory',
        'joint_names': [f'joint_a{i}' for i in range(1, 7)],
    },
    'kr8': {
        'controller': '/kr6_arm_controller/follow_joint_trajectory',
        'joint_names': [f'joint_{i}' for i in range(1, 7)],
    },
    'ur5': {
        'controller': '/ur5_arm_controller/follow_joint_trajectory',
        'joint_names': [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
        ],
    },
}


def send_trajectory_gazebo(trajectory, dt=0.15, robot='ur20'):
    """Drive Gazebo joints via gz transport topics (no ROS2 required)."""
    import time
    GZ_BIN = "/opt/homebrew/bin/gz" if os.path.exists("/opt/homebrew/bin/gz") else "gz"
    joints = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
              'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
    topics = [f"/model/{robot}/joint/{j}/0/cmd_pos" for j in joints]

    step_dt = dt / max(len(trajectory), 1)
    for q in trajectory:
        cmds = [f"{GZ_BIN} topic -t {topics[i]} -m gz.msgs.Double -p 'data: {float(v)}'"
                for i, v in enumerate(q)]
        os.system("(" + " & ".join(cmds) + " & wait) >/dev/null 2>&1")
        time.sleep(step_dt)
    return True


def send_trajectory_ros(trajectory, dt=0.15, robot='kuka'):
    """Send a joint trajectory via ROS2 FollowJointTrajectory action.

    ROS1 equivalent used actionlib.SimpleActionClient + FollowJointTrajectoryGoal.
    ROS2 uses rclpy.action.ActionClient (matches refuel_mission_commander.py pattern).
    """
    _ensure_ros_path()
    import rclpy # type: ignore
    from rclpy.action import ActionClient # type: ignore
    from control_msgs.action import FollowJointTrajectory # type: ignore
    from trajectory_msgs.msg import JointTrajectoryPoint # type: ignore
    from builtin_interfaces.msg import Duration as BuiltinDuration # type: ignore

    cfg    = ROS_CONFIG[robot]
    client = ActionClient(_ROS_NODE, FollowJointTrajectory, cfg['controller'])
    if not client.wait_for_server(timeout_sec=5.0):
        print(f"  [send_trajectory_ros] Action server not available: {cfg['controller']}")
        return None

    goal_msg = FollowJointTrajectory.Goal()
    goal_msg.trajectory.joint_names = cfg['joint_names']

    for i, q in enumerate(trajectory):
        pt = JointTrajectoryPoint()
        pt.positions  = q.tolist()
        pt.velocities = [0.0] * 6
        t = i * dt
        pt.time_from_start = BuiltinDuration(
            sec=int(t),
            nanosec=int((t % 1) * 1_000_000_000))
        goal_msg.trajectory.points.append(pt)

    send_future = client.send_goal_async(goal_msg)
    # Wait for goal acceptance with 10s timeout
    rclpy.spin_until_future_complete(_ROS_NODE, send_future, timeout_sec=10.0)
    if not send_future.done():
        print("  [send_trajectory_ros] Goal send TIMEOUT")
        return None
    goal_handle = send_future.result()

    if not goal_handle.accepted:
        print("  [send_trajectory_ros] Goal REJECTED by controller")
        return None

    result_future = goal_handle.get_result_async()
    # Wait for execution with 600s timeout (Wall Clock)
    # Note: 30s was too short if Gazebo RTF is low.
    rclpy.spin_until_future_complete(_ROS_NODE, result_future, timeout_sec=600.0)
    if not result_future.done():
        print("  [send_trajectory_ros] Execution TIMEOUT (600s Wall-Time exceeded)")
        return False
    return result_future.result().result


_ROS_NODE        = None   # rclpy.Node — set in main() when --ros is active
_ADMITTANCE_NODE = None

def ros_sleep(duration_sec: float):
    """Wait for simulation time to pass using the ROS node clock."""
    if _ROS_NODE is None:
        import time as _time
        _time.sleep(duration_sec)
        return

    start_t = _ROS_NODE.get_clock().now()
    target_ns = int(duration_sec * 1e9)
    
    # Use a spinning wait loop to keep the node active while time passes
    import rclpy # type: ignore
    while (_ROS_NODE.get_clock().now() - start_t).nanoseconds < target_ns:
        rclpy.spin_once(_ROS_NODE, timeout_sec=0.01)


def send_trajectory_compliant(trajectory, dt=0.05, robot='ur5'):
    """Send trajectory through the admittance controller for force-compliant execution.

    Used for fine insertion/extraction phases where contact forces must be
    accommodated. The admittance node yields when external forces exceed
    15N and aborts if they exceed 50N.

    Returns True on success, False on abort.
    """
    global _ADMITTANCE_NODE
    _ensure_ros_path()

    if _ADMITTANCE_NODE is None:
        # AdmittanceNode is now a proper rclpy.Node subclass — just instantiate it.
        # rclpy.init() has already been called in main() before we get here.
        from admittance_node import AdmittanceNode  # type: ignore
        _ADMITTANCE_NODE = AdmittanceNode()
        import time
        time.sleep(0.5)  # Let subscribers connect

    return _ADMITTANCE_NODE.execute_trajectory(trajectory, dt=dt)


_RVIZ_PUB = None

def send_trajectory_rviz(trajectory, dt=0.15, robot='kuka'):
    """Publish joint states for RViz visualization (no Gazebo physics).

    ROS1 used rospy.Publisher + rospy.Rate + rospy.Time.now().
    ROS2 uses _ROS_NODE.create_publisher + time.sleep + node.get_clock().now().
    """
    global _RVIZ_PUB
    _ensure_ros_path()
    import time # type: ignore
    from sensor_msgs.msg import JointState # type: ignore

    if _RVIZ_PUB is None:
        _RVIZ_PUB = _ROS_NODE.create_publisher(JointState, '/joint_states', 10)
        time.sleep(0.5)

    cfg = ROS_CONFIG[robot]
    msg = JointState()
    msg.name = cfg['joint_names']

    for q in trajectory:
        msg.header.stamp = _ROS_NODE.get_clock().now().to_msg()
        msg.position     = q.tolist()
        _RVIZ_PUB.publish(msg)
        time.sleep(dt)

    return True


# ── RViz markers ──────────────────────────────────────────────────

def publish_markers(target_xyz, obs_list, segments, kin=None):
    _ensure_ros_path()
    import time # type: ignore
    from visualization_msgs.msg import Marker, MarkerArray # type: ignore
    from geometry_msgs.msg import Point # type: ignore

    pub = _ROS_NODE.create_publisher(MarkerArray, '/visualization_marker_array', 10)
    time.sleep(0.5)
    ma = MarkerArray()

    # Green target
    m = Marker()
    m.header.frame_id = "world"; m.ns = "mission"; m.id = 1
    m.type = Marker.CUBE; m.action = Marker.ADD # type: ignore
    m.pose.position.x, m.pose.position.y, m.pose.position.z = target_xyz
    m.pose.orientation.w = 1.0
    m.scale.x = 0.06; m.scale.y = 0.06; m.scale.z = 0.06
    m.color.r = 0.0; m.color.g = 0.9; m.color.b = 0.0; m.color.a = 1.0
    ma.markers.append(m)

    # Blue obstacles
    for k, (c, r) in enumerate(obs_list):
        m2 = Marker()
        m2.header.frame_id = "world"; m2.ns = "mission"; m2.id = 10 + k
        m2.type = Marker.SPHERE; m2.action = Marker.ADD
        m2.pose.position.x, m2.pose.position.y, m2.pose.position.z = c
        m2.pose.orientation.w = 1.0
        m2.scale.x = m2.scale.y = m2.scale.z = r * 2
        m2.color.r = 0.0; m2.color.g = 0.0; m2.color.b = 1.0; m2.color.a = 0.7
        ma.markers.append(m2)

    # Trajectory trace (white)
    m_path = Marker()
    m_path.header.frame_id = "world"; m_path.ns = "trajectory"; m_path.id = 100
    m_path.type = Marker.LINE_STRIP; m_path.action = Marker.ADD # type: ignore
    m_path.pose.orientation.w = 1.0
    m_path.scale.x = 0.008
    m_path.color.r = 1.0; m_path.color.g = 1.0; m_path.color.b = 1.0; m_path.color.a = 0.8
    for label, traj, _ in segments:
        if traj is not None:
            for q in traj:
                _, p = fwd_kinematics(q, kin=kin)
                m_path.points.append(Point(x=p[0], y=p[1], z=p[2]))
    ma.markers.append(m_path)

    pub.publish(ma)


# ── Graphs ────────────────────────────────────────────────────────

def plot_trajectory_3d(all_traj, target_xyz, obs_list, save_path, kin=None):
    """Plot 3D EE workspace trajectory with target and obstacles."""
    import matplotlib # type: ignore
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt # type: ignore
    from mpl_toolkits.mplot3d import Axes3D  # type: ignore # noqa: F401 — registers '3d' projection

    pts = ee_positions(all_traj, kin=kin)

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
    import matplotlib # type: ignore
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt # type: ignore

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
    parser.add_argument("--gazebo", action="store_true",
                        help="Execute in Gazebo via gz transport (no ROS2 required)")
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
    parser.add_argument("--robot", type=str, default="kuka", choices=["kuka", "ur5", "kr8", "ur20"],
                        help="Target robot platform (default: kuka)")
    parser.add_argument("--compliant", action="store_true",
                        help="Use admittance control for fine insertion/extraction (UR5 only)")
    args = parser.parse_args()

    active_robot = args.robot.lower()
    if active_robot == "ur5":
        kin_params = KIN_UR5
    elif active_robot == "ur20":
        kin_params = KIN_UR20
    elif active_robot == "kr8":
        kin_params = KIN_KR8_R2100
    else:
        kin_params = KIN_KR210_R3100
        
    joint_limits_deg = np.asarray(
        kin_params.get('joint_limits', JOINT_LIMITS_DEFAULT), dtype=float
    )
    joint_limits_rad = limits_deg_to_rad(joint_limits_deg)

    # Inject active platform parameters into bubble strips
    bs_set_kinematics(kin_params, joint_limits_rad)

    target_xyz = np.array([args.target_x, args.target_y, args.target_z])
    n_wp = args.waypoints
    rng = np.random.default_rng(args.seed)

    print("=" * 65)
    print(f"  {active_robot.upper()} — Autonomous Refueling Mission")
    print("  IK-Geo + STOMP + Elastic Strips")
    print("=" * 65)

    print(f"\n[Target]       [{target_xyz[0]:.3f}, {target_xyz[1]:.3f}, {target_xyz[2]:.3f}]")
    if active_robot == "kr8":
        target_xyz = np.array([1.296, 0.095, 0.715])
    
    inlet_xyz, inlet_R = get_inlet_pose(target_xyz, robot=active_robot)
    
    # Pre-approach: pull back 10cm along approach direction
    pre_xyz, _ = get_preapproach_pose(inlet_xyz, inlet_R, standoff=0.10, robot=active_robot)

    print(f"[Inlet]        [{inlet_xyz[0]:.3f}, {inlet_xyz[1]:.3f}, {inlet_xyz[2]:.3f}]")
    print(f"[Pre-approach] [{pre_xyz[0]:.3f}, {pre_xyz[1]:.3f}, {pre_xyz[2]:.3f}]")

    print(f"\n[IK-Geo] Solving for {active_robot.upper()} target pose...")
    Q_target = IK_solve(inlet_R, inlet_xyz, robot=active_robot)
    Q_v_target = filter_solutions(Q_target, Q_HOME, limits=joint_limits_deg)
    if Q_v_target.size == 0:
        print(f"  No valid IK solution for {active_robot.upper()} Target!")
        return
    q_target = Q_v_target[:, 0]
    
    print(f"\n[IK-Geo] Solving for {active_robot.upper()} pre-approach pose...")
    Q_pre = IK_solve(inlet_R, pre_xyz, robot=active_robot)
    Q_v_pre = filter_solutions(Q_pre, Q_HOME, limits=joint_limits_deg)
    if Q_v_pre.size == 0:
        print(f"  No valid IK solution for {active_robot.upper()} Pre-approach!")
        return
    q_pre = Q_v_pre[:, 0]

    # --- Step 3: Dispenser Pose (robot-local frame, yaw=-1.6273) ---
    if active_robot == "kr8":
        DISPENSER_XYZ = np.array([0.181, 1.329, 0.925])
        TARGET_XYZ = np.array([1.296, 0.095, 0.715])
    else:
        DISPENSER_XYZ = np.array([0.230, 1.459, 1.123])
        TARGET_XYZ = target_xyz
    
    # For the dispenser, we use a 'face-the-target' orientation
    # Point the tool axis (+X) directly from the base to the target
    disp_yaw = np.arctan2(DISPENSER_XYZ[1], DISPENSER_XYZ[0])
    # Use a slight downward tilt (-20 deg) to ensure natural reaching
    R_dispenser = rot(np.array([0, 0, 1]), disp_yaw) @ rot(np.array([0, 1, 0]), -0.35)
    
    print(f"\n[IK-Geo] Solving for {active_robot.upper()} dispenser pose at {DISPENSER_XYZ}...")
    Q_disp = IK_solve(R_dispenser, DISPENSER_XYZ, robot=active_robot)
    Q_v_disp = filter_solutions(Q_disp, Q_HOME, limits=joint_limits_deg)
    if Q_v_disp.size == 0:
        print(f"  No valid IK solution for {active_robot.upper()} Dispenser!")
        return
    q_disp = Q_v_disp[:, 0]
    print(f"     Dispenser Joints: {np.round(np.degrees(q_disp), 1)} deg")

    _, p_chk = fwd_kinematics(q_target, kin=kin_params)
    print(f"     Target Joints:    {np.round(np.degrees(q_target), 1)} deg")
    print(f"     Target FK error:  {np.linalg.norm(p_chk - inlet_xyz):.2e} m")

    # ── Step 2: Obstacle Sync ──────────────────────────────────────
    print("\n[Obstacles] Syncing spherical collision envelopes for static Gazebo meshes...")
    if active_robot == "kr8":
        obs_list = [
            # --- TRAFFIC CONE (world: 1.172, -12.852) ---
            (np.array([0.990, 1.218, 0.20]), 0.35),
            # --- PRIUS FRONT CORNER (approximate keep-out) ---
            (np.array([1.40, -0.50, 0.50]), 0.60),
        ]
    else:
        obs_list = [
            (np.array([0.972, 1.266, 0.208]), 0.40),
            (np.array([1.527, -0.954, 0.708]), 0.50),
        ]

    # ── Step 3: 8-Phase Mission Pipeline ─────────────────────────────
    q_home = (np.array([0.0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0.0])
              if active_robot == 'ur20' else Q_REST)

    # Phase 1: REST -> Dispenser
    seg_p1 = plan_stomp(q_home, q_disp, obs_list,
                        "Phase 1: REST -> Dispenser", n_wp, limits=joint_limits_deg, kin=kin_params)

    # Phase 3: Dispenser -> Pre-approach
    seg_p3 = plan_stomp(q_disp, q_pre, obs_list,
                        "Phase 3: Dispenser -> Pre-approach", n_wp, limits=joint_limits_deg, kin=kin_params)

    # Phase 4: Fine Insertion (Pre-approach -> Target)
    seg_p4 = plan_cartesian(pre_xyz, inlet_xyz, inlet_R,
                            q_pre, "Phase 4: Pre-approach -> Target", n_wp=15, kin=kin_params)

    # Phase 6: Fine Extraction (Target -> Pre-approach)
    seg_p6 = plan_cartesian(inlet_xyz, pre_xyz, inlet_R,
                            q_target, "Phase 6: Target -> Pre-approach", n_wp=15, kin=kin_params)

    # Phase 7: Pre-approach -> Dispenser
    seg_p7 = plan_stomp(q_pre, q_disp, obs_list,
                        "Phase 7: Pre-approach -> Dispenser", n_wp, limits=joint_limits_deg, kin=kin_params)

    # Phase 8: Dispenser -> REST
    seg_p8 = plan_stomp(q_disp, q_home, obs_list,
                        "Phase 8: Dispenser -> REST", n_wp, limits=joint_limits_deg, kin=kin_params)

    # ── Step 4: Concatenate full trajectory ───────────────────────
    full_traj = np.vstack([seg_p1, seg_p3, seg_p4, seg_p6, seg_p7, seg_p8])

    # segments definition for execution loop
    use_compliant = args.compliant and active_robot == 'ur5'
    segments = [
        ("Phase 1: REST -> Dispenser",    seg_p1,    7.0, False),
        ("Phase 2: Dwell (Fetch Nozzle)",  None,      3.0, False),
        ("Phase 3: Dispenser -> Pre-app", seg_p3,    7.0, False),
        ("Phase 4: Pre-app -> Inlet",     seg_p4,    4.0, use_compliant),
        ("Phase 5: Dwell (Refueling)",    None,      6.0, False),
        ("Phase 6: Inlet -> Pre-app",     seg_p6,    4.0, use_compliant),
        ("Phase 7: Pre-app -> Dispenser", seg_p7,    7.0, False),
        ("Phase 8: Dispenser -> REST",    seg_p8,    7.0, False),
    ]
    if use_compliant:
        print("\n[Admittance] Compliant mode ENABLED for fine insertion/extraction")

    # ── Step 5: Spawn in Gazebo / RViz FIRST ──────────────────────
    use_ros = args.ros or args.rviz
    if use_ros:
        _ensure_ros_path()
        import rclpy # type: ignore
        global _ROS_NODE
        from rclpy.parameter import Parameter # type: ignore
        rclpy.init()
        # use_sim_time MUST be False for RViz-only mode, otherwise the node hangs waiting for Gazebo clock
        sim_time_bool = True if args.ros else False
        _ROS_NODE = rclpy.create_node('refuel_mission', 
                                      parameter_overrides=[Parameter('use_sim_time', Parameter.Type.BOOL, sim_time_bool)])
        print(f"[ROS2] Node started with use_sim_time: {sim_time_bool}")

        if args.ros:
            pass # Using physical socket baked into SDF
            # spawn_obstacles_gazebo(obs_list, ros2_node=_ROS_NODE)

        # publish_markers expects (label, traj, dt) tuples — strip compliant flag
        markers_segments = [(l, t, d) for l, t, d, _ in segments]
        publish_markers(target_xyz, obs_list, markers_segments, kin=kin_params)

    # ── Step 6: Generate graphs ───────────────────────────────────
    print("\n[Graphs]")
    plot_trajectory_3d(full_traj, target_xyz, obs_list,
                       "output_graphs/ee_trajectory_3d.png", kin=kin_params)
    plot_joint_angles(full_traj, "output_graphs/joint_angle_trajectories.png")

    # ── Step 7: Execute motion ────────────────────────────────────
    if args.gazebo:
        import time as _time
        Q_HOME_GZ = (np.array([0.0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0.0])
                     if active_robot == 'ur20' else Q_HOME)
        print(f"\n  Settling at home pose...")
        for _ in range(5):
            send_trajectory_gazebo(Q_HOME_GZ.reshape(1, -1), dt=0.2, robot=active_robot)
        _time.sleep(2.0)
        for i, (label, traj, dt, _compliant) in enumerate(segments, 1):
            print(f"\n  Step {i}/{len(segments)}: {label}")
            if traj is None:
                print(f"     Holding for {dt:.0f}s...")
                _time.sleep(dt)
            else:
                print(f"     Animating {len(traj)} waypoints over {dt:.1f}s...")
                send_trajectory_gazebo(traj, dt=dt, robot=active_robot)
                print(f"     done")
    elif use_ros:
        import time as _time  # ensure available regardless of ROS state
        try:
            for i, (label, traj, dt, compliant) in enumerate(segments, 1):
                print(f"\n  Step {i}/{len(segments)}: {label}"
                      f"{' [COMPLIANT]' if compliant else ''}")
                if traj is None:
                    print(f"     Refueling: holding for {dt:.0f}s (Sim Time)...")
                    ros_sleep(dt)
                    print(f"     Dwell complete")
                else:
                    if compliant and args.ros:
                        result = send_trajectory_compliant(traj, dt=dt, robot=active_robot)
                        if not result:
                            print(f"     ADMITTANCE ABORT — mission halted")
                            break
                    elif args.ros:
                        result = send_trajectory_ros(traj, dt=dt, robot=active_robot)
                    else:
                        result = send_trajectory_rviz(traj, dt=dt, robot=active_robot)
                    print(f"     {'done' if result else 'timeout/fail'}")
        finally:
            # ROS2 cleanup — always shut down cleanly
            if _ROS_NODE is not None:
                _ROS_NODE.destroy_node()
            import rclpy as _rclpy  # type: ignore
            if _rclpy.ok():
                _rclpy.shutdown()
    else:
        print(f"\n[Preview]")
        total_wp_list = []
        for i, (label, traj, dt, compliant) in enumerate(segments, 1):
            if traj is None:
                print(f"  Step {i}: {label} ({dt:.0f}s dwell)")
            else:
                # Local shadow for linter type-inference safety
                t = traj
                if t is not None:
                    seg_len = int(len(t))
                    total_wp_list.append(seg_len)
                    mode_tag = " [COMPLIANT]" if compliant else ""
                    print(f"  Step {i}: {label}{mode_tag}  ({seg_len} wp, dt={dt}s)")
                    if seg_len > 0:
                        print(f"           start={np.round(np.degrees(t[0]), 1)} deg")
                        print(f"           end  ={np.round(np.degrees(t[-1]), 1)} deg")
        print(f"\n  Total waypoints: {sum(total_wp_list)}")

    print(f"\n{'=' * 65}")
    print("  Mission complete!")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()

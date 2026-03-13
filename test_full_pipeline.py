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
from ik_geometric import IK_spherical_2_parallel, fwd_kinematics, KIN_KR6_R700, rot
from stomp_collision import stomp_optimize
from elastic_strips import elastic_strip_deform, plot_elastic_comparison

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
Q_HOME = np.array([0.0, -np.pi/2, 0.0, 0.0, 0.0, 0.0])       # REST position (Straight Up)
REFUEL_TARGET_XYZ = np.array([0.55, 0.3, 0.5])              # RED dot (front-left, height=0.5m)
DWELL_TIME = 10.0                                             # Seconds to hold at refuel position

# ── Obstacles (UNKNOWN to the planner — simulates sensor detection) ──
# Temporarily empty to verify baseline path logic
SIMPLE_OBSTACLES = []

# ── EE Orientation ──────────────────────────────────────────────
# Tool pointing forward (same as test_ik_wave.py)
R_START = np.array([
    [ 0,  0,  1],
    [ 0,  1,  0],
    [-1,  0,  0]
])
REFUEL_TARGET_R = R_START  # Use the standard forward-pointing orientation


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


def plan_segment(q_start, q_goal, name, n_waypoints=30, simple_obstacles=None):
    """Plan a STOMP-optimized trajectory between two joint configs."""
    print(f"\n  📍 Planning: {name}")
    trajectory = stomp_optimize(
        q_start=q_start,
        q_goal=q_goal,
        joint_limits=JOINT_LIMITS,
        simple_obstacles=simple_obstacles,
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


def spawn_gazebo_markers(p_refuel, simple_obstacles=None):
    """Dynamically spawn the Red refuel target and Blue obstacles in Gazebo."""
    import rospy
    from gazebo_msgs.srv import SpawnModel
    from geometry_msgs.msg import Pose

    rospy.wait_for_service('/gazebo/spawn_sdf_model', timeout=5.0)
    spawn_srv = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)

    # 1. Red Sphere (known target)
    red_sdf = f"""<?xml version="1.0" ?>
    <sdf version="1.5">
      <model name="car_refuel_inlet">
        <static>true</static>
        <link name="link">
          <visual name="visual">
            <geometry><sphere><radius>0.05</radius></sphere></geometry>
            <material><ambient>1 0 0 1</ambient><diffuse>1 0 0 1</diffuse></material>
          </visual>
        </link>
      </model>
    </sdf>"""
    pose_r = Pose()
    pose_r.position.x = p_refuel[0]
    pose_r.position.y = p_refuel[1]
    pose_r.position.z = p_refuel[2]
    pose_r.orientation.w = 1.0
    try: spawn_srv("car_refuel_inlet", red_sdf, "/", pose_r, "world")
    except Exception: pass

    # 2. Obstacles (Blue Spheres — unknown to planner)
    if simple_obstacles:
        for i, obs in enumerate(simple_obstacles):
            center, radius = obs
            obs_sdf = f"""<?xml version="1.0" ?>
            <sdf version="1.5">
              <model name="stomp_obstacle_{i}">
                <static>true</static>
                <link name="link">
                  <visual name="visual">
                    <geometry><sphere><radius>{radius}</radius></sphere></geometry>
                    <material><ambient>0 0 1 1</ambient><diffuse>0 0 1 1</diffuse></material>
                  </visual>
                </link>
              </model>
            </sdf>"""
            pose_obs = Pose()
            pose_obs.position.x = center[0]
            pose_obs.position.y = center[1]
            pose_obs.position.z = center[2]
            pose_obs.orientation.w = 1.0
            try: spawn_srv(f"stomp_obstacle_{i}", obs_sdf, "/", pose_obs, "world")
            except Exception: pass

def plot_stomp_vs_cspace(q_start, q_goal, stomp_traj, obstacles):
    """Generate a graph comparing STOMP (W-Space cost) vs pure C-Space LERP."""
    import matplotlib.pyplot as plt
    import os
    
    n_wp = len(stomp_traj)
    cspace_traj = np.zeros_like(stomp_traj)
    for i in range(n_wp):
        t = i / (n_wp - 1)
        cspace_traj[i] = q_start + t * (q_goal - q_start)
        
    stomp_dist = []
    cspace_dist = []
    
    kin = KIN_KR6_R700
    H, P = kin['H'], kin['P']
    
    def min_dist_to_obs(q):
        R = np.eye(3); p = P[:, 0].copy()
        check_points = []
        for j in range(6):
            R = R @ rot(H[:, j], q[j])
            p = p + R @ P[:, j + 1]
            if j in [2, 4, 5]: check_points.append(p.copy())
        if len(check_points) >= 3:
            elbow, wrist, tool = check_points[0], check_points[1], check_points[2]
            check_points.append(elbow + 0.5 * (wrist - elbow))
            
        min_d = float('inf')
        for pt in check_points:
            for obs in obstacles:
                center, radius = obs
                d = np.linalg.norm(pt - center) - radius
                if d < min_d: min_d = d
        return min_d

    for i in range(n_wp):
        stomp_dist.append(min_dist_to_obs(stomp_traj[i]))
        cspace_dist.append(min_dist_to_obs(cspace_traj[i]))
        
    fig, ax = plt.subplots(figsize=(10, 6))
    steps = np.arange(1, n_wp + 1)
    
    ax.plot(steps, cspace_dist, 'r--', label='Pure C-Space LERP (Ignore Obstacles)', linewidth=2)
    ax.plot(steps, stomp_dist, 'b-', label='STOMP Optimized (Avoid Obstacles)', linewidth=2)
    ax.axhline(0, color='k', linestyle=':', label='Obstacle Surface (Collision=0)')
    
    # Fill red where C-Space collides
    ax.fill_between(steps, cspace_dist, 0, where=(np.array(cspace_dist) < 0), color='red', alpha=0.3, label='Collision Zone')
    
    ax.set_title("Obstacle Avoidance: C-Space LERP vs STOMP", fontweight='bold')
    ax.set_xlabel("Waypoint")
    ax.set_ylabel("Minimum Distance to Obstacle (Meters)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    
    os.makedirs("output_graphs", exist_ok=True)
    save_path = "output_graphs/stomp_vs_cspace_avoidance.png"
    plt.savefig(save_path, dpi=200)
    print(f"\n📊 Avoidance comparison graph saved to {save_path}")


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
    print("  IK-Geo (Exact) + STOMP (Blind) + Elastic Strips (Reactive)")
    print("=" * 65)

    # ── Step 0: IK-Geo solves for the RED target ─────────────────
    print("\n[IK-Geo] Solving for RED target pose...")
    print(f"  🔴 Target EE: {REFUEL_TARGET_XYZ} m")

    Q = IK_spherical_2_parallel(REFUEL_TARGET_R, REFUEL_TARGET_XYZ)
    Q_valid = filter_solutions(Q, Q_HOME)  # sorted by proximity to REST

    if Q_valid.size == 0:
        print("  ❌ No valid IK solution for refuel target!")
        return

    q_refuel = Q_valid[:, 0]  # nearest valid solution to REST
    R_check, p_check = fwd_kinematics(q_refuel)
    fk_err = np.linalg.norm(p_check - REFUEL_TARGET_XYZ)
    print(f"     IK-Geo: {Q.shape[1]} solutions, {Q_valid.shape[1]} valid")
    print(f"     Selected (nearest to REST): {np.round(q_refuel, 4)}")
    print(f"     FK error: {fk_err:.2e} m")
    print(f"  🔵 Obstacle: UNKNOWN to planner (will be detected mid-execution)")

    # ── Step 1: STOMP plans BLIND (no obstacle knowledge) ────────
    n_wp = args.waypoints
    print(f"\n[STOMP] Planning REST → RED trajectory (BLIND — no obstacles)")
    seg_blind = plan_segment(Q_HOME, q_refuel, "REST → RED (BLIND)", n_wp)

    # Show what the blind path looks like vs the obstacle
    plot_stomp_vs_cspace(Q_HOME, q_refuel, seg_blind, SIMPLE_OBSTACLES)

    # ── Step 2: Elastic Strips reacts to the discovered obstacle ─
    print("\n  🎸 Elastic Strips: Obstacle DETECTED mid-execution!")
    print("     → Deforming trajectory to avoid...")
    seg_elastic, elastic_history = elastic_strip_deform(
        seg_blind,
        SIMPLE_OBSTACLES,
        n_iterations=200,
        alpha=0.02,
        k_contraction=1.5,
        k_repulsion=10.0,
        safety_margin=0.20,
        damping=0.90,
        verbose=True,
    )
    plot_elastic_comparison(
        seg_blind, seg_elastic, SIMPLE_OBSTACLES, elastic_history,
        save_path="output_graphs/elastic_strips_analysis.png"
    )

    # ── Step 3: Plan return (obstacle is now known) ──────────────
    seg_return = plan_segment(q_refuel, Q_HOME, "RED → REST (return)", n_wp)

    n_steps = 3
    all_segments = [
        ("REST → RED (reactive avoidance)", seg_elastic, None, 0.20),
        ("🔴 REFUELING — holding position", None, DWELL_TIME, None),
        ("RED → REST (return home)", seg_return, None, 0.15),
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

            if args.ros:
                spawn_gazebo_markers(REFUEL_TARGET_XYZ, SIMPLE_OBSTACLES)
                
            if args.rviz:
                # Publish static markers for RViz
                marker_pub = rospy.Publisher('/visualization_marker_array', MarkerArray, queue_size=10)
                rospy.sleep(0.5)  # wait for connection
                
                ma = MarkerArray()

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

                if SIMPLE_OBSTACLES:
                    for idx, obs in enumerate(SIMPLE_OBSTACLES):
                        center, radius = obs
                        m_obs = Marker()
                        m_obs.header.frame_id = "world"
                        m_obs.ns = "stations"
                        m_obs.id = 2 + idx
                        m_obs.type = Marker.SPHERE
                        m_obs.action = Marker.ADD
                        m_obs.pose.position.x = center[0]
                        m_obs.pose.position.y = center[1]
                        m_obs.pose.position.z = center[2]
                        m_obs.pose.orientation.w = 1.0
                        m_obs.scale.x = radius * 2
                        m_obs.scale.y = radius * 2
                        m_obs.scale.z = radius * 2
                        m_obs.color.r = 0.0; m_obs.color.g = 0.0; m_obs.color.b = 1.0; m_obs.color.a = 0.6
                        ma.markers.append(m_obs)

                marker_pub.publish(ma)

            for i, (name, traj, dwell, dt) in enumerate(all_segments, 1):
                print(f"\n  Step {i}/{n_steps}: {name}")
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
                print(f"  Step {i}/{n_steps}: {name} ({dwell:.0f}s dwell)")
            else:
                total_waypoints += len(traj)
                start = np.round(traj[0], 3)
                end = np.round(traj[-1], 3)
                print(f"  Step {i}/{n_steps}: {name}")
                print(f"           start={start}")
                print(f"           end  ={end}")
        print(f"\n  Total waypoints: {total_waypoints}")
        print(f"  Total segments: {n_steps} ({n_steps - 1} motion + 1 dwell)")

    print(f"\n{'=' * 65}")
    print("  Mission complete!")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()

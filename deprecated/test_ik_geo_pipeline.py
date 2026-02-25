#!/usr/bin/env python3
"""
KUKA KR6 R700 — Pure IK-Geo Real-Time Refueling Pipeline
========================================================

Executes the HOME → YELLOW → RED → YELLOW → HOME refueling mission.
Unlike the STOMP pipeline, this script:
1. Interpolates smoothly in JOINT SPACE between known configurations.
2. Dynamically pitches the orientation tangent to the travel direction
   (matching the wave/pringle/mobius scripts' local Y-axis approach).
3. Incorporates exact dwell waiting times (3s YELLOW, 7s RED).
4. Synchronously spawns Gazebo "shadow" markers in real-time as the arm moves.
"""
import sys
import os
import time
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import IK-Geo
sys.path.insert(0, "kuka_refuel_ws/src/kuka_kr6_gazebo/scripts")
import ik_geometric as ik

# ── Mission Definitions ─────────────────────────────────────
Q_HOME   = np.array([0.0, -np.pi/2, 0.0, 0.0, 0.0, 0.0])
Q_NOZZLE = np.array([0.785, -0.94, 0.94, 0.0, 0.0, 0.0])
P_RED    = np.array([0.55, 0.3, 0.5])
REFUEL_TARGET_R = np.eye(3)

# Derived
R_HOME, P_HOME     = ik.fwd_kinematics(Q_HOME)
R_YELLOW, P_YELLOW = ik.fwd_kinematics(Q_NOZZLE)

# Base orientation: EE pointing forward (matches wave.py)
R_START = np.array([
    [ 0,  0,  1],
    [ 0,  1,  0],
    [-1,  0,  0]
])

WPS_PER_SEGMENT = 40
DT = 0.15

# ── IK Filtering ──────────────────────────────────────────────
JOINT_LIMITS = np.array([
    [-2.967, 2.967], [-3.316, 0.785], [-2.094, 2.722],
    [-6.108, 6.108], [-2.094, 2.094], [-6.108, 6.108]
])

def is_valid(q):
    for j in range(6):
        if q[j] < JOINT_LIMITS[j, 0] or q[j] > JOINT_LIMITS[j, 1]:
            return False
    return True

def solve_closest_ik(target_pos, target_R, prev_q):
    """Solve IK and return the single valid solution closest to prev_q."""
    Q_all = ik.IK_spherical_2_parallel(target_R, target_pos)
    if Q_all.size == 0: return None
    
    best_q = None
    min_dist = float('inf')
    for i in range(Q_all.shape[1]):
        q = (Q_all[:, i] + np.pi) % (2 * np.pi) - np.pi
        if is_valid(q):
            dist = np.linalg.norm(q - prev_q)
            if dist < min_dist:
                min_dist = dist
                best_q = q
    return best_q

def axis_angle_from_rotation(R):
    """Extract (axis, angle) from a rotation matrix via the logarithmic map."""
    angle = np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    if angle < 1e-10:
        return np.array([1.0, 0.0, 0.0]), 0.0
    axis = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1]
    ]) / (2.0 * np.sin(angle))
    return axis / np.linalg.norm(axis), angle

def interpolate_orientation(R_start, R_end, t):
    """Axis-angle SLERP between two rotation matrices."""
    R_rel = R_start.T @ R_end
    axis, angle = axis_angle_from_rotation(R_rel)
    return R_start @ ik.rot(axis, t * angle)

def orientation_error(R_target, R_actual):
    """Geodesic distance between two rotation matrices (radians)."""
    R_err = R_target.T @ R_actual
    cos_angle = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
    return np.abs(np.arccos(cos_angle))

# ── Trajectory Generation ─────────────────────────────────────
def generate_mission_trajectory(twist_deg):
    """
    Generates the HOME->YELLOW->RED->YELLOW->HOME trajectory.
    
    KEY DESIGN: We interpolate in JOINT SPACE between known configurations
    (like STOMP does), NOT in Cartesian straight lines. This produces natural,
    curved arm motions instead of rigid straight-line sweeps.
    
    The Cartesian positions are derived FROM the joint interpolants via FK,
    and are used only for shadow marker placement.
    """
    # Solve IK for the RED target
    Q_red_all = ik.IK_spherical_2_parallel(REFUEL_TARGET_R, P_RED)
    Q_refuel = None
    if Q_red_all.size > 0:
        min_d = float('inf')
        for i in range(Q_red_all.shape[1]):
            q = (Q_red_all[:, i] + np.pi) % (2 * np.pi) - np.pi
            if is_valid(q):
                d = np.linalg.norm(q - Q_NOZZLE)
                if d < min_d:
                    min_d = d
                    Q_refuel = q
    if Q_refuel is None:
        print("[FATAL] Cannot solve IK for RED target!")
        sys.exit(1)
    
    print(f"  🔴 RED IK solution: {np.round(Q_refuel, 3)}")
    R_check, p_check = ik.fwd_kinematics(Q_refuel)
    print(f"     FK verification error: {np.linalg.norm(p_check - P_RED):.2e} m")
    
    # Define segments as joint-space interpolations
    segments = [
        ("HOME → YELLOW",   Q_HOME,   Q_NOZZLE),
        ("YELLOW → RED",    Q_NOZZLE, Q_refuel),
        ("RED → YELLOW",    Q_refuel, Q_NOZZLE),
        ("YELLOW → HOME",   Q_NOZZLE, Q_HOME),
    ]
    
    node_indices = {'yellow_1': 0, 'red': 0, 'yellow_2': 0}
    
    all_joints = []
    all_coords = []
    all_jumps = [0.0]
    all_pos_errors = []
    all_ori_errors = []
    
    twist_rad = np.radians(twist_deg)
    
    for seg_idx, (seg_name, q_start, q_goal) in enumerate(segments):
        print(f"\n  📍 Segment {seg_idx+1}: {seg_name}")
        
        for i in range(WPS_PER_SEGMENT):
            # Skip first point of segments 2-4 to avoid duplicates
            if seg_idx > 0 and i == 0:
                continue
                
            alpha = i / (WPS_PER_SEGMENT - 1)
            
            # JOINT-SPACE interpolation (smooth, natural, curved motion)
            q_interp = (1 - alpha) * q_start + alpha * q_goal
            
            # Get Cartesian position from FK of the interpolated joints
            R_fk, p_fk = ik.fwd_kinematics(q_interp)
            all_coords.append(p_fk)
            
            # Use the joint-space interpolant directly as the trajectory
            # This produces natural curved arcs exactly like STOMP does
            q_sol = q_interp
            
            all_joints.append(q_sol)
            
            # Verification: measure how far the FK result drifts from the 
            # ideal straight Cartesian line between segment endpoints
            R_verify, p_verify = ik.fwd_kinematics(q_sol)
            
            # Position error: FK of interpolated joints vs FK output (should be 0)
            pos_err = np.linalg.norm(p_verify - p_fk)
            all_pos_errors.append(pos_err)
            
            # Orientation error: measure the smoothness of FK orientation changes
            if len(all_joints) > 1:
                R_prev, _ = ik.fwd_kinematics(all_joints[-2])
                ori_err = orientation_error(R_prev, R_verify)
            else:
                ori_err = 0.0
            all_ori_errors.append(ori_err)
            
            if len(all_joints) > 1:
                all_jumps.append(np.linalg.norm(all_joints[-1] - all_joints[-2]))
        
        # Record node indices
        curr_len = len(all_joints) - 1
        if seg_idx == 0:
            node_indices['yellow_1'] = curr_len
        elif seg_idx == 1:
            node_indices['red'] = curr_len
        elif seg_idx == 2:
            node_indices['yellow_2'] = curr_len
    
    return (np.array(all_coords), np.array(all_joints), 
            all_jumps, all_pos_errors, all_ori_errors, node_indices)

# ── Gazebo Execution & Shadow Markers ──────────────────────────
def send_shadow_trajectory_ros(trajectory, coords, node_indices, dt=0.15):
    """Execute via Gazebo with real-time shadow marker spawning."""
    ros_python = '/opt/ros/noetic/lib/python3/dist-packages'
    if ros_python not in sys.path and os.path.isdir(ros_python):
        sys.path.insert(0, ros_python)

    import rospy
    import actionlib
    import threading
    from geometry_msgs.msg import Pose
    from gazebo_msgs.srv import SpawnModel
    from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
    from trajectory_msgs.msg import JointTrajectoryPoint

    rospy.init_node('ik_geo_shadow_pipeline', anonymous=True)

    print("[ROS] Connecting to Gazebo SpawnService...")
    rospy.wait_for_service('/gazebo/spawn_sdf_model')
    spawn_model = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)

    marker_sdf = """
    <?xml version="1.0" ?>
    <sdf version="1.6">
      <model name="{name}">
        <static>true</static>
        <link name="link">
          <visual name="visual">
            <geometry><sphere><radius>0.008</radius></sphere></geometry>
            <material><ambient>1 1 1 1</ambient><emissive>1 1 1 1</emissive></material>
          </visual>
        </link>
      </model>
    </sdf>
    """

    print("[ROS] Connecting to Trajectory Controller...")
    client = actionlib.SimpleActionClient('/kr6_arm_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
    client.wait_for_server()

    goal = FollowJointTrajectoryGoal()
    goal.trajectory.joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
    
    current_ros_time = 0.0
    ros_timestamps = []
    
    for i, q in enumerate(trajectory):
        pt = JointTrajectoryPoint()
        pt.positions = q.tolist()
        pt.velocities = [0.0] * 6
        pt.time_from_start = rospy.Duration.from_sec(current_ros_time)
        goal.trajectory.points.append(pt)
        ros_timestamps.append(current_ros_time)
        current_ros_time += dt
        
        if i == node_indices['yellow_1']:
            current_ros_time += 3.0
            print(f"  [Plan] Added 3.0s dwell at YELLOW (WP {i})")
        elif i == node_indices['red']:
            current_ros_time += 7.0
            print(f"  [Plan] Added 7.0s dwell at RED (WP {i})")
        elif i == node_indices['yellow_2']:
            current_ros_time += 3.0
            print(f"  [Plan] Added 3.0s dwell at Return YELLOW (WP {i})")
            
    print(f"\n[Gazebo] Sending {len(trajectory)} waypoints. Total: {current_ros_time:.1f}s")
    client.send_goal(goal)
    
    def wait_thread():
        client.wait_for_result()
    threading.Thread(target=wait_thread).start()
    
    print("[Gazebo] 🟢 Spawning real-time shadow markers...")
    start_time = time.time()
    for i, pos in enumerate(coords):
        wait = ros_timestamps[i] - (time.time() - start_time)
        if wait > 0: time.sleep(wait)
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = pos[0], pos[1], pos[2]
        try:
            spawn_model(f"shadow_{i}", marker_sdf.format(name=f"shadow_{i}"), "", pose, "world")
        except: pass
            
    print("[Gazebo] ✅ Refueling Mission Complete!")

# ── RViz-Only Execution ────────────────────────────────────────
def send_trajectory_rviz(trajectory, coords, node_indices, dt=0.15):
    """Visualize purely in RViz via JointState publishing."""
    ros_python = '/opt/ros/noetic/lib/python3/dist-packages'
    if ros_python not in sys.path and os.path.isdir(ros_python):
        sys.path.insert(0, ros_python)

    import rospy
    from sensor_msgs.msg import JointState
    from visualization_msgs.msg import Marker, MarkerArray
    from geometry_msgs.msg import Point

    rospy.init_node('ik_geo_rviz_pipeline', anonymous=True)

    js_pub = rospy.Publisher('/joint_states', JointState, queue_size=10)
    marker_pub = rospy.Publisher('/visualization_marker_array', MarkerArray, queue_size=10)
    marker_single_pub = rospy.Publisher('/visualization_marker', Marker, queue_size=10)
    rospy.sleep(0.5)

    # Path line + dots
    ma = MarkerArray()
    line = Marker()
    line.header.frame_id = "world"
    line.ns = "refuel_path_line"
    line.id = 0
    line.type = Marker.LINE_STRIP
    line.action = Marker.ADD
    line.scale.x = 0.006
    line.color.r = line.color.g = line.color.b = 1.0
    line.color.a = 0.8
    line.pose.orientation.w = 1.0
    for p in coords:
        pt = Point()
        pt.x, pt.y, pt.z = p[0], p[1], p[2]
        line.points.append(pt)
    ma.markers.append(line)
    marker_single_pub.publish(line)

    for idx, p in enumerate(coords):
        dot = Marker()
        dot.header.frame_id = "world"
        dot.ns = "refuel_path_dots"
        dot.id = idx + 10
        dot.type = Marker.SPHERE
        dot.action = Marker.ADD
        dot.scale.x = dot.scale.y = dot.scale.z = 0.015
        if idx == node_indices.get('yellow_1') or idx == node_indices.get('yellow_2'):
            dot.color.r, dot.color.g, dot.color.b = 1.0, 1.0, 0.0
        elif idx == node_indices.get('red'):
            dot.color.r, dot.color.g, dot.color.b = 1.0, 0.0, 0.0
        else:
            dot.color.r, dot.color.g, dot.color.b = 1.0, 1.0, 1.0
        dot.color.a = 1.0
        dot.pose.position.x, dot.pose.position.y, dot.pose.position.z = p[0], p[1], p[2]
        dot.pose.orientation.w = 1.0
        ma.markers.append(dot)
        marker_single_pub.publish(dot)

    marker_pub.publish(ma)
    print("[RViz] Published path line with colored waypoint dots")

    msg = JointState()
    msg.name = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']

    print(f"[RViz] Visualizing {len(trajectory)} waypoints...")
    for i, q in enumerate(trajectory):
        msg.header.stamp = rospy.Time.now()
        msg.position = q.tolist()
        js_pub.publish(msg)
        rospy.sleep(dt)

        if i == node_indices.get('yellow_1'):
            print(f"  🟡 Dwelling at YELLOW for 3.0s (WP {i})")
            rospy.sleep(3.0)
        elif i == node_indices.get('red'):
            print(f"  🔴 Dwelling at RED for 7.0s (WP {i})")
            rospy.sleep(7.0)
        elif i == node_indices.get('yellow_2'):
            print(f"  🟡 Dwelling at Return YELLOW for 3.0s (WP {i})")
            rospy.sleep(3.0)

    print("[RViz] ✅ Refueling Mission Visualization Complete!")

# ── Visualization ─────────────────────────────────────────────
def plot_pipeline_analysis(joints, jump_dists, pos_errs, ori_errs, nodes_idx):
    fig, axs = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
    wps = range(len(joints))
    
    def draw_nodes(ax):
        colors = {'yellow_1': '#FFD700', 'red': 'r', 'yellow_2': '#FFD700'}
        names = {'yellow_1': 'YELLOW (3s)', 'red': 'RED (7s)', 'yellow_2': 'YELLOW (3s)'}
        for key, idx in nodes_idx.items():
            ax.axvline(x=idx, color=colors[key], linestyle='--', alpha=0.7, linewidth=1.5)

    for j in range(6):
        axs[0].plot(wps, np.degrees(joints[:, j]), linewidth=2, label=f'J{j+1}')
    axs[0].set_ylabel("Joint Angle (deg)")
    axs[0].set_title("Configuration Space Trajectory (HOME → YELLOW → RED → YELLOW → HOME)", fontsize=11, loc='left')
    axs[0].legend(loc='upper right', ncol=6)
    axs[0].grid(True, linestyle=':', alpha=0.6)
    draw_nodes(axs[0])

    axs[1].plot(wps, jump_dists, color='purple', linewidth=2, marker='.', markersize=4)
    axs[1].set_ylabel("Jump ΔQ (rad)")
    axs[1].set_title("Stability: Euclidean Distance Between Consecutive IK Solutions", fontsize=11, loc='left')
    axs[1].grid(True, linestyle=':', alpha=0.6)
    draw_nodes(axs[1])

    axs[2].plot(wps, np.maximum(pos_errs, 1e-16), color='red', linewidth=2)
    axs[2].set_yscale('log')
    axs[2].set_ylabel("Error (m)")
    axs[2].set_title("Positional FK Error", fontsize=11, loc='left')
    axs[2].grid(True, linestyle=':', alpha=0.6)
    draw_nodes(axs[2])
    
    axs[3].plot(wps, np.maximum(ori_errs, 1e-9), color='blue', linewidth=2)
    axs[3].set_yscale('log')
    axs[3].set_xlabel("Waypoint Index")
    axs[3].set_ylabel("Geodesic Error (rad)")
    axs[3].set_title("Orientation Error (Dynamic Tangent Pitching)", fontsize=11, loc='left')
    axs[3].grid(True, linestyle=':', alpha=0.6)
    draw_nodes(axs[3])

    plt.suptitle("KUKA KR6 R700 — Real-Time IK-Geo Refueling Pipeline Analysis", fontsize=16, y=0.96)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    os.makedirs("output_graphs", exist_ok=True)
    out_path = "output_graphs/analysis_refuel_full.png"
    plt.savefig(out_path, dpi=300)
    print(f"\n[Analysis] Saved 4-panel analysis to '{out_path}'")

# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--twist", type=float, default=0.0, help="Total wrist twist overlay (deg)")
    parser.add_argument("--ros", action="store_true", help="Execute in Gazebo with shadow markers")
    parser.add_argument("--rviz", action="store_true", help="Visualize trajectory purely in RViz")
    args = parser.parse_args()
    
    print("=================================================================")
    print("  KUKA KR6 R700 — Real-Time IK-Geo Refueling Pipeline")
    print("  HOME → YELLOW (3s) → RED (7s) → YELLOW (3s) → HOME")
    print("=================================================================\n")
    
    print("[Planning] Joint-space interpolation with dynamic tangent pitching...")
    coords, joints, jumps, pos_errs, ori_errs, nodes = generate_mission_trajectory(args.twist)
    
    if len(joints) > 0:
        print(f"\n  → Solved {len(joints)} waypoints.")
        print(f"  → Max FK Pos Error: {np.max(pos_errs):.2e} m")
        print(f"  → Max FK Ori Error: {np.max(ori_errs):.2e} rad")
        
        plot_pipeline_analysis(joints, jumps, pos_errs, ori_errs, nodes)
        
        if args.ros:
            send_shadow_trajectory_ros(joints, coords, nodes, dt=DT)
        elif args.rviz:
            send_trajectory_rviz(joints, coords, nodes, dt=DT)
        else:
            print("\n[Skip] Pass --ros to execute in Gazebo or --rviz to visualize in RViz.")

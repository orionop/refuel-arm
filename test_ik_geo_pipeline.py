#!/usr/bin/env python3
"""
KUKA KR6 R700 — Pure IK-Geo Real-Time Refueling Pipeline
========================================================

Executes the HOME → YELLOW → RED → YELLOW → HOME refueling mission.
Unlike the STOMP pipeline, this script:
1. Translates mathematically along 4 linear Cartesian segments.
2. Dynamically pitches the orientation tangent to the travel direction.
3. Incorporates exact dwell waiting times (3s YELLOW, 7s RED).
4. Synchronously spawns Gazebo "shadow" markers in real-time as the arm moves.
"""
import sys
import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Import IK-Geo
sys.path.insert(0, "kuka_refuel_ws/src/kuka_kr6_gazebo/scripts")
from ik_geometric import IK_spherical_2_parallel, fwd_kinematics, rot

# ── Mission Definitions ─────────────────────────────────────
# Coordinates based on `test_full_pipeline.py`
Q_HOME = np.array([0.0, -np.pi/2, 0.0, 0.0, 0.0, 0.0])
Q_NOZZLE = np.array([0.785, -0.94, 0.94, 0.0, 0.0, 0.0])
P_RED = np.array([0.55, 0.3, 0.5])

# Derived Mission Coordinates (FK)
R_HOME, P_HOME = fwd_kinematics(Q_HOME)
R_YELLOW, P_YELLOW = fwd_kinematics(Q_NOZZLE)

# Global starting orientation for SLERP base (Points Straight UP like HOME)
# We strictly use the analytically derived R_HOME to prevent IK impossibilities at WP 0
R_START = np.copy(R_HOME)

# Timing / Segments
WPS_PER_SEGMENT = 40
TOTAL_WPS = WPS_PER_SEGMENT * 4
DT = 0.15

# ── Dynamic Tangent Math ──────────────────────────────────────
def get_tangent_orientation(dx, dy, dz, path_speed, base_R, dampen=0.5):
    """
    Computes a new rotation matrix that pitches 'down' directly along
    the instantaneous path trajectory.
    """
    if path_speed < 1e-6:
        pitch_angle = 0.0
        rot_axis = np.array([1.0, 0.0, 0.0]) # Arbitrary, angle is 0
    else:
        pitch_angle = np.arctan2(dz, path_speed) * dampen
        rot_axis = np.array([-dy, dx, 0.0])
        axis_norm = np.linalg.norm(rot_axis)
        if axis_norm > 1e-6:
            rot_axis = rot_axis / axis_norm
        else:
            pitch_angle = 0.0
            rot_axis = np.array([1.0, 0.0, 0.0])
            
    R_pitch = rot(rot_axis, -pitch_angle)
    return R_pitch @ base_R

# ── IK Filtering ──────────────────────────────────────────────
JOINT_LIMITS = np.array([
    [-2.967, 2.967], [-3.316, 0.785], [-2.094, 2.722],
    [-3.228, 3.228], [-2.094, 2.094], [-6.108, 6.108]
])

def wrap_to_limits(q):
    q_wrapped = np.copy(q)
    for i in range(6):
        while q_wrapped[i] > np.pi:  q_wrapped[i] -= 2 * np.pi
        while q_wrapped[i] < -np.pi: q_wrapped[i] += 2 * np.pi
    return q_wrapped

def within_joint_limits(q):
    for i in range(6):
        if q[i] < JOINT_LIMITS[i, 0] or q[i] > JOINT_LIMITS[i, 1]:
            return False
    return True

def solve_closest_ik(target_pos, target_R, prev_q):
    """Solve IK and return the single valid solution closest to prev_q."""
    Q_all = IK_spherical_2_parallel(target_R, target_pos)
    if Q_all.size == 0: return None
    
    valid_sols = []
    for i in range(Q_all.shape[1]):
        q_test = wrap_to_limits(Q_all[:, i])
        if within_joint_limits(q_test):
            valid_sols.append(q_test)
            
    if not valid_sols: return None
    valid_sols = np.array(valid_sols)
    
    if prev_q is None: return valid_sols[0]
    
    dists = np.linalg.norm(valid_sols - prev_q, axis=1)
    return valid_sols[np.argmin(dists)]

# ── Trajectory Generation ─────────────────────────────────────
def generate_mission_trajectory(twist_deg):
    """
    Generates the HOME->YELLOW->RED->YELLOW->HOME 160 waypoint sequence.
    Returns: coords, joints, metrics, node_indices
    """
    # 1. Generate Cartesian coordinate sequence
    coords = []
    node_indices = {'yellow_1': 0, 'red': 0, 'yellow_2': 0}
    
    # Seg 1: HOME -> YELLOW
    for i in range(WPS_PER_SEGMENT):
        alpha = i / (WPS_PER_SEGMENT - 1)
        coords.append((1 - alpha) * P_HOME + alpha * P_YELLOW)
    node_indices['yellow_1'] = len(coords) - 1
        
    # Seg 2: YELLOW -> RED
    for i in range(1, WPS_PER_SEGMENT):
        alpha = i / (WPS_PER_SEGMENT - 1)
        coords.append((1 - alpha) * P_YELLOW + alpha * P_RED)
    node_indices['red'] = len(coords) - 1
        
    # Seg 3: RED -> YELLOW
    for i in range(1, WPS_PER_SEGMENT):
        alpha = i / (WPS_PER_SEGMENT - 1)
        coords.append((1 - alpha) * P_RED + alpha * P_YELLOW)
    node_indices['yellow_2'] = len(coords) - 1
        
    # Seg 4: YELLOW -> HOME
    for i in range(1, WPS_PER_SEGMENT):
        alpha = i / (WPS_PER_SEGMENT - 1)
        coords.append((1 - alpha) * P_YELLOW + alpha * P_HOME)
        
    coords = np.array(coords)
    total_wps = len(coords)
    
    # 2. Twist Goal
    twist_rad = np.radians(twist_deg)
    R_END = R_START @ rot(np.array([1, 0, 0]), twist_rad)
    R_rel = R_START.T @ R_END
    twist_angle = np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1.0, 1.0))
    # Handle zero-twist axis safely
    if twist_angle < 1e-6:
        twist_axis = np.array([1.0, 0.0, 0.0])
    else:
        twist_axis = np.array([R_rel[2, 1] - R_rel[1, 2],
                               R_rel[0, 2] - R_rel[2, 0],
                               R_rel[1, 0] - R_rel[0, 1]]) / (2 * np.sin(twist_angle))
                               
    # 3. Compute IK Iteratively
    joints = []
    jump_distances = [0.0]
    pos_errors = []
    ori_errors = []
    
    prev_q = Q_HOME
    R_target = R_START
    
    for i in range(total_wps):
        P_curr = coords[i]
        
        # Calculate localized dx, dy, dz for tangent tracking
        if i < total_wps - 1:
            P_next = coords[i + 1]
        else:
            P_next = P_curr # At the end, don't pitch further
            
        dx = P_next[0] - P_curr[0]
        dy = P_next[1] - P_curr[1]
        dz = P_next[2] - P_curr[2]
        path_speed = np.hypot(dx, dy)
        
        # SLERP Twist
        alpha_t = i / (total_wps - 1)
        R_twist = R_START @ rot(twist_axis, alpha_t * twist_angle)
        
        # Dynamic Tangent Pitch
        R_target_pitch = get_tangent_orientation(dx, dy, dz, path_speed, R_twist, dampen=0.5)
        
        # ── Pitch Blending to prevent HOME Singularity ──
        # The tool cannot be fully pitched when the arm is vertical at HOME.
        
        # Segment 1: HOME -> YELLOW (Fade IN the pitch)
        if i < WPS_PER_SEGMENT:
            beta = i / (WPS_PER_SEGMENT - 1)
            # SLERP from R_START to R_target_pitch
            R_rel = R_START.T @ R_target_pitch
            angle = np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1.0, 1.0))
            if angle > 1e-6:
                axis = np.array([
                    R_rel[2, 1] - R_rel[1, 2],
                    R_rel[0, 2] - R_rel[2, 0],
                    R_rel[1, 0] - R_rel[0, 1]
                ]) / (2 * np.sin(angle))
                R_target = R_START @ rot(axis, beta * angle)
            else:
                R_target = R_START
                
        # Segment 4: YELLOW -> HOME (Fade OUT the pitch)
        elif i >= total_wps - WPS_PER_SEGMENT:
            beta = (i - (total_wps - WPS_PER_SEGMENT)) / (WPS_PER_SEGMENT - 1)
            beta = np.clip(beta, 0.0, 1.0)
            # SLERP from R_target_pitch back to R_START
            R_rel = R_target_pitch.T @ R_START
            angle = np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1.0, 1.0))
            if angle > 1e-6:
                axis = np.array([
                    R_rel[2, 1] - R_rel[1, 2],
                    R_rel[0, 2] - R_rel[2, 0],
                    R_rel[1, 0] - R_rel[0, 1]
                ]) / (2 * np.sin(angle))
                R_target = R_target_pitch @ rot(axis, beta * angle)
            else:
                R_target = R_START
                
        # Segment 2 and 3: Fully engaged tangent pitching
        else:
            R_target = R_target_pitch
        
        # Solve
        q_sol = solve_closest_ik(P_curr, R_target, prev_q)
        if q_sol is None:
            print(f"[FATAL] IK Failed at waypoint {i} -> {P_curr}. Stopping.")
            break
            
        joints.append(q_sol)
        
        # Verification & Errors
        R_fk, P_fk = fwd_kinematics(q_sol)
        err_pos = np.linalg.norm(P_fk - P_curr)
        pos_errors.append(err_pos)
        
        # Orientation Geodesic Error
        R_err = R_target.T @ R_fk
        cos_theta = np.clip((np.trace(R_err) - 1) / 2, -1.0, 1.0)
        ori_errors.append(np.arccos(cos_theta))
        
        if i > 0:
            jump_distances.append(np.linalg.norm(q_sol - prev_q))
            
        prev_q = q_sol
        
    return coords, np.array(joints), jump_distances, pos_errors, ori_errors, node_indices

# ── Gazebo Execution & Shadow Markers ──────────────────────────
def send_shadow_trajectory_ros(trajectory, coords, node_indices, dt=0.15):
    """
    Executes the ROS trajectory AND asynchronously spawns Gazebo shadow markers 
    in real-time as the arm moves.
    """
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

    # Base SDF for shadow dots
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

    # Build the full trajectory with Dwell Times
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
        
        # Add Dwell Delays AFTER reaching the node WP
        current_ros_time += dt # Base time step
        
        if i == node_indices['yellow_1']:
            current_ros_time += 3.0
            print(f"  [Plan] Added 3.0s dwell at YELLOW (WP {i})")
        elif i == node_indices['red']:
            current_ros_time += 7.0
            print(f"  [Plan] Added 7.0s dwell at RED (WP {i})")
        elif i == node_indices['yellow_2']:
            current_ros_time += 3.0
            print(f"  [Plan] Added 3.0s dwell at Return YELLOW (WP {i})")
            
    print(f"\n[Gazebo] Sending {len(trajectory)} waypoints. Total Execution Time: {current_ros_time:.1f}s")
    client.send_goal(goal)
    
    # -- Real-Time Shadow Spawner Loop --
    print("[Gazebo] 🟢 Execution started. Spawning real-time shadow markers...")
    
    # Thread the wait for result so we don't block the spawn loop
    def wait_for_result_thread():
        client.wait_for_result()
    threading.Thread(target=wait_for_result_thread).start()
    
    start_time = time.time()
    for i, pos in enumerate(coords):
        target_time = ros_timestamps[i]
        
        # Sleep exactly until the robot is about to hit this waypoint
        time_to_wait = target_time - (time.time() - start_time)
        if time_to_wait > 0:
            time.sleep(time_to_wait)
            
        # Spawn the shadow directly under the arm!
        pose = Pose()
        pose.position.x = pos[0]
        pose.position.y = pos[1]
        pose.position.z = pos[2]
        
        try:
            # We don't block execution waiting for gazebo to catch up printing these
            spawn_model(f"shadow_{i}", marker_sdf.format(name=f"shadow_{i}"), "", pose, "world")
        except rospy.ServiceException as e:
            pass
            
    print("[Gazebo] ✅ Refueling Mission Complete!")
    return True

# ── Visualization ─────────────────────────────────────────────
def plot_pipeline_analysis(joints, jump_dists, pos_errs, ori_errs, nodes_idx):
    # To fix matplotlib warning about gui thread
    import matplotlib
    matplotlib.use("Agg")
    
    fig, axs = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
    wps = range(len(joints))
    
    # Helper to draw vertical lines at the dwell nodes
    def draw_nodes(ax):
        colors = {'yellow_1': 'y', 'red': 'r', 'yellow_2': 'y'}
        names = {'yellow_1': 'YELLOW (3s)', 'red': 'RED (7s)', 'yellow_2': 'YELLOW (3s)'}
        for key, idx in nodes_idx.items():
            ax.axvline(x=idx, color=colors[key], linestyle='--', alpha=0.7)
            # ax.text(idx, ax.get_ylim()[1], names[key], rotation=90, verticalalignment='top', fontsize=8)

    # Plot 1: Joint Angles
    for j in range(6):
        axs[0].plot(wps, np.degrees(joints[:, j]), linewidth=2, label=f'J{j+1}')
    axs[0].set_ylabel("Joint Angle (deg)")
    axs[0].set_title("Configuration Space Trajectory (HOME → YELLOW → RED → YELLOW → HOME)", fontsize=11, loc='left')
    axs[0].legend(loc='upper right', ncol=6)
    axs[0].grid(True, linestyle=':', alpha=0.6)
    draw_nodes(axs[0])

    # Plot 2: Jump Distances
    axs[1].plot(wps, jump_dists, color='purple', linewidth=2, marker='.', markersize=4)
    axs[1].set_ylabel("Jump ΔQ (rad)")
    axs[1].set_title("Stability: Euclidean Distance Between Consecutive IK Solutions", fontsize=11, loc='left')
    axs[1].grid(True, linestyle=':', alpha=0.6)
    draw_nodes(axs[1])

    # Plot 3: Pos Error
    axs[2].plot(wps, pos_errs, color='red', linewidth=2)
    axs[2].set_yscale('log')
    axs[2].set_ylabel("Error (m)")
    axs[2].set_title("Positional Forward Kinematics Error vs True Cartesian Plan", fontsize=11, loc='left')
    axs[2].grid(True, linestyle=':', alpha=0.6)
    draw_nodes(axs[2])
    
    # Plot 4: Ori Error
    axs[3].plot(wps, ori_errs, color='blue', linewidth=2)
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
    print(f"\n[Analysis] Saved 4-panel mathematical proof to '{out_path}'")

# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--twist", type=float, default=0.0, help="Total wrist twist overlay (deg)")
    parser.add_argument("--ros", action="store_true", help="Execute in Gazebo with shadow markers")
    args = parser.parse_args()
    
    print("=================================================================")
    print("  KUKA KR6 R700 — Real-Time IK-Geo Refueling Pipeline")
    print("  HOME → YELLOW (3s) → RED (7s) → YELLOW (3s) → HOME")
    print("=================================================================\n")
    
    print("[Planning] Generating 160-waypoint sequence with dynamic tangent pitching...")
    coords, joints, jumps, pos_errs, ori_errs, nodes = generate_mission_trajectory(args.twist)
    
    if len(joints) > 0:
        print(f"  → Solved {len(joints)} waypoints analytically.")
        print(f"  → Max FK Pos Error: {np.max(pos_errs):.2e} m")
        print(f"  → Max FK Ori Error: {np.max(ori_errs):.2e} rad")
        
        plot_pipeline_analysis(joints, jumps, pos_errs, ori_errs, nodes)
        
        if args.ros:
            send_shadow_trajectory_ros(joints, coords, nodes, dt=DT)
        else:
            print("\n[Skip] Pass --ros to execute in Gazebo and spawn shadow markers.")

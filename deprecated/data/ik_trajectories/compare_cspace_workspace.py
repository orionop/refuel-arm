#!/usr/bin/env python3
"""
KUKA KR6 R700 — Configuration Space vs. Workspace Comparison
=============================================================

This script compares two fundamental motion planning approaches:
1. Workspace (W-Space): The solver MUST know the path. It calculates IK 
   at every single waypoint along a predefined straight 3D line.
2. Configuration Space (C-Space): The solver ONLY knows Start and End.
   It calculates IK exactly twice. All intermediate motion is a simple
   linear slide in joint angles (LERP).
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Import IK-Geo
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'kuka_refuel_ws', 'src', 'kuka_kr6_gazebo', 'scripts')))
import ik_geometric as ik

# --- TARGET POSES (THE ONLY INPUTS FOR C-SPACE) ---
START_POS = np.array([0.3, 0.4, 0.5])
END_POS   = np.array([0.706, 0.000, 0.413])
TWIST_DEG = 0.0 
END_PITCH = 15.0 # Degrees

# Resolution for Graphing/Execution
PLOT_RESOLUTION = 60 

# Base orientation: Forward looking slightly down
R_START = np.array([
    [ 0,  0,  1],
    [ 0,  1,  0],
    [-1,  0,  0]
])

JOINT_LIMITS = np.array([
    [-2.967,  2.967], [-3.316,  0.785], [-2.094,  2.722],
    [-6.108,  6.108], [-2.094,  2.094], [-6.108,  6.108],
])

Q_HOME = np.array([0.0, -np.pi/2, 0.0, 0.0, 0.0, 0.0])

def is_valid(q):
    for j in range(6):
        if q[j] < JOINT_LIMITS[j, 0] or q[j] > JOINT_LIMITS[j, 1]:
            return False
    return True

def solve_closest_ik(target_pos, target_R, prev_q):
    Q_all = ik.IK_spherical_2_parallel(target_R, target_pos)
    if Q_all.size == 0: return None
    best_q, min_dist = None, float('inf')
    for i in range(Q_all.shape[1]):
        q = (Q_all[:, i] + np.pi) % (2 * np.pi) - np.pi
        if is_valid(q):
            dist = np.linalg.norm(q - prev_q)
            if dist < min_dist:
                min_dist = dist
                best_q = q
    return best_q

def axis_angle_from_rotation(R):
    angle = np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    if angle < 1e-10: return np.array([1.0, 0.0, 0.0]), 0.0
    axis = np.array([R[2, 1]-R[1, 2], R[0, 2]-R[2, 0], R[1, 0]-R[0, 1]]) / (2.0*np.sin(angle))
    return axis / np.linalg.norm(axis), angle

def interpolate_orientation(R_start, R_end, t):
    R_rel = R_start.T @ R_end
    axis, angle = axis_angle_from_rotation(R_rel)
    return R_start @ ik.rot(axis, t * angle)

def run_comparison():
    # 1. Setup orientations for the two end states
    R_target_start = R_START
    R_target_end = ik.rot(np.array([0.0, 1.0, 0.0]), np.radians(END_PITCH))

    # --- STRATEGY 1: WORKSPACE PLANNING (LINE-AWARE) ---
    print("[1/2] Generating Workspace Trajectory...")
    ws_q_traj = []
    ws_pos_traj = []
    current_q = Q_HOME
    for i in range(PLOT_RESOLUTION):
        t = i / (PLOT_RESOLUTION - 1)
        pos = START_POS + t * (END_POS - START_POS)
        rot_t = interpolate_orientation(R_target_start, R_target_end, t)
        q = solve_closest_ik(pos, rot_t, current_q)
        ws_q_traj.append(q)
        ws_pos_traj.append(pos)
        current_q = q

    # --- STRATEGY 2: C-SPACE PLANNING (PATH-AGNOSTIC) ---
    print("[2/2] Generating C-Space Trajectory...")
    q_start = solve_closest_ik(START_POS, R_target_start, Q_HOME)
    q_goal  = solve_closest_ik(END_POS, R_target_end, q_start)
    
    cs_q_traj = []
    cs_pos_traj = []
    for i in range(PLOT_RESOLUTION):
        t = i / (PLOT_RESOLUTION - 1)
        q_t = q_start + t * (q_goal - q_start)
        _, p_actual = ik.fwd_kinematics(q_t)
        cs_q_traj.append(q_t)
        cs_pos_traj.append(p_actual)

    # --- ERROR ANALYSIS ---
    ws_pos_errs = []
    ws_ori_errs = []
    cs_pos_errs = []
    cs_ori_errs = []
    
    for i in range(PLOT_RESOLUTION):
        t = i / (PLOT_RESOLUTION - 1)
        p_ideal = START_POS + t * (END_POS - START_POS)
        R_ideal = interpolate_orientation(R_target_start, R_target_end, t)
        
        # Workspace actuals
        q_ws = ws_q_traj[i]
        R_ws, p_ws = ik.fwd_kinematics(q_ws)
        
        # C-Space actuals
        q_cs = cs_q_traj[i]
        R_cs, p_cs = ik.fwd_kinematics(q_cs)
        
        # Position Errors
        ws_pos_errs.append(np.linalg.norm(p_ws - p_ideal))
        cs_pos_errs.append(np.linalg.norm(p_cs - p_ideal))
        
        # Orientation Errors
        _, angle_ws = axis_angle_from_rotation(R_ws.T @ R_ideal)
        _, angle_cs = axis_angle_from_rotation(R_cs.T @ R_ideal)
        ws_ori_errs.append(np.degrees(angle_ws))
        cs_ori_errs.append(np.degrees(angle_cs))

    return (np.array(ws_q_traj), np.array(ws_pos_traj), np.array(ws_pos_errs), np.array(ws_ori_errs),
            np.array(cs_q_traj), np.array(cs_pos_traj), np.array(cs_pos_errs), np.array(cs_ori_errs),
            q_start, q_goal)

def plot_comparison(ws_q, ws_p, ws_pe, ws_oe, cs_q, cs_p, cs_pe, cs_oe, q_start, q_goal):
    steps = np.arange(1, PLOT_RESOLUTION + 1)
    fig, axs = plt.subplots(2, 2, figsize=(14, 11)) # Reverting to 2 rows (C-Space vs W-Space)
    
    start_pitch = 90.0 
    q_start_str = ", ".join([f"{v:.2f}" for v in q_start])
    q_goal_str  = ", ".join([f"{v:.2f}" for v in q_goal])
    
    info_str = (f"Target 1 (Start): {START_POS} | Target 2 (Goal): {END_POS}\n"
                f"Start Pitch: {start_pitch}° | Target Pitch: {END_PITCH}°\n"
                f"Selected q_start: [{q_start_str}] rad\n"
                f"Selected q_goal:  [{q_goal_str}] rad")
    
    fig.suptitle('Motion Strategy Comparison: Workspace (Linear) vs. Configuration Space (Linear)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    fig.text(0.5, 0.92, info_str, ha='center', fontsize=11, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.4))
    
    j_labels = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6']

    # --- Row 1: Joint Angles ---
    axs[0, 0].set_title("Workspace Strategy: Solver solves IK @ every step", color='navy', fontweight='bold', pad=20)
    for j in range(6): axs[0, 0].plot(steps, ws_q[:, j], label=j_labels[j], linewidth=2)
    axs[0, 0].set_ylabel("Radians"); axs[0, 0].grid(True, alpha=0.3); axs[0, 0].legend(fontsize=8, loc='upper right')

    axs[0, 1].set_title("C-Space Strategy: Solver solves IK @ Start/End ONLY", color='darkgreen', fontweight='bold', pad=20)
    for j in range(6): axs[0, 1].plot(steps, cs_q[:, j], label=j_labels[j], linewidth=2)
    axs[0, 1].set_ylabel("Radians"); axs[0, 1].grid(True, alpha=0.3)

    # --- Row 2: Cartesian Paths ---
    axs[1, 0].set_title("Resulting Workspace Path (Constraint: Straight Line)", color='navy', fontweight='bold')
    axs[1, 0].plot(steps, ws_p[:, 0], 'r', label='X'); axs[1, 0].plot(steps, ws_p[:, 1], 'g', label='Y'); axs[1, 0].plot(steps, ws_p[:, 2], 'b', label='Z')
    axs[1, 0].set_ylabel("Meters"); axs[1, 0].grid(True, alpha=0.3); axs[1, 0].legend()

    axs[1, 1].set_title("Resulting C-Space Path (Result: Dynamic Arc)", color='darkgreen', fontweight='bold')
    axs[1, 1].plot(steps, cs_p[:, 0], 'r', label='X'); axs[1, 1].plot(steps, cs_p[:, 1], 'g', label='Y'); axs[1, 1].plot(steps, cs_p[:, 2], 'b', label='Z')
    axs[1, 1].set_ylabel("Meters"); axs[1, 1].grid(True, alpha=0.3)

    # --- Row 3: Quantitative Error Analysis (COMMENTED OUT) ---
    # axs[2, 0].set_title("Position Error (L2 Norm to Straight Line)", color='purple', fontweight='bold')
    # axs[2, 0].plot(steps, ws_pe, 'b', label='Workspace Error', linewidth=2)
    # axs[2, 0].plot(steps, cs_pe, 'r--', label='C-Space Deviation', linewidth=2)
    # axs[2, 0].set_ylabel("Error (Meters)"); axs[2, 0].set_xlabel("Step Number"); axs[2, 0].grid(True, alpha=0.3); axs[2, 0].legend()

    # axs[2, 1].set_title("Orientation Error (Geodesic Distance to Target Rot)", color='purple', fontweight='bold')
    # axs[2, 1].plot(steps, ws_oe, 'b', label='Workspace Error', linewidth=2)
    # axs[2, 1].plot(steps, cs_oe, 'r--', label='C-Space Deviation', linewidth=2)
    # axs[2, 1].set_ylabel("Error (Degrees)"); axs[2, 1].set_xlabel("Step Number"); axs[2, 1].grid(True, alpha=0.3); axs[2, 1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.88])
    save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output_graphs', 'cspace_vs_wspace_comparison.png'))
    plt.savefig(save_path, dpi=200)
    print(f"\n[Success] Comparison graph saved to: {save_path}")

def execute_ros(trajectory, cartesian_points, mode="ros"):
    import rospy
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from sensor_msgs.msg import JointState
    from visualization_msgs.msg import Marker
    rospy.init_node('ik_comparison_tracker', anonymous=True)
    marker_pub = rospy.Publisher('/visualization_marker', Marker, queue_size=10)
    rospy.sleep(0.5)
    
    line = Marker(); line.header.frame_id = "world"; line.ns = "comp_path"; line.id = 0; line.type = Marker.LINE_STRIP; line.action = Marker.ADD
    line.scale.x = 0.008; line.color.r = 0.0; line.color.g = 0.5; line.color.b = 1.0; line.color.a = 1.0; line.pose.orientation.w = 1.0
    for p in cartesian_points:
        from geometry_msgs.msg import Point
        line.points.append(Point(x=p[0], y=p[1], z=p[2]))
    marker_pub.publish(line)

    if mode == "ros":
        pub = rospy.Publisher('/kr6_arm_controller/command', JointTrajectory, queue_size=10)
        rospy.sleep(0.5)
        msg = JointTrajectory(); msg.joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
        time_from_start = 0.0; prev_q = Q_HOME
        for q in trajectory:
            dt = max(0.20, np.linalg.norm(q - prev_q) * 2.5)
            time_from_start += dt
            pt = JointTrajectoryPoint(); pt.positions = q.tolist(); pt.time_from_start = rospy.Duration.from_sec(time_from_start)
            msg.points.append(pt); prev_q = q
        pub.publish(msg)
        rospy.sleep(time_from_start + 1.0)
    else:
        pub = rospy.Publisher('/joint_states', JointState, queue_size=10)
        rospy.sleep(0.5)
        msg = JointState(); msg.name = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
        for q in trajectory:
            msg.header.stamp = rospy.Time.now(); msg.position = q.tolist(); pub.publish(msg); rospy.sleep(0.15)

def main():
    parser = argparse.ArgumentParser(description="IK Comparison")
    parser.add_argument("--ros", action="store_true")
    parser.add_argument("--rviz", action="store_true")
    parser.add_argument("--ws", action="store_true")
    parser.add_argument("--cs", action="store_true")
    args = parser.parse_args()

    (ws_q, ws_p, ws_pe, ws_oe, 
     cs_q, cs_p, cs_pe, cs_oe, 
     q_s, q_g) = run_comparison()
    
    plot_comparison(ws_q, ws_p, ws_pe, ws_oe, 
                    cs_q, cs_p, cs_pe, cs_oe, 
                    q_s, q_g)

    if args.ros or args.rviz:
        import os
        ros_python = '/opt/ros/noetic/lib/python3/dist-packages'
        if ros_python not in sys.path and os.path.isdir(ros_python): sys.path.insert(0, ros_python)
        mode = "ros" if args.ros else "rviz"
        if args.ws: execute_ros(ws_q, ws_p, mode=mode)
        elif args.cs: execute_ros(cs_q, cs_p, mode=mode)

if __name__ == "__main__":
    main()

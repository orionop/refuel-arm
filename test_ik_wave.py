#!/usr/bin/env python3
"""
KUKA KR6 R700 — Pure IK-Geo Cartesian Line Tracking
===================================================

Demonstrates plotting a straight 3D line in Cartesian space, decomposing it
into dense waypoints, and smoothly tracking it using purely algebraic IK-Geo.

No STOMP. No IKFlow. Just math!
"""
import sys
import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Import IK-Geo
sys.path.insert(0, "kuka_refuel_ws/src/kuka_kr6_gazebo/scripts")
import ik_geometric as ik

# ── Default Configuration ─────────────────────────────────────
DEFAULT_START = [0.3, 0.4, 0.5]
DEFAULT_END   = [0.65, -0.25, 0.45]
Z_AMPLITUDE   = 0.10  # Peak height of the sine wave in meters
NUM_WAYPOINTS = 60
DT = 0.15  # Time per waypoint for execution (seconds)
DEFAULT_TWIST_DEG = 45.0  # Default wrist twist in degrees

# Base orientation: EE pointing forward, looking slightly down
R_START = np.array([
    [ 0,  0,  1],
    [ 0,  1,  0],
    [-1,  0,  0]
])

# Official Joint Limits
JOINT_LIMITS = np.array([
    [-2.967,  2.967],  # joint_1
    [-3.316,  0.785],  # joint_2
    [-2.094,  2.722],  # joint_3
    [-6.108,  6.108],  # joint_4
    [-2.094,  2.094],  # joint_5
    [-6.108,  6.108],  # joint_6
])

# For the very first point, start searching from a straight-up HOME pose
Q_HOME = np.array([0.0, -np.pi/2, 0.0, 0.0, 0.0, 0.0])


def is_valid(q):
    """Check if joint angles within official URDF limits."""
    for j in range(6):
        if q[j] < JOINT_LIMITS[j, 0] or q[j] > JOINT_LIMITS[j, 1]:
            return False
    return True


def orientation_error(R_target, R_actual):
    """Geodesic distance between two rotation matrices (radians)."""
    R_err = R_target.T @ R_actual
    cos_angle = (np.trace(R_err) - 1.0) / 2.0
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Numerical safety
    return np.abs(np.arccos(cos_angle))


def axis_angle_from_rotation(R):
    """Extract (axis, angle) from a rotation matrix via the logarithmic map."""
    angle = np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    if angle < 1e-10:
        return np.array([1.0, 0.0, 0.0]), 0.0  # Identity rotation
    axis = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1]
    ]) / (2.0 * np.sin(angle))
    return axis / np.linalg.norm(axis), angle


def interpolate_orientation(R_start, R_end, t):
    """Axis-angle SLERP: interpolate between two rotation matrices at parameter t ∈ [0, 1]."""
    R_rel = R_start.T @ R_end
    axis, angle = axis_angle_from_rotation(R_rel)
    return R_start @ ik.rot(axis, t * angle)


def solve_closest_ik(target_pos, target_R, prev_q):
    """Solve IK-Geo for a Cartesian point + orientation, pick the valid pose closest to prev_q."""
    Q_all = ik.IK_spherical_2_parallel(target_R, target_pos)
    
    if Q_all.size == 0:
        return None

    best_q = None
    min_dist = float('inf')

    # Q_all is 6xN matrix of solutions
    valid_count = 0
    for i in range(Q_all.shape[1]):
        q = Q_all[:, i]
        # Normalize between -pi and pi
        q = (q + np.pi) % (2 * np.pi) - np.pi
        
        status = "INVALID (Limits)"
        if is_valid(q):
            status = "VALID"
            valid_count += 1
            # L2 norm (Euclidean distance in joint space)
            dist = np.linalg.norm(q - prev_q)
            if dist < min_dist:
                min_dist = dist
                best_q = q
            print(f"  Sol {i+1}: {np.round(q, 3)} | Dist to prev: {dist:.3f} rad | [{status}]")
        else:
            print(f"  Sol {i+1}: {np.round(q, 3)} | [{status}]")
            
    print(f"  -> Total Valid: {valid_count}/{Q_all.shape[1]}")

    return best_q


def generate_wave_trajectory(start_pt, end_pt, twist_deg=45.0):
    """Interpolate Cartesian sine wave with SLERP orientation and solve IK for each point."""
    print(f"\n[Planning] Interpolating {NUM_WAYPOINTS} wave waypoints from {start_pt} to {end_pt}...")
    print(f"[Planning] End-effector twist: {twist_deg}° over trajectory")
    
    trajectory = []
    cartesian_points = []
    jump_distances = []
    fk_errors = []
    orient_errors = []
    
    # Generate X, Y coordinates linearly spaced
    x_vals = np.linspace(start_pt[0], end_pt[0], NUM_WAYPOINTS)
    y_vals = np.linspace(start_pt[1], end_pt[1], NUM_WAYPOINTS)
    
    # Generate Z coordinates with a multi-cycle sine wave (audio-like)
    # base_z linearly drops from 0.5 to 0.45.
    base_z = np.linspace(start_pt[2], end_pt[2], NUM_WAYPOINTS)
    # sin(0) to sin(4*pi) creates 2 full wave cycles (up, down, up, down).
    wave_freq = 4 * np.pi
    wave_z = Z_AMPLITUDE * np.sin(np.linspace(0, wave_freq, NUM_WAYPOINTS))
    z_vals = base_z + wave_z

    # Compute R_END by applying the twist around the tool X-axis (first column of R_START)
    twist_rad = np.radians(twist_deg)
    tool_x_axis = R_START[:, 0]  # The approach direction of the end-effector
    R_END = R_START @ ik.rot(np.array([1.0, 0.0, 0.0]), twist_rad)

    current_q = Q_HOME
    prev_wp_pos = None

    for i in range(NUM_WAYPOINTS):
        wp_pos = np.array([x_vals[i], y_vals[i], z_vals[i]])
        cartesian_points.append(wp_pos)
        
        # Calculate dynamic orientation (Real-time trajectory shape tracking)
        # We calculate the instantaneous slope (dz/dx) directly from the adjacent coordinate points,
        # ensuring the robot "feels" the shape of the path dynamically without relying on hardcoded functions.
        if i < NUM_WAYPOINTS - 1:
            dx = x_vals[i+1] - x_vals[i]
            dz = z_vals[i+1] - z_vals[i]
        else:
            dx = x_vals[i] - x_vals[i-1]
            dz = z_vals[i] - z_vals[i-1]
            
        instantaneous_pitch = np.arctan2(dz, dx) * 0.5
        
        # 1. Base rotation tangent to the instantaneous path slope
        R_tangent = R_START @ ik.rot(np.array([0.0, 1.0, 0.0]), -instantaneous_pitch) # Negative pitch rotates tool 'up' when slope is positive
        
        # 2. Compute SLERP Twist over the newly pitched frame
        twist_rad = np.radians(twist_deg)
        R_END_tangent = R_tangent @ ik.rot(np.array([1.0, 0.0, 0.0]), twist_rad) # Twist around local X (tool pointing axis)
        
        t = i / max(NUM_WAYPOINTS - 1, 1)
        # 3. Combine both: The frame is pitched correctly, now SLERP the twist across it
        R_target = interpolate_orientation(R_tangent, R_END_tangent, t)
        
        # Calculate Cartesian chunk distance
        if prev_wp_pos is not None:
            cart_dist = np.linalg.norm(wp_pos - prev_wp_pos)
        else:
            cart_dist = 0.0
            
        print(f"\n--- WP {i+1:2d} | Target XYZ: {np.round(wp_pos, 3)} | Twist: {t*twist_deg:.1f}° | Dist from WP{i}: {cart_dist:.4f} m ---")
        prev_wp_pos = wp_pos
        
        # 1. Solve IK for this waypoint (with per-waypoint orientation)
        solved_q = solve_closest_ik(wp_pos, R_target, current_q)
        
        if solved_q is None:
            print(f"\n[WARNING] IK-Geo failed to find valid solution for Waypoint {i+1} at XYZ={np.round(wp_pos, 3)}")
            print("[WARNING] The line has left the reachable kinematic workspace or violates joint limits.")
            print(f"[WARNING] Truncating trajectory to only execute the first {i} safe waypoints.")
            cartesian_points.pop() # Remove the unreachable point
            break
            
        # 2. Check joint jump distance to prevent 'flips'
        jump = np.linalg.norm(solved_q - current_q)
        if i > 0 and jump > 1.0: # 1 radian jump between millimeters is physically impossible, denotes a flip
            print(f"\n[WARNING] Massive joint jump detected at Waypoint {i+1}: {jump:.2f} rad")
            print("[WARNING] IK-Geo had to flip configuration (elbow up/down shifted).")
            print(f"[WARNING] Truncating trajectory to safely execute up to Waypoint {i}.")
            cartesian_points.pop()
            break
            
        # 3. Verify via Forward Kinematics (FK)
        R_check, p_check = ik.fwd_kinematics(solved_q)
        pos_err = np.linalg.norm(p_check - wp_pos)
        rot_err = orientation_error(R_target, R_check)
        
        if pos_err > 1e-4:
            print(f"\n[ERROR] FK Verification mathematically failed at Waypoint {i+1}!")
            print(f"Target: {wp_pos}, FK Output: {p_check}")
            print(f"[WARNING] Truncating trajectory to safely execute up to Waypoint {i}.")
            cartesian_points.pop()
            break
            
        print(f"  => SELECTED WP {i+1}: {np.round(solved_q, 3)} | FK Pos Err: {pos_err:.2e} m | Rot Err: {rot_err:.2e} rad")
        
        trajectory.append(solved_q)
        
        # We only record the jump from WP(i-1) to WP(i). The first point from HOME is excluded from the plot to keep scale readable.
        if i > 0:
            jump_distances.append(jump)
        fk_errors.append(pos_err)
        orient_errors.append(rot_err)
        
        current_q = solved_q  # Update reference for next waypoint

    if len(trajectory) == 0:
        print("\n[FATAL] Failed to solve even the first waypoint. Execution cancelled.")
        sys.exit(1)
        
    print(f"\n[Planning Complete] Successfully planned {len(trajectory)} / {NUM_WAYPOINTS} waypoints.")
    return trajectory, cartesian_points, jump_distances, fk_errors, orient_errors

def plot_trajectory_analysis(trajectory, jump_distances, fk_errors, orient_errors, shape_name="line"):
    """Generates a 4-panel matplotlib figure proving tracking smoothness and FK precision."""
    print(f"\n[Analysis] Rendering {shape_name} trajectory error plots...")
    
    # Extract joint angles (N_waypoints x 6_joints)
    waypoints_q = np.array(trajectory)
    steps = np.arange(1, len(trajectory) + 1)
    
    fig, axs = plt.subplots(4, 1, figsize=(10, 16), gridspec_kw={'height_ratios': [3, 1.5, 1.5, 1.5]})
    fig.canvas.manager.set_window_title(f"IK-Geo {shape_name.title()} Trajectory Analysis")
    fig.suptitle(f'KUKA KR6 R700 — {shape_name.title()} Trajectory: 6-DOF Cartesian Tracking Analysis', fontsize=14, fontweight='bold', y=0.96)

    # --- Plot 1: Joint Angles vs Time (Smoothness Proof) ---
    axs[0].set_title("Kinematic Profile: Joint Angles vs. Waypoints\n(Proves smooth tracking without elbow-flips)", fontsize=11, loc='left')
    colors = ['r', 'g', 'b', 'c', 'm', 'y']
    labels = ['J1 (Base)', 'J2 (Shoulder)', 'J3 (Elbow)', 'J4 (Wrist 1)', 'J5 (Wrist 2)', 'J6 (Wrist 3)']
    
    for j in range(6):
        axs[0].plot(steps, waypoints_q[:, j], label=labels[j], color=colors[j], linewidth=2, marker='.')
        
    axs[0].set_ylabel("Joint Angle (radians)", fontsize=10)
    axs[0].grid(True, linestyle='--', alpha=0.6)
    axs[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    axs[0].set_xlim(1, len(trajectory))

    # --- Plot 2: Euclidean Jump Distance (Least Squares Norm Profile) ---
    axs[1].set_title("Configuration Stability: Joint Space Jump Distance ($\Delta Q$)\n(Proves minimal Euclidean distance selection)", fontsize=11, loc='left')
    # Pad the jump array since there is no 'jump' for the very first WP
    padded_jumps = [0.0] + jump_distances 
    axs[1].plot(steps, padded_jumps, color='darkorange', linewidth=2, drawstyle='steps-mid', fillstyle='bottom')
    axs[1].fill_between(steps, padded_jumps, 0, color='darkorange', alpha=0.2, step='mid')
    axs[1].set_ylabel(r"$\Delta Q$ Norm (rad)", fontsize=10)
    axs[1].grid(True, linestyle='--', alpha=0.6)
    axs[1].set_xlim(1, len(trajectory))
    # Add an absolute safety limit line representing an instant flip tolerance
    axs[1].axhline(y=1.0, color='r', linestyle=':', label='Max Safety Tolerance')

    # --- Plot 3: Position FK Error ---
    axs[2].set_title("Position Precision: Forward Kinematics Translational Error\n(Algebraic exactness: error bounded by IEEE 754 FP64 noise)", fontsize=11, loc='left')
    
    # Clamp the values to the precision floor (1e-16) to keep the graph visually flat.
    printable_pos_errors = np.maximum(fk_errors, 1e-16)
    
    axs[2].plot(steps, printable_pos_errors, color='crimson', marker='o', markersize=3, linestyle='-', linewidth=1.5)
    axs[2].set_ylabel("Position Error (m)", fontsize=10)
    axs[2].set_yscale('log')
    axs[2].grid(True, which="both", linestyle='--', alpha=0.6)
    axs[2].set_xlim(1, len(trajectory))
    
    import matplotlib.ticker as ticker
    axs[2].yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=5))
    axs[2].set_ylim(bottom=1e-17, top=1e-13)

    # --- Plot 4: Orientation FK Error ---
    axs[3].set_title("Orientation Precision: Forward Kinematics Rotational Error\n(Geodesic angular distance between target and actual rotation)", fontsize=11, loc='left')
    
    printable_orient_errors = np.maximum(orient_errors, 1e-9)
    
    axs[3].plot(steps, printable_orient_errors, color='dodgerblue', marker='s', markersize=3, linestyle='-', linewidth=1.5)
    axs[3].set_xlabel("Cartesian Waypoint Number", fontsize=10)
    axs[3].set_ylabel("Orientation Error (rad)", fontsize=10)
    axs[3].set_yscale('log')
    axs[3].grid(True, which="both", linestyle='--', alpha=0.6)
    axs[3].set_xlim(1, len(trajectory))
    axs[3].yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=5))
    axs[3].set_ylim(bottom=1e-10, top=1e-6)
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.94])
    
    # Save with trajectory-type naming convention in a dedicated output folder
    save_dir = "output_graphs"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    safe_path = os.path.join(save_dir, f"analysis_{shape_name}_full.png")
    plt.savefig(safe_path, dpi=300, bbox_inches='tight')
    print(f"  -> High-Res plot saved to: {safe_path}")
    
    # Unblock the code by explicitly bringing window to front, showing, but not freezing
    plt.show(block=False)
    plt.pause(2.0)

def spawn_gazebo_markers(cartesian_points):
    """Force-spawns native spherical SDF models into Gazebo to guarantee visibility, skipping the flaky RViz marker plugin."""
    import rospy
    from gazebo_msgs.srv import SpawnModel
    from geometry_msgs.msg import Pose
    
    print("\n[Visuals] Spawning 30 native white spheres into Gazebo physics engine...", end="")
    rospy.wait_for_service('/gazebo/spawn_sdf_model', timeout=5.0)
    spawn_model_prox = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
    
    # Generic white glowing sphere SDF
    sdf_xml = """
    <?xml version="1.0" ?>
    <sdf version="1.5">
      <model name="dot_{id}">
        <static>true</static>
        <link name="link">
          <visual name="visual">
            <geometry><sphere><radius>0.015</radius></sphere></geometry>
            <material><ambient>1 1 1 1</ambient><diffuse>1 1 1 1</diffuse><emissive>1 1 1 1</emissive></material>
          </visual>
        </link>
      </model>
    </sdf>
    """
    
    for idx, p in enumerate(cartesian_points):
        pose = Pose()
        pose.position.x = p[0]
        pose.position.y = p[1]
        pose.position.z = p[2]
        pose.orientation.w = 1.0
        
        try:
            spawn_model_prox(f"ik_line_wp_{idx}", sdf_xml.replace("{id}", str(idx)), "ik_path", pose, "world")
        except Exception as e:
            pass
            
    print(" Done!")


def execute_ros(trajectory, cartesian_points, mode="ros"):
    """Send joint trajectory to Gazebo / RViz and draw the blue line."""
    import rospy
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from sensor_msgs.msg import JointState
    from visualization_msgs.msg import Marker, MarkerArray

    rospy.init_node('pure_ik_line_tracker', anonymous=True)
    
    marker_pub = rospy.Publisher('/visualization_marker_array', MarkerArray, queue_size=10)
    rospy.sleep(0.5)
    
    ma = MarkerArray()
    
    line = Marker()
    line.header.frame_id = "world"
    line.ns = "ik_path_line"
    line.id = 0
    line.type = Marker.LINE_STRIP
    line.action = Marker.ADD
    line.scale.x = 0.008 # Line width
    
    # Neon Blue Line
    line.color.r = 0.0
    line.color.g = 0.5
    line.color.b = 1.0
    line.color.a = 1.0
    line.pose.orientation.w = 1.0
    
    # Gazebo doesn't natively support MarkerArrays well without an external plugin. 
    # But it does support single markers over standard topic `visualization_marker`.
    # Let's publish individual sphere markers in a loop specifically so Gazebo natively renders them.
    marker_pub_single = rospy.Publisher('/visualization_marker', Marker, queue_size=10)
    
    # Send the main strip first
    marker_pub_single.publish(line)
    
    # Send individual spheres instead of SPHERE_LIST for better Gazebo compatibility
    for idx, p in enumerate(cartesian_points):
        dot = Marker()
        dot.header.frame_id = "world"
        dot.ns = "ik_path_dots"
        dot.id = idx + 10  # Offset ID so they don't overwrite each other
        dot.type = Marker.SPHERE
        dot.action = Marker.ADD
        dot.scale.x = 0.02
        dot.scale.y = 0.02
        dot.scale.z = 0.02
        dot.color.r = 1.0
        dot.color.g = 1.0
        dot.color.b = 1.0
        dot.color.a = 1.0
        dot.pose.position.x = p[0]
        dot.pose.position.y = p[1]
        dot.pose.position.z = p[2]
        dot.pose.orientation.w = 1.0
        
        # Publish to single topic
        marker_pub_single.publish(dot)
        
        # Also append to array for RViz
        ma.markers.append(dot)
    
    ma.markers.append(line)
    marker_pub.publish(ma)
    print("\n[Visuals] Published blue Cartesian path line with white waypoint dots to RViz")

    # ── 2. Execute Trajectory ──
    if mode == "ros":
        # Spawn literal models inside Gazebo so we don't have to rely on buggy plugins
        try:
            spawn_gazebo_markers(cartesian_points)
        except Exception as e:
            print(f"[Warning] Failed to spawn Gazebo markers natively: {e}")
            
        pub = rospy.Publisher('/kr6_arm_controller/command', JointTrajectory, queue_size=10)
        rospy.sleep(0.5)
        
        msg = JointTrajectory()
        msg.joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
        
        time_from_start = 0.0
        prev_q = Q_HOME
        for i, q in enumerate(trajectory):
            # Calculate dynamic time based on joint distance
            # If it's a huge jump (like returning HOME), give it more time!
            dist = np.linalg.norm(q - prev_q)
            
            # Base dt is 0.2s for tiny micro-movements, but add 2.5 seconds per radian for big jumps
            dt = max(0.20, dist * 2.5)
            time_from_start += dt
            
            pt = JointTrajectoryPoint()
            pt.positions = q.tolist()
            pt.time_from_start = rospy.Duration.from_sec(time_from_start)
            msg.points.append(pt)
            
            prev_q = q
            
        print(f"[Execute] Sending {len(trajectory)} waypoints to ROS Gazebo JointTrajectoryController...")
        pub.publish(msg)
        
        duration = time_from_start + 1.0
        print(f"Waiting {duration:.1f}s for trajectory to finish...")
        rospy.sleep(duration)
    else:
        # RViz mode
        pub = rospy.Publisher('/joint_states', JointState, queue_size=10)
        rospy.sleep(0.5)
        
        msg = JointState()
        msg.name = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
        
        print(f"[Execute] Visualizing {len(trajectory)} waypoints directly in RViz...")
        for q in trajectory:
            msg.header.stamp = rospy.Time.now()
            msg.position = q.tolist()
            pub.publish(msg)
            rospy.sleep(DT)
            
    print("Done!")

    
def main():
    parser = argparse.ArgumentParser(description="Pure IK-Geo Sine Wave Tracking")
    parser.add_argument("--ros", action="store_true", help="Execute in Gazebo via ROS")
    parser.add_argument("--rviz", action="store_true", help="Execute purely in RViz (no physics)")
    parser.add_argument("--start", nargs=3, type=float, default=DEFAULT_START, help="Start XYZ coordinates (m)")
    parser.add_argument("--end", nargs=3, type=float, default=DEFAULT_END, help="End XYZ coordinates (m)")
    parser.add_argument("--twist", type=float, default=DEFAULT_TWIST_DEG, help="End-effector twist angle in degrees (default: 45°)")
    args = parser.parse_args()

    line_start = np.array(args.start)
    line_end = np.array(args.end)

    print("=" * 65)
    print("  KUKA KR6 R700 — Pure IK-Geo 6-DOF Sine Wave Tracking")
    print(f"  Cartesian wave (Amplitude {Z_AMPLITUDE}m) + {args.twist}° wrist twist")
    print("=" * 65)

    # 1. Plan trajectory with orientation
    trajectory, cartesian_points, jump_distances, fk_errors, orient_errors = generate_wave_trajectory(line_start, line_end, twist_deg=args.twist)
    
    # Generate the Matplotlib Analysis Graphs
    plot_trajectory_analysis(trajectory, jump_distances, fk_errors, orient_errors, shape_name="wave")
    
    # 2. Add HOME block at the front and back for safety
    print("\nAdding HOME sequence for safe deployment...")
    safe_traj = [Q_HOME] + trajectory + [Q_HOME]
    
    # We need to prepend/append HOME to cartesian points too so the blue line connects down
    safe_pts = [ik.fwd_kinematics(Q_HOME)[1]] + cartesian_points + [ik.fwd_kinematics(Q_HOME)[1]]
    
    # 3. Execute
    if args.ros or args.rviz:
        import os
        ros_python = '/opt/ros/noetic/lib/python3/dist-packages'
        if ros_python not in sys.path and os.path.isdir(ros_python):
            sys.path.insert(0, ros_python)
        
        mode = "ros" if args.ros else "rviz"
        execute_ros(safe_traj, safe_pts, mode=mode)
    else:
        print("\n[Preview] Trajectory planned successfully. Run with --ros or --rviz to execute.")


if __name__ == "__main__":
    main()

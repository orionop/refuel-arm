#!/usr/bin/env python3
"""
KUKA KR6 R700 — IK-Geo Multi-Solution Visualizer (Ghost Arms)
==============================================================

For a given EE pose, IK-Geo produces up to 8 closed-form solutions.
This script:
  1. Solves IK for the target pose.
  2. Filters valid solutions (within joint limits).
  3. Publishes each valid solution as a separate "ghost arm" in RViz
     using TF-prefixed robot_state_publishers.

Usage:
  Terminal 1: roslaunch kuka_kr6_gazebo multi_ik_viz.launch
  Terminal 2: python3 visualize_ik_solutions.py
              python3 visualize_ik_solutions.py --x 0.55 --y 0.3 --z 0.5 --pitch 15
"""
import sys
import os
import subprocess
import signal
import argparse
import numpy as np

# Import IK-Geo
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'kuka_refuel_ws', 'src', 'kuka_kr6_gazebo', 'scripts')))
import ik_geometric as ik

# ROS
ros_python = '/opt/ros/noetic/lib/python3/dist-packages'
if ros_python not in sys.path and os.path.isdir(ros_python):
    sys.path.insert(0, ros_python)

import rospy
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker, MarkerArray

# ── Config ────────────────────────────────────────────────────────
JOINT_LIMITS = np.array([
    [-2.967,  2.967], [-3.316,  0.785], [-2.094,  2.722],
    [-3.228,  3.228], [-2.094,  2.094], [-6.108,  6.108],
])

JOINT_NAMES = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']

# Color palette for ghost arms (R, G, B)
GHOST_COLORS = [
    (0.2, 0.8, 0.2),   # Green  — Solution 0
    (0.2, 0.2, 1.0),   # Blue   — Solution 1
    (1.0, 0.2, 0.2),   # Red    — Solution 2
    (1.0, 1.0, 0.2),   # Yellow — Solution 3
    (1.0, 0.5, 0.0),   # Orange — Solution 4
    (0.8, 0.2, 0.8),   # Purple — Solution 5
    (0.0, 1.0, 1.0),   # Cyan   — Solution 6
    (1.0, 1.0, 1.0),   # White  — Solution 7
]

URDF_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), 'kuka_refuel_ws', 'src', 'kuka_kr6_gazebo', 'urdf', 'kr6_r700_2_clean.urdf'))


def is_valid(q):
    for j in range(6):
        if q[j] < JOINT_LIMITS[j, 0] or q[j] > JOINT_LIMITS[j, 1]:
            return False
    return True


def solve_all_ik(target_pos, target_R):
    """Solve IK and return ALL solutions with validity flags."""
    Q_all = ik.IK_spherical_2_parallel(target_R, target_pos)
    if Q_all.size == 0:
        return []
    
    solutions = []
    for i in range(Q_all.shape[1]):
        q_raw = Q_all[:, i]
        q_wrapped = (q_raw + np.pi) % (2 * np.pi) - np.pi
        
        R_check, p_check = ik.fwd_kinematics(q_wrapped)
        fk_err = np.linalg.norm(p_check - target_pos)
        valid = is_valid(q_wrapped)
        
        solutions.append({
            'index': i,
            'q': q_wrapped,
            'fk_pos': p_check,
            'fk_err': fk_err,
            'valid': valid,
        })
    
    return solutions


def print_solution_table(solutions, target_pos):
    """Pretty-print all IK solutions."""
    print(f"\n{'='*90}")
    print(f"  IK-Geo Multi-Solution Analysis for Target: {target_pos}")
    print(f"{'='*90}")
    print(f"  {'Sol':>3} | {'Valid':>5} | {'FK Error (m)':>12} | Joint Angles (rad)")
    print(f"  {'-'*3}-+-{'-'*5}-+-{'-'*12}-+-{'-'*50}")
    
    valid_count = 0
    for s in solutions:
        q_str = ', '.join([f'{v:+7.3f}' for v in s['q']])
        status = '  ✅' if s['valid'] else '  ❌'
        print(f"  {s['index']:3d} | {status} | {s['fk_err']:12.2e} | [{q_str}]")
        if s['valid']:
            valid_count += 1
    
    print(f"\n  Total: {len(solutions)} solutions, {valid_count} valid (within joint limits)")
    print(f"{'='*90}\n")
    return valid_count


def load_urdf_string(tf_prefix):
    """Load URDF and modify link/joint names for TF prefix namespacing."""
    with open(URDF_PATH, 'r') as f:
        urdf = f.read()
    
    # For TF-prefixed ghost arms, we don't modify the URDF itself.
    # Instead, robot_state_publisher uses the tf_prefix parameter to
    # automatically prefix all TF frames.
    return urdf


def spawn_ghost_publishers(valid_solutions):
    """Spawn a robot_state_publisher per valid solution with TF prefix."""
    processes = []
    urdf_str = load_urdf_string("")
    
    for i, sol in enumerate(valid_solutions):
        ns = f"ghost_{i}"
        tf_prefix = f"ghost_{i}"
        
        # Set the robot_description on the parameter server under namespace
        rospy.set_param(f'/{ns}/robot_description', urdf_str)
        rospy.set_param(f'/{ns}/tf_prefix', tf_prefix)
        
        # Launch robot_state_publisher in the ghost namespace
        cmd = [
            'rosrun', 'robot_state_publisher', 'robot_state_publisher',
            f'__ns:={ns}',
            f'_tf_prefix:={tf_prefix}',
            f'robot_description:=/{ns}/robot_description',
        ]
        
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        processes.append(proc)
        rospy.loginfo(f"  🤖 Ghost {i} (Sol #{sol['index']}): robot_state_publisher started [TF: {tf_prefix}/]")
    
    return processes


def publish_ghost_joint_states(valid_solutions, publishers):
    """Publish joint states for each ghost arm."""
    for i, (sol, pub) in enumerate(zip(valid_solutions, publishers)):
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = ''
        msg.name = JOINT_NAMES
        msg.position = sol['q'].tolist()
        pub.publish(msg)


def publish_target_marker(target_pos, marker_pub):
    """Publish a sphere at the target position."""
    m = Marker()
    m.header.frame_id = "world"
    m.header.stamp = rospy.Time.now()
    m.ns = "ik_target"
    m.id = 0
    m.type = Marker.SPHERE
    m.action = Marker.ADD
    m.pose.position.x = target_pos[0]
    m.pose.position.y = target_pos[1]
    m.pose.position.z = target_pos[2]
    m.pose.orientation.w = 1.0
    m.scale.x = 0.06; m.scale.y = 0.06; m.scale.z = 0.06
    m.color.r = 1.0; m.color.g = 0.0; m.color.b = 0.0; m.color.a = 1.0
    marker_pub.publish(m)


def publish_solution_labels(valid_solutions, marker_pub):
    """Publish text labels near each ghost's EE."""
    ma = MarkerArray()
    for i, sol in enumerate(valid_solutions):
        m = Marker()
        m.header.frame_id = "world"
        m.header.stamp = rospy.Time.now()
        m.ns = "solution_labels"
        m.id = i + 10
        m.type = Marker.TEXT_VIEW_FACING
        m.action = Marker.ADD
        m.pose.position.x = sol['fk_pos'][0]
        m.pose.position.y = sol['fk_pos'][1]
        m.pose.position.z = sol['fk_pos'][2] + 0.08
        m.pose.orientation.w = 1.0
        m.scale.z = 0.04
        r, g, b = GHOST_COLORS[i % len(GHOST_COLORS)]
        m.color.r = r; m.color.g = g; m.color.b = b; m.color.a = 1.0
        m.text = f"Sol {sol['index']}"
        ma.markers.append(m)
    marker_pub.publish(ma)


def main():
    parser = argparse.ArgumentParser(description="IK-Geo Multi-Solution Ghost Arm Visualizer")
    parser.add_argument("--x", type=float, default=0.55, help="Target X (m)")
    parser.add_argument("--y", type=float, default=0.3, help="Target Y (m)")
    parser.add_argument("--z", type=float, default=0.5, help="Target Z (m)")
    parser.add_argument("--pitch", type=float, default=15.0, help="Target pitch (degrees)")
    args = parser.parse_args()

    target_pos = np.array([args.x, args.y, args.z])
    target_R = ik.rot(np.array([0.0, 1.0, 0.0]), np.radians(args.pitch))

    # 1. Solve IK
    solutions = solve_all_ik(target_pos, target_R)
    if not solutions:
        print("❌ IK-Geo returned 0 solutions for this pose!")
        return
    
    valid_count = print_solution_table(solutions, target_pos)
    if valid_count == 0:
        print("❌ No valid solutions within joint limits!")
        return
    
    valid_solutions = [s for s in solutions if s['valid']]
    
    # 2. Initialize ROS
    rospy.init_node('ik_multi_solution_viz', anonymous=True)
    
    # 3. Spawn ghost arm publishers
    print("[ROS] Spawning ghost arm publishers...")
    processes = spawn_ghost_publishers(valid_solutions)
    rospy.sleep(1.0)  # Wait for publishers to connect
    
    # 4. Create joint_state publishers for each ghost
    js_publishers = []
    for i in range(len(valid_solutions)):
        pub = rospy.Publisher(f'/ghost_{i}/joint_states', JointState, queue_size=10)
        js_publishers.append(pub)
    
    marker_pub = rospy.Publisher('/visualization_marker', Marker, queue_size=10)
    marker_array_pub = rospy.Publisher('/visualization_marker_array', MarkerArray, queue_size=10)
    rospy.sleep(0.5)
    
    # 5. Main loop: continuously publish joint states
    print(f"\n[ROS] Publishing {len(valid_solutions)} ghost arms. Press Ctrl+C to stop.\n")
    print("  💡 In RViz, add a 'RobotModel' display for each ghost:")
    for i, sol in enumerate(valid_solutions):
        r, g, b = GHOST_COLORS[i % len(GHOST_COLORS)]
        print(f"     Ghost {i} (Sol #{sol['index']}): "
              f"Robot Description: /ghost_{i}/robot_description | "
              f"TF Prefix: ghost_{i} | "
              f"Color: ({r},{g},{b})")
    print()
    
    rate = rospy.Rate(10)  # 10 Hz
    
    def shutdown_handler(sig, frame):
        print("\n[Shutdown] Killing ghost publishers...")
        for p in processes:
            p.terminate()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, shutdown_handler)
    
    while not rospy.is_shutdown():
        publish_ghost_joint_states(valid_solutions, js_publishers)
        publish_target_marker(target_pos, marker_pub)
        publish_solution_labels(valid_solutions, marker_array_pub)
        rate.sleep()


if __name__ == "__main__":
    main()

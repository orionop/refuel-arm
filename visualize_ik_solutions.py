#!/usr/bin/env python3
"""
KUKA KR6 R700 — IK-Geo Multi-Solution Visualizer (Ghost Arms)
==============================================================

For a given EE pose, IK-Geo produces up to 8 closed-form solutions.
This script:
  1. Solves IK for the target pose.
  2. Filters valid solutions (within joint limits).
  3. Draws each valid config as a colored "ghost arm" in RViz
     using FK-computed link positions rendered as Line/Sphere markers.

Usage:
  Terminal 1: roslaunch kuka_kr6_gazebo rviz.launch
  Terminal 2: python3 visualize_ik_solutions.py
              python3 visualize_ik_solutions.py --x 0.55 --y 0.3 --z 0.5 --pitch 15
"""
import sys
import os
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
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA

# ── Config ────────────────────────────────────────────────────────
JOINT_LIMITS = np.array([
    [-2.967,  2.967], [-3.316,  0.785], [-2.094,  2.722],
    [-3.228,  3.228], [-2.094,  2.094], [-6.108,  6.108],
])

JOINT_NAMES = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']

# Color palette for ghost arms (R, G, B)
GHOST_COLORS = [
    (0.2, 0.9, 0.2),   # Green
    (0.3, 0.3, 1.0),   # Blue
    (1.0, 0.3, 0.3),   # Red
    (1.0, 1.0, 0.2),   # Yellow
    (1.0, 0.5, 0.0),   # Orange
    (0.8, 0.2, 0.8),   # Purple
    (0.0, 1.0, 1.0),   # Cyan
    (1.0, 1.0, 1.0),   # White
]


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


def compute_link_positions(q):
    """Compute all joint/link positions via FK chain for visualization."""
    kin = ik.KIN_KR6_R700
    H, P = kin['H'], kin['P']
    
    positions = []
    R = np.eye(3)
    p = P[:, 0].copy()
    positions.append(p.copy())  # Base
    
    for j in range(6):
        R = R @ ik.rot(H[:, j], q[j])
        p = p + R @ P[:, j + 1]
        positions.append(p.copy())
    
    return positions


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


def build_ghost_markers(valid_solutions):
    """Build MarkerArray with ghost arms rendered as colored line strips + joint spheres."""
    ma = MarkerArray()
    
    for i, sol in enumerate(valid_solutions):
        r, g, b = GHOST_COLORS[i % len(GHOST_COLORS)]
        link_positions = compute_link_positions(sol['q'])
        
        # Line strip connecting all joints
        line = Marker()
        line.header.frame_id = "world"
        line.ns = f"ghost_arm_{i}"
        line.id = i * 100
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.scale.x = 0.02  # Line width
        line.color = ColorRGBA(r=r, g=g, b=b, a=0.7)
        line.pose.orientation.w = 1.0
        
        for p in link_positions:
            line.points.append(Point(x=p[0], y=p[1], z=p[2]))
        ma.markers.append(line)
        
        # Spheres at each joint
        for j, p in enumerate(link_positions):
            sphere = Marker()
            sphere.header.frame_id = "world"
            sphere.ns = f"ghost_joints_{i}"
            sphere.id = i * 100 + j + 1
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position.x = p[0]
            sphere.pose.position.y = p[1]
            sphere.pose.position.z = p[2]
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = 0.04
            sphere.scale.y = 0.04
            sphere.scale.z = 0.04
            sphere.color = ColorRGBA(r=r, g=g, b=b, a=0.9)
            ma.markers.append(sphere)
        
        # Text label at the EE position
        label = Marker()
        label.header.frame_id = "world"
        label.ns = "ghost_labels"
        label.id = i * 100 + 50
        label.type = Marker.TEXT_VIEW_FACING
        label.action = Marker.ADD
        ee = link_positions[-1]
        label.pose.position.x = ee[0]
        label.pose.position.y = ee[1]
        label.pose.position.z = ee[2] + 0.08
        label.pose.orientation.w = 1.0
        label.scale.z = 0.05
        label.color = ColorRGBA(r=r, g=g, b=b, a=1.0)
        q_str = ', '.join([f'{v:.1f}°' for v in np.degrees(sol['q'])])
        label.text = f"Sol {sol['index']}\n[{q_str}]"
        ma.markers.append(label)
    
    return ma


def build_target_marker(target_pos):
    """Build a bright red sphere at the target."""
    m = Marker()
    m.header.frame_id = "world"
    m.ns = "ik_target"
    m.id = 999
    m.type = Marker.SPHERE
    m.action = Marker.ADD
    m.pose.position.x = target_pos[0]
    m.pose.position.y = target_pos[1]
    m.pose.position.z = target_pos[2]
    m.pose.orientation.w = 1.0
    m.scale.x = 0.08; m.scale.y = 0.08; m.scale.z = 0.08
    m.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
    return m


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
    
    marker_pub = rospy.Publisher('/visualization_marker_array', MarkerArray, queue_size=10)
    target_pub = rospy.Publisher('/visualization_marker', Marker, queue_size=10)
    rospy.sleep(1.0)
    
    # 3. Build markers
    ghost_markers = build_ghost_markers(valid_solutions)
    target_marker = build_target_marker(target_pos)
    
    print(f"[RViz] Publishing {len(valid_solutions)} ghost arms as colored markers.")
    print("  💡 Make sure RViz has 'Marker' and 'MarkerArray' displays enabled!")
    print("  Press Ctrl+C to stop.\n")
    
    # 4. Main loop
    rate = rospy.Rate(5)
    
    def shutdown(sig, frame):
        print("\n[Shutdown] Done.")
        sys.exit(0)
    signal.signal(signal.SIGINT, shutdown)
    
    while not rospy.is_shutdown():
        # Update timestamps
        now = rospy.Time.now()
        for m in ghost_markers.markers:
            m.header.stamp = now
        target_marker.header.stamp = now
        
        marker_pub.publish(ghost_markers)
        target_pub.publish(target_marker)
        rate.sleep()


if __name__ == "__main__":
    main()

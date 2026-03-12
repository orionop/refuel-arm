#!/usr/bin/env python3
"""
KUKA KR6 R700 — IK-Geo Multi-Solution Visualizer (Gazebo Ghost Arms)
=====================================================================

This script connects directly to a running Gazebo simulation, solves IK
for a given target pose, and spawns up to 8 fully rendered robotic arms in Gazebo.
These "ghost arms" are static (frozen in place) and visualized in different colors
to demonstrate the multimodal reachability of the IK-Geo solver in 3D physical space.

Usage:
  Terminal 1: roslaunch kuka_kr6_gazebo kr6_main.launch
  Terminal 2: python3 visualize_ik_solutions.py
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
from gazebo_msgs.srv import SpawnModel, DeleteModel, SetModelConfiguration
from geometry_msgs.msg import Pose

# ── Config ────────────────────────────────────────────────────────
JOINT_LIMITS = np.array([
    [-2.967,  2.967], [-3.316,  0.785], [-2.094,  2.722],
    [-3.228,  3.228], [-2.094,  2.094], [-6.108,  6.108],
])

JOINT_NAMES = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']

GAZEBO_COLORS = [
    "Gazebo/Green",
    "Gazebo/Blue",
    "Gazebo/Red",
    "Gazebo/Yellow",
    "Gazebo/Orange",
    "Gazebo/Purple",
]

URDF_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), 'kuka_refuel_ws', 'src', 'kuka_kr6_gazebo', 'urdf', 'kr6_r700_2_clean.urdf'))

SPAWNED_MODELS = []


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


def get_ghost_urdf(color_name="Gazebo/Green"):
    """Modify the URDF in-memory to be a static, collision-free, colored ghost."""
    import xml.etree.ElementTree as ET
    tree = ET.parse(URDF_PATH)
    root = tree.getroot()
    
    # Force the entire robot to be static in Gazebo so it doesn't fall due to gravity
    # The world joint prevents falling, but this ensures no jitter
    gazebo_static = ET.SubElement(root, 'gazebo')
    ET.SubElement(gazebo_static, 'static').text = 'true'
    
    for link in root.findall('link'):
        # 1. Remove collisions and inertia
        coll = link.find('collision')
        if coll is not None: link.remove(coll)
        inert = link.find('inertial')
        if inert is not None: link.remove(inert)
        
        # 2. Add color
        name = link.get('name')
        if name != 'world':
            gz = ET.SubElement(root, 'gazebo', {'reference': name})
            mat = ET.SubElement(gz, 'material')
            mat.text = color_name

    # Remove the gazebo_ros_control plugin so the ghost doesn't try to conflict with the real robot
    for gz in root.findall('gazebo'):
        for plugin in gz.findall('plugin'):
            if 'gazebo_ros_control' in plugin.get('name', ''):
                gz.remove(plugin)
                
    return ET.tostring(root, encoding='utf8').decode('utf8')


def cleanup_ghosts():
    """Delete all spawned ghosts from Gazebo."""
    if not SPAWNED_MODELS:
        return
    print("\n[Gazebo] Cleaning up ghost arms...")
    try:
        delete_srv = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        for model in SPAWNED_MODELS:
            delete_srv(model)
            print(f"  🗑️ Deleted {model}")
    except Exception as e:
        print(f"Failed to delete models: {e}")


def main():
    parser = argparse.ArgumentParser(description="IK-Geo Ghost Arms in Gazebo")
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
    
    valid_solutions = [s for s in solutions if s['valid']]
    print(f"\n[IK-Geo] Found {len(valid_solutions)} valid solutions for target {target_pos}.")
    
    # 2. Setup ROS and Services
    rospy.init_node('ik_gazebo_ghost_spawner', anonymous=True)
    
    print("[ROS] Waiting for Gazebo /spawn_urdf_model service...")
    try:
        rospy.wait_for_service('/gazebo/spawn_urdf_model', timeout=5.0)
        rospy.wait_for_service('/gazebo/set_model_configuration', timeout=5.0)
    except rospy.ROSException:
        print("❌ Gazebo services not found. Is Gazebo running? (roslaunch kuka_kr6_gazebo kr6_main.launch)")
        return
    
    spawn_srv = rospy.ServiceProxy('/gazebo/spawn_urdf_model', SpawnModel)
    set_config_srv = rospy.ServiceProxy('/gazebo/set_model_configuration', SetModelConfiguration)
    
    def shutdown_handler(sig, frame):
        cleanup_ghosts()
        sys.exit(0)
    signal.signal(signal.SIGINT, shutdown_handler)
    
    # 3. Spawn Ghosts
    pose = Pose()
    pose.orientation.w = 1.0
    
    for i, sol in enumerate(valid_solutions):
        model_name = f"ghost_arm_sol_{sol['index']}"
        color = GAZEBO_COLORS[i % len(GAZEBO_COLORS)]
        
        print(f"  👻 Spawning {model_name} in {color}...")
        
        # Get custom URDF
        urdf_xml = get_ghost_urdf(color)
        
        # Spawn!
        resp = spawn_srv(model_name, urdf_xml, f"/ghost_{i}", pose, "world")
        if resp.success:
            SPAWNED_MODELS.append(model_name)
            
            # Set the exact joint angles immediately
            rospy.sleep(0.5)  # Let model load
            q_list = sol['q'].tolist()
            cfg_resp = set_config_srv(model_name, "kr6_r700", JOINT_NAMES, q_list)
            if not cfg_resp.success:
                print(f"    ⚠️ Failed to set joint angles for {model_name}: {cfg_resp.status_message}")
        else:
            print(f"    ❌ Failed to spawn: {resp.status_message}")
    
    print(f"\n✅ All {len(valid_solutions)} ghost arms spawned in Gazebo.")
    print("Hit Ctrl+C to delete them and exit.")
    
    rospy.spin()

if __name__ == "__main__":
    main()

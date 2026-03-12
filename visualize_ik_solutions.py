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
import actionlib
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectoryPoint
from gazebo_msgs.srv import SpawnModel, DeleteModel
from geometry_msgs.msg import Pose

# ── Config ────────────────────────────────────────────────────────
JOINT_LIMITS = np.array([
    [-2.967,  2.967], [-3.316,  0.785], [-2.094,  2.722],
    [-3.228,  3.228], [-2.094,  2.094], [-6.108,  6.108],
])

JOINT_NAMES = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
Q_HOME = [0.0, -1.570796326, 0.0, 0.0, 0.0, 0.0]  # REST position

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


def get_ghost_urdf(q, color_name="Gazebo/Green"):
    """Modify the URDF in-memory: fix joints to specific angles, remove physics."""
    import xml.etree.ElementTree as ET
    from scipy.spatial.transform import Rotation
    
    tree = ET.parse(URDF_PATH)
    root = tree.getroot()
    
    gazebo_static = ET.SubElement(root, 'gazebo')
    ET.SubElement(gazebo_static, 'static').text = 'true'
    
    for link in root.findall('link'):
        coll = link.find('collision')
        if coll is not None: link.remove(coll)
        
        name = link.get('name')
        if name != 'world':
            gz = ET.SubElement(root, 'gazebo', {'reference': name})
            ET.SubElement(gz, 'material').text = color_name
            ET.SubElement(gz, 'turnGravityOff').text = 'true'

    for gz in root.findall('gazebo'):
        for plugin in gz.findall('plugin'):
            gz.remove(plugin)
            
    # Bake joint angles into fixed joints
    for j in range(6):
        joint = root.find(f".//joint[@name='joint_{j+1}']")
        if joint is not None:
            joint.set('type', 'fixed')
            origin = joint.find('origin')
            axis = joint.find('axis')
            
            rpy = [0.0, 0.0, 0.0]
            if origin is not None and 'rpy' in origin.attrib:
                rpy = [float(x) for x in origin.attrib['rpy'].split()]
                
            ax = [0.0, 0.0, 1.0]
            if axis is not None and 'xyz' in axis.attrib:
                ax = [float(x) for x in axis.attrib['xyz'].split()]
            
            # Base rotation R_origin
            R_base = Rotation.from_euler('xyz', rpy).as_matrix()
            
            # Angle rotation R_joint(q) around axis
            theta = q[j]
            v = np.array(ax)
            K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            R_joint = np.eye(3) + np.sin(theta)*K + (1-np.cos(theta))*(K@K)
            
            # Combined and back to Euler
            R_total = R_base @ R_joint
            new_rpy = Rotation.from_matrix(R_total).as_euler('xyz')
            
            if origin is None:
                origin = ET.SubElement(joint, 'origin')
                origin.set('xyz', '0 0 0')
            origin.set('rpy', f"{new_rpy[0]:.6f} {new_rpy[1]:.6f} {new_rpy[2]:.6f}")
            
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


def execute_trajectory(q_target, duration=2.0):
    """Wait for the action server and send the target joint configuration."""
    client = actionlib.SimpleActionClient('/kr6_arm_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
    if not client.wait_for_server(rospy.Duration(2.0)):
        print("    ⚠️ Could not connect to /kr6_arm_controller action server. Is the robot fully spawned?")
        return False
        
    goal = FollowJointTrajectoryGoal()
    goal.trajectory.joint_names = JOINT_NAMES
    
    point = JointTrajectoryPoint()
    point.positions = q_target
    point.time_from_start = rospy.Duration(duration)
    
    goal.trajectory.points.append(point)
    client.send_goal(goal)
    client.wait_for_result()
    return True


def spawn_target_marker(target_pos):
    """Spawn a simple black sphere at the target EE location."""
    xml = f\"\"\"<?xml version="1.0" ?>
    <sdf version="1.5">
      <model name="ik_target_marker">
        <static>true</static>
        <link name="link">
          <visual name="visual">
            <geometry><sphere><radius>0.05</radius></sphere></geometry>
            <material>
              <ambient>0 0 0 1</ambient>
              <diffuse>0 0 0 1</diffuse>
            </material>
          </visual>
        </link>
      </model>
    </sdf>\"\"\"
    pose = Pose()
    pose.position.x = target_pos[0]
    pose.position.y = target_pos[1]
    pose.position.z = target_pos[2]
    pose.orientation.w = 1.0
    try:
        spawn_srv = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        resp = spawn_srv("ik_target_marker", xml, "/", pose, "world")
        if resp.success: SPAWNED_MODELS.append("ik_target_marker")
    except Exception as e:
        print(f"Failed to spawn target marker: {e}")


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
        rospy.wait_for_service('/gazebo/spawn_sdf_model', timeout=5.0)
    except rospy.ROSException:
        print("❌ Gazebo services not found. Is Gazebo running? (roslaunch kuka_kr6_gazebo gazebo.launch)")
        return
    
    spawn_srv = rospy.ServiceProxy('/gazebo/spawn_urdf_model', SpawnModel)
    
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
        
        print(f"  👻 Spawning ghost {model_name} in {color}...")
        
        # Get custom URDF with baked joints
        urdf_xml = get_ghost_urdf(sol['q'], color)
        
        # Spawn! No need for SetModelConfiguration anymore because it's baked!
        resp = spawn_srv(model_name, urdf_xml, f"/ghost_{i}", pose, "world")
        if resp.success:
            SPAWNED_MODELS.append(model_name)
        else:
            print(f"    ❌ Failed to spawn ghost: {resp.status_message}")
            
    print(f"\n✅ All {len(valid_solutions)} ghost arms spawned in Gazebo.")
    
    # 4. Target Marker
    spawn_target_marker(target_pos)
    
    # 5. Animation Sequence
    print(f"\n[Animation] Moving the real robot to each ghost configuration ONE BY ONE...")
    rospy.sleep(2.0)
    
    for i, sol in enumerate(valid_solutions):
        print(f"\n  👉 Animating to Sol #{sol['index']} (Ghost {i+1}/{len(valid_solutions)})...")
        # Go to ghost pose
        execute_trajectory(sol['q'].tolist(), duration=1.5)
        # Admire
        rospy.sleep(1.0)
        
        print("  👈 Returning to REST...")
        # Return to REST
        execute_trajectory(Q_HOME, duration=1.5)
        # Settle
        rospy.sleep(0.5)

    print("\n✅ Animation complete. Hit Ctrl+C to delete ghosts and exit.")
    
    rospy.spin()

if __name__ == "__main__":
    main()

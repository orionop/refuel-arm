#!/usr/bin/env python3
"""
Target Pose: green marker spawn and EE orientation computation.
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), 'kuka_refuel_ws', 'src',
    'kuka_kr6_gazebo', 'scripts')))
from ik_geometric import rot

# Target: Base of a 20cm-deep socket (Center X=0.62 + 0.10)
# Shifted target rightward (Y=0.40) so the arm (at Y=0) spawns further to the right.
TARGET_XYZ_DEFAULT = np.array([0.52, 0.40, 0.50])

# EE orientation: tool axis pointing +Y (sideways into the socket)
R_TOOL_INTO_CAR_KUKA = np.array([
    [0., -1.,  0.],
    [1.,  0.,  0.],
    [0.,  0.,  1.]])

R_TOOL_INTO_CAR_UR5 = np.array([
    [ 0.,  0., -1.],
    [ 0.,  1.,  0.],
    [ 1.,  0.,  0.]])

INLET_SIZE = (0.06, 0.005, 0.06)

TARGET_MARKER_SDF_TEMPLATE = """<?xml version="1.0" ?>
<sdf version="1.9">
  <model name="refuel_target">
    <static>true</static>
    <pose>{x} {y} {z} 0 0 0</pose>
    <link name="link">
      <visual name="vis">
        <geometry><box><size>0.06 0.005 0.06</size></box></geometry>
        <material><ambient>0.0 0.85 0.0 1</ambient><diffuse>0.0 0.90 0.0 1</diffuse></material>
      </visual>
    </link>
  </model>
</sdf>"""


def get_inlet_pose(target_xyz, yaw=0.0, robot="kuka"):
    """Compute EE target position and orientation matrix."""
    R_yaw = rot(np.array([0., 0., 1.]), yaw)
    if robot == "ur5":
        inlet_R = R_yaw @ R_TOOL_INTO_CAR_UR5
    else:
        inlet_R = R_yaw @ R_TOOL_INTO_CAR_KUKA
    return target_xyz.copy(), inlet_R


def get_preapproach_pose(inlet_xyz, inlet_R, standoff=0.08, robot="kuka"):
    """Pull back from the target along the approach normal."""
    if robot == "ur5":
        approach_dir = inlet_R[:, 1]  # UR5 flange Y-axis is the tool axis
    else:
        approach_dir = inlet_R[:, 0]  # KUKA flange X-axis is the tool axis
    preapproach_xyz = inlet_xyz - standoff * approach_dir
    return preapproach_xyz, inlet_R


def spawn_target_marker(target_xyz, ros2_node=None):
    """Spawn a green square marker at the target pose in Gazebo (ROS2).

    Parameters
    ----------
    target_xyz : array-like
        [x, y, z] position for the marker in world frame.
    ros2_node : rclpy.node.Node
        Active rclpy node used to call the /spawn_entity service.
        Callers must pass their node reference; this function does not
        call rclpy.init() or create a node itself.

    ROS1 equivalent: rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
    Deprecated ROS1 version: deprecated/car_model.ros1.py
    """
    import subprocess

    if ros2_node is None:
        print("  [car_model] spawn_target_marker: ros2_node is required in ROS2 mode")
        return

    # Gz Sim: spawn via ros_gz_sim create command
    x, y, z = float(target_xyz[0]), float(target_xyz[1]), float(target_xyz[2])

    # Delete previous marker (silent fail on first run)
    subprocess.run(
        ['gz', 'service', '-s', '/world/refuel_world/remove',
         '--reqtype', 'gz.msgs.Entity',
         '--reptype', 'gz.msgs.Boolean',
         '--timeout', '1000',
         '--req', 'name: "refuel_target" type: 2'],
        capture_output=True, text=True, timeout=5,
    )

    # Render exactly at target center (back of the hollow socket)
    sdf_str = TARGET_MARKER_SDF_TEMPLATE.format(x=x, y=y, z=z)

    result = subprocess.run(
        ['ros2', 'run', 'ros_gz_sim', 'create',
         '-string', sdf_str,
         '-name', 'refuel_target',
         '-x', str(x), '-y', str(y), '-z', str(z)],
        capture_output=True, text=True, timeout=15,
    )
    if result.returncode == 0:
        print(f"  Green target marker visible at socket base [{x:.2f}, {y:.2f}, {z:.2f}]")
    else:
        print(f"  Target marker spawn note: {result.stderr.strip()}")

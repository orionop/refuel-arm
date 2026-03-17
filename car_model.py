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

# ── Default target (reachable by KUKA KR6 R700) ──────────────────
TARGET_XYZ_DEFAULT = np.array([0.55, 0.30, 0.50])

# EE orientation: tool axis pointing +Y (toward the target)
R_TOOL_INTO_CAR = np.array([
    [0.,  0.,  1.],
    [1.,  0.,  0.],
    [0.,  1.,  0.]])

INLET_SIZE = (0.06, 0.005, 0.06)

# Green marker SDF
TARGET_MARKER_SDF = """<?xml version="1.0" ?>
<sdf version="1.5">
  <model name="refuel_target">
    <static>true</static>
    <link name="link">
      <visual name="vis">
        <geometry><box><size>0.06 0.005 0.06</size></box></geometry>
        <material><ambient>0.0 0.85 0.0 1</ambient><diffuse>0.0 0.90 0.0 1</diffuse></material>
      </visual>
    </link>
  </model>
</sdf>"""


def get_inlet_pose(target_xyz, yaw=0.0):
    """Compute EE target position and orientation matrix."""
    R_yaw = rot(np.array([0., 0., 1.]), yaw)
    inlet_R = R_yaw @ R_TOOL_INTO_CAR
    return target_xyz.copy(), inlet_R


def get_preapproach_pose(inlet_xyz, inlet_R, standoff=0.08):
    """Pull back from the target along the approach normal."""
    approach_dir = inlet_R[:, 0]
    preapproach_xyz = inlet_xyz - standoff * approach_dir
    return preapproach_xyz, inlet_R


def spawn_target_marker(target_xyz):
    """Spawn a green square marker at the target pose in Gazebo."""
    import rospy
    from gazebo_msgs.srv import SpawnModel
    from geometry_msgs.msg import Pose

    rospy.wait_for_service('/gazebo/spawn_sdf_model', timeout=5.0)
    spawn = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)

    p = Pose()
    p.position.x = target_xyz[0]
    p.position.y = target_xyz[1]
    p.position.z = target_xyz[2]
    p.orientation.w = 1.0
    try:
        spawn("refuel_target", TARGET_MARKER_SDF, "/", p, "world")
        print(f"  Green target marker at [{target_xyz[0]:.2f}, "
              f"{target_xyz[1]:.2f}, {target_xyz[2]:.2f}]")
    except Exception as e:
        print(f"  Target marker spawn note: {e}")

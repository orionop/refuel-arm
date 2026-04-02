"""
UR5 — RViz Visualization Launch (ROS2 Humble)

Replaces: ur5_rviz.launch (ROS1 XML) — see deprecated/ros1_launch/ur5_rviz.ros1.launch
No Gazebo physics — pure visualization via joint_states.

Usage:
    ros2 launch ur5_gazebo ur5_rviz.launch.py
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    ur_desc_pkg = get_package_share_directory('ur_description')
    ur5_pkg     = get_package_share_directory('ur5_gazebo')

    # Load UR5 URDF
    urdf_path = os.path.join(ur_desc_pkg, 'urdf', 'ur5.urdf')
    with open(urdf_path, 'r') as f:
        robot_description = {'robot_description': f.read()}

    # Robot State Publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='both',
        parameters=[robot_description],
    )

    # RViz2
    rviz_config = os.path.join(ur5_pkg, 'config', 'ur5_rviz.rviz')
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_config],
        output='screen',
    )

    return LaunchDescription([
        robot_state_publisher,
        rviz,
    ])

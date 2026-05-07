"""
KUKA KR6 R700 — RViz Visualization Launch (ROS2 Humble)

Replaces: rviz.launch (ROS1 XML) — see deprecated/ros1_launch/kuka_rviz.ros1.launch
No Gazebo physics — pure visualization via joint_states.

Usage:
    ros2 launch kuka_kr6_gazebo rviz.launch.py
"""
import os
import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('kuka_kr6_gazebo')

    # Process xacro → robot_description
    xacro_file = os.path.join(pkg, 'urdf', 'kr6_r700.gazebo.xacro')
    robot_description = {'robot_description': xacro.process_file(xacro_file).toxml()}

    # Robot State Publisher (publishes TF from /joint_states)
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='both',
        parameters=[robot_description, {'use_sim_time': False}],
    )

    # Joint State Publisher (provides default pose if mission script isn't running)
    # This prevents the robot from being 'invisible' initially
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        parameters=[{'use_sim_time': False}],
    )

    # RViz2
    rviz_config = os.path.join(pkg, 'config', 'kuka_rviz.rviz')
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_config],
        output='screen',
        parameters=[{'use_sim_time': False}],
    )

    return LaunchDescription([
        robot_state_publisher,
        joint_state_publisher,
        rviz,
    ])

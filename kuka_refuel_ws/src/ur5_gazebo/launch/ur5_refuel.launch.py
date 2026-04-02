"""
UR5 — Gazebo Simulation Launch for Autonomous Refueling (ROS2 Humble)

Replaces: ur5_refuel.launch + ur5.launch (ROS1 XML)
  see deprecated/ros1_launch/ur5_refuel.ros1.launch
  see deprecated/ros1_launch/ur5.ros1.launch

Usage:
    ros2 launch ur5_gazebo ur5_refuel.launch.py
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, RegisterEventHandler, SetEnvironmentVariable
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    ur_desc_pkg = get_package_share_directory('ur_description')
    kuka_pkg    = get_package_share_directory('kuka_kr6_gazebo')
    ur5_pkg     = get_package_share_directory('ur5_gazebo')

    # 1. Load UR5 URDF and resolve $(find ur5_gazebo) — plain URDFs don't go
    #    through xacro so the substitution must be done here in Python.
    urdf_path = os.path.join(ur_desc_pkg, 'urdf', 'ur5.urdf')
    with open(urdf_path, 'r') as f:
        urdf_content = f.read()
    urdf_content = urdf_content.replace('$(find ur5_gazebo)', ur5_pkg)
    robot_description = {'robot_description': urdf_content}

    # 2. Robot State Publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='both',
        parameters=[robot_description, {'use_sim_time': True}],
    )

    # 3. Gazebo (Classic) with refuel world (shared with KUKA package)
    world_path = os.path.join(kuka_pkg, 'worlds', 'refuel_world.world')
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')
        ]),
        launch_arguments={'world': world_path}.items(),
    )

    # 4. Spawn UR5 — shoulder_lift pre-bent to -90 deg (matches ROS1 behaviour)
    spawn_robot = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'ur5',
            '-z', '0.05',
        ],
        output='screen',
    )

    # 5. Controllers — use spawner nodes (proper ROS2 approach; auto-waits for
    #    controller_manager which is started by libgazebo_ros2_control.so)
    load_joint_state_broadcaster = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster',
                   '--controller-manager', '/controller_manager'],
        output='screen',
    )

    load_arm_controller = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['ur5_arm_controller',
                   '--controller-manager', '/controller_manager'],
        output='screen',
    )

    plugin_path = '/opt/ros/humble/lib'
    set_plugin_path = SetEnvironmentVariable(
        name='GAZEBO_PLUGIN_PATH',
        value=plugin_path + ':' + os.environ.get('GAZEBO_PLUGIN_PATH', ''),
    )

    return LaunchDescription([
        set_plugin_path,
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=spawn_robot,
                on_exit=[load_joint_state_broadcaster],
            )
        ),
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=load_joint_state_broadcaster,
                on_exit=[load_arm_controller],
            )
        ),
        gazebo,
        robot_state_publisher,
        spawn_robot,
    ])

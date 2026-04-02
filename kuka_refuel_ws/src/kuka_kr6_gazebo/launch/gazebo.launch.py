"""
KUKA KR6 R700 — Gazebo Simulation Launch (ROS2 Humble)

Replaces: gazebo.launch (ROS1 XML) — see deprecated/ros1_launch/kuka_gazebo.ros1.launch

Usage:
    ros2 launch kuka_kr6_gazebo gazebo.launch.py
"""
import os
import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription, RegisterEventHandler, SetEnvironmentVariable
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('kuka_kr6_gazebo')

    # 1. Process xacro → robot_description
    xacro_file = os.path.join(pkg, 'urdf', 'kr6_r700.gazebo.xacro')
    robot_description = {'robot_description': xacro.process_file(xacro_file).toxml()}

    # 2. Robot State Publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='both',
        parameters=[robot_description],
    )

    # 3. Gazebo (Classic) with refuel world
    world_path = os.path.join(pkg, 'worlds', 'refuel_world.world')
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')
        ]),
        launch_arguments={'world': world_path}.items(),
    )

    # 4. Spawn KUKA KR6 R700 — shoulder joint pre-bent to -90 deg
    spawn_robot = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'kr6_r700',
            '-z', '0.05',
        ],
        output='screen',
    )

    # 5. Controllers (loaded in sequence after spawn completes)
    load_joint_state_broadcaster = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active',
             'joint_state_broadcaster'],
        output='screen',
    )

    load_arm_controller = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active',
             'kr6_arm_controller'],
        output='screen',
    )

    # Ensure Gazebo finds gazebo_ros2_control plugin regardless of shell env
    plugin_path = '/opt/ros/humble/lib'
    set_plugin_path = SetEnvironmentVariable(
        name='GAZEBO_PLUGIN_PATH',
        value=plugin_path + ':' + os.environ.get('GAZEBO_PLUGIN_PATH', ''),
    )

    return LaunchDescription([
        set_plugin_path,
        # Sequence: spawn → joint_state_broadcaster → arm_controller
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

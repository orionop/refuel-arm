"""
KUKA KR6 R700 — Gazebo Simulation Launch (ROS2 Humble)

Replaces: gazebo.launch (ROS1 XML) — see deprecated/ros1_launch/kuka_gazebo.ros1.launch

ROS1 flow was:
  1. param robot_description = xacro output
  2. include gazebo empty_world.launch
  3. spawn_model -param robot_description
  4. robot_state_publisher
  5. rosparam load controllers.yaml
  6. controller_manager/spawner

ROS2 equivalent:
  1. xacro.process_file() → urdf_xml string
  2. robot_state_publisher (holds robot_description as param + publishes topic)
  3. include gazebo_ros gazebo.launch.py
  4. spawn_entity.py -topic /robot_description
  5. controller YAML path is baked into URDF plugin tag at xacro time
  6. controller_manager spawner nodes (chained after spawn completes)

Usage:
    ros2 launch kuka_kr6_gazebo gazebo.launch.py
"""
import os
import subprocess
import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription,
    LogInfo,
    RegisterEventHandler,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('kuka_kr6_gazebo')

    # ── 1. Process xacro → robot_description ──────────────────────────
    xacro_file = os.path.join(pkg, 'urdf', 'kr6_r700.gazebo.xacro')
    yaml_path  = os.path.join(pkg, 'config', 'ros2_controllers.yaml')
    urdf_xml   = xacro.process_file(
        xacro_file,
        mappings={'ros2_controllers_yaml': yaml_path}
    ).toxml()
    robot_description = {'robot_description': urdf_xml}

    # ── Diagnostics ───────────────────────────────────────────────────
    has_plugin = 'libgazebo_ros2_control.so' in urdf_xml
    plugin_lib = '/opt/ros/humble/lib/libgazebo_ros2_control.so'
    plugin_exists = os.path.isfile(plugin_lib)
    current_gazebo_plugin_path = os.environ.get('GAZEBO_PLUGIN_PATH', '')

    diag_msg = (
        f'\n===== LAUNCH DIAGNOSTICS =====\n'
        f'  URDF contains gazebo_ros2_control plugin tag: {has_plugin}\n'
        f'  Controller YAML path: {yaml_path}\n'
        f'  Controller YAML exists: {os.path.isfile(yaml_path)}\n'
        f'  Plugin lib exists at {plugin_lib}: {plugin_exists}\n'
        f'  GAZEBO_PLUGIN_PATH (before launch): {current_gazebo_plugin_path}\n'
        f'==============================\n'
    )
    print(diag_msg)

    # ── 2. Environment ────────────────────────────────────────────────
    # Gazebo needs to find libgazebo_ros2_control.so
    set_plugin_path = SetEnvironmentVariable(
        name='GAZEBO_PLUGIN_PATH',
        value='/opt/ros/humble/lib:' + current_gazebo_plugin_path,
    )
    # Gazebo needs to resolve package:// mesh URIs → set model path to
    # the share directory's parent so model://kuka_kr6_gazebo/... works
    share_parent = os.path.dirname(pkg)  # .../share/
    current_model_path = os.environ.get('GAZEBO_MODEL_PATH', '')
    set_model_path = SetEnvironmentVariable(
        name='GAZEBO_MODEL_PATH',
        value=share_parent + ':' + current_model_path,
    )

    # ── 3. Robot State Publisher ──────────────────────────────────────
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='both',
        parameters=[robot_description, {'use_sim_time': True}],
    )

    # ── 4. Gazebo (Classic) ───────────────────────────────────────────
    world_path = os.path.join(pkg, 'worlds', 'refuel_world.world')
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory('gazebo_ros'),
                         'launch', 'gazebo.launch.py')
        ]),
        launch_arguments={
            'world': world_path,
            'verbose': 'true',
        }.items(),
    )

    # ── 5. Spawn robot ────────────────────────────────────────────────
    # Use -topic so spawn_entity reads the full URDF (with <gazebo> tags)
    # from robot_state_publisher. Delayed to let Gazebo fully start.
    spawn_robot = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', '/robot_description',
            '-entity', 'kr6_r700',
            '-z', '0.05',
        ],
        output='screen',
    )

    delayed_spawn = TimerAction(period=5.0, actions=[spawn_robot])

    # ── 6. Controller spawners (chained after spawn) ──────────────────
    load_jsb = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster'],
        output='screen',
    )

    load_arm = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['kr6_arm_controller'],
        output='screen',
    )

    return LaunchDescription([
        # Environment
        set_plugin_path,
        set_model_path,
        LogInfo(msg=diag_msg),
        # Nodes
        robot_state_publisher,
        gazebo,
        delayed_spawn,
        # Chain: spawn done → jsb → arm controller
        RegisterEventHandler(OnProcessExit(
            target_action=spawn_robot,
            on_exit=[load_jsb],
        )),
        RegisterEventHandler(OnProcessExit(
            target_action=load_jsb,
            on_exit=[load_arm],
        )),
    ])

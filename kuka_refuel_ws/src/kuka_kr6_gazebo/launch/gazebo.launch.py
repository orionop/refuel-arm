"""
KUKA KR6 R700 — Gz Sim Launch (ROS2 Jazzy + Gazebo Harmonic)

Replaces: gazebo.launch (ROS1 XML) — see deprecated/ros1_launch/kuka_gazebo.ros1.launch

Usage:
    ros2 launch kuka_kr6_gazebo gazebo.launch.py
"""
import os
import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription,
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
    yaml_path = os.path.join(pkg, 'config', 'ros2_controllers.yaml')
    urdf_xml = xacro.process_file(
        xacro_file,
        mappings={'ros2_controllers_yaml': yaml_path}
    ).toxml()
    robot_description = {'robot_description': urdf_xml}

    # ── 2. Environment ────────────────────────────────────────────────
    # Gz Sim needs to find gz_ros2_control plugin and mesh resources
    set_plugin_path = SetEnvironmentVariable(
        name='GZ_SIM_SYSTEM_PLUGIN_PATH',
        value='/opt/ros/jazzy/lib:' + os.environ.get('GZ_SIM_SYSTEM_PLUGIN_PATH', ''),
    )
    share_parent = os.path.dirname(pkg)  # .../install/.../share/
    set_resource_path = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=share_parent + ':' + os.environ.get('GZ_SIM_RESOURCE_PATH', ''),
    )

    # ── 3. Robot State Publisher ──────────────────────────────────────
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='both',
        parameters=[robot_description, {'use_sim_time': True}],
    )

    # ── 4. Gz Sim ─────────────────────────────────────────────────────
    world_path = os.path.join(pkg, 'worlds', 'refuel_world.sdf')
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('ros_gz_sim'),
                         'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': ['-r ', world_path]}.items(),
    )

    # ── 5. Spawn robot ────────────────────────────────────────────────
    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-topic', '/robot_description',
            '-name', 'kr6_r700',
            '-z', '0.05',
            '--joint-names', 'joint_1,joint_2,joint_3,joint_4,joint_5,joint_6',
            '--joint-positions', '0.0,-1.5708,0.0,0.0,0.0,0.0',
        ],
        output='screen',
    )

    delayed_spawn = TimerAction(period=5.0, actions=[spawn_robot])

    # ── 6. Bridge Gz Sim → ROS2 ───────────────────────────────────────
    # - /clock: required for use_sim_time
    # - /tf and /tf_static: allows seeing the robot/world state in RViz
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V',
            '/tf_static@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V',
        ],
        output='screen',
    )

    # ── 7. Controller spawners (chained after spawn) ──────────────────
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
        set_plugin_path,
        set_resource_path,
        robot_state_publisher,
        gz_sim,
        bridge,
        delayed_spawn,
        RegisterEventHandler(OnProcessExit(
            target_action=spawn_robot,
            on_exit=[load_jsb],
        )),
        RegisterEventHandler(OnProcessExit(
            target_action=load_jsb,
            on_exit=[load_arm],
        )),
    ])

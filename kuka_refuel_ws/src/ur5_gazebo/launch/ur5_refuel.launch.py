"""
UR5 — Gz Sim Launch for Autonomous Refueling (ROS2 Jazzy + Gazebo Harmonic)

Replaces: ur5_refuel.launch + ur5.launch (ROS1 XML)
  see deprecated/ros1_launch/ur5_refuel.ros1.launch
  see deprecated/ros1_launch/ur5.ros1.launch

Usage:
    ros2 launch ur5_gazebo ur5_refuel.launch.py
"""
import os
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
    ur_desc_pkg = get_package_share_directory('ur_description')
    kuka_pkg = get_package_share_directory('kuka_kr6_gazebo')
    ur5_pkg = get_package_share_directory('ur5_gazebo')

    # 1. Load UR5 URDF and resolve $(find ...) substitutions
    urdf_path = os.path.join(ur_desc_pkg, 'urdf', 'ur5.urdf')
    with open(urdf_path, 'r') as f:
        urdf_content = f.read()
    urdf_content = urdf_content.replace('$(find ur5_gazebo)', ur5_pkg)
    robot_description = {'robot_description': urdf_content}

    # 2. Environment
    set_plugin_path = SetEnvironmentVariable(
        name='GZ_SIM_SYSTEM_PLUGIN_PATH',
        value='/opt/ros/jazzy/lib:' + os.environ.get('GZ_SIM_SYSTEM_PLUGIN_PATH', ''),
    )
    share_parent = os.path.dirname(ur_desc_pkg)
    set_resource_path = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=share_parent + ':' + os.environ.get('GZ_SIM_RESOURCE_PATH', ''),
    )

    # 3. Robot State Publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='both',
        parameters=[robot_description, {'use_sim_time': True}],
    )

    # 4. Gz Sim with refuel world (shared with KUKA package)
    world_path = os.path.join(kuka_pkg, 'worlds', 'refuel_world.sdf')
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('ros_gz_sim'),
                         'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': ['-r ', world_path]}.items(),
    )

    # 5. Spawn UR5
    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-topic', '/robot_description',
            '-name', 'ur5',
            '-z', '0.0',
        ],
        output='screen',
    )

    delayed_spawn = TimerAction(period=5.0, actions=[spawn_robot])

    # 6. Bridge Gz Sim → ROS2
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

    # 7. Controllers
    load_jsb = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster'],
        output='screen',
    )
    load_arm = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['ur5_arm_controller'],
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

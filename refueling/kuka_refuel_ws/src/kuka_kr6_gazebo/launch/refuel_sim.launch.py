"""
KUKA KR6 R700 — Refueling Simulation Launch (ROS2 Jazzy + Gz Sim)

Simplified launch: Gz Sim + robot + controllers. No diagnostics.

Usage:
    ros2 launch kuka_kr6_gazebo refuel_sim.launch.py
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
    ExecuteProcess,
)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('kuka_kr6_gazebo')

    # 1. Process xacro
    xacro_file = os.path.join(pkg, 'urdf', 'kr6_r700.gazebo.xacro')
    yaml_path = os.path.join(pkg, 'config', 'ros2_controllers.yaml')
    urdf_xml = xacro.process_file(
        xacro_file,
        mappings={'ros2_controllers_yaml': yaml_path}
    ).toxml()
    robot_description = {'robot_description': urdf_xml}

    # 2. Environment
    set_plugin_path = SetEnvironmentVariable(
        name='GZ_SIM_SYSTEM_PLUGIN_PATH',
        value='/opt/ros/jazzy/lib:' + os.environ.get('GZ_SIM_SYSTEM_PLUGIN_PATH', ''),
    )
    share_parent = os.path.dirname(pkg)
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

    # 4. Gz Sim
    world_path = os.path.join(pkg, 'worlds', 'refuel_gas_station.sdf')
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('ros_gz_sim'),
                         'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': ['-r ', world_path]}.items(),
    )

    # 5. Spawn robot
    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-topic', '/robot_description',
            '-name', 'kr6_r700',
            '-x', '0.2896',
            '-y', '-12.1261',
            '-z', '0.1'
        ],
        output='screen',
    )

    # Re-added small safety delay for Mac stability
    delayed_spawn = TimerAction(period=2.0, actions=[spawn_robot])

    # 6. Bridge /clock from Gz Sim → ROS2 (required for use_sim_time)
    clock_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'],
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
        arguments=['kr6_arm_controller'],
        output='screen',
    )

    # 8. Command upright candle pose
    candle_pose_cmd = ExecuteProcess(
        cmd=[
            'ros2', 'topic', 'pub', '--once', '/kr6_arm_controller/joint_trajectory',
            'trajectory_msgs/msg/JointTrajectory',
            '{"joint_names": ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"], "points": [{"positions": [0.0, -1.5708, 0.0, 0.0, 0.0, 0.0], "time_from_start": {"sec": 1, "nanosec": 0}}]}'
        ],
        output='screen'
    )

    return LaunchDescription([
        set_plugin_path,
        set_resource_path,
        robot_state_publisher,
        gz_sim,
        clock_bridge,
        delayed_spawn, # 2s delay
        RegisterEventHandler(OnProcessExit(
            target_action=spawn_robot,
            on_exit=[load_jsb],
        )),
        RegisterEventHandler(OnProcessExit(
            target_action=load_jsb,
            on_exit=[load_arm],
        )),
        RegisterEventHandler(OnProcessExit(
            target_action=load_arm,
            on_exit=[candle_pose_cmd],
        )),
    ])

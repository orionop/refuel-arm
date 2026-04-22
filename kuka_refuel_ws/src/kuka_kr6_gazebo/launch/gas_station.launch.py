import os
import shutil
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (IncludeLaunchDescription, RegisterEventHandler,
                             SetEnvironmentVariable, TimerAction, ExecuteProcess)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    pkg = get_package_share_directory('kuka_kr6_gazebo')

    # KR8 is already embedded in refuel_gas_station.sdf — do NOT spawn a
    # separate robot. Publish its URDF so the controller_manager (started
    # inside Gz Sim by the gz_ros2_control plugin) can resolve hardware
    # interfaces.  Using the KR8 model.urdf (not the KR6 xacro) so link
    # names match what RViz/tf expects.
    kr8_urdf_path = os.path.join(pkg, 'models', 'kr8_r2100', 'model.urdf')
    with open(kr8_urdf_path, 'r') as f:
        kr8_urdf = f.read()
    robot_description = {'robot_description': kr8_urdf}

    # gz_ros2_control reads <parameters> as a literal path passed to rcl --params-file.
    # $ENV{} substitution does not work inside <plugin> content in libsdformat.
    # Copy the yaml to a fixed absolute path so model.sdf can reference it unconditionally.
    kr8_ctrl_cfg_src = os.path.join(pkg, 'models', 'kr8_r2100', 'config', 'ros2_controllers.yaml')
    shutil.copy2(kr8_ctrl_cfg_src, '/tmp/kr8_ros2_controllers.yaml')

    # ── Environment ─────────────────────────────────────────────────
    set_plugin_path = SetEnvironmentVariable(
        name='GZ_SIM_SYSTEM_PLUGIN_PATH',
        value='/opt/ros/jazzy/lib:' + os.environ.get('GZ_SIM_SYSTEM_PLUGIN_PATH', ''),
    )
    models_path = os.path.join(pkg, 'models')
    share_parent = os.path.dirname(pkg)
    set_resource_path = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=models_path + ':' + share_parent + ':' + os.environ.get('GZ_SIM_RESOURCE_PATH', ''),
    )

    # ── Nodes ────────────────────────────────────────────────────────
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[robot_description, {'use_sim_time': True}],
        output='screen',
    )

    world_path = os.path.join(pkg, 'worlds', 'refuel_gas_station.sdf')
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': ['-r ', world_path]}.items(),
    )

    clock_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'],
    )

    # Controllers — wait 10 s for Gz Sim + gz_ros2_control to finish init
    load_jsb = Node(
        package='controller_manager', executable='spawner',
        arguments=['joint_state_broadcaster'], output='screen',
    )
    load_arm = Node(
        package='controller_manager', executable='spawner',
        arguments=['kr6_arm_controller'], output='screen',
    )
    delayed_controllers = TimerAction(period=10.0, actions=[load_jsb])

    # Move KR8 to upright candle pose immediately after arm controller is ready.
    # Prevents gravity from pulling the arm down before refuel_mission.py runs.
    move_to_candle = ExecuteProcess(
        cmd=[
            'ros2', 'action', 'send_goal',
            '/kr6_arm_controller/follow_joint_trajectory',
            'control_msgs/action/FollowJointTrajectory',
            '{trajectory: {joint_names: [joint_1, joint_2, joint_3,'
            ' joint_4, joint_5, joint_6],'
            ' points: [{positions: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],'
            ' time_from_start: {sec: 3, nanosec: 0}}]}}',
        ],
        output='screen',
    )

    return LaunchDescription([
        set_plugin_path,
        set_resource_path,
        robot_state_publisher,
        gz_sim,
        clock_bridge,
        delayed_controllers,
        RegisterEventHandler(OnProcessExit(target_action=load_jsb, on_exit=[load_arm])),
        RegisterEventHandler(OnProcessExit(target_action=load_arm, on_exit=[move_to_candle])),
    ])

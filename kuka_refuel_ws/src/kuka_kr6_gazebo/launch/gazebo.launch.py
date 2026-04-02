"""
KUKA KR6 R700 — Gazebo Simulation Launch (ROS2 Humble)

Replaces: gazebo.launch (ROS1 XML) — see deprecated/ros1_launch/kuka_gazebo.ros1.launch

Usage:
    ros2 launch kuka_kr6_gazebo gazebo.launch.py
"""
import os
import tempfile
import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, RegisterEventHandler, SetEnvironmentVariable
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('kuka_kr6_gazebo')

    # 1. Process xacro → robot_description
    #    Pass yaml path as absolute so $(find ...) inside xacro is not needed
    xacro_file = os.path.join(pkg, 'urdf', 'kr6_r700.gazebo.xacro')
    yaml_path = os.path.join(pkg, 'config', 'ros2_controllers.yaml')
    urdf_xml = xacro.process_file(
        xacro_file,
        mappings={'ros2_controllers_yaml': yaml_path}
    ).toxml()
    robot_description = {'robot_description': urdf_xml}

    # Write processed URDF to temp file so spawn_entity can load it directly
    # (using -file instead of -topic ensures Gazebo processes <gazebo><plugin> tags)
    urdf_file = os.path.join(tempfile.gettempdir(), 'kr6_r700.urdf')
    with open(urdf_file, 'w') as f:
        f.write(urdf_xml)

    # 2. Robot State Publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='both',
        parameters=[robot_description, {'use_sim_time': True}],
    )

    # 3. Gazebo (Classic) with refuel world
    world_path = os.path.join(pkg, 'worlds', 'refuel_world.world')
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')
        ]),
        launch_arguments={'world': world_path, 'verbose': 'true'}.items(),
    )

    # 4. Spawn KUKA KR6 R700 from file (not topic) so Gazebo loads model plugins
    spawn_robot = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-file', urdf_file,
            '-entity', 'kr6_r700',
            '-z', '0.05',
        ],
        output='screen',
    )

    # 5. Controllers — use spawner nodes (proper ROS2 approach; auto-waits for
    #    controller_manager which is started by libgazebo_ros2_control.so)
    load_joint_state_broadcaster = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster'],
        output='screen',
    )

    load_arm_controller = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['kr6_arm_controller'],
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

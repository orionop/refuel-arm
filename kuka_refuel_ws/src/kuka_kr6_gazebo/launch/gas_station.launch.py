import os
import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (IncludeLaunchDescription, RegisterEventHandler, SetEnvironmentVariable, TimerAction, ExecuteProcess)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    pkg = get_package_share_directory('kuka_kr6_gazebo')
    xacro_file = os.path.join(pkg, 'urdf', 'kr6_r700.gazebo.xacro')
    yaml_path = os.path.join(pkg, 'config', 'ros2_controllers.yaml')
    urdf_xml = xacro.process_file(xacro_file, mappings={'ros2_controllers_yaml': yaml_path}).toxml()
    robot_description = {'robot_description': urdf_xml}

    # Environment for Fuel models
    set_plugin_path = SetEnvironmentVariable(
        name='GZ_SIM_SYSTEM_PLUGIN_PATH',
        value='/opt/ros/jazzy/lib:' + os.environ.get('GZ_SIM_SYSTEM_PLUGIN_PATH', ''),
    )
    share_parent = os.path.dirname(pkg)
    set_resource_path = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=share_parent + ':' + os.environ.get('GZ_SIM_RESOURCE_PATH', ''),
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[robot_description, {'use_sim_time': True}],
    )

    # Launch the NEW Gas Station world
    world_path = os.path.join(pkg, 'worlds', 'refuel_gas_station.sdf')
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': ['-r ', world_path]}.items(),
    )

    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=['-topic', '/robot_description', '-name', 'kr6_r700_gas'],
        output='screen',
    )
    delayed_spawn = TimerAction(period=5.0, actions=[spawn_robot])

    clock_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'],
    )

    load_jsb = Node(package='controller_manager', executable='spawner', arguments=['joint_state_broadcaster'])
    load_arm = Node(package='controller_manager', executable='spawner', arguments=['kr6_arm_controller'])

    return LaunchDescription([
        set_plugin_path,
        set_resource_path,
        robot_state_publisher,
        gz_sim,
        clock_bridge,
        delayed_spawn,
        RegisterEventHandler(OnProcessExit(target_action=spawn_robot, on_exit=[load_jsb])),
        RegisterEventHandler(OnProcessExit(target_action=load_jsb, on_exit=[load_arm])),
    ])

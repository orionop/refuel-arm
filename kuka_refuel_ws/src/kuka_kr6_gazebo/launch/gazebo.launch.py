"""
KUKA KR6 R700 — Gz Sim Launch (ROS2 Jazzy + Gazebo Harmonic)

Replaces: gazebo.launch (ROS1 XML) — see deprecated/ros1_launch/kuka_gazebo.ros1.launch

Usage:
    ros2 launch kuka_kr6_gazebo gazebo.launch.py
"""
import os
import subprocess
import tempfile
import xml.etree.ElementTree as ET

import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    ExecuteProcess,
    IncludeLaunchDescription,
    RegisterEventHandler,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

# Joint angles for candle (upright) spawn pose.
# Gz Sim's URDF parser does NOT handle <initial_position> in <gazebo reference>
# tags — it copies them verbatim instead of placing them inside <axis>.
# We fix this by converting URDF→SDF ourselves and patching the SDF.
INITIAL_JOINT_POSITIONS = {
    'joint_2': -1.5708,
}


def _urdf_to_sdf_with_initial_positions(urdf_xml):
    """Convert URDF to SDF using ``gz sdf -p`` and fix initial_position placement.

    The Gz Sim URDF parser places <initial_position> as a direct child of
    <joint> (invalid).  SDF requires it inside <joint><axis>.  This function
    moves it to the correct location and injects any entries from
    INITIAL_JOINT_POSITIONS that are missing.
    """
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.urdf', delete=False
    ) as tmp:
        tmp.write(urdf_xml)
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            ['gz', 'sdf', '-p', tmp_path],
            capture_output=True, text=True, timeout=15,
        )
    finally:
        os.unlink(tmp_path)

    if result.returncode != 0:
        raise RuntimeError(f'gz sdf -p failed: {result.stderr}')

    root = ET.fromstring(result.stdout)

    for joint_elem in root.iter('joint'):
        jname = joint_elem.get('name')

        # Remove misplaced <initial_position> (copied verbatim by URDF parser)
        for ip in joint_elem.findall('initial_position'):
            joint_elem.remove(ip)

        # Insert correct <initial_position> inside <axis>
        if jname in INITIAL_JOINT_POSITIONS:
            axis = joint_elem.find('axis')
            if axis is not None:
                ip_elem = ET.SubElement(axis, 'initial_position')
                ip_elem.text = str(INITIAL_JOINT_POSITIONS[jname])

    return ET.tostring(root, encoding='unicode', xml_declaration=True)


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

    # ── 2. Convert URDF → SDF with corrected initial joint positions ──
    sdf_xml = _urdf_to_sdf_with_initial_positions(urdf_xml)

    # ── 3. Environment ────────────────────────────────────────────────
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

    # ── 4. Robot State Publisher ──────────────────────────────────────
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='both',
        parameters=[robot_description, {'use_sim_time': True}],
    )

    # ── 5. Gz Sim ─────────────────────────────────────────────────────
    world_path = os.path.join(pkg, 'worlds', 'refuel_world.sdf')
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('ros_gz_sim'),
                         'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': ['-r ', world_path]}.items(),
    )

    # ── 6. Spawn robot from fixed SDF (not URDF topic) ───────────────
    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-string', sdf_xml,
            '-name', 'kr6_r700',
            '-z', '0.05',
        ],
        output='screen',
    )

    delayed_spawn = TimerAction(period=5.0, actions=[spawn_robot])

    # ── 7. Bridge Gz Sim → ROS2 ───────────────────────────────────────
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

    # ── 8. Controller spawners (chained after spawn) ──────────────────
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

    # After arm controller is active, immediately command candle position.
    # Gz Sim's <initial_position> is broken for URDF models, so this is
    # the only reliable way to reach candle before the user runs a mission.
    move_to_candle = ExecuteProcess(
        cmd=[
            'ros2', 'action', 'send_goal',
            '/kr6_arm_controller/follow_joint_trajectory',
            'control_msgs/action/FollowJointTrajectory',
            '{trajectory: {joint_names: [joint_1, joint_2, joint_3,'
            ' joint_4, joint_5, joint_6],'
            ' points: [{positions: [0.0, -1.5708, 0.0, 0.0, 0.0, 0.0],'
            ' time_from_start: {sec: 2, nanosec: 0}}]}}',
        ],
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
        RegisterEventHandler(OnProcessExit(
            target_action=load_arm,
            on_exit=[move_to_candle],
        )),
    ])

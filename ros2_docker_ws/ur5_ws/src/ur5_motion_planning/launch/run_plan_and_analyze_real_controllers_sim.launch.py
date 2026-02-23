# Launch UR5 with real ros2_control controllers + mock hardware (simulation, no Gazebo).
# Same execution path as hardware: controller_manager + scaled_joint_trajectory_controller.
# Then move_group + plan_and_analyze node. Pre-hardware step before GUI (Gazebo/CoppeliaSim/RViz).

import os
import subprocess
import tempfile
import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _build_driver_urdf(use_mock_hardware=True):
    """Build driver URDF with ros2_control (mock or real) so move_group gets same model."""
    ur_desc_share = get_package_share_directory("ur_description")
    ur_driver_share = get_package_share_directory("ur_robot_driver")
    ur_client_share = get_package_share_directory("ur_client_library")
    ur_type = "ur5"
    config_dir = os.path.join(ur_desc_share, "config", ur_type)
    # In binary installs, the main URDF xacro lives in ur_description
    xacro_path = os.path.join(ur_desc_share, "urdf", "ur.urdf.xacro")
    script_path = os.path.join(ur_client_share, "resources", "external_control.urscript")
    input_recipe = os.path.join(ur_driver_share, "resources", "rtde_input_recipe.txt")
    output_recipe = os.path.join(ur_driver_share, "resources", "rtde_output_recipe.txt")

    args = [
        "xacro",
        xacro_path,
        "robot_ip:=192.168.56.101",
        "joint_limit_params:=" + os.path.join(config_dir, "joint_limits.yaml"),
        "kinematics_params:=" + os.path.join(config_dir, "default_kinematics.yaml"),
        "physical_params:=" + os.path.join(config_dir, "physical_parameters.yaml"),
        "visual_params:=" + os.path.join(config_dir, "visual_parameters.yaml"),
        "safety_limits:=false",
        "safety_pos_margin:=0.15",
        "safety_k_position:=20",
        "name:=ur",  # match SRDF robot name
        "script_filename:=" + script_path,
        "input_recipe_filename:=" + input_recipe,
        "output_recipe_filename:=" + output_recipe,
        "tf_prefix:=",
        "use_mock_hardware:=" + ("true" if use_mock_hardware else "false"),
        "mock_sensor_commands:=false",
        "headless_mode:=true",
        "use_tool_communication:=false",
    ]
    result = subprocess.run(args, capture_output=True, text=True, env=os.environ)
    if result.returncode != 0:
        raise RuntimeError(f"xacro failed: {result.stderr}")
    return result.stdout


def _process_srdf_xacro(ur_moveit_share):
    srdf_path = os.path.join(ur_moveit_share, "srdf", "ur.srdf.xacro")
    with open(srdf_path) as f:
        content = f.read()
    content = content.replace("$(find ur_moveit_config)", ur_moveit_share)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xacro", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    try:
        import xacro
        doc = xacro.process_file(tmp_path, mappings={"name": "ur"})
        return doc.toxml()
    finally:
        os.unlink(tmp_path)


def generate_launch_description():
    use_real_controllers_sim = LaunchConfiguration("use_real_controllers_sim", default="true")
    ur_moveit_share = get_package_share_directory("ur_moveit_config")

    # Build driver URDF (with mock hardware) for move_group so it matches controller_manager's model
    robot_description = _build_driver_urdf(use_mock_hardware=True)
    robot_description_semantic = _process_srdf_xacro(ur_moveit_share)
    kinematics_yaml = _load_yaml(os.path.join(ur_moveit_share, "config", "kinematics.yaml"))
    joint_limits_yaml = _load_yaml(os.path.join(ur_moveit_share, "config", "joint_limits.yaml"))
    ompl_yaml = _load_yaml(os.path.join(ur_moveit_share, "config", "ompl_planning.yaml"))
    ompl_yaml = dict(ompl_yaml)
    ompl_yaml["ur_manipulator"] = {
        "enforce_constrained_state_space": True,
        "projection_evaluator": "joints(shoulder_pan_joint,shoulder_lift_joint)",
    }
    moveit_controllers_yaml = _load_yaml(
        os.path.join(ur_moveit_share, "config", "moveit_controllers.yaml")
    )
    moveit_controllers_without_manage = dict(moveit_controllers_yaml)
    moveit_controllers_without_manage["moveit_manage_controllers"] = False
    if "trajectory_execution" not in moveit_controllers_without_manage:
        moveit_controllers_without_manage["trajectory_execution"] = {}
    moveit_controllers_without_manage["trajectory_execution"]["allowed_goal_duration_margin"] = 30.0

    move_group_params = [
        {"robot_description": robot_description},
        {"robot_description_semantic": robot_description_semantic},
        kinematics_yaml,
        joint_limits_yaml,
        moveit_controllers_without_manage,
        {"planning_pipelines": {"ompl": ompl_yaml}},
        {"pipeline_names": ["ompl"]},
        {"default_planning_pipeline": "ompl"},
        {"publish_robot_description_semantic": True},
    ]
    robot_description_kinematics = {"robot_description_kinematics": kinematics_yaml}
    node_params = [
        {"robot_description": robot_description},
        {"robot_description_semantic": robot_description_semantic},
        robot_description_kinematics,
        kinematics_yaml,
    ]

    # UR driver stack: controller_manager + ur_rsp + scaled_joint_trajectory_controller (real controller, mock hardware)
    ur_control_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                os.path.join(
                    get_package_share_directory("ur_robot_driver"),
                    "launch",
                    "ur_control.launch.py",
                )
            ]
        ),
        launch_arguments={
            "ur_type": "ur5",
            "robot_ip": "192.168.56.101",
            "use_mock_hardware": "true",
            "mock_sensor_commands": "false",
            "initial_joint_controller": "scaled_joint_trajectory_controller",
            "activate_joint_controller": "true",
            "launch_rviz": "false",
        }.items(),
        condition=IfCondition(use_real_controllers_sim),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "use_real_controllers_sim",
            default_value="true",
            description="Use real ros2_control controllers with mock hardware (true) or legacy fake controller (false).",
        ),
        ur_control_launch,
        Node(
            package="moveit_ros_move_group",
            executable="move_group",
            output="screen",
            parameters=move_group_params,
            condition=IfCondition(use_real_controllers_sim),
        ),
        TimerAction(
            period=15.0,
            actions=[
                Node(
                    package="ur5_motion_planning",
                    executable="ur5_plan_and_analyze_node",
                    name="ur5_plan_and_analyze",
                    output="screen",
                    parameters=node_params,
                ),
            ],
            condition=IfCondition(use_real_controllers_sim),
        ),
    ])

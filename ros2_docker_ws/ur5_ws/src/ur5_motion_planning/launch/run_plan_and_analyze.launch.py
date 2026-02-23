# Launch move_group + plan-and-analyze node so planning has an action server (no "unknown goal response").
# Single launch: starts move_group (and deps) then the plan_and_analyze node after a short delay.

import os
import tempfile
import xacro
import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import TimerAction
from launch_ros.actions import Node


def _load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _process_urdf_xacro(ur_desc_share, ur_type="ur5"):
    xacro_path = os.path.join(ur_desc_share, "urdf", "ur.urdf.xacro")
    config_dir = os.path.join(ur_desc_share, "config", ur_type)
    with open(xacro_path) as f:
        content = f.read()
    content = content.replace("$(find ur_description)", ur_desc_share)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xacro", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    try:
        doc = xacro.process_file(
            tmp_path,
            mappings={
                "name": "ur",
                "ur_type": ur_type,
                "joint_limit_params": os.path.join(config_dir, "joint_limits.yaml"),
                "kinematics_params": os.path.join(config_dir, "default_kinematics.yaml"),
                "physical_params": os.path.join(config_dir, "physical_parameters.yaml"),
                "visual_params": os.path.join(config_dir, "visual_parameters.yaml"),
            },
        )
        return doc.toxml()
    finally:
        os.unlink(tmp_path)


def _process_srdf_xacro(ur_moveit_share):
    srdf_path = os.path.join(ur_moveit_share, "srdf", "ur.srdf.xacro")
    with open(srdf_path) as f:
        content = f.read()
    content = content.replace("$(find ur_moveit_config)", ur_moveit_share)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xacro", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    try:
        doc = xacro.process_file(tmp_path, mappings={"name": "ur"})
        return doc.toxml()
    finally:
        os.unlink(tmp_path)


def generate_launch_description():
    ur_desc_share = get_package_share_directory("ur_description")
    ur_moveit_share = get_package_share_directory("ur_moveit_config")

    robot_description = _process_urdf_xacro(ur_desc_share, ur_type="ur5")
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
    # Allow long execution so fake controller can finish before move_group times out
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

    return LaunchDescription([
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="world_to_base",
            arguments=["0", "0", "0", "0", "0", "0", "world", "base_link"],
        ),
        Node(
            package="ur5_minimal_moveit",
            executable="fake_joint_trajectory_controller",
            name="fake_joint_trajectory_controller",
            output="screen",
        ),
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            output="screen",
            parameters=[{"robot_description": robot_description}],
        ),
        Node(
            package="moveit_ros_move_group",
            executable="move_group",
            output="screen",
            parameters=move_group_params,
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
        ),
    ])

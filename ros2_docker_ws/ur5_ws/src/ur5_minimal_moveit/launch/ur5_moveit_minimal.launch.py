# Copyright 2024
# Minimal MoveIt 2 launch for UR5: robot_state_publisher + move_group only.
# No ros2_control, no hardware, no UR vendor driver. Planning-only (e.g. for IK-Geo).

import os
import tempfile
import xacro
import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def _load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _process_urdf_xacro(ur_desc_share, ur_type="ur5"):
    """Process ur.urdf.xacro with explicit paths so $(find ...) is not required."""
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
    """Process ur.srdf.xacro; replace $(find ur_moveit_config) for include resolution."""
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
    # Enable OMPL constrained state space for path constraints (e.g. workspace box)
    ompl_yaml = dict(ompl_yaml)
    ompl_yaml["ur_manipulator"] = {
        "enforce_constrained_state_space": True,
        "projection_evaluator": "joints(shoulder_pan_joint,shoulder_lift_joint)",
    }
    moveit_controllers_yaml = _load_yaml(
        os.path.join(ur_moveit_share, "config", "moveit_controllers.yaml")
    )

    move_group_params = [
        {"robot_description": robot_description},
        {"robot_description_semantic": robot_description_semantic},
        kinematics_yaml,
        joint_limits_yaml,
        moveit_controllers_yaml,
        {"planning_pipelines": {"ompl": ompl_yaml}},
        {"pipeline_names": ["ompl"]},
        {"default_planning_pipeline": "ompl"},
        {"publish_robot_description_semantic": True},
    ]

    return LaunchDescription([
        Node(
            package="joint_state_publisher",
            executable="joint_state_publisher",
            name="joint_state_publisher",
            output="screen",
            parameters=[
                {"robot_description": robot_description},
                {
                    "zeros.shoulder_pan_joint": 0.0,
                    "zeros.shoulder_lift_joint": -1.5707,
                    "zeros.elbow_joint": 0.0,
                    "zeros.wrist_1_joint": 0.0,
                    "zeros.wrist_2_joint": 0.0,
                    "zeros.wrist_3_joint": 0.0,
                },
            ],
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
    ])

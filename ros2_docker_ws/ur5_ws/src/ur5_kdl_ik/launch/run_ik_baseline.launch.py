# Launch UR5 KDL IK baseline node with robot_description and robot_description_semantic.
# Run while move_group is active, or standalone (node only needs the params).
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
    # MoveIt loader expects robot_description_kinematics for plugin loading
    robot_description_kinematics = {"robot_description_kinematics": kinematics_yaml}

    node_params = [
        {"robot_description": robot_description},
        {"robot_description_semantic": robot_description_semantic},
        robot_description_kinematics,
        kinematics_yaml,
    ]

    return LaunchDescription([
        Node(
            package="ur5_kdl_ik",
            executable="ur5_kdl_ik_baseline",
            name="ur5_kdl_ik_baseline",
            output="screen",
            parameters=node_params,
        ),
    ])

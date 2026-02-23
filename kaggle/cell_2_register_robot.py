#!/usr/bin/env python3
"""
Cell 2: Generate URDF + Register KUKA KR6 R700 in jrl
Run on Kaggle with: %run refuel-arm/kaggle/cell_2_register_robot.py
"""
import os
import numpy as np
import jrl
from jrl.robot import Robot

jrl_path = os.path.dirname(jrl.__file__)
urdf_dir = os.path.join(jrl_path, "urdfs", "kuka_kr6")
os.makedirs(urdf_dir, exist_ok=True)

URDF_CONTENT = """<?xml version="1.0" ?>
<robot name="kuka_kr6_r700">

  <link name="base_link">
    <inertial><mass value="7.5"/><origin xyz="0 0 0.1"/><inertia ixx="0.05" ixy="0" ixz="0" iyy="0.05" iyz="0" izz="0.05"/></inertial>
    <collision><origin xyz="0 0 0.1"/><geometry><cylinder radius="0.08" length="0.2"/></geometry></collision>
  </link>

  <link name="link_1">
    <inertial><mass value="5.0"/><origin xyz="0 0 0.1"/><inertia ixx="0.04" ixy="0" ixz="0" iyy="0.04" iyz="0" izz="0.04"/></inertial>
    <collision><origin xyz="0 0 0.05"/><geometry><cylinder radius="0.06" length="0.15"/></geometry></collision>
  </link>
  <joint name="joint_1" type="revolute">
    <origin rpy="3.141592653589793 0 0" xyz="0 0 0.400"/>
    <parent link="base_link"/><child link="link_1"/>
    <axis xyz="0 0 1"/>
    <limit effort="186" lower="-2.967059725" upper="2.967059725" velocity="6.283"/>
  </joint>

  <link name="link_2">
    <inertial><mass value="4.0"/><origin xyz="0.15 0 0"/><inertia ixx="0.03" ixy="0" ixz="0" iyy="0.03" iyz="0" izz="0.03"/></inertial>
    <collision><origin xyz="0.16 0 0"/><geometry><cylinder radius="0.05" length="0.315"/></geometry></collision>
  </link>
  <joint name="joint_2" type="revolute">
    <origin rpy="-1.5707963267948966 0 0" xyz="0.025 0 0"/>
    <parent link="link_1"/><child link="link_2"/>
    <axis xyz="0 0 1"/>
    <limit effort="169" lower="-3.316125575" upper="0.785398163" velocity="5.236"/>
  </joint>

  <link name="link_3">
    <inertial><mass value="3.0"/><origin xyz="0.15 0 0"/><inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.02"/></inertial>
    <collision><origin xyz="0.05 0 0"/><geometry><cylinder radius="0.045" length="0.1"/></geometry></collision>
  </link>
  <joint name="joint_3" type="revolute">
    <origin rpy="0 0 0" xyz="0.315 0 0"/>
    <parent link="link_2"/><child link="link_3"/>
    <axis xyz="0 0 1"/>
    <limit effort="82" lower="-2.094395100" upper="2.722713630" velocity="6.283"/>
  </joint>

  <link name="link_4">
    <inertial><mass value="2.5"/><origin xyz="0.1 0 0"/><inertia ixx="0.015" ixy="0" ixz="0" iyy="0.015" iyz="0" izz="0.015"/></inertial>
    <collision><origin xyz="0.18 0 0"/><geometry><cylinder radius="0.04" length="0.365"/></geometry></collision>
  </link>
  <joint name="joint_4" type="revolute">
    <origin rpy="0 1.5707963267948966 0" xyz="0 0 0"/>
    <parent link="link_3"/><child link="link_4"/>
    <axis xyz="0 0 1"/>
    <limit effort="49" lower="-3.228859113" upper="3.228859113" velocity="7.854"/>
  </joint>

  <link name="link_5">
    <inertial><mass value="1.5"/><origin xyz="0 0 0"/><inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial>
    <collision><origin xyz="0 0 0"/><geometry><cylinder radius="0.035" length="0.08"/></geometry></collision>
  </link>
  <joint name="joint_5" type="revolute">
    <origin rpy="-1.5707963267948966 0 0" xyz="0.365 0 0"/>
    <parent link="link_4"/><child link="link_5"/>
    <axis xyz="0 0 1"/>
    <limit effort="40" lower="-2.094395100" upper="2.094395100" velocity="7.854"/>
  </joint>

  <link name="link_6">
    <inertial><mass value="0.5"/><origin xyz="0 0 0"/><inertia ixx="0.005" ixy="0" ixz="0" iyy="0.005" iyz="0" izz="0.005"/></inertial>
    <collision><origin xyz="0 0 0"/><geometry><cylinder radius="0.03" length="0.06"/></geometry></collision>
  </link>
  <joint name="joint_6" type="revolute">
    <origin rpy="1.5707963267948966 0 0" xyz="0 0 0"/>
    <parent link="link_5"/><child link="link_6"/>
    <axis xyz="0 0 1"/>
    <limit effort="27" lower="-6.108652375" upper="6.108652375" velocity="9.425"/>
  </joint>

  <link name="flange">
    <collision><origin xyz="0.04 0 0"/><geometry><cylinder radius="0.025" length="0.08"/></geometry></collision>
  </link>
  <joint name="link6-flange" type="fixed">
    <origin xyz="0.080 0 0"/>
    <parent link="link_6"/><child link="flange"/>
  </joint>

</robot>
"""

urdf_path = os.path.join(urdf_dir, "kr6_r700.urdf")
with open(urdf_path, "w") as f:
    f.write(URDF_CONTENT)
print(f"✅ URDF written to: {urdf_path}")


# ── Robot class ──────────────────────────────────────────────────
class KukaKR6(Robot):
    name = "kuka_kr6"
    formal_robot_name = "KUKA KR6 R700"
    POSITIONAL_REPEATABILITY_MM = 0.03
    ROTATIONAL_REPEATABILITY_DEG = 0.1

    def __init__(self, verbose=False):
        Robot.__init__(
            self,
            KukaKR6.name,
            os.path.join(urdf_dir, "kr6_r700.urdf"),
            ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"],
            "base_link",
            "flange",
            [   # ignored collision pairs (non-adjacent links that can't collide)
                ("base_link", "link_2"), ("base_link", "link_3"),
                ("link_1", "link_3"), ("link_1", "link_4"),
                ("link_2", "link_4"), ("link_2", "link_5"),
                ("link_3", "link_5"), ("link_3", "link_6"),
                ("link_4", "link_6"),
            ],
            None,  # collision_capsules_by_link — not needed for training
            batch_fk_enabled=True,
            verbose=verbose,
        )


# ── Test it ──────────────────────────────────────────────────────
robot = KukaKR6(verbose=True)

print(f"\n✅ Robot registered: {robot.name}, DOF: {robot.ndof}")
print(f"   Joints: {robot.actuated_joint_names}")
print(f"   Joint limits:")
for i, (lo, hi) in enumerate(robot.actuated_joints_limits):
    print(f"     J{i+1}: [{lo:.3f}, {hi:.3f}] rad  ({np.degrees(lo):.0f}° to {np.degrees(hi):.0f}°)")

# Quick FK sanity check
import torch
q_zero = torch.zeros(1, 6)
pose_home = robot.forward_kinematics(q_zero)
print(f"\n   FK at HOME (all zeros): {pose_home[0,:3].numpy().round(4)}")
print(f"   ✅ Robot is ready for dataset generation!")

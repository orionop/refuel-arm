#!/usr/bin/env python3
"""
IKFlow Training Notebook for KUKA KR6 R700
============================================

Run this on Kaggle with T4 x2 GPUs.

Steps:
  1. Install dependencies (jrl, ikflow)
  2. Convert KUKA xacro to standalone URDF
  3. Register KR6 R700 in jrl
  4. Generate IK training dataset (25M samples)
  5. Train the normalizing flow model
  6. Download the trained weights
"""

# ╔════════════════════════════════════════════════════════════════╗
# ║  CELL 1: Install Dependencies                                 ║
# ║  Run this cell FIRST, then restart the kernel if prompted.     ║
# ╚════════════════════════════════════════════════════════════════╝

import subprocess
import sys

def run(cmd):
    print(f"➤ {cmd}")
    subprocess.check_call(cmd, shell=True)

# Kaggle already has torch, numpy, pandas, matplotlib pre-installed.
# Only install the missing deps that ikflow/jrl need.
run(f"{sys.executable} -m pip install -q pytorch-lightning freia wandb tqdm tabulate")

# Clone the repo
run("git clone https://github.com/orionop/refuel-arm.git || true")

# Install jrl first (ikflow's pyproject.toml pins Python <3.12, so we install jrl directly)
run(f"{sys.executable} -m pip install -q git+https://github.com/jstmn/Jrl.git@2ba7c3995b36b32886a8aa021a00c73b2cd55b2c")

# Install ikflow without deps (to avoid the python version conflict), then patch manually
run(f"{sys.executable} -m pip install -q --no-deps -e refuel-arm/ikflow/")

print("\n✅ All dependencies installed!")


# ╔════════════════════════════════════════════════════════════════╗
# ║  CELL 2: Convert KUKA Xacro to Standalone URDF                ║
# ╚════════════════════════════════════════════════════════════════╝

# The KUKA URDF is in xacro format. We need a standalone .urdf for jrl.
# If xacro is not available, we generate a minimal URDF directly from
# the known kinematic parameters.

import os
import numpy as np

# Create the URDF directory in jrl's package
import jrl
jrl_path = os.path.dirname(jrl.__file__)
urdf_dir = os.path.join(jrl_path, "urdfs", "kuka_kr6")
os.makedirs(urdf_dir, exist_ok=True)

# Generate a standalone URDF for the KR6 R700 from known parameters
# Joint limits from: kuka_robot_descriptions/kuka_agilus_support/urdf/kr6_r700_2_macro.xacro
URDF_CONTENT = """<?xml version="1.0" ?>
<robot name="kuka_kr6_r700" xmlns:xacro="http://www.ros.org/wiki/xacro">

  <!-- Base -->
  <link name="base_link">
    <inertial>
      <mass value="7.5"/>
      <origin rpy="0 0 0" xyz="0 0 0.1"/>
      <inertia ixx="0.05" ixy="0" ixz="0" iyy="0.05" iyz="0" izz="0.05"/>
    </inertial>
  </link>

  <!-- Link 1 -->
  <link name="link_1">
    <inertial>
      <mass value="5.0"/>
      <origin rpy="0 0 0" xyz="0 0 0.1"/>
      <inertia ixx="0.04" ixy="0" ixz="0" iyy="0.04" iyz="0" izz="0.04"/>
    </inertial>
  </link>

  <joint name="joint_1" type="revolute">
    <origin rpy="3.141592653589793 0 0" xyz="0 0 0.400"/>
    <parent link="base_link"/>
    <child link="link_1"/>
    <axis xyz="0 0 1"/>
    <limit effort="186.0" lower="-2.967059725" upper="2.967059725" velocity="6.283185307179586"/>
  </joint>

  <!-- Link 2 -->
  <link name="link_2">
    <inertial>
      <mass value="4.0"/>
      <origin rpy="0 0 0" xyz="0.15 0 0"/>
      <inertia ixx="0.03" ixy="0" ixz="0" iyy="0.03" iyz="0" izz="0.03"/>
    </inertial>
  </link>

  <joint name="joint_2" type="revolute">
    <origin rpy="-1.5707963267948966 0 0" xyz="0.025 0 0"/>
    <parent link="link_1"/>
    <child link="link_2"/>
    <axis xyz="0 0 1"/>
    <limit effort="168.9" lower="-3.316125575" upper="0.785398163" velocity="5.235987755982989"/>
  </joint>

  <!-- Link 3 -->
  <link name="link_3">
    <inertial>
      <mass value="3.0"/>
      <origin rpy="0 0 0" xyz="0.15 0 0"/>
      <inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.02"/>
    </inertial>
  </link>

  <joint name="joint_3" type="revolute">
    <origin rpy="0 0 0" xyz="0.315 0 0"/>
    <parent link="link_2"/>
    <child link="link_3"/>
    <axis xyz="0 0 1"/>
    <limit effort="81.5" lower="-2.094395100" upper="2.722713630" velocity="6.283185307179586"/>
  </joint>

  <!-- Link 4 -->
  <link name="link_4">
    <inertial>
      <mass value="2.5"/>
      <origin rpy="0 0 0" xyz="0.1 0 0"/>
      <inertia ixx="0.015" ixy="0" ixz="0" iyy="0.015" iyz="0" izz="0.015"/>
    </inertial>
  </link>

  <joint name="joint_4" type="revolute">
    <origin rpy="0 1.5707963267948966 0" xyz="0 0 0"/>
    <parent link="link_3"/>
    <child link="link_4"/>
    <axis xyz="0 0 1"/>
    <limit effort="48.9" lower="-3.228859113" upper="3.228859113" velocity="7.853981633974483"/>
  </joint>

  <!-- Link 5 -->
  <link name="link_5">
    <inertial>
      <mass value="1.5"/>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="joint_5" type="revolute">
    <origin rpy="-1.5707963267948966 0 0" xyz="0.365 0 0"/>
    <parent link="link_4"/>
    <child link="link_5"/>
    <axis xyz="0 0 1"/>
    <limit effort="40.1" lower="-2.094395100" upper="2.094395100" velocity="7.853981633974483"/>
  </joint>

  <!-- Link 6 -->
  <link name="link_6">
    <inertial>
      <mass value="0.5"/>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <inertia ixx="0.005" ixy="0" ixz="0" iyy="0.005" iyz="0" izz="0.005"/>
    </inertial>
  </link>

  <joint name="joint_6" type="revolute">
    <origin rpy="1.5707963267948966 0 0" xyz="0 0 0"/>
    <parent link="link_5"/>
    <child link="link_6"/>
    <axis xyz="0 0 1"/>
    <limit effort="27.4" lower="-6.108652375" upper="6.108652375" velocity="9.42477796076938"/>
  </joint>

  <!-- Flange (tool mounting point) -->
  <link name="flange"/>

  <joint name="link6-flange" type="fixed">
    <origin rpy="0 0 0" xyz="0.080 0 0"/>
    <parent link="link_6"/>
    <child link="flange"/>
  </joint>

</robot>
"""

urdf_path = os.path.join(urdf_dir, "kr6_r700.urdf")
with open(urdf_path, "w") as f:
    f.write(URDF_CONTENT)
print(f"✅ URDF written to: {urdf_path}")


# ╔════════════════════════════════════════════════════════════════╗
# ║  CELL 3: Register KR6 R700 in jrl                             ║
# ╚════════════════════════════════════════════════════════════════╝

# Monkey-patch jrl to add the KR6 R700 robot
from jrl.robot import Robot
from jrl.utils import get_filepath

class KukaKR6(Robot):
    """KUKA KR6 R700-2 — 6-DOF industrial manipulator (Agilus family)"""
    name = "kuka_kr6"
    formal_robot_name = "KUKA KR6 R700"
    POSITIONAL_REPEATABILITY_MM = 0.03  # From KUKA datasheet
    ROTATIONAL_REPEATABILITY_DEG = 0.1

    def __init__(self, verbose=False):
        active_joints = [
            "joint_1",  # (-2.967, 2.967)
            "joint_2",  # (-3.316, 0.785)
            "joint_3",  # (-2.094, 2.723)
            "joint_4",  # (-3.229, 3.229)
            "joint_5",  # (-2.094, 2.094)
            "joint_6",  # (-6.109, 6.109)
        ]

        urdf_filepath = os.path.join(urdf_dir, "kr6_r700.urdf")
        base_link = "base_link"
        end_effector_link_name = "flange"

        # Collision pairs — conservative: skip adjacent links that can't collide
        ignored_collision_pairs = [
            ("base_link", "link_2"),
            ("base_link", "link_3"),
            ("link_1", "link_3"),
            ("link_1", "link_4"),
            ("link_2", "link_4"),
            ("link_2", "link_5"),
            ("link_3", "link_5"),
            ("link_3", "link_6"),
            ("link_4", "link_6"),
        ]

        # No collision capsules for now — train without self-collision filtering
        # To enable: generate capsules using jrl's capsule fitting tool
        Robot.__init__(
            self,
            KukaKR6.name,
            urdf_filepath,
            active_joints,
            base_link,
            end_effector_link_name,
            ignored_collision_pairs,
            batch_fk_enabled=True,
            verbose=verbose,
        )

# Register the robot in jrl's module
import jrl.robots as jrl_robots
jrl_robots.KukaKR6 = KukaKR6

# Also patch the ROBOT_DICT that maps names to classes
if hasattr(jrl_robots, 'get_robot'):
    _orig_get_robot = jrl_robots.get_robot
    def patched_get_robot(name):
        if name == "kuka_kr6":
            return KukaKR6()
        return _orig_get_robot(name)
    jrl_robots.get_robot = patched_get_robot

# Test the registration
robot = KukaKR6(verbose=True)
print(f"\n✅ Robot registered: {robot.name}")
print(f"   DOF: {robot.ndof}")
print(f"   Active joints: {robot.actuated_joint_names}")
print(f"   Joint limits:")
for i, (lo, hi) in enumerate(robot.actuated_joints_limits):
    print(f"     J{i+1}: [{lo:.3f}, {hi:.3f}] rad  ({np.degrees(lo):.1f}° to {np.degrees(hi):.1f}°)")


# ╔════════════════════════════════════════════════════════════════╗
# ║  CELL 4: Generate IK Training Dataset                         ║
# ╚════════════════════════════════════════════════════════════════╝

# The dataset consists of (joint_config, cartesian_pose) pairs.
# IKFlow learns to map cartesian_pose → joint_config.

import torch
from tqdm import tqdm

DATASET_SIZE = 10_000_000  # 10M for faster training, increase to 25M for production
BATCH_SIZE = 10000

print(f"\n📊 Generating {DATASET_SIZE:,} FK samples for IKFlow training...")

all_configs = []
all_poses = []

for batch_start in tqdm(range(0, DATASET_SIZE, BATCH_SIZE)):
    batch_sz = min(BATCH_SIZE, DATASET_SIZE - batch_start)

    # Random joint configs within limits
    limits = torch.tensor(robot.actuated_joints_limits)
    q_rand = torch.rand(batch_sz, 6) * (limits[:, 1] - limits[:, 0]) + limits[:, 0]

    # Forward kinematics → (position, quaternion) for each config
    poses = robot.forward_kinematics(q_rand)
    # poses shape: (batch_sz, 7) — [x, y, z, qw, qx, qy, qz]

    all_configs.append(q_rand)
    all_poses.append(poses)

configs = torch.cat(all_configs, dim=0)
poses = torch.cat(all_poses, dim=0)

print(f"✅ Dataset generated: {configs.shape[0]:,} samples")
print(f"   Config shape: {configs.shape}")
print(f"   Pose shape:   {poses.shape}")

# Save dataset
dataset_dir = "refuel-arm/ikflow_dataset"
os.makedirs(dataset_dir, exist_ok=True)
torch.save({"configs": configs, "poses": poses}, f"{dataset_dir}/kuka_kr6_dataset.pt")
print(f"   Saved to: {dataset_dir}/kuka_kr6_dataset.pt")


# ╔════════════════════════════════════════════════════════════════╗
# ║  CELL 5: Train IKFlow                                         ║
# ╚════════════════════════════════════════════════════════════════╝

from ikflow.model import IkflowModelParameters, glow_cNF_model
from ikflow.ikflow_solver import IKFlowSolver
from ikflow.training.lt_model import IkfLitModel
import pytorch_lightning as pl

# Model hyperparameters (matching the Panda config that works well)
model_params = IkflowModelParameters(
    nb_nodes=12,               # 12 coupling layers
    dim_latent_space=6,         # 6 DOF = 6-dim latent space
    coeff_fn_config=3,          # Coefficient function type
    coeff_fn_internal_size=1024, # Hidden layer size
    rnvp_clamp=2.5,             # RealNVP clamping
)

# Create solver
solver = IKFlowSolver(model_params, robot)
print(f"\n🧠 IKFlow Model:")
print(f"   Parameters: {sum(p.numel() for p in solver.nn_model.parameters()):,}")
print(f"   Architecture: {model_params.nb_nodes} coupling layers, {model_params.coeff_fn_internal_size} hidden")

# Create Lightning training module
lit_model = IkfLitModel(
    model_params,
    robot,
    learning_rate=5e-4,
    gamma=0.975,
    softflow_noise_scale=0.001,
    dataset_tags=["kuka_kr6_custom"],
)

# Load our custom dataset
lit_model.training_data = configs
lit_model.training_poses = poses

# Train
trainer = pl.Trainer(
    max_epochs=50,
    accelerator="gpu",
    devices=1,            # Use 1 GPU (T4)
    precision="16-mixed",  # Mixed precision for T4
    log_every_n_steps=100,
    val_check_interval=0.25,
)

print("\n🚀 Starting training...")
trainer.fit(lit_model)

# Save the trained model
model_save_path = f"{dataset_dir}/kuka_kr6_ikflow_trained.pt"
torch.save(solver.nn_model.state_dict(), model_save_path)
print(f"\n✅ Model saved to: {model_save_path}")
print("   Download this file and place it in your refuel-arm repo!")


# ╔════════════════════════════════════════════════════════════════╗
# ║  CELL 6: Quick Validation                                     ║
# ╚════════════════════════════════════════════════════════════════╝

# Test the trained model
print("\n🔍 Validating trained IKFlow model...")

# Generate 1000 random test poses
test_configs = torch.rand(1000, 6) * (limits[:, 1] - limits[:, 0]) + limits[:, 0]
test_poses = robot.forward_kinematics(test_configs)

# Solve IK using IKFlow
with torch.no_grad():
    ik_solutions = solver.generate_ik_solutions(
        test_poses,
        latent=torch.randn(1000, 6) * 0.5,
        clamp_to_joint_limits=True,
    )

# Check accuracy: FK of IK solutions vs target poses
ik_poses = robot.forward_kinematics(ik_solutions)
pos_errors = torch.norm(ik_poses[:, :3] - test_poses[:, :3], dim=1)

print(f"   Mean position error: {pos_errors.mean():.4f} m")
print(f"   Max position error:  {pos_errors.max():.4f} m")
print(f"   Median position error: {pos_errors.median():.4f} m")
print(f"   % within 1cm: {(pos_errors < 0.01).float().mean() * 100:.1f}%")
print(f"   % within 1mm: {(pos_errors < 0.001).float().mean() * 100:.1f}%")

print("\n✅ Training complete! Download the model weights file.")
print("   Next: Register in CppFlow → Run trajectory planning → Gazebo!")

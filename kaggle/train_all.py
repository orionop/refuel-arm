#!/usr/bin/env python3
"""
KUKA KR6 R700 — IKFlow Training (All-in-One)
=============================================
Run on Kaggle after Cell 1 installs:
    %run refuel-arm/kaggle/train_all.py
"""
import os
import numpy as np
import torch
from tqdm import tqdm

# Patch StepLR to remove deprecated 'verbose' kwarg (removed in PyTorch 2.4)
_OrigStepLR = torch.optim.lr_scheduler.StepLR
class _PatchedStepLR(_OrigStepLR):
    def __init__(self, *args, **kwargs):
        kwargs.pop("verbose", None)
        super().__init__(*args, **kwargs)
torch.optim.lr_scheduler.StepLR = _PatchedStepLR

# ═══════════════════════════════════════════════════════════════════
# PART 1: Generate URDF + Register Robot
# ═══════════════════════════════════════════════════════════════════

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
  <joint name="joint_1" type="revolute"><origin rpy="3.141592653589793 0 0" xyz="0 0 0.400"/><parent link="base_link"/><child link="link_1"/><axis xyz="0 0 1"/><limit effort="186" lower="-2.967059725" upper="2.967059725" velocity="6.283"/></joint>
  <link name="link_2">
    <inertial><mass value="4.0"/><origin xyz="0.15 0 0"/><inertia ixx="0.03" ixy="0" ixz="0" iyy="0.03" iyz="0" izz="0.03"/></inertial>
    <collision><origin xyz="0.16 0 0"/><geometry><cylinder radius="0.05" length="0.315"/></geometry></collision>
  </link>
  <joint name="joint_2" type="revolute"><origin rpy="-1.5707963267948966 0 0" xyz="0.025 0 0"/><parent link="link_1"/><child link="link_2"/><axis xyz="0 0 1"/><limit effort="169" lower="-3.316125575" upper="0.785398163" velocity="5.236"/></joint>
  <link name="link_3">
    <inertial><mass value="3.0"/><origin xyz="0.15 0 0"/><inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.02"/></inertial>
    <collision><origin xyz="0.05 0 0"/><geometry><cylinder radius="0.045" length="0.1"/></geometry></collision>
  </link>
  <joint name="joint_3" type="revolute"><origin rpy="0 0 0" xyz="0.315 0 0"/><parent link="link_2"/><child link="link_3"/><axis xyz="0 0 1"/><limit effort="82" lower="-2.094395100" upper="2.722713630" velocity="6.283"/></joint>
  <link name="link_4">
    <inertial><mass value="2.5"/><origin xyz="0.1 0 0"/><inertia ixx="0.015" ixy="0" ixz="0" iyy="0.015" iyz="0" izz="0.015"/></inertial>
    <collision><origin xyz="0.18 0 0"/><geometry><cylinder radius="0.04" length="0.365"/></geometry></collision>
  </link>
  <joint name="joint_4" type="revolute"><origin rpy="0 1.5707963267948966 0" xyz="0 0 0"/><parent link="link_3"/><child link="link_4"/><axis xyz="0 0 1"/><limit effort="49" lower="-3.228859113" upper="3.228859113" velocity="7.854"/></joint>
  <link name="link_5">
    <inertial><mass value="1.5"/><origin xyz="0 0 0"/><inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial>
    <collision><origin xyz="0 0 0"/><geometry><cylinder radius="0.035" length="0.08"/></geometry></collision>
  </link>
  <joint name="joint_5" type="revolute"><origin rpy="-1.5707963267948966 0 0" xyz="0.365 0 0"/><parent link="link_4"/><child link="link_5"/><axis xyz="0 0 1"/><limit effort="40" lower="-2.094395100" upper="2.094395100" velocity="7.854"/></joint>
  <link name="link_6">
    <inertial><mass value="0.5"/><origin xyz="0 0 0"/><inertia ixx="0.005" ixy="0" ixz="0" iyy="0.005" iyz="0" izz="0.005"/></inertial>
    <collision><origin xyz="0 0 0"/><geometry><cylinder radius="0.03" length="0.06"/></geometry></collision>
  </link>
  <joint name="joint_6" type="revolute"><origin rpy="1.5707963267948966 0 0" xyz="0 0 0"/><parent link="link_5"/><child link="link_6"/><axis xyz="0 0 1"/><limit effort="27" lower="-6.108652375" upper="6.108652375" velocity="9.425"/></joint>
  <link name="flange">
    <collision><origin xyz="0.04 0 0"/><geometry><cylinder radius="0.025" length="0.08"/></geometry></collision>
  </link>
  <joint name="link6-flange" type="fixed"><origin xyz="0.080 0 0"/><parent link="link_6"/><child link="flange"/></joint>
</robot>
"""

with open(os.path.join(urdf_dir, "kr6_r700.urdf"), "w") as f:
    f.write(URDF_CONTENT)

class KukaKR6(Robot):
    name = "kuka_kr6"
    formal_robot_name = "KUKA KR6 R700"
    POSITIONAL_REPEATABILITY_MM = 0.03
    ROTATIONAL_REPEATABILITY_DEG = 0.1
    def __init__(self, verbose=False):
        Robot.__init__(self, KukaKR6.name,
            os.path.join(urdf_dir, "kr6_r700.urdf"),
            ["joint_1","joint_2","joint_3","joint_4","joint_5","joint_6"],
            "base_link", "flange",
            [("base_link","link_2"),("base_link","link_3"),("link_1","link_3"),
             ("link_1","link_4"),("link_2","link_4"),("link_2","link_5"),
             ("link_3","link_5"),("link_3","link_6"),("link_4","link_6")],
            None, batch_fk_enabled=True, verbose=verbose)

robot = KukaKR6(verbose=True)
print(f"\n✅ Robot: {robot.name}, DOF: {robot.ndof}")
for i,(lo,hi) in enumerate(robot.actuated_joints_limits):
    print(f"  J{i+1}: [{lo:.3f}, {hi:.3f}] rad  ({np.degrees(lo):.0f}° to {np.degrees(hi):.0f}°)")

q_zero = torch.zeros(1, 6)
pose_home = robot.forward_kinematics(q_zero)
print(f"\n  FK at HOME: {pose_home[0,:3].cpu().numpy().round(4)}")


# ═══════════════════════════════════════════════════════════════════
# PART 2: Generate FK Dataset
# ═══════════════════════════════════════════════════════════════════

DATASET_SIZE = 5_000_000
BATCH_SIZE = 50_000
SAVE_DIR = "/kaggle/working/ikflow_dataset"
os.makedirs(SAVE_DIR, exist_ok=True)

print(f"\n{'='*60}")
print(f"📊 Generating {DATASET_SIZE:,} FK samples...")

limits = torch.tensor(robot.actuated_joints_limits)
lo = limits[:, 0]
hi = limits[:, 1]

all_configs = []
all_poses = []

for batch_start in tqdm(range(0, DATASET_SIZE, BATCH_SIZE), desc="Generating"):
    batch_sz = min(BATCH_SIZE, DATASET_SIZE - batch_start)
    q_rand = torch.rand(batch_sz, 6) * (hi - lo) + lo
    poses = robot.forward_kinematics(q_rand)
    all_configs.append(q_rand.cpu())
    all_poses.append(poses.cpu())

configs = torch.cat(all_configs, dim=0)
poses = torch.cat(all_poses, dim=0)

save_path = os.path.join(SAVE_DIR, "kuka_kr6_dataset.pt")
torch.save({"configs": configs, "poses": poses}, save_path)

print(f"✅ Dataset: {configs.shape[0]:,} samples")
print(f"   Configs: {configs.shape}, Poses: {poses.shape}")
print(f"   Saved: {save_path} ({os.path.getsize(save_path)/1e6:.1f} MB)")

pos = poses[:, :3]
print(f"   Workspace X: [{pos[:,0].min():.3f}, {pos[:,0].max():.3f}]")
print(f"   Workspace Y: [{pos[:,1].min():.3f}, {pos[:,1].max():.3f}]")
print(f"   Workspace Z: [{pos[:,2].min():.3f}, {pos[:,2].max():.3f}]")


# ═══════════════════════════════════════════════════════════════════
# PART 3: Train IKFlow (using official IkfLitModel + PyTorch Lightning)
# ═══════════════════════════════════════════════════════════════════

from ikflow.model import IkflowModelParameters
from ikflow.ikflow_solver import IKFlowSolver
from ikflow.training.lt_model import IkfLitModel
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl

# Use default model params (same as official IKFlow)
model_params = IkflowModelParameters()
model_params.nb_nodes = 12
model_params.dim_latent_space = 6   # 6 DOF robot
model_params.coeff_fn_config = 3
model_params.coeff_fn_internal_size = 1024
model_params.rnvp_clamp = 2.5

# Create solver + model
solver = IKFlowSolver(model_params, robot)
n_params = sum(p.numel() for p in solver.nn_model.parameters())

print(f"\n{'='*60}")
print(f"🧠 IKFlow: {n_params:,} params, {model_params.nb_nodes} layers")
print(f"   dim_latent_space: {model_params.dim_latent_space}")
print(f"   softflow_enabled: {model_params.softflow_enabled}")

# Create the Lightning training module (handles loss, optimizer, scheduler internally)
BATCH_SIZE_TRAIN = 512
N_EPOCHS = 20

lit_model = IkfLitModel(
    ik_solver=solver,
    base_hparams=model_params,
    learning_rate=1e-4,
    checkpoint_every=100000,
    gamma=0.9794578299341784,
    log_every=int(1e10),  # disable wandb logging
    gradient_clip=1.0,
    optimizer_name="adamw",
    weight_decay=1.8e-05,
)

# Create DataLoader from our generated dataset
# IkfLitModel expects batch = (joint_configs, ee_poses)
from jrl.config import DEVICE as JRL_DEVICE
train_dataset = TensorDataset(configs.to(JRL_DEVICE), poses.to(JRL_DEVICE))
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE_TRAIN,
    shuffle=True,
    drop_last=True,
    generator=torch.Generator(device=JRL_DEVICE),
)

# Validation set (small subset)
val_configs = configs[:500].to(JRL_DEVICE)
val_poses = poses[:500].to(JRL_DEVICE)
val_dataset = TensorDataset(val_configs, val_poses)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, drop_last=True)

print(f"   Training batches/epoch: {len(train_loader):,}")
print(f"   Estimated time: {len(train_loader) * N_EPOCHS * 0.007 / 60:.0f}-{len(train_loader) * N_EPOCHS * 0.015 / 60:.0f} minutes")
print(f"\n🚀 Training on: {JRL_DEVICE}")

# Train using PyTorch Lightning (same as official IKFlow train.py)
SAVE_DIR = "/kaggle/working/ikflow_dataset"
trainer = pl.Trainer(
    max_epochs=N_EPOCHS,
    devices=[0],
    accelerator="gpu",
    log_every_n_steps=5000,
    val_check_interval=min(20000, len(train_loader)),
    enable_progress_bar=True,
)

trainer.fit(lit_model, train_loader, val_loader)

# Save the trained model weights
import pickle
best_path = os.path.join(SAVE_DIR, "kuka_kr6_ikflow_best.pt")
with open(best_path, "wb") as f:
    pickle.dump(solver.nn_model.state_dict(), f)
print(f"\n✅ Model saved to: {best_path}")


# ═══════════════════════════════════════════════════════════════════
# PART 4: Validate
# ═══════════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print("🔍 Validating trained model...")
solver.nn_model.eval()
solver._model_weights_loaded = True  # We trained it ourselves, not loaded from file

test_q = torch.rand(1000, 6, device="cpu") * (hi - lo) + lo
test_poses = robot.forward_kinematics(test_q)

with torch.no_grad():
    ik_solutions = solver.generate_ik_solutions(
        test_poses,
        latent=torch.randn(1000, model_params.dim_latent_space, device=JRL_DEVICE) * 0.5,
        clamp_to_joint_limits=True,
    )

ik_poses = robot.forward_kinematics(ik_solutions.cpu())
pos_errors = torch.norm(ik_poses[:, :3].cpu() - test_poses[:, :3].cpu(), dim=1)

print(f"   Mean error:   {pos_errors.mean():.4f} m")
print(f"   Median error: {pos_errors.median():.4f} m")
print(f"   % < 1cm:      {(pos_errors < 0.01).float().mean()*100:.1f}%")
print(f"   % < 1mm:      {(pos_errors < 0.001).float().mean()*100:.1f}%")

print(f"\n✅ Done! Download weights from: {SAVE_DIR}/kuka_kr6_ikflow_best.pt")

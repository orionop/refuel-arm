#!/usr/bin/env python3
"""
Cell 3: Generate FK dataset for IKFlow training
Run on Kaggle with: %run refuel-arm/kaggle/cell_3_generate_dataset.py

Requires: cell_2_register_robot.py to have been run first.
"""
import os
import torch
import numpy as np
from tqdm import tqdm

# The robot object should already exist from cell_2
try:
    _ = robot.name
    print(f"Using robot: {robot.name}")
except NameError:
    print("❌ Run cell_2_register_robot.py first!")
    raise

# ── Config ───────────────────────────────────────────────────────
DATASET_SIZE = 10_000_000   # 10M samples (increase to 25M for production)
BATCH_SIZE = 50_000
SAVE_DIR = "/kaggle/working/ikflow_dataset"
os.makedirs(SAVE_DIR, exist_ok=True)

# ── Generate ─────────────────────────────────────────────────────
print(f"\n📊 Generating {DATASET_SIZE:,} FK samples...")
print(f"   Batch size: {BATCH_SIZE:,}")

limits = torch.tensor(robot.actuated_joints_limits)  # (6, 2)
lo = limits[:, 0]  # (6,)
hi = limits[:, 1]  # (6,)

all_configs = []
all_poses = []

for batch_start in tqdm(range(0, DATASET_SIZE, BATCH_SIZE), desc="Generating"):
    batch_sz = min(BATCH_SIZE, DATASET_SIZE - batch_start)

    # Random joint configs uniformly within limits
    q_rand = torch.rand(batch_sz, 6) * (hi - lo) + lo

    # Forward kinematics → 7D pose (x, y, z, qw, qx, qy, qz)
    poses = robot.forward_kinematics(q_rand)

    all_configs.append(q_rand)
    all_poses.append(poses)

configs = torch.cat(all_configs, dim=0)
poses = torch.cat(all_poses, dim=0)

# ── Save ─────────────────────────────────────────────────────────
save_path = os.path.join(SAVE_DIR, "kuka_kr6_dataset.pt")
torch.save({"configs": configs, "poses": poses}, save_path)

print(f"\n✅ Dataset generated and saved!")
print(f"   Configs: {configs.shape}  (joint angles)")
print(f"   Poses:   {poses.shape}  (x,y,z,qw,qx,qy,qz)")
print(f"   File:    {save_path}")
print(f"   Size:    {os.path.getsize(save_path) / 1e6:.1f} MB")

# Workspace stats
pos = poses[:, :3]
print(f"\n   Workspace bounds:")
print(f"     X: [{pos[:,0].min():.3f}, {pos[:,0].max():.3f}] m")
print(f"     Y: [{pos[:,1].min():.3f}, {pos[:,1].max():.3f}] m")
print(f"     Z: [{pos[:,2].min():.3f}, {pos[:,2].max():.3f}] m")

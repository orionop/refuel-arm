#!/usr/bin/env python3
"""
Cell 4: Train IKFlow normalizing flow model
Run on Kaggle with: %run refuel-arm/kaggle/cell_4_train.py

Requires: cell_2 + cell_3 to have been run first.
"""
import os
import torch
import numpy as np

from ikflow.model import IkflowModelParameters
from ikflow.ikflow_solver import IKFlowSolver

# ── Check prerequisites ──────────────────────────────────────────
try:
    _ = robot.name
except NameError:
    print("❌ Run cell_2_register_robot.py first!")
    raise

DATASET_PATH = "/kaggle/working/ikflow_dataset/kuka_kr6_dataset.pt"
if not os.path.exists(DATASET_PATH):
    print("❌ Run cell_3_generate_dataset.py first!")
    raise FileNotFoundError(DATASET_PATH)

# ── Load dataset ─────────────────────────────────────────────────
print("📂 Loading dataset...")
data = torch.load(DATASET_PATH)
configs = data["configs"]
poses = data["poses"]
print(f"   Loaded {configs.shape[0]:,} samples")

# ── Create model ─────────────────────────────────────────────────
model_params = IkflowModelParameters(
    nb_nodes=12,                # 12 coupling layers (matches Panda config)
    dim_latent_space=6,         # 6 DOF robot → 6D latent
    coeff_fn_config=3,          # Coefficient function type
    coeff_fn_internal_size=1024, # Hidden layer width
    rnvp_clamp=2.5,             # RealNVP clamping value
)

solver = IKFlowSolver(model_params, robot)
nn_model = solver.nn_model

n_params = sum(p.numel() for p in nn_model.parameters())
print(f"\n🧠 IKFlow Model:")
print(f"   Parameters:    {n_params:,}")
print(f"   Coupling layers: {model_params.nb_nodes}")
print(f"   Hidden size:   {model_params.coeff_fn_internal_size}")
print(f"   Latent dim:    {model_params.dim_latent_space}")

# ── Training loop ────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🚀 Training on: {device}")

nn_model = nn_model.to(device)
nn_model.train()

optimizer = torch.optim.Adam(nn_model.parameters(), lr=5e-4)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.975)

BATCH_SIZE = 512
N_EPOCHS = 50
N_BATCHES = configs.shape[0] // BATCH_SIZE

# Shuffle indices
perm = torch.randperm(configs.shape[0])
configs = configs[perm]
poses = poses[perm]

print(f"   Epochs: {N_EPOCHS}")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Batches/epoch: {N_BATCHES:,}")
print(f"   Total steps: {N_EPOCHS * N_BATCHES:,}")

best_loss = float("inf")
SAVE_DIR = "/kaggle/working/ikflow_dataset"

for epoch in range(N_EPOCHS):
    epoch_loss = 0.0
    n_batches_done = 0

    for i in range(0, configs.shape[0] - BATCH_SIZE, BATCH_SIZE):
        q_batch = configs[i:i+BATCH_SIZE].to(device)       # (B, 6)
        pose_batch = poses[i:i+BATCH_SIZE].to(device)      # (B, 7)

        # IKFlow conditioning: pose + dummy softflow noise
        softflow_noise = torch.zeros(BATCH_SIZE, 1, device=device)
        conditional = torch.cat([pose_batch, softflow_noise], dim=1)  # (B, 8)

        # Forward pass through normalizing flow
        # Input: joint configs, Condition: poses → Output: latent vectors
        z, log_det_J = nn_model(q_batch, c=conditional)

        # Negative log-likelihood loss
        # z should be standard normal if the model is well-trained
        nll = 0.5 * torch.sum(z**2, dim=1) - log_det_J
        loss = torch.mean(nll)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(nn_model.parameters(), 1.0)
        optimizer.step()

        epoch_loss += loss.item()
        n_batches_done += 1

    scheduler.step()
    avg_loss = epoch_loss / max(n_batches_done, 1)
    lr = scheduler.get_last_lr()[0]

    # Save best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(nn_model.state_dict(), os.path.join(SAVE_DIR, "kuka_kr6_ikflow_best.pt"))

    print(f"  Epoch {epoch+1:3d}/{N_EPOCHS} | Loss: {avg_loss:.4f} | Best: {best_loss:.4f} | LR: {lr:.6f}")

# Save final model
final_path = os.path.join(SAVE_DIR, "kuka_kr6_ikflow_final.pt")
torch.save(nn_model.state_dict(), final_path)
print(f"\n✅ Training complete!")
print(f"   Best model: {SAVE_DIR}/kuka_kr6_ikflow_best.pt")
print(f"   Final model: {final_path}")

# ── Quick validation ─────────────────────────────────────────────
print("\n🔍 Validating...")
nn_model.eval()

test_q = torch.rand(1000, 6) * (torch.tensor(robot.actuated_joints_limits)[:, 1] - torch.tensor(robot.actuated_joints_limits)[:, 0]) + torch.tensor(robot.actuated_joints_limits)[:, 0]
test_poses = robot.forward_kinematics(test_q)

with torch.no_grad():
    ik_solutions = solver.generate_ik_solutions(
        test_poses.to(device),
        latent=torch.randn(1000, 6, device=device) * 0.5,
        clamp_to_joint_limits=True,
    )

ik_poses = robot.forward_kinematics(ik_solutions.cpu())
pos_errors = torch.norm(ik_poses[:, :3] - test_poses[:, :3], dim=1)

print(f"   Mean error:   {pos_errors.mean():.4f} m")
print(f"   Median error: {pos_errors.median():.4f} m")
print(f"   % < 1cm:      {(pos_errors < 0.01).float().mean() * 100:.1f}%")
print(f"   % < 1mm:      {(pos_errors < 0.001).float().mean() * 100:.1f}%")

print("\n📥 Download the model weights from Kaggle Output:")
print(f"   {SAVE_DIR}/kuka_kr6_ikflow_best.pt")

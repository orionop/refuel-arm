#!/usr/bin/env python3
"""
KUKA KR6 R700 — Configuration Space (Joint Space) Interpolation
===============================================================

Demonstrates linear interpolation in joint space (C-space) between two configurations.
This results in a "curved" path in the Cartesian workspace but "smooth" joint velocities.
"""
import sys
import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Import IK-Geo
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'kuka_refuel_ws', 'src', 'kuka_kr6_gazebo', 'scripts')))
import ik_geometric as ik

# ── Default Configuration (Matches test_ik_line.py for comparison) ────────────────
DEFAULT_START = [0.3, 0.4, 0.5]
DEFAULT_END   = [0.65, -0.25, 0.45]
NUM_WAYPOINTS = 60
DT = 0.15 
DEFAULT_TWIST_DEG = 45.0

# Base orientation: EE pointing forward, looking slightly down
R_START = np.array([
    [ 0,  0,  1],
    [ 0,  1,  0],
    [-1,  0,  0]
])

# Official Joint Limits
JOINT_LIMITS = np.array([
    [-2.967,  2.967],  # joint_1
    [-3.316,  0.785],  # joint_2
    [-2.094,  2.722],  # joint_3
    [-6.108,  6.108],  # joint_4
    [-2.094,  2.094],  # joint_5
    [-6.108,  6.108],  # joint_6
])

Q_HOME = np.array([0.0, -np.pi/2, 0.0, 0.0, 0.0, 0.0])

def is_valid(q):
    for j in range(6):
        if q[j] < JOINT_LIMITS[j, 0] or q[j] > JOINT_LIMITS[j, 1]:
            return False
    return True

def get_best_ik(target_pos, target_R, ref_q):
    """Find the valid IK solution closest to ref_q."""
    Q_all = ik.IK_spherical_2_parallel(target_R, target_pos)
    if Q_all.size == 0:
        return None
    best_q = None
    min_dist = float('inf')
    for i in range(Q_all.shape[1]):
        q = Q_all[:, i]
        q = (q + np.pi) % (2 * np.pi) - np.pi # Normalize
        if is_valid(q):
            dist = np.linalg.norm(q - ref_q)
            if dist < min_dist:
                min_dist = dist
                best_q = q
    return best_q

def generate_joint_space_trajectory(start_pt, end_pt, twist_deg=45.0):
    print(f"\n[C-Space Planning] Starting Joint-Space Interpolation...")
    
    # 1. Define Orientations for Start and End
    # For start, we assume a slightly pitched down orientation (like test_ik_line)
    # Actually, let's keep it simple: Start at R_START, End at R_START + Twist
    R_target_start = R_START
    twist_rad = np.radians(twist_deg)
    R_target_end = R_START @ ik.rot(np.array([1.0, 0.0, 0.0]), twist_rad)

    # 2. Solve IK for Start and End poses
    q_start = get_best_ik(start_pt, R_target_start, Q_HOME)
    if q_start is None:
        print("[Error] Could not find valid IK for start pose.")
        sys.exit(1)
        
    q_goal = get_best_ik(end_pt, R_target_end, q_start)
    if q_goal is None:
        print("[Error] Could not find valid IK for end pose.")
        sys.exit(1)

    print(f"Goal Q: {np.round(q_goal, 3)}")
    print(f"Start Q: {np.round(q_start, 3)}")

    # 3. Linearly interpolate in Joint Space
    trajectory = []
    cartesian_points = []
    fk_errors = [] # In this script, FK error is 0 by definition, but we'll use it to track path
    
    for i in range(NUM_WAYPOINTS):
        t = i / (NUM_WAYPOINTS - 1)
        # LERP in q
        q_t = q_start + t * (q_goal - q_start)
        trajectory.append(q_t)
        
        # Calculate resulting Cartesian path via FK
        R_actual, p_actual = ik.fwd_kinematics(q_t)
        cartesian_points.append(p_actual)
        
    print(f"[C-Space Planning] Generated {len(trajectory)} waypoints.")
    return trajectory, cartesian_points

def plot_analysis(trajectory, cartesian_points):
    """Plot joint angles and the resulting Cartesian 'curved' path."""
    waypoints_q = np.array(trajectory)
    cart_pts = np.array(cartesian_points)
    steps = np.arange(1, len(trajectory) + 1)
    
    fig, axs = plt.subplots(2, 1, figsize=(8, 6))
    fig.suptitle('KUKA KR6 R700 — Configuration Space Interpolation', fontsize=12, fontweight='bold')

    # Plot 1: Joint Angles vs Waypoints (Linear by definition)
    axs[0].set_title("Joint Angles vs. Waypoints (Linear ramps)", fontsize=10)
    labels = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6']
    for j in range(6):
        axs[0].plot(steps, waypoints_q[:, j], label=labels[j], linewidth=2)
    axs[0].set_ylabel("Radians", fontsize=9)
    axs[0].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
    axs[0].grid(True, alpha=0.3)

    # Plot 2: Cartesian Path components (Curved)
    axs[1].set_title("Resulting Cartesian Path (Curved in Workspace)", fontsize=10)
    axs[1].plot(steps, cart_pts[:, 0], label='X', color='r')
    axs[1].plot(steps, cart_pts[:, 1], label='Y', color='g')
    axs[1].plot(steps, cart_pts[:, 2], label='Z', color='b')
    axs[1].set_ylabel("Position (m)", fontsize=9)
    axs[1].legend(fontsize=8)
    axs[1].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output_graphs'))
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, "analysis_cspace_linear.png"), dpi=100)
    print(f"[Analysis] Plot saved to output_graphs/analysis_cspace_linear.png")
    plt.show(block=False)
    plt.pause(2.0)

def main():
    parser = argparse.ArgumentParser(description="Joint Space Interpolation")
    parser.add_argument("--start", nargs=3, type=float, default=DEFAULT_START)
    parser.add_argument("--end", nargs=3, type=float, default=DEFAULT_END)
    parser.add_argument("--twist", type=float, default=DEFAULT_TWIST_DEG)
    args = parser.parse_args()

    line_start = np.array(args.start)
    line_end = np.array(args.end)

    print("=" * 65)
    print("  KUKA KR6 R700 — Configuration Space (Joint Space) Interpolation")
    print("=" * 65)

    trajectory, cart_pts = generate_joint_space_trajectory(line_start, line_end, twist_deg=args.twist)
    plot_analysis(trajectory, cart_pts)

if __name__ == "__main__":
    main()

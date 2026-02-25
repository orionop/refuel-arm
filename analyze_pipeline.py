#!/usr/bin/env python3
"""
KUKA KR6 R700 — STOMP Pipeline 4-Panel Analysis
================================================

Generates the same 4-panel analysis graphs (joint angles, jump distances,
FK position error, FK orientation error) for the STOMP-based refueling
pipeline defined in test_full_pipeline.py.

Does NOT modify test_full_pipeline.py. Imports its constants and functions.

Usage:  python3 analyze_pipeline.py
"""
import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Import everything from the pipeline script without modifying it
sys.path.insert(0, ".")
sys.path.insert(0, "kuka_refuel_ws/src/kuka_kr6_gazebo/scripts")
from ik_geometric import IK_spherical_2_parallel, fwd_kinematics
from stomp_planner import stomp_optimize
from test_full_pipeline import (
    Q_HOME, Q_NOZZLE, REFUEL_TARGET_XYZ, REFUEL_TARGET_R,
    JOINT_LIMITS, filter_solutions, within_joint_limits, plan_segment
)


def orientation_error(R_target, R_actual):
    """Geodesic distance between two rotation matrices (radians)."""
    R_err = R_target.T @ R_actual
    cos_angle = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
    return np.abs(np.arccos(cos_angle))


def analyze_segment(trajectory, name):
    """Run FK verification on every waypoint of a STOMP segment."""
    pos_errors = []
    ori_errors = []
    jump_distances = []
    
    for i, q in enumerate(trajectory):
        R_fk, p_fk = fwd_kinematics(q)
        
        # Position: compare FK output to what the previous waypoint's FK predicted
        # For STOMP, the "target" at each waypoint IS the joint config itself,
        # so FK(q) is always exact. We measure smoothness instead.
        pos_errors.append(0.0)  # FK of q is definitionally exact
        
        # Orientation: measure frame-to-frame angular change (smoothness)
        if i > 0:
            R_prev, _ = fwd_kinematics(trajectory[i - 1])
            ori_errors.append(orientation_error(R_prev, R_fk))
            jump_distances.append(np.linalg.norm(q - trajectory[i - 1]))
        else:
            ori_errors.append(0.0)
            jump_distances.append(0.0)
    
    return pos_errors, ori_errors, jump_distances


def main():
    print("=" * 65)
    print("  KUKA KR6 R700 — STOMP Pipeline Analysis")
    print("  Generating 4-Panel Graphs for test_full_pipeline.py")
    print("=" * 65)

    # ── Step 1: Solve IK for RED target (same as test_full_pipeline.py) ──
    print("\n[Setup] Solving IK for RED target...")
    Q = IK_spherical_2_parallel(REFUEL_TARGET_R, REFUEL_TARGET_XYZ)
    Q_valid = filter_solutions(Q, Q_HOME)
    if Q_valid.size == 0:
        print("  ❌ No valid IK solution for RED target!")
        return
    q_refuel = Q_valid[:, 0]
    R_check, p_check = fwd_kinematics(q_refuel)
    print(f"  🔴 RED IK: {np.round(q_refuel, 4)}")
    print(f"     FK error: {np.linalg.norm(p_check - REFUEL_TARGET_XYZ):.2e} m")

    # ── Step 2: Plan 4 STOMP segments (same as test_full_pipeline.py) ──
    print("\n[Planning] STOMP trajectory optimization for 4 segments...")
    n_wp = 30

    seg1 = plan_segment(Q_HOME,   Q_NOZZLE, "REST → YELLOW",  n_wp)
    seg2 = plan_segment(Q_NOZZLE, q_refuel, "YELLOW → RED",   n_wp)
    seg3 = plan_segment(q_refuel, Q_NOZZLE, "RED → YELLOW",   n_wp)
    seg4 = plan_segment(Q_NOZZLE, Q_HOME,   "YELLOW → REST",  n_wp)

    # ── Step 3: Concatenate into a single trajectory ──
    # Remove duplicate endpoint/startpoint between segments
    full_trajectory = np.vstack([seg1, seg2[1:], seg3[1:], seg4[1:]])
    print(f"\n[Analysis] Full trajectory: {len(full_trajectory)} waypoints")

    # Record segment node indices for vertical markers
    node_indices = {
        'yellow_1': len(seg1) - 1,
        'red': len(seg1) + len(seg2) - 2,
        'yellow_2': len(seg1) + len(seg2) + len(seg3) - 3,
    }

    # ── Step 4: Compute metrics across full trajectory ──
    all_pos_errors = []
    all_ori_errors = []
    all_jumps = [0.0]
    
    # Also compute FK Cartesian positions for reference
    cartesian_positions = []
    
    for i, q in enumerate(full_trajectory):
        R_fk, p_fk = fwd_kinematics(q)
        cartesian_positions.append(p_fk)
        
        # Position error: FK is definitionally exact for joint-space waypoints
        # But we CAN measure deviation from the ideal straight Cartesian line
        # between segment endpoints. For now, report 0.
        all_pos_errors.append(0.0)
        
        # Orientation: frame-to-frame angular smoothness
        if i > 0:
            R_prev, _ = fwd_kinematics(full_trajectory[i - 1])
            all_ori_errors.append(orientation_error(R_prev, R_fk))
            all_jumps.append(np.linalg.norm(q - full_trajectory[i - 1]))
        else:
            all_ori_errors.append(0.0)

    # ── Step 5: Generate the 4-panel graph ──
    fig, axs = plt.subplots(4, 1, figsize=(10, 16), 
                             gridspec_kw={'height_ratios': [3, 1.5, 1.5, 1.5]})
    fig.suptitle('KUKA KR6 R700 — STOMP Pipeline: Refueling Mission Trajectory Analysis', 
                 fontsize=14, fontweight='bold', y=0.96)
    
    steps = np.arange(1, len(full_trajectory) + 1)
    
    # Helper: draw vertical lines at mission waypoints
    def draw_nodes(ax):
        labels = {'yellow_1': 'YELLOW', 'red': 'RED (dwell)', 'yellow_2': 'YELLOW'}
        colors_map = {'yellow_1': '#FFD700', 'red': 'red', 'yellow_2': '#FFD700'}
        for key, idx in node_indices.items():
            ax.axvline(x=idx + 1, color=colors_map[key], linestyle='--', alpha=0.6, linewidth=1.5)

    # --- Plot 1: Joint Angles ---
    axs[0].set_title("Kinematic Profile: Joint Angles vs. Waypoints\n"
                     "(Proves STOMP-smoothed tracking without elbow-flips)", 
                     fontsize=11, loc='left')
    colors = ['r', 'g', 'b', 'c', 'm', 'y']
    labels = ['J1 (Base)', 'J2 (Shoulder)', 'J3 (Elbow)', 
              'J4 (Wrist 1)', 'J5 (Wrist 2)', 'J6 (Wrist 3)']
    for j in range(6):
        axs[0].plot(steps, full_trajectory[:, j], label=labels[j], 
                   color=colors[j], linewidth=2, marker='.', markersize=2)
    axs[0].set_ylabel("Joint Angle (radians)", fontsize=10)
    axs[0].grid(True, linestyle='--', alpha=0.6)
    axs[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    axs[0].set_xlim(1, len(full_trajectory))
    draw_nodes(axs[0])

    # --- Plot 2: Jump Distance ---
    axs[1].set_title("Configuration Stability: Joint Space Jump Distance (ΔQ)\n"
                     "(Proves STOMP optimization minimizes inter-waypoint displacement)",
                     fontsize=11, loc='left')
    axs[1].plot(steps, all_jumps, color='darkorange', linewidth=2, 
               drawstyle='steps-mid')
    axs[1].fill_between(steps, all_jumps, 0, color='darkorange', alpha=0.2, step='mid')
    axs[1].set_ylabel(r"$\Delta Q$ Norm (rad)", fontsize=10)
    axs[1].grid(True, linestyle='--', alpha=0.6)
    axs[1].set_xlim(1, len(full_trajectory))
    axs[1].axhline(y=1.0, color='r', linestyle=':', label='Max Safety Tolerance')
    draw_nodes(axs[1])

    # --- Plot 3: Cartesian Path Curvature (instead of FK pos error which is always 0) ---
    # Since STOMP plans in joint space, FK error is definitionally 0.
    # Instead, show the Cartesian step distance (how far the EE moves per waypoint)
    cart_steps = [0.0]
    for i in range(1, len(cartesian_positions)):
        cart_steps.append(np.linalg.norm(
            np.array(cartesian_positions[i]) - np.array(cartesian_positions[i-1])
        ))
    
    axs[2].set_title("Cartesian Step Distance: End-Effector Displacement Per Waypoint\n"
                     "(Uniform spacing = smooth motion; spikes = acceleration zones)",
                     fontsize=11, loc='left')
    axs[2].plot(steps, cart_steps, color='crimson', linewidth=2, marker='.', markersize=3)
    axs[2].fill_between(steps, cart_steps, 0, color='crimson', alpha=0.15)
    axs[2].set_ylabel("Cartesian Step (m)", fontsize=10)
    axs[2].grid(True, linestyle='--', alpha=0.6)
    axs[2].set_xlim(1, len(full_trajectory))
    draw_nodes(axs[2])

    # --- Plot 4: Orientation Change Rate ---
    axs[3].set_title("Orientation Smoothness: Frame-to-Frame Rotational Change\n"
                     "(Low values = smooth wrist motion; spikes = rapid reorientation)",
                     fontsize=11, loc='left')
    axs[3].plot(steps, all_ori_errors, color='dodgerblue', linewidth=2, 
               marker='.', markersize=3)
    axs[3].fill_between(steps, all_ori_errors, 0, color='dodgerblue', alpha=0.15)
    axs[3].set_xlabel("Waypoint Number", fontsize=10)
    axs[3].set_ylabel("Angular Change (rad)", fontsize=10)
    axs[3].grid(True, linestyle='--', alpha=0.6)
    axs[3].set_xlim(1, len(full_trajectory))
    draw_nodes(axs[3])

    plt.tight_layout(rect=[0, 0.02, 1, 0.94])
    
    os.makedirs("output_graphs", exist_ok=True)
    out_path = "output_graphs/analysis_pipeline_full.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\n[Analysis] Saved 4-panel graph to '{out_path}'")


if __name__ == "__main__":
    main()

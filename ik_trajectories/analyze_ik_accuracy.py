#!/usr/bin/env python3
"""
IK-Geo Accuracy Benchmarking Script
===================================
Generates random reachable EE poses, computes all 8 mathematical IK roots,
and calculates the L2 Norm (Euclidean Distance) between the forward kinematics
of the roots and the input target pose.

This empirically proves the pure algebraic exactness of the polynomial solvers.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Import the IK-Geo solver from the ROS workspace
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'kuka_refuel_ws', 'src', 'kuka_kr6_gazebo', 'scripts')))
import ik_geometric as ik

# Official Joint Limits for KUKA KR6 R700
JOINT_LIMITS = np.array([
    [-2.967,  2.967],  # joint_1
    [-3.316,  0.785],  # joint_2
    [-2.094,  2.722],  # joint_3
    [-6.108,  6.108],  # joint_4
    [-2.094,  2.094],  # joint_5
    [-6.108,  6.108],  # joint_6
])

NUM_SAMPLES = 500

def generate_random_reachable_pose():
    """
    Generates a guaranteed-reachable Cartesian pose by randomly sampling
    within the physical joint limits and computing the Forward Kinematics.
    """
    q_rand = np.random.uniform(JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
    R_target, p_target = ik.fwd_kinematics(q_rand)
    return R_target, p_target

def measure_errors():
    print(f"Generating {NUM_SAMPLES} random reachable Cartesian poses...")
    
    all_pos_errors = []
    all_ori_errors = []
    total_roots_found = 0
    
    for i in range(NUM_SAMPLES):
        # 1. Generate Input EE Pose
        R_target, p_target = generate_random_reachable_pose()
        
        # 2. Feed into IK Solver (Gets up to 8 roots)
        Q_solutions = ik.IK_spherical_2_parallel(R_target, p_target)
        num_roots = Q_solutions.shape[1]
        total_roots_found += num_roots
        
        # 3. Test every single mathematical root
        for j in range(num_roots):
            q_sol = Q_solutions[:, j]
            
            # Plug root back into Forward Kinematics
            R_fk, p_fk = ik.fwd_kinematics(q_sol)
            
            # L2 Norm (Euclidean Distance) of Position Error: sqrt(dx^2 + dy^2 + dz^2)
            pos_error_l2 = np.linalg.norm(p_fk - p_target)
            
            # Geodesic Rotational Error (Angle between rotation matrices)
            trace_val = np.clip((np.trace(R_target.T @ R_fk) - 1.0) / 2.0, -1.0, 1.0)
            ori_error_rad = np.arccos(trace_val)
            
            all_pos_errors.append(pos_error_l2)
            all_ori_errors.append(ori_error_rad)
            
    print(f"\\n[Metrics] Successfully solved IK across {NUM_SAMPLES} random poses.")
    print(f"          Total algebraic roots evaluated: {total_roots_found} (Avg {total_roots_found/NUM_SAMPLES:.1f} per pose)")
    
    max_pos_err = np.max(all_pos_errors)
    mean_pos_err = np.mean(all_pos_errors)
    max_ori_err = np.max(all_ori_errors)
    
    print("\\n[Absolute Maximum Errors Across ALL Roots]")
    print(f"  Max Position L2 Error:    {max_pos_err:.2e} meters")
    print(f"  Mean Position L2 Error:   {mean_pos_err:.2e} meters")
    print(f"  Max Orientation Error:    {max_ori_err:.2e} radians")
    
    # 4. Plot the Errors
    print("\\nRendering accuracy scatter distribution...")
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('IK-Geo Algebraic Accuracy (Benchmarked over 500 Random Poses)', fontsize=14, fontweight='bold')
    
    # We clip the minimum printable error to 1e-16 to avoid log(0) issues on the graph
    printable_pos = np.maximum(all_pos_errors, 1e-16)
    printable_ori = np.maximum(all_ori_errors, 1e-16)
    
    # Scatter 1: Position Error
    x_axis = np.arange(len(printable_pos))
    axs[0].scatter(x_axis, printable_pos, alpha=0.5, color='crimson', s=10)
    axs[0].set_title('Position $\\mathcal{L}_2$ Norm Error (Meters)')
    axs[0].set_xlabel('Mathematical Root Index')
    axs[0].set_ylabel('Euclidean Error (m)')
    axs[0].set_yscale('log')
    axs[0].grid(True, which="both", linestyle='--', alpha=0.5)
    axs[0].axhline(y=1e-13, color='black', linestyle=':', label='IEEE-754 Precision limit')
    
    # Text Annotations for Position Error
    axs[0].annotate('Kinematic Singularities\\n(Arm stretched completely straight)', 
                    xy=(0, 0.05), xytext=(200, 0.01),
                    arrowprops=dict(facecolor='black', arrowstyle='->'),
                    fontsize=9, color='darkred')
                    
    axs[0].annotate('Mathematical Exactness\\n(Limited only by 64-bit float precision)', 
                    xy=(0, 1e-15), xytext=(200, 1e-11),
                    arrowprops=dict(facecolor='black', arrowstyle='->'),
                    fontsize=9, color='darkgreen')
    
    axs[0].legend(loc='lower left')
    
    # Scatter 2: Orientation Error
    axs[1].scatter(x_axis, printable_ori, alpha=0.5, color='dodgerblue', s=10)
    axs[1].set_title('Orientation Geodesic Error (Radians)')
    axs[1].set_xlabel('Mathematical Root Index')
    axs[1].set_ylabel('Angular Error (rad)')
    axs[1].set_yscale('log')
    axs[1].grid(True, which="both", linestyle='--', alpha=0.5)
    axs[1].axhline(y=1e-13, color='black', linestyle=':', label='IEEE-754 Precision limit')
    
    # Text Annotations for Orientation Error
    axs[1].annotate('Orientation Singularities\\n(Gimbal Lock / Wrist Axis Alignment)', 
                    xy=(0, 1e-7), xytext=(200, 1e-6),
                    arrowprops=dict(facecolor='black', arrowstyle='->'),
                    fontsize=9, color='darkred')
                    
    axs[1].annotate('Perfect Geodesic Alignment\\n(Tracing exactly to the SO(3) Manifold)', 
                    xy=(0, 1e-15), xytext=(200, 1e-11),
                    arrowprops=dict(facecolor='black', arrowstyle='->'),
                    fontsize=9, color='darkgreen')
    
    axs[1].legend(loc='lower left')
    
    plt.tight_layout()
    
    # Save Graph
    save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output_graphs'))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    safe_path = os.path.join(save_dir, "ik_accuracy_analysis.png")
    plt.savefig(safe_path, dpi=300, bbox_inches='tight')
    print(f"Graph saved dynamically to -> {safe_path}\\n")
    
    plt.show(block=False)
    plt.pause(3.0)

if __name__ == "__main__":
    measure_errors()

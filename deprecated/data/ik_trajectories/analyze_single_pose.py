#!/usr/bin/env python3
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Import the IK-Geo solver
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

def check_limits(q):
    for i in range(6):
        if q[i] < JOINT_LIMITS[i, 0] or q[i] > JOINT_LIMITS[i, 1]:
            return False
    return True

def analyze_pose():
    # 1. Define Target
    p_target = np.array([0.706, 0.000, 0.413])
    pitch_rad = np.radians(15.0)
    
    # Rotation matrix (Rotation about Y as per report)
    R_target = np.array([
        [np.cos(pitch_rad), 0, np.sin(pitch_rad)],
        [0, 1, 0],
        [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]
    ])
    
    print("-" * 60)
    print(f"TARGET POSE ANALYSIS")
    print(f"Position: {p_target}")
    print(f"Pitch: 15.0 degrees")
    print("-" * 60)
    
    # 2. Solve IK
    # Note: Using the family 2 parallel as per KR6 R700 structure
    Q_solutions = ik.IK_spherical_2_parallel(R_target, p_target)
    num_roots = Q_solutions.shape[1]
    
    pos_errors = []
    ori_errors = []
    valid_mask = []
    
    print(f"{'Sol #':<6} | {'Status':<10} | {'Pos Err (mm)':<12} | {'Ori Err (deg)':<12}")
    print("-" * 60)
    
    for i in range(num_roots):
        q = Q_solutions[:, i]
        
        # Check if imaginary (ik_geometric returns NaN for imaginary components usually or handles it)
        is_imaginary = np.any(np.isnan(q))
        is_in_limits = False if is_imaginary else check_limits(q)
        
        status = "VALID" if (not is_imaginary and is_in_limits) else ("LIMIT" if not is_in_limits else "IMAG")
        if is_imaginary: status = "IMAGINARY"
        elif not is_in_limits: status = "OOB_LIMITS"
        
        # FK Verification
        if not is_imaginary:
            R_fk, p_fk = ik.fwd_kinematics(q)
            
            # Position Error (L2 Norm)
            pos_err = np.linalg.norm(p_fk - p_target) * 1000.0 # to mm
            
            # Orientation Error (Geodesic)
            trace_val = np.clip((np.trace(R_target.T @ R_fk) - 1.0) / 2.0, -1.0, 1.0)
            ori_err_deg = np.degrees(np.arccos(trace_val))
        else:
            pos_err = np.nan
            ori_err_deg = np.nan
            
        pos_errors.append(pos_err)
        ori_errors.append(ori_err_deg)
        valid_mask.append(status == "VALID")
        
        print(f"{i+1:<6} | {status:<10} | {pos_err:12.6f} | {ori_err_deg:12.8f}")
        if not is_imaginary:
            print(f"       Joints: {np.degrees(q)}")

    # 3. Plotting
    indices = np.arange(1, num_roots + 1)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
    fig.suptitle(f"IK Root Comparison: [0.706, 0.0, 0.413] @ 15° Pitch", fontsize=12, fontweight='bold')
    
    # Clip errors to a minimum floor for log-scale visibility
    printable_pos = np.maximum(np.array(pos_errors), 1e-16)
    printable_ori = np.maximum(np.array(ori_errors), 1e-16)
    
    # Set colors
    colors = ['limegreen' if v else 'crimson' for v in valid_mask]
    
    # Plot Position Error
    ax1.bar(indices, printable_pos, color=colors, alpha=0.7)
    ax1.set_ylabel("Pos Error (mm)")
    ax1.set_yscale('log')
    ax1.set_ylim(1e-17, 1)
    ax1.set_title("L2 Norm Position Deviation (Log Scale)")
    ax1.set_xticks(indices)
    ax1.grid(True, which="both", linestyle='--', alpha=0.4)
    
    # Plot Orientation Error
    ax2.bar(indices, printable_ori, color=colors, alpha=0.7)
    ax2.set_ylabel("Ori Error (deg)")
    ax2.set_xlabel("IK Root Index")
    ax2.set_yscale('log')
    ax2.set_ylim(1e-17, 1)
    ax2.set_title("Geodesic Orientation Deviation (Log Scale)")
    ax2.set_xticks(indices)
    ax2.grid(True, which="both", linestyle='--', alpha=0.4)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='limegreen', lw=4, label='Valid Solution'),
                       Line2D([0], [0], color='crimson', lw=4, label='Invalid (Limits/Imaginary)')]
    ax1.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output_graphs', 'single_pose_analysis.png'))
    plt.savefig(save_path, dpi=300)
    print(f"\nAnalysis plot saved to: {save_path}")
    plt.show(block=False)
    plt.pause(2)

if __name__ == "__main__":
    from matplotlib.lines import Line2D # fixing the import in the function
    analyze_pose()

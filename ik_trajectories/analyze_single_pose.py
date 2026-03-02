import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Import the IK-Geo solver from the ROS workspace
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'kuka_refuel_ws', 'src', 'kuka_kr6_gazebo', 'scripts')))
import ik_geometric as ik

def main():
    # Target Pose: (X = 0.706, Y = 0.000, Z = 0.413, Roll = 0, Pitch = 15 deg, Yaw = 0)
    p_target = np.array([[0.7057], [0.0], [0.4134]])
    
    # 15 degrees in radians is ~0.2618. cos(15) = 0.9659, sin(15) = 0.2588
    R_target = np.array([
        [ 0.9659, 0.0,  0.2588],
        [ 0.0,    1.0,  0.0   ],
        [-0.2588, 0.0,  0.9659]
    ])
    
    Q_solutions = ik.IK_spherical_2_parallel(R_target, p_target)
    num_roots = Q_solutions.shape[1]
    
    pos_errors = []
    ori_errors = []
    
    for j in range(num_roots):
        q_sol = Q_solutions[:, j]
        
        # Plug root back into Forward Kinematics
        R_fk, p_fk = ik.fwd_kinematics(q_sol)
        
        # L2 Norm (Euclidean Distance) of Position Error: sqrt(dx^2 + dy^2 + dz^2)
        pos_error_l2 = np.linalg.norm(p_fk - p_target)
        
        # Geodesic Rotational Error (Angle between rotation matrices)
        trace_val = np.clip((np.trace(R_target.T @ R_fk) - 1.0) / 2.0, -1.0, 1.0)
        ori_error_rad = np.arccos(trace_val)
        
        pos_errors.append(pos_error_l2)
        ori_errors.append(ori_error_rad)
        
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('IK-Geo Algebraic Accuracy (Single Target Pose: X=0.706, Y=0.0, Z=0.413, Pitch=15°)', fontsize=14, fontweight='bold')
    
    x_labels = [f"q{i+1}" for i in range(num_roots)]
    x_pos = np.arange(num_roots)
    
    # We clip the minimum printable error to 1e-16 to avoid log(0) issues on the graph
    printable_pos = np.maximum(pos_errors, 1e-16)
    printable_ori = np.maximum(ori_errors, 1e-16)
    
    # Scatter 1: Position Error
    axs[0].scatter(x_pos, printable_pos, s=100, alpha=0.8, color='crimson')
    axs[0].set_title('Position $\\mathcal{L}_2$ Norm Error (Meters)')
    axs[0].set_xticks(x_pos)
    axs[0].set_xticklabels(x_labels)
    axs[0].set_xlabel('IK Solutions')
    axs[0].set_ylabel('Euclidean Error (m)')
    axs[0].set_yscale('log')
    axs[0].grid(True, which="both", axis='y', linestyle='--', alpha=0.5)
    axs[0].axhline(y=1e-13, color='black', linestyle=':', label='IEEE-754 Precision limit')
    axs[0].legend(loc='upper right')
    
    # Scatter 2: Orientation Error
    axs[1].scatter(x_pos, printable_ori, s=100, alpha=0.8, color='dodgerblue')
    axs[1].set_title('Orientation Geodesic Error (Radians)')
    axs[1].set_xticks(x_pos)
    axs[1].set_xticklabels(x_labels)
    axs[1].set_xlabel('IK Solutions')
    axs[1].set_ylabel('Angular Error (rad)')
    axs[1].set_yscale('log')
    axs[1].grid(True, which="both", axis='y', linestyle='--', alpha=0.5)
    axs[1].axhline(y=1e-13, color='black', linestyle=':', label='IEEE-754 Precision limit')
    axs[1].legend(loc='upper right')
    
    plt.tight_layout()
    
    # Save Graph
    save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output_graphs'))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # User requested it at the same path output_graphs/ik_accuracy_analysis.png but I'll write it there
    safe_path = os.path.join(save_dir, "ik_accuracy_analysis.png")
    plt.savefig(safe_path, dpi=300, bbox_inches='tight')
    print(f"Graph saved dynamically to -> {safe_path}\n")

if __name__ == "__main__":
    main()

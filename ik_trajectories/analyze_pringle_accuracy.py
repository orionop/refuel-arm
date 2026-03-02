import sys
import os
import matplotlib.pyplot as plt
import numpy as np

# Import the IK-Geo solver from the ROS workspace
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'kuka_refuel_ws', 'src', 'kuka_kr6_gazebo', 'scripts')))
import ik_geometric as ik

# Import the exact generation logic from the pringle script directly
from test_ik_pringle import generate_pringle_trajectory, DEFAULT_CENTER, RADIUS, DEFAULT_TWIST_DEG

def main():
    print("Generating Pringle trajectory to calculate dynamic waypoint errors...")
    center_pt = np.array(DEFAULT_CENTER)
    
    # 1. Plan trajectory with orientation (uses closest IK root from previous waypoint)
    trajectory, cartesian_points, jump_distances, fk_errors, orient_errors = generate_pringle_trajectory(center_pt, float(RADIUS), float(DEFAULT_TWIST_DEG))
    
    num_waypoints = len(trajectory)
    
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'IK-Geo Algebraic Accuracy (Sequential Pringle Trajectory Tracking, N={num_waypoints})', fontsize=14, fontweight='bold')
    
    x_labels = [f"WP {i+1}" for i in range(num_waypoints)]
    x_pos = np.arange(num_waypoints)
    
    # Clip errors to the 64-bit precision floor (1e-16) to avoid log(0)
    printable_pos = np.maximum(fk_errors, 1e-16)
    printable_ori = np.maximum(orient_errors, 1e-16)
    
    # Scatter 1: Position Error
    axs[0].scatter(x_pos, printable_pos, s=50, alpha=0.8, color='crimson', zorder=5)
    axs[0].plot(x_pos, printable_pos, alpha=0.4, color='crimson', linestyle='-', zorder=4)
    axs[0].set_title('Position $\\mathcal{L}_2$ Norm Error (Meters)')
    
    # Show fewer ticks so text doesn't overlap
    step = max(1, num_waypoints // 15)
    axs[0].set_xticks(x_pos[::step])
    axs[0].set_xticklabels(x_labels[::step], rotation=45, ha='right')
    
    axs[0].set_xlabel('Waypoints in Sequential Order (Prev -> Target)')
    axs[0].set_ylabel('Euclidean Error (m)')
    axs[0].set_yscale('log')
    axs[0].grid(True, which="both", linestyle='--', alpha=0.5)
    axs[0].axhline(y=1e-13, color='black', linestyle=':', label='IEEE-754 Precision limit')
    axs[0].legend(loc='lower left')
    
    # Scatter 2: Orientation Error
    axs[1].scatter(x_pos, printable_ori, s=50, alpha=0.8, color='dodgerblue', zorder=5)
    axs[1].plot(x_pos, printable_ori, alpha=0.4, color='dodgerblue', linestyle='-', zorder=4)
    axs[1].set_title('Orientation Geodesic Error (Radians)')
    axs[1].set_xticks(x_pos[::step])
    axs[1].set_xticklabels(x_labels[::step], rotation=45, ha='right')
    axs[1].set_xlabel('Waypoints in Sequential Order (Prev -> Target)')
    axs[1].set_ylabel('Angular Error (rad)')
    axs[1].set_yscale('log')
    axs[1].grid(True, which="both", linestyle='--', alpha=0.5)
    axs[1].axhline(y=1e-13, color='black', linestyle=':', label='IEEE-754 Precision limit')
    axs[1].legend(loc='lower left')
    
    plt.tight_layout()
    
    save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output_graphs'))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    safe_path = os.path.join(save_dir, "pringle_analysis.png")
    plt.savefig(safe_path, dpi=300, bbox_inches='tight')
    print(f"\nGraph successfully saved dynamically to -> {safe_path}")

if __name__ == "__main__":
    main()

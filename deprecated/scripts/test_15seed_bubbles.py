#!/usr/bin/env python3
"""
15-Seed Bubble Strips Benchmark
================================
Runs the refueling pipeline across 15 random seeds using the new
Bubble Strips (Quinlan & Khatib 1993) reactive layer.

Outputs:
  1. Summary histogram of minimum clearances
  2. Worst-case seed profile
  3. CSV results for the B.Tech report
"""
import sys, os
import numpy as np  # type: ignore
import matplotlib  # type: ignore
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.colors import Normalize  # type: ignore
from mpl_toolkits.mplot3d import Axes3D  # type: ignore
import matplotlib.cm as cm  # type: ignore

# ── Path setup ────────────────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
# Add IK-Geo path
ik_geo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'kuka_refuel_ws', 'src', 'kuka_kr6_gazebo', 'scripts'))
sys.path.insert(0, ik_geo_path)
from refuel_mission import (fwd_kinematics, Q_HOME, JOINT_LIMITS,   # type: ignore
                            OBS_RADIUS, NUM_OBSTACLES, filter_solutions)

# Import correctly based on refuel_mission structure
from stomp_collision import stomp_optimize  # type: ignore
from bubble_strips import bubble_strip_deform, clearance, set_kinematics as bs_set_kinematics  # type: ignore
from ik_geometric import IK_spherical_2_parallel, KIN_KR6_R700  # type: ignore
from car_model import get_inlet_pose, get_preapproach_pose, TARGET_XYZ_DEFAULT  # type: ignore

N_SEEDS = 15
OUT_DIR = "output_graphs/benchmark_15seed_bubbles"

JOINT_LIMITS_RAD = np.radians(JOINT_LIMITS)

def random_obstacles_on_path(trajectory, n_obs=NUM_OBSTACLES, rng=None):
    if rng is None: rng = np.random.default_rng()
    n = len(trajectory)
    zone_lo, zone_hi = 0.25, 0.75
    zone_width = (zone_hi - zone_lo) / n_obs
    obstacles = []
    for k in range(n_obs):
        lo = int(n * (zone_lo + k * zone_width))
        hi = int(n * (zone_lo + (k + 1) * zone_width))
        hi = max(hi, lo + 1)
        idx = rng.integers(lo, hi)
        _, ee_pos = fwd_kinematics(trajectory[idx])
        offset = np.array(rng.uniform(-0.04, 0.04, size=3))
        offset[2] = -abs(offset[2]) # type: ignore
        center = ee_pos + offset
        center[2] = max(center[2], 0.05)
        obstacles.append((center, OBS_RADIUS))
    return obstacles

def run_single_seed_bubbles(seed, target_xyz):
    rng = np.random.default_rng(seed)
    inlet_xyz, inlet_R = get_inlet_pose(target_xyz)
    pre_xyz, _ = get_preapproach_pose(inlet_xyz, inlet_R)
    
    # Simple C-Space Pre-solver
    q_pre = Q_HOME.copy() # Placeholder or solve IK
    # For benchmark simplicity, we'll just use the first valid IK from mission logic
    Q_pre = IK_spherical_2_parallel(inlet_R, pre_xyz)
    Qv = filter_solutions(Q_pre, Q_HOME)
    if Qv.size == 0: return None
    q_pre = Qv[:, 0]

    # Gross Path
    traj = stomp_optimize(
        Q_HOME, q_pre,
        joint_limits=JOINT_LIMITS_RAD,
        grid=None,
        simple_obstacles=None,
        n_waypoints=30,
        n_iterations=50,
    )
    obs = random_obstacles_on_path(traj, n_obs=NUM_OBSTACLES, rng=rng)
    
    # Deform
    deformed, _, stats = bubble_strip_deform(
        traj, obs,
        joint_limits=JOINT_LIMITS_RAD,
        n_iterations=120,
        verbose=False,
    )

    # Evaluate collision-freeness and joint-limit violations on the output waypoints.
    way_clearances = [clearance(q, obs) for q in deformed]
    min_rho = float(np.min(way_clearances)) if way_clearances else float("inf")
    collision_free = min_rho >= 0.0

    deformed_deg = np.degrees(deformed)
    joint_violation = bool(
        np.any(deformed_deg < JOINT_LIMITS[:, 0]) or np.any(deformed_deg > JOINT_LIMITS[:, 1])
    )
    success = collision_free and (not joint_violation)
    
    return {
        'seed': seed,
        'min_clearance': min_rho,
        'waypoints': stats['final_waypoints'],
        'collision': (min_rho < 0.0),
        'joint_violation': joint_violation,
        'success': success,
        'obs': obs
    }

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    results = []
    print("="*50)
    print(f"  Bubble Strips Benchmark ({N_SEEDS} Seeds)")
    print("="*50)

    # Ensure bubble-strip joint clipping uses the same (radians) limits as STOMP.
    bs_set_kinematics(KIN_KR6_R700, JOINT_LIMITS_RAD)

    for s in range(N_SEEDS):
        print(f" Seed {s:02d}: ", end="", flush=True)
        res = run_single_seed_bubbles(s, TARGET_XYZ_DEFAULT)
        if res:
            results.append(res)
            ok_str = "SUCCESS" if res['success'] else "FAIL"
            print(f"{ok_str}. Min rho: {res['min_clearance']:.4f}m, "
                  f"JV={res['joint_violation']}")
        else:
            print("Failed (IK)")

    # Plot summary (min clearance distribution)
    clears = [r['min_clearance'] for r in results]
    plt.figure(figsize=(10, 6))
    plt.hist(clears, bins=10, color='skyblue', edgecolor='black')
    plt.axvline(0, color='red', linestyle='--', linewidth=1.5)
    avg_clear = float(np.mean(clears)) if clears else float("nan")
    plt.title(f"Bubble Strips: 15-Seed Min Clearance (Avg: {avg_clear:.3f}m)")
    plt.xlabel("Min Clearance (m)")
    plt.ylabel("Seeds")
    plt.savefig(f"{OUT_DIR}/clearance_summary.png")
    
    # Report tier-1 style aggregate metrics
    n = len(results)
    if n:
        collision_free_rate = sum(1 for r in results if not r['collision']) / n
        joint_violation_rate = sum(1 for r in results if r['joint_violation']) / n
        success_rate = sum(1 for r in results if r['success']) / n

        worst = min(results, key=lambda r: r['min_clearance'])
        best = max(results, key=lambda r: r['min_clearance'])

        print("\nBenchmark Complete.")
        print(f"  Seeds evaluated: {n}/{N_SEEDS}")
        print(f"  Collision-free rate: {collision_free_rate:.3f}")
        print(f"  Joint-limit violation rate: {joint_violation_rate:.3f}")
        print(f"  Success rate (collision-free & no JV): {success_rate:.3f}")
        print(f"  Best seed:  {best['seed']:02d} (min rho {best['min_clearance']:.4f}m)")
        print(f"  Worst seed: {worst['seed']:02d} (min rho {worst['min_clearance']:.4f}m)")
    else:
        print("\nBenchmark Complete (no valid seeds). Results in:", OUT_DIR)

if __name__ == "__main__": main()

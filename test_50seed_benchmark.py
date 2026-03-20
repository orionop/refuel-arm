#!/usr/bin/env python3
"""
50-Seed Obstacle Avoidance Benchmark
=====================================
Runs the refueling pipeline across 50 random seeds (100 total obstacles)
and computes per-link obstacle clearance metrics.

Outputs:
  1. Per-seed 3D trajectory with obstacle spheres + clearance heatmap
  2. Clearance vs. waypoint index for worst-case seed
  3. Summary histogram of minimum clearances across all 50 seeds

Usage:
    python test_50seed_benchmark.py
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

# ── Path setup (same as refuel_mission.py) ────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), 'kuka_refuel_ws', 'src',
    'kuka_kr6_gazebo', 'scripts')))

from ik_geometric import (IK_spherical_2_parallel, fwd_kinematics,
                           rot, KIN_KR6_R700)
from stomp_collision import stomp_optimize
from elastic_strips import elastic_strip_deform
from car_model import get_inlet_pose, get_preapproach_pose, TARGET_XYZ_DEFAULT

# ── Constants from refuel_mission.py ──────────────────────────────
JOINT_LIMITS = np.array([
    [-2.967059725,  2.967059725],
    [-3.316125575,  0.785398163],
    [-2.094395100,  2.722713630],
    [-3.228859113,  3.228859113],
    [-2.094395100,  2.094395100],
    [-6.108652375,  6.108652375],
])
Q_HOME     = np.array([0.0, -np.pi / 2, 0.0, 0.0, 0.0, 0.0])
OBS_RADIUS = 0.05
NUM_OBSTACLES = 2
N_SEEDS = 50
OUT_DIR = "output_graphs/benchmark_50seed"


# ── Link FK: compute positions of all 7 link frames ──────────────
def all_link_positions(q, kin=None):
    """Return (7, 3) array of positions for base + 6 link frames."""
    if kin is None:
        kin = KIN_KR6_R700
    H = kin['H']
    P = kin['P']
    positions = np.zeros((7, 3))
    R = np.eye(3)
    p = P[:, 0].copy()
    positions[0] = p.copy()
    for i in range(6):
        R = R @ rot(H[:, i], q[i])
        p = p + R @ P[:, i + 1]
        positions[i + 1] = p.copy()
    return positions


def min_clearance_at_waypoint(q, obstacles):
    """Compute the minimum distance from any link to any obstacle surface."""
    link_pos = all_link_positions(q)
    min_d = np.inf
    for lp in link_pos:
        for (obs_center, obs_radius) in obstacles:
            d = np.linalg.norm(lp - obs_center) - obs_radius
            if d < min_d:
                min_d = d
    return min_d


def compute_clearance_profile(trajectory, obstacles):
    """Compute clearance at every waypoint. Returns (N,) array."""
    N = len(trajectory)
    clearances = np.zeros(N)
    for i in range(N):
        clearances[i] = min_clearance_at_waypoint(trajectory[i], obstacles)
    return clearances


# ── Reuse obstacle & planning functions from refuel_mission ───────
def wrap_to_limits(q):
    q_w = np.copy(q)
    for i in range(6):
        while q_w[i] > np.pi:  q_w[i] -= 2 * np.pi
        while q_w[i] < -np.pi: q_w[i] += 2 * np.pi
    return q_w

def within_joint_limits(q):
    for i in range(6):
        if q[i] < JOINT_LIMITS[i, 0] or q[i] > JOINT_LIMITS[i, 1]:
            return False
    return True

def delta_wrap(d):
    return (d + np.pi) % (2 * np.pi) - np.pi

def filter_solutions(Q, history=None):
    if Q.size == 0:
        return np.empty((6, 0))
    valid = []
    for i in range(Q.shape[1]):
        q = wrap_to_limits(Q[:, i])
        if within_joint_limits(q):
            valid.append(q)
    if not valid:
        return np.empty((6, 0))
    valid = np.array(valid).T
    if history is not None:
        if isinstance(history, np.ndarray):
            history = [history]
        if len(history) > 0:
            scores = np.zeros(valid.shape[1])
            W_v, W_a, W_j = 1.0, 5.0, 10.0
            for i in range(valid.shape[1]):
                q = valid[:, i]
                score = 0.0
                v_k = delta_wrap(q - history[-1])
                score += W_v * np.linalg.norm(v_k)
                if len(history) >= 2:
                    v_k_1 = delta_wrap(history[-1] - history[-2])
                    a_k = v_k - v_k_1
                    score += W_a * np.linalg.norm(a_k)
                    if len(history) >= 3:
                        v_k_2 = delta_wrap(history[-2] - history[-3])
                        a_k_1 = v_k_1 - v_k_2
                        j_k = a_k - a_k_1
                        score += W_j * np.linalg.norm(j_k)
                scores[i] = score
            valid = valid[:, np.argsort(scores)]
    return valid

def random_obstacles_on_path(trajectory, n_obs=NUM_OBSTACLES, rng=None):
    if rng is None:
        rng = np.random.default_rng()
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
        offset = rng.uniform(-0.04, 0.04, size=3)
        offset[2] = -abs(offset[2])
        center = ee_pos + offset
        center[2] = max(center[2], 0.05)
        obstacles.append((center, OBS_RADIUS))
    return obstacles

def plan_cartesian(start_xyz, end_xyz, R_target, q_prev, name, n_wp=20):
    waypoints_xyz = np.linspace(start_xyz, end_xyz, n_wp)
    traj = []
    history = [q_prev]
    for i, pt in enumerate(waypoints_xyz):
        Q = IK_spherical_2_parallel(R_target, pt)
        Q_valid = filter_solutions(Q, history[-3:])
        if Q_valid.size == 0:
            traj.append(history[-1])
            continue
        q = Q_valid[:, 0]
        traj.append(q)
        history.append(q)
    return np.array(traj)


# ── Run a single seed ─────────────────────────────────────────────
def run_single_seed(seed, target_xyz, n_wp=30):
    """Run the full 4-phase pipeline for one seed. Returns dict of results."""
    rng = np.random.default_rng(seed)
    
    inlet_xyz, inlet_R = get_inlet_pose(target_xyz)
    preapproach_xyz, _ = get_preapproach_pose(inlet_xyz, inlet_R)
    
    # IK solve
    Q_target = IK_spherical_2_parallel(inlet_R, inlet_xyz)
    Q_valid_target = filter_solutions(Q_target, Q_HOME)
    if Q_valid_target.size == 0:
        return None
    q_target = Q_valid_target[:, 0]
    
    Q_pre = IK_spherical_2_parallel(inlet_R, preapproach_xyz)
    Q_valid_pre = filter_solutions(Q_pre, Q_HOME)
    if Q_valid_pre.size == 0:
        return None
    q_pre = Q_valid_pre[:, 0]
    
    # Blind STOMP to determine path shape
    seg_blind = stomp_optimize(
        q_start=Q_HOME, q_goal=q_pre,
        joint_limits=JOINT_LIMITS, simple_obstacles=None,
        n_waypoints=n_wp, n_iterations=50, n_rollouts=5,
        noise_stddev=0.08, verbose=False)
    
    # Place obstacles on that path
    obs_list = random_obstacles_on_path(seg_blind, n_obs=NUM_OBSTACLES, rng=rng)
    
    # Phase 1: STOMP + Elastic Strips (Gross Approach)
    seg_approach = stomp_optimize(
        q_start=Q_HOME, q_goal=q_pre,
        joint_limits=JOINT_LIMITS,
        simple_obstacles=obs_list or None,
        n_waypoints=n_wp, n_iterations=50, n_rollouts=5,
        noise_stddev=0.08, verbose=False)
    if obs_list:
        seg_approach, _ = elastic_strip_deform(
            seg_approach, obs_list,
            safety_margin=0.15, k_repulsion=5.0, n_iterations=80, verbose=False)
    
    # Phase 2: Cartesian insertion
    seg_insert = plan_cartesian(preapproach_xyz, inlet_xyz, inlet_R,
                                q_pre, "Insert", n_wp=20)
    
    # Phase 3: Cartesian extraction
    seg_extract = plan_cartesian(inlet_xyz, preapproach_xyz, inlet_R,
                                 q_target, "Extract", n_wp=20)
    
    # Phase 4: STOMP + Elastic Strips (Return)
    seg_return = stomp_optimize(
        q_start=q_pre, q_goal=Q_HOME,
        joint_limits=JOINT_LIMITS,
        simple_obstacles=obs_list or None,
        n_waypoints=n_wp, n_iterations=50, n_rollouts=5,
        noise_stddev=0.08, verbose=False)
    if obs_list:
        seg_return, _ = elastic_strip_deform(
            seg_return, obs_list,
            safety_margin=0.15, k_repulsion=5.0, n_iterations=80, verbose=False)
    
    # Full trajectory
    full_traj = np.vstack([seg_approach, seg_insert, seg_extract, seg_return])
    
    # Compute clearance profile
    clearance = compute_clearance_profile(full_traj, obs_list)
    
    # EE positions for 3D plot
    ee_pts = np.zeros((len(full_traj), 3))
    for i, q in enumerate(full_traj):
        _, p = fwd_kinematics(q)
        ee_pts[i] = p
    
    # FK error at target
    _, p_achieved = fwd_kinematics(q_target)
    fk_error = np.linalg.norm(p_achieved - inlet_xyz)
    
    return {
        'seed': seed,
        'full_traj': full_traj,
        'obstacles': obs_list,
        'clearance': clearance,
        'min_clearance': np.min(clearance),
        'mean_clearance': np.mean(clearance),
        'collision': np.min(clearance) < 0,
        'ee_pts': ee_pts,
        'fk_error': fk_error,
        'target_xyz': target_xyz,
    }


# ── Visualization ─────────────────────────────────────────────────

def plot_3d_trajectory(result, save_path):
    """3D trajectory with obstacles + clearance heatmap."""
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    ee = result['ee_pts']
    clearance = result['clearance']
    obs_list = result['obstacles']
    
    # Color-code path by clearance
    norm = Normalize(vmin=max(0, np.min(clearance)), vmax=np.max(clearance))
    colors = cm.RdYlGn(norm(clearance))
    
    for i in range(len(ee) - 1):
        ax.plot(ee[i:i+2, 0], ee[i:i+2, 1], ee[i:i+2, 2],
                color=colors[i], linewidth=2.5)
    
    # Draw obstacle spheres
    for k, (center, radius) in enumerate(obs_list):
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 15)
        x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
        y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
        z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
        ax.plot_surface(x, y, z, alpha=0.3, color='red')
    
    # Mark target
    t = result['target_xyz']
    ax.scatter(*t, s=100, c='gold', marker='*', zorder=5,
               label='Target', edgecolors='black')
    
    sm = cm.ScalarMappable(cmap='RdYlGn', norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label('Link-Obstacle Clearance (m)', fontsize=10)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'Seed {result["seed"]} — Min Clearance: '
                 f'{result["min_clearance"]:.4f} m', fontsize=13)
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_clearance_profile(result, save_path):
    """Clearance vs waypoint index for a single seed."""
    fig, ax = plt.subplots(figsize=(14, 5))
    clearance = result['clearance']
    N = len(clearance)
    
    colors = ['green' if c > 0.05 else 'orange' if c > 0 else 'red'
              for c in clearance]
    ax.bar(range(N), clearance, color=colors, width=1.0, edgecolor='none')
    ax.axhline(y=0, color='red', linewidth=2, linestyle='--', label='Collision Boundary')
    ax.axhline(y=0.05, color='orange', linewidth=1.5, linestyle=':', label='Safety Margin (5 cm)')
    
    ax.set_xlabel('Waypoint Index', fontsize=12)
    ax.set_ylabel('Min Link-Obstacle Clearance (m)', fontsize=12)
    ax.set_title(f'Seed {result["seed"]} — Clearance Profile '
                 f'(Min: {result["min_clearance"]:.4f} m)', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_summary_histogram(all_results, save_path):
    """Histogram of minimum clearances across all seeds."""
    min_clearances = [r['min_clearance'] for r in all_results]
    collisions = sum(1 for c in min_clearances if c < 0)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors_hist = ['red' if c < 0 else 'orange' if c < 0.05 else '#2ecc71'
                   for c in sorted(min_clearances)]
    
    n, bins, patches = ax.hist(min_clearances, bins=25, edgecolor='black',
                                linewidth=0.8, alpha=0.85)
    
    # Color bins by safety
    for patch, left_edge in zip(patches, bins[:-1]):
        if left_edge < 0:
            patch.set_facecolor('#e74c3c')
        elif left_edge < 0.05:
            patch.set_facecolor('#f39c12')
        else:
            patch.set_facecolor('#2ecc71')
    
    ax.axvline(x=0, color='red', linewidth=2, linestyle='--', label='Collision Boundary')
    ax.axvline(x=0.05, color='orange', linewidth=1.5, linestyle=':', label='Safety Margin (5 cm)')
    
    # Stats annotation
    stats_text = (
        f"Seeds: {len(all_results)}\n"
        f"Obstacles: {len(all_results) * NUM_OBSTACLES}\n"
        f"Collisions: {collisions}/{len(all_results)}\n"
        f"Global Min: {min(min_clearances):.4f} m\n"
        f"Mean Min: {np.mean(min_clearances):.4f} m\n"
        f"Std: {np.std(min_clearances):.4f} m"
    )
    ax.text(0.97, 0.95, stats_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                      edgecolor='gray', alpha=0.9),
            fontfamily='monospace')
    
    ax.set_xlabel('Minimum Link-Obstacle Clearance per Seed (m)', fontsize=12)
    ax.set_ylabel('Number of Seeds', fontsize=12)
    ax.set_title(f'50-Seed Obstacle Avoidance Benchmark — '
                 f'{collisions} Collisions out of {len(all_results)} Runs',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ── Main ──────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    target_xyz = np.array(TARGET_XYZ_DEFAULT)
    
    print("=" * 65)
    print("  50-Seed Obstacle Avoidance Benchmark")
    print(f"  {N_SEEDS} seeds × {NUM_OBSTACLES} obstacles = "
          f"{N_SEEDS * NUM_OBSTACLES} total obstacles")
    print("=" * 65)
    
    all_results = []
    worst_seed_result = None
    worst_clearance = np.inf
    
    for i, seed in enumerate(range(N_SEEDS)):
        print(f"\n[Seed {seed:3d}] ({i+1}/{N_SEEDS}) ", end="", flush=True)
        try:
            result = run_single_seed(seed, target_xyz)
            if result is None:
                print("SKIP (no valid IK)")
                continue
            
            all_results.append(result)
            mc = result['min_clearance']
            status = "COLLISION!" if mc < 0 else f"OK (min={mc:.4f}m)"
            print(status, end="", flush=True)
            
            if mc < worst_clearance:
                worst_clearance = mc
                worst_seed_result = result
                
        except Exception as e:
            print(f"ERROR: {e}")
            continue
    
    print(f"\n\n{'=' * 65}")
    print(f"  Benchmark Complete: {len(all_results)}/{N_SEEDS} seeds succeeded")
    collisions = sum(1 for r in all_results if r['collision'])
    print(f"  Collisions: {collisions}/{len(all_results)}")
    print(f"  Global worst clearance: {worst_clearance:.4f} m")
    print(f"  Mean min clearance: {np.mean([r['min_clearance'] for r in all_results]):.4f} m")
    print(f"{'=' * 65}")
    
    # ── Generate visualizations ───────────────────────────────────
    print("\n[Plots] Generating visualizations...")
    
    # 1. Summary histogram (all 50 seeds)
    hist_path = os.path.join(OUT_DIR, "clearance_histogram_50seeds.png")
    plot_summary_histogram(all_results, hist_path)
    print(f"  Saved: {hist_path}")
    
    # 2. Worst-case seed: 3D trajectory
    if worst_seed_result:
        traj3d_path = os.path.join(OUT_DIR, "worst_seed_3d_trajectory.png")
        plot_3d_trajectory(worst_seed_result, traj3d_path)
        print(f"  Saved: {traj3d_path}")
        
        # 3. Worst-case seed: clearance profile
        profile_path = os.path.join(OUT_DIR, "worst_seed_clearance_profile.png")
        plot_clearance_profile(worst_seed_result, profile_path)
        print(f"  Saved: {profile_path}")
    
    # 4. Also save a CSV summary
    csv_path = os.path.join(OUT_DIR, "benchmark_results.csv")
    with open(csv_path, 'w') as f:
        f.write("seed,min_clearance_m,mean_clearance_m,collision,fk_error_m\n")
        for r in all_results:
            f.write(f"{r['seed']},{r['min_clearance']:.6f},"
                    f"{r['mean_clearance']:.6f},"
                    f"{r['collision']},{r['fk_error']:.2e}\n")
    print(f"  Saved: {csv_path}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()

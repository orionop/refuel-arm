#!/usr/bin/env python3
"""
Benchmark Framework for collision avoidance methods.
Compares: Baseline (No Avoidance), Bubble Strips, Tangent Bug.
Metrics:
- Path Length (C-Space)
- Execution Time
- Min Clearance
- Smoothness (Avg Jerk)
"""
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt

# Add ref_env to path if run from root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ik_geometric import IK_solve, KIN_KR6_R700
from refuel_mission_v2 import get_realistic_obstacles, plan_stomp
from bubble_strips import bubble_strip_deform, clearance, set_kinematics
from elastic_strips import elastic_strip_deform, set_kinematics as es_set_kinematics
from tangent_bug import tangent_bug_optimize

def get_path_length(trajectory):
    """Calculate C-space path length."""
    if len(trajectory) < 2: return 0.0
    diffs = np.diff(trajectory, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))

def get_min_clearance(trajectory, obstacles):
    """Calculate minimum workspace clearance across trajectory."""
    min_c = float('inf')
    for q in trajectory:
        c = clearance(q, obstacles)
        min_c = min(min_c, c)
    return min_c

def get_smoothness(trajectory, dt=0.1):
    """Calculate average jerk magnitude."""
    if len(trajectory) < 4: return 0.0
    vel = np.diff(trajectory, axis=0) / dt
    acc = np.diff(vel, axis=0) / dt
    jerk = np.diff(acc, axis=0) / dt
    return float(np.mean(np.linalg.norm(jerk, axis=1)))

def run_benchmark():
    # Setup
    q_start = np.array([0.0, -1.5708, 0.0, 0.0, 0.0, 0.0]) # Q_HOME
    q_goal = np.array([0.3, -1.0, 1.2, -0.2, 1.0, -0.5]) # Sample target behind obstacles
    obstacles = get_realistic_obstacles()
    kin = KIN_KR6_R700
    limits = kin.get('joint_limits', np.array([
        [-170.0, 170.0], [-190.0, 45.0], [-120.0, 156.0],
        [-185.0, 185.0], [-120.0, 120.0], [-350.0, 350.0]
    ]))
    limits_rad = np.radians(limits)
    set_kinematics(kin, limits_rad)
    es_set_kinematics(kin, limits_rad)

    n_wp = 30
    dt_step = 0.15 # seconds per step

    results = {}

    print("=" * 60)
    print("  Collision Avoidance Benchmark Framework")
    print("=" * 60)

    # 1. Baseline (C-Space LERP - No Avoidance)
    print("\n[Running Baseline: C-Space Linear]")
    t0 = time.time()
    traj_base = np.zeros((n_wp, 6))
    for i in range(n_wp):
        alpha = i / (n_wp - 1)
        traj_base[i] = (1 - alpha) * q_start + alpha * q_goal
    t_base = time.time() - t0
    
    results['Baseline'] = {
        'compute_time': t_base,
        'exec_time': len(traj_base) * dt_step,
        'path_length': get_path_length(traj_base),
        'clearance': get_min_clearance(traj_base, obstacles),
        'smoothness': get_smoothness(traj_base, dt_step)
    }

    # 2. Bubble Strips
    print("\n[Running Bubble Strips]")
    t0 = time.time()
    traj_stomp = plan_stomp(q_start, q_goal, None, "Init", n_wp, limits=limits, kin=kin)
    traj_bubble, _, _ = bubble_strip_deform(
        traj_stomp, obstacles, n_iterations=100,
        k_contraction=0.5, k_repulsion=30.0, rho_0=0.25, verbose=False
    )
    t_bubble = time.time() - t0

    results['Bubble Strips'] = {
        'compute_time': t_bubble,
        'exec_time': len(traj_bubble) * dt_step,
        'path_length': get_path_length(traj_bubble),
        'clearance': get_min_clearance(traj_bubble, obstacles),
        'smoothness': get_smoothness(traj_bubble, dt_step)
    }

    # 3. Elastic Strips (Brock & Khatib 2002)
    print("\n[Running Elastic Strips]")
    t0 = time.time()
    traj_stomp2 = plan_stomp(q_start, q_goal, None, "Init (Elastic)", n_wp, limits=limits, kin=kin)
    traj_elastic, _, _ = elastic_strip_deform(
        traj_stomp2, obstacles, n_iterations=100,
        k_contraction=2.0, k_repulsion=8.0, safety_margin=0.25,
        damping=0.92, verbose=False
    )
    t_elastic = time.time() - t0

    results['Elastic Strips'] = {
        'compute_time': t_elastic,
        'exec_time': len(traj_elastic) * dt_step,
        'path_length': get_path_length(traj_elastic),
        'clearance': get_min_clearance(traj_elastic, obstacles),
        'smoothness': get_smoothness(traj_elastic, dt_step)
    }

    # 4. Tangent Bug
    print("\n[Running Tangent Bug]")
    t0 = time.time()
    traj_bug = tangent_bug_optimize(q_start, q_goal, obstacles, max_steps=150, kin=kin)
    t_bug = time.time() - t0

    results['Tangent Bug'] = {
        'compute_time': t_bug,
        'exec_time': len(traj_bug) * dt_step,
        'path_length': get_path_length(traj_bug),
        'clearance': get_min_clearance(traj_bug, obstacles),
        'smoothness': get_smoothness(traj_bug, dt_step)
    }

    # Print Results Table
    methods = ['Baseline', 'Bubble Strips', 'Elastic Strips', 'Tangent Bug']
    col_w = 17
    print("\n" + "=" * (22 + col_w * len(methods)))
    header = f"{'Metric':<20} |" + "|".join(f" {m:^{col_w-1}}" for m in methods)
    print(header)
    print("-" * (22 + col_w * len(methods)))

    metrics = [
        ('Path Length (rad)', 'path_length', '.2f'),
        ('Exec Time (s)', 'exec_time', '.1f'),
        ('Compute Time (s)', 'compute_time', '.3f'),
        ('Min Clearance (m)', 'clearance', '.3f'),
        ('Avg Jerk', 'smoothness', '.1f')
    ]

    for label, key, fmt in metrics:
        vals = []
        for m in methods:
            v = format(results[m][key], fmt)
            if key == 'clearance':
                v = v + (" (X)" if results[m][key] < 0 else " (OK)")
            vals.append(v)
        row = f"{label:<20} |" + "|".join(f" {v:^{col_w-1}}" for v in vals)
        print(row)

if __name__ == "__main__":
    run_benchmark()

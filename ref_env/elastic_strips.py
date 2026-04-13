#!/usr/bin/env python3
"""
Elastic Strips: Real-Time Reactive Obstacle Avoidance for 6-DOF Arms
=====================================================================

Based on: Brock, O., & Khatib, O. (2002). "Elastic Strips: A Framework
for Motion Generation in Human Environments." IJRR 21(12), 1031-1052.

This module takes a pre-planned trajectory (e.g. from STOMP) and treats
it as a physical rubber band. When dynamic obstacles intrude, the band
stretches away smoothly using:
  - Internal Forces: Spring tension between adjacent waypoints (smoothness)
  - External Forces: Workspace repulsion translated to joint-space via J^T

Adapted from deprecated/elastic_strips.py with multi-primitive obstacle
support (box, cylinder, sphere) to match refuel_world_v2.sdf.
"""
import sys
import os
import numpy as np

# ── IK-Geo import (local ref_env) ───────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ik_geometric import rot, fwd_kinematics, KIN_KR6_R700, KIN_UR5  # type: ignore

# ── Kinematics Constants ─────────────────────────────────────────
KIN = KIN_KR6_R700
H_AXES = KIN['H']
P_VECS = KIN['P']

JOINT_LIMITS = np.array([
    [-2.967060285,  2.967060285],
    [-3.316125571,  0.785398163],
    [-2.094395100,  2.722713630],
    [-3.228859113,  3.228859113],
    [-2.094395100,  2.094395100],
    [-6.108652375,  6.108652375],
])

# Links to check for collisions (indices into the 6 joints)
CHECK_JOINTS = [0, 1, 2, 3, 5]


def set_kinematics(kin_dict, limits=None):
    """Inject platform-specific kinematics before running the module."""
    global KIN, H_AXES, P_VECS, JOINT_LIMITS
    KIN = kin_dict
    H_AXES = KIN['H']
    P_VECS = KIN['P']
    if limits is not None:
        JOINT_LIMITS = np.array(limits)


# ═══════════════════════════════════════════════════════════════════
#  Core FK & Jacobian
# ═══════════════════════════════════════════════════════════════════

def fk_checkpoints(q):
    """
    Run Forward Kinematics for a 6-DOF config and return Cartesian
    positions of key arm checkpoints and midpoints.
    Returns list of (joint_index, position_3d) tuples.
    """
    R = np.eye(3)
    p = P_VECS[:, 0].copy()
    raw_pts = []
    for j in range(6):
        R = R @ rot(H_AXES[:, j], q[j])
        p = p + R @ P_VECS[:, j + 1]
        if j in CHECK_JOINTS:
            raw_pts.append((j, p.copy()))

    points = []
    for k in range(len(raw_pts) - 1):
        jA, pA = raw_pts[k]
        jB, pB = raw_pts[k + 1]
        points.append((jA, pA))
        points.append((jB, 0.5 * (pA + pB)))
    points.append(raw_pts[-1])

    return points


def fk_segment_endpoints(q):
    """FK endpoints for continuous collision modeling (matches bubble_strips)."""
    R = np.eye(3); p = P_VECS[:, 0].copy()
    pts = [p.copy()]
    for j in range(6):
        R = R @ rot(H_AXES[:, j], q[j])
        p = p + R @ P_VECS[:, j + 1]
        pts.append(p.copy())

    NOZZLE_LENGTH = 0.15
    is_kuka = H_AXES[2, 0] < 0
    tool_axis = R[:, 0] if is_kuka else R[:, 1]
    p_tip = pts[6] + tool_axis * NOZZLE_LENGTH
    pts.append(p_tip)
    return pts


def numerical_jacobian(q, link_idx, eps=1e-6):
    """
    Compute the 3x6 positional Jacobian for a specific link checkpoint
    using finite differences.
    """
    J = np.zeros((3, 6))
    _, p0 = _fk_single_link(q, link_idx)
    for j in range(6):
        q_plus = q.copy()
        q_plus[j] += eps
        _, p_plus = _fk_single_link(q_plus, link_idx)
        J[:, j] = (p_plus - p0) / eps
    return J


def _fk_single_link(q, target_link_idx):
    """FK up to a specific joint index, returning (R, p)."""
    R = np.eye(3)
    p = P_VECS[:, 0].copy()
    for j in range(6):
        R = R @ rot(H_AXES[:, j], q[j])
        p = p + R @ P_VECS[:, j + 1]
        if j == target_link_idx:
            return R, p.copy()
    return R, p.copy()


# ═══════════════════════════════════════════════════════════════════
#  Multi-Primitive Distance Functions
# ═══════════════════════════════════════════════════════════════════

def _closest_point_on_box(pt, center, size, yaw):
    """Return the closest point on an oriented box surface to pt, and the distance."""
    R_rot = np.array([
        [np.cos(-yaw), -np.sin(-yaw), 0],
        [np.sin(-yaw),  np.cos(-yaw), 0],
        [0, 0, 1]])
    local = R_rot @ (pt - center)
    half = np.asarray(size) / 2.0
    clamped = np.clip(local, -half, half)
    # Closest point in world frame
    R_inv = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0, 0, 1]])
    closest_world = center + R_inv @ clamped
    dist = np.linalg.norm(pt - closest_world)
    return closest_world, dist


def _closest_point_on_cylinder(pt, center, radius, height):
    """Return the closest point on a vertical cylinder surface to pt, and the distance."""
    # Project onto XY plane relative to center
    dx = pt[0] - center[0]
    dy = pt[1] - center[1]
    dz = pt[2] - center[2]
    d_xy = np.sqrt(dx**2 + dy**2)

    # Clamp to cylinder surface
    if d_xy < 1e-8:
        cx, cy = center[0] + radius, center[1]
    else:
        scale = min(radius, d_xy) / d_xy
        cx = center[0] + dx * scale
        cy = center[1] + dy * scale
    cz = center[2] + np.clip(dz, -height / 2, height / 2)

    closest = np.array([cx, cy, cz])
    dist = np.linalg.norm(pt - closest)
    return closest, dist


def _closest_point_on_sphere(pt, center, radius):
    """Return the closest point on sphere surface to pt, and the distance."""
    vec = pt - center
    d = np.linalg.norm(vec)
    if d < 1e-8:
        closest = center + np.array([radius, 0, 0])
    else:
        closest = center + vec / d * radius
    return closest, max(0.0, d - radius)


def point_obstacle_distance(pt, obs):
    """Compute distance and repulsion direction from a point to a multi-primitive obstacle.

    Parameters
    ----------
    pt : (3,) array — point in workspace
    obs : (type, center, dims, yaw) tuple

    Returns
    -------
    dist : float — signed distance (negative = penetration)
    direction : (3,) array — unit vector pointing AWAY from obstacle
    """
    obs_type, center, dims, yaw = obs

    if obs_type == 'sphere':
        closest, dist = _closest_point_on_sphere(pt, center, dims)
    elif obs_type == 'box':
        closest, dist = _closest_point_on_box(pt, center, dims, yaw)
    elif obs_type == 'cylinder':
        closest, dist = _closest_point_on_cylinder(pt, center, dims[0], dims[1])
    else:
        return 1.0, np.array([0.0, 0.0, 1.0])

    vec = pt - closest
    d = np.linalg.norm(vec)
    if d > 1e-8:
        direction = vec / d
    else:
        direction = np.array([0.0, 0.0, 1.0])

    return dist, direction


# ═══════════════════════════════════════════════════════════════════
#  Clearance (matches bubble_strips API for benchmark)
# ═══════════════════════════════════════════════════════════════════

def clearance(q, obstacles):
    """Compute minimum workspace clearance across all arm segments and obstacles."""
    pts = fk_segment_endpoints(q)
    min_rho = float('inf')
    for obs in obstacles:
        for k in range(len(pts) - 1):
            # Sample along segment
            for alpha in [0, 0.25, 0.5, 0.75, 1.0]:
                p = pts[k] + alpha * (pts[k + 1] - pts[k])
                d, _ = point_obstacle_distance(p, obs)
                if d < min_rho:
                    min_rho = d
    return min_rho


# ═══════════════════════════════════════════════════════════════════
#  Force Calculations
# ═══════════════════════════════════════════════════════════════════

def internal_force(q_prev, q_curr, q_next, k_contraction=1.0):
    """
    Spring-like contraction force pulling waypoint toward its neighbors.
    F_int = k_c * (q_{i-1} + q_{i+1} - 2*q_i)
    This is a discrete Laplacian — it smooths the trajectory.
    """
    return k_contraction * (q_prev + q_next - 2.0 * q_curr)


def external_force(q, obstacles, safety_margin=0.25, k_repulsion=1.0):
    """
    Compute the total repulsive force in joint-space from all obstacles.

    For each checkpoint on the arm:
      1. Calculate the Cartesian repulsion vector (away from obstacle)
      2. Translate to joint-space torques via Jacobian Transpose: tau = J^T . F_ws

    Supports box, cylinder, and sphere obstacles.

    Returns a 6-DOF joint-space force vector.
    """
    tau_total = np.zeros(6)

    checkpoints = fk_checkpoints(q)
    for link_idx, pt in checkpoints:
        for obs in obstacles:
            dist, direction = point_obstacle_distance(pt, obs)
            penetration = safety_margin - dist

            if penetration > 0:
                magnitude = k_repulsion * (penetration / safety_margin) ** 2
                F_workspace = magnitude * direction

                J = numerical_jacobian(q, link_idx)
                tau_total += J.T @ F_workspace

    return tau_total


# ═══════════════════════════════════════════════════════════════════
#  Elastic Strips Optimizer
# ═══════════════════════════════════════════════════════════════════

def elastic_strip_deform(
    trajectory,
    obstacles,
    joint_limits=None,
    n_iterations=100,
    alpha=0.02,
    k_contraction=2.0,
    k_repulsion=5.0,
    safety_margin=0.25,
    damping=0.95,
    verbose=True,
):
    """
    Deform a pre-planned trajectory to avoid multi-primitive obstacles.

    Parameters
    ----------
    trajectory : np.ndarray, shape (N, 6)
    obstacles : list of (type, center, dims, yaw)
    joint_limits : np.ndarray, shape (6, 2), optional
    n_iterations : int
    alpha : float — step size
    k_contraction : float — internal spring stiffness
    k_repulsion : float — external repulsion gain
    safety_margin : float — repulsion activation distance (m)
    damping : float — velocity damping (0-1)
    verbose : bool

    Returns
    -------
    deformed : np.ndarray, shape (N, 6)
    history : dict with 'min_clearance' list
    stats : dict with summary statistics
    """
    if joint_limits is None:
        joint_limits = JOINT_LIMITS

    N = len(trajectory)
    deformed = trajectory.copy()
    velocities = np.zeros_like(deformed)

    q_start = deformed[0].copy()
    q_goal = deformed[-1].copy()

    history = {'min_clearance': []}
    total_jacobian_calls = 0

    for it in range(n_iterations):
        min_dist_iter = float('inf')

        for i in range(1, N - 1):
            F_int = internal_force(
                deformed[i - 1], deformed[i], deformed[i + 1],
                k_contraction=k_contraction)

            F_ext = external_force(
                deformed[i], obstacles,
                safety_margin=safety_margin,
                k_repulsion=k_repulsion)

            if np.linalg.norm(F_ext) > 0:
                total_jacobian_calls += len(fk_checkpoints(deformed[i]))

            total_force = F_int + F_ext
            velocities[i] = damping * velocities[i] + alpha * total_force
            deformed[i] = deformed[i] + velocities[i]

            for j in range(6):
                deformed[i, j] = np.clip(
                    deformed[i, j], joint_limits[j, 0], joint_limits[j, 1])

            rho = clearance(deformed[i], obstacles)
            if rho < min_dist_iter:
                min_dist_iter = rho

        deformed[0] = q_start
        deformed[-1] = q_goal

        history['min_clearance'].append(min_dist_iter)

        if verbose and (it % 20 == 0 or it == n_iterations - 1):
            print(f"  [Elastic iter {it:3d}/{n_iterations}] "
                  f"min_dist={min_dist_iter:.4f}m")

    stats = {
        'initial_waypoints': N,
        'final_waypoints': N,
        'output_waypoints': N,
        'total_insertions': 0,
        'total_deletions': 0,
        'total_jacobian_calls': total_jacobian_calls,
        'min_clearance': min(history['min_clearance']) if history['min_clearance'] else 0.0,
        'final_min_clearance': history['min_clearance'][-1] if history['min_clearance'] else 0.0,
    }

    return deformed, history, stats


# ═══════════════════════════════════════════════════════════════════
#  Visualization
# ═══════════════════════════════════════════════════════════════════

def plot_elastic_comparison(original_traj, deformed_traj, obstacles, history,
                            save_path="output_graphs/elastic_strips_analysis.png"):
    """Generate a 3-panel analysis comparing original vs deformed trajectories."""
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    N = len(original_traj)
    steps = np.arange(1, N + 1)

    # Panel 1: Joint Angle Comparison
    ax = axs[0]
    j_labels = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6']
    colors = plt.cm.tab10(np.linspace(0, 1, 6))
    for j in range(6):
        ax.plot(steps, original_traj[:, j], '--', color=colors[j], alpha=0.4,
                label=f'{j_labels[j]} (STOMP)')
        ax.plot(steps, deformed_traj[:, j], '-', color=colors[j], linewidth=2,
                label=f'{j_labels[j]} (Elastic)')
    ax.set_title("Joint Angles: STOMP vs Elastic", fontweight='bold')
    ax.set_xlabel("Waypoint")
    ax.set_ylabel("Radians")
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[6:], [l.replace(' (Elastic)', '') for l in labels[6:]],
              fontsize=8, loc='upper right', title='Elastic')

    # Panel 2: Min Distance Over Iterations
    ax = axs[1]
    mc = history['min_clearance']
    ax.plot(mc, 'b-', linewidth=2)
    ax.axhline(0, color='r', linestyle=':', linewidth=1.5, label='Collision')
    ax.fill_between(range(len(mc)), mc, 0,
                     where=(np.array(mc) < 0), color='red', alpha=0.3)
    ax.set_title("Elastic Convergence: Min Clearance", fontweight='bold')
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Distance (m)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Panel 3: 3D EE Path
    ax = fig.add_subplot(1, 3, 3, projection='3d')
    axs[2].remove()

    orig_ee, def_ee = [], []
    for i in range(N):
        _, p_orig = fwd_kinematics(original_traj[i], kin=KIN)
        _, p_def = fwd_kinematics(deformed_traj[i], kin=KIN)
        orig_ee.append(p_orig)
        def_ee.append(p_def)
    orig_ee = np.array(orig_ee)
    def_ee = np.array(def_ee)

    ax.plot(orig_ee[:, 0], orig_ee[:, 1], orig_ee[:, 2],
            'r--', linewidth=1.5, alpha=0.5, label='STOMP (Original)')
    ax.plot(def_ee[:, 0], def_ee[:, 1], def_ee[:, 2],
            'b-', linewidth=2.5, label='Elastic (Deformed)')

    for obs in obstacles:
        obs_type, center, dims, yaw = obs
        if obs_type == 'sphere':
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 15)
            x = center[0] + dims * np.outer(np.cos(u), np.sin(v))
            y = center[1] + dims * np.outer(np.sin(u), np.sin(v))
            z = center[2] + dims * np.outer(np.ones_like(u), np.cos(v))
            ax.plot_surface(x, y, z, alpha=0.25, color='red')

    ax.set_title("3D EE Path", fontweight='bold')
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    print(f"\n  Elastic Strips analysis saved to {save_path}")


# ═══════════════════════════════════════════════════════════════════
#  Standalone Test
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("  Elastic Strips — Multi-Primitive Test")
    print("  Brock & Khatib (2002) + KR6 R700 FK + J^T")
    print("=" * 65)

    Q_START = np.array([0.785, -0.94, 0.94, 0.0, 0.0, 0.0])
    Q_GOAL = np.array([-0.4911, -0.7409, 0.9101, -0.0578, 1.5099, -0.4844])

    N_WP = 30
    base_traj = np.zeros((N_WP, 6))
    for i in range(N_WP):
        t = i / (N_WP - 1)
        base_traj[i] = Q_START + t * (Q_GOAL - Q_START)

    obstacles = [
        ('sphere', np.array([0.52, 0.05, 0.45]), 0.12, 0.0),
    ]

    print(f"\n[Setup] Base trajectory: {N_WP} waypoints (C-Space LERP)")
    print(f"[Setup] Obstacle: sphere center={obstacles[0][1]}, r={obstacles[0][2]}")

    print("\n[Pre-Deform] Checking initial trajectory...")
    initial_collisions = sum(1 for q in base_traj if clearance(q, obstacles) < 0)
    print(f"  {initial_collisions}/{N_WP} waypoints penetrating obstacles")

    print("\n[Elastic Strips] Running reactive deformation...")
    deformed, history, stats = elastic_strip_deform(
        base_traj, obstacles,
        n_iterations=150, alpha=0.015,
        k_contraction=2.0, k_repulsion=8.0,
        safety_margin=0.20, damping=0.92, verbose=True)

    print("\n[Post-Deform] Checking deformed trajectory...")
    post_collisions = sum(1 for q in deformed if clearance(q, obstacles) < 0)
    print(f"  {post_collisions}/{N_WP} waypoints penetrating obstacles")

    print(f"\n[Stats]")
    print(f"  Min clearance:       {stats['min_clearance']:.4f}m")
    print(f"  Final min clearance: {stats['final_min_clearance']:.4f}m")
    print(f"  Jacobian calls:      {stats['total_jacobian_calls']}")

    plot_elastic_comparison(base_traj, deformed, obstacles, history)

    print("\n" + "=" * 65)
    print("  Elastic Strips test complete!")
    print("=" * 65)

#!/usr/bin/env python3
"""
Bubble Strips: Elastic Bands with Free-Space Bubbles for 6-DOF Arms
====================================================================

Based on: Quinlan, S. & Khatib, O. (1993). "Elastic Bands: Connecting
Path Planning and Control." IEEE ICRA, pp. 802-807.

Each waypoint on the band carries a "bubble" — a C-space free region
whose radius equals the workspace clearance scaled by the lever-arm
metric.  Waypoints that remain inside their bubble are guaranteed
collision-free and skip expensive FK / distance recomputation.

Key differences from Brock & Khatib (2002) Elastic Strips:
  - Bubbles are C-space free regions (not workspace spheres)
  - Forces computed directly in C-space via clearance gradient
  - Normalized contraction springs (not raw Laplacian)
  - Dynamic waypoint insertion/deletion to maintain overlap
  - Adaptive step size proportional to bubble radius
  - Tangential force removal to prevent oscillation

Standalone implementation for the KUKA KR6 R700.
"""
import sys
import os
import time
import numpy as np

# ── IK-Geo import ────────────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), 'kuka_refuel_ws', 'src',
    'kuka_kr6_gazebo', 'scripts')))
try:
    import ik_geometric as ik
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', 'kuka_refuel_ws', 'src',
        'kuka_kr6_gazebo', 'scripts')))
    import ik_geometric as ik


# ═══════════════════════════════════════════════════════════════════
#  Kinematics Constants (KUKA KR 8 R2100 arc HW)
# ═══════════════════════════════════════════════════════════════════

KIN = ik.KIN_KR8_R2100
H_AXES = KIN['H']
P_VECS = KIN['P']
rot = ik.rot  # Rodrigues rotation

# KR8 R2100 joint limits (from URDF xacro)
JOINT_LIMITS = np.array([
    [-3.228859,  3.228859],   # A1: ±185°
    [-3.228859,  1.134464],   # A2: -185° / 65°
    [-2.408554,  3.054326],   # A3: -138° / 175°
    [-2.879793,  2.879793],   # A4: ±165°
    [-2.007129,  2.443461],   # A5: -115° / 140°
    [-6.108652,  6.108652],   # A6: ±350°
])


# Links to check for collisions (matches STOMP dense segments)
CHECK_JOINTS = [0, 1, 2, 3, 5]

# Placeholder for precomputed globals
LEVER_ARMS = None
L_MAX = 0.0


# ═══════════════════════════════════════════════════════════════════
#  FK & Clearance
# ═══════════════════════════════════════════════════════════════════

def fk_checkpoints(q):
    """
    Forward Kinematics for a 6-DOF config. Returns Cartesian positions
    of key arm checkpoints and midpoints, mirroring STOMP collision segments.
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
        jB, pB = raw_pts[k+1]
        points.append((jA, pA))
        # midpoint theoretically belongs to the distal joint jB
        points.append((jB, 0.5 * (pA + pB)))
    points.append(raw_pts[-1])
    
    return points


def fk_segment_endpoints(q):
    """
    Forward Kinematics endpoints for continuous collision modeling.

    Returns Cartesian positions at joints in `CHECK_JOINTS` order:
      [shoulder, elbow, wrist1, wrist2, ee]
    """
    R = np.eye(3)
    p = P_VECS[:, 0].copy()
    raw_pts = []
    for j in range(6):
        R = R @ rot(H_AXES[:, j], q[j])
        p = p + R @ P_VECS[:, j + 1]
        if j in CHECK_JOINTS:
            raw_pts.append(p.copy())
    return raw_pts


def _pt_to_segment_dist(pA: np.ndarray, pB: np.ndarray, pC: np.ndarray) -> float:
    """Shortest distance between line segment AB and point C."""
    AB = pB - pA
    AC = pC - pA
    ab_sq = np.dot(AB, AB)
    if ab_sq < 1e-8:
        return float(np.linalg.norm(AC))
    t = np.clip(np.dot(AC, AB) / ab_sq, 0.0, 1.0)
    closest = pA + t * AB
    return float(np.linalg.norm(pC - closest))


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


def clearance(q, obstacles):
    """
    Compute clearance rho(q) using the same *continuous segment-to-sphere*
    distance model as STOMP's `simple_obstacle_cost`.

    rho(q) = min_{segments,obstacles} (dist(point, segment) - radius)
      Positive => free, negative => penetrating.
    """
    seg_pts = fk_segment_endpoints(q)  # 5 endpoints => 4 continuous segments
    min_rho = float('inf')
    for obs_center, obs_radius in obstacles:
        obs_center = np.asarray(obs_center, dtype=float)
        for k in range(len(seg_pts) - 1):
            d = _pt_to_segment_dist(seg_pts[k], seg_pts[k + 1], obs_center)
            surf_dist = d - float(obs_radius)
            if surf_dist < min_rho:
                min_rho = surf_dist
    return min_rho


def clearance_gradient(q, obstacles, h=1e-4):
    """
    C-space gradient of the clearance function via central finite differences.
    Returns a 6-DOF vector pointing in the direction of increasing clearance.
    Requires 12 FK + distance evaluations (2 per joint).
    """
    grad = np.zeros(6)
    for j in range(6):
        q_plus = q.copy()
        q_plus[j] += h
        q_minus = q.copy()
        q_minus[j] -= h
        grad[j] = (clearance(q_plus, obstacles) - clearance(q_minus, obstacles)) / (2.0 * h)
    return grad


def _compute_lever_arms():
    """
    Compute max workspace displacement per radian for each joint.
    L_j = max displacement of any checked link when joint j rotates by eps.
    Evaluated at the home configuration (conservative estimate).
    """
    q0 = np.zeros(6)
    eps = 1e-4
    L = np.zeros(6)
    pts_0 = [p for _, p in fk_checkpoints(q0)]
    for j in range(6):
        q_pert = q0.copy()
        q_pert[j] += eps
        pts_pert = [p for _, p in fk_checkpoints(q_pert)]
        max_disp = max(np.linalg.norm(p1 - p0)
                       for p0, p1 in zip(pts_0, pts_pert))
        L[j] = max_disp / eps
    return L


    return L


# Precompute globals
def set_kinematics(kin_dict, limits=None):
    """Inject platform-specific kinematics before running the module."""
    global KIN, H_AXES, P_VECS, JOINT_LIMITS, LEVER_ARMS, L_MAX
    KIN = kin_dict
    H_AXES = KIN['H']
    P_VECS = KIN['P']
    if limits is not None:
        JOINT_LIMITS = np.array(limits)
        
    LEVER_ARMS = _compute_lever_arms()
    L_MAX = float(np.max(LEVER_ARMS))

# Initialize defaults at module load
set_kinematics(ik.KIN_KR6_R700, JOINT_LIMITS)


def cspace_displacement(dq):
    """
    Weighted C-space displacement metric (Quinlan & Khatib Sec. 5):
      D(dq) = sum_j |dq_j| * L_j
    Bounds the maximum workspace displacement of any robot point.
    """
    return float(np.sum(np.abs(dq) * LEVER_ARMS))


# ═══════════════════════════════════════════════════════════════════
#  Bubble Data Structure
# ═══════════════════════════════════════════════════════════════════

class Bubble:
    """
    A free-space sphere in C-space around a configuration.

    Attributes
    ----------
    config : np.ndarray (6,)
        Joint configuration at the center of the bubble.
    rho : float
        Workspace clearance rho(q) — min distance from any link to obstacle.
    rho_cspace : float
        Conservative C-space bubble radius = rho / L_MAX.
    """
    __slots__ = ['config', 'rho', 'rho_cspace']

    def __init__(self, config, rho):
        self.config = config.copy()
        self.rho = rho
        self.rho_cspace = max(rho / L_MAX, 0.0) if L_MAX > 0 else 0.0

    def __repr__(self):
        return (f"Bubble(rho={self.rho:.4f}, r_cs={self.rho_cspace:.4f})")


def compute_bubble(q, obstacles):
    """Create a Bubble at configuration q."""
    rho = clearance(q, obstacles)
    return Bubble(q, rho)


# ═══════════════════════════════════════════════════════════════════
#  Band Operations (Insert / Delete / Overlap)
# ═══════════════════════════════════════════════════════════════════

def bubbles_overlap(b_i, b_j):
    """
    Check whether two consecutive bubbles overlap in C-space.
    Overlap guarantees a collision-free straight-line path between them.
    Uses the weighted displacement metric D(dq) for a tighter bound.
    """
    dq = b_i.config - b_j.config
    dist = cspace_displacement(dq)
    return dist < (b_i.rho + b_j.rho)


def insert_bubble(band, idx, obstacles):
    """
    Insert a new bubble at the midpoint between band[idx] and band[idx+1].
    Returns the updated band (list of Bubbles).
    """
    q_mid = 0.5 * (band[idx].config + band[idx + 1].config)
    new_bubble = compute_bubble(q_mid, obstacles)
    band.insert(idx + 1, new_bubble)
    return band


def remove_redundant(band, min_band_length=5):
    """
    Remove bubbles whose neighbors overlap each other (the bubble is
    redundant for maintaining the collision-free invariant).
    Never removes the first or last bubble (pinned endpoints).
    Maintains at least min_band_length waypoints.
    Returns the updated band.
    """
    i = 1
    while i < len(band) - 1:
        if len(band) <= min_band_length:
            break
        if bubbles_overlap(band[i - 1], band[i + 1]):
            band.pop(i)
        else:
            i += 1
    return band


def maintain_overlap(band, obstacles, max_insertions_per_pass=50):
    """
    Ensure consecutive bubbles overlap. Insert midpoints where gaps exist,
    then remove redundant bubbles.

    Parameters
    ----------
    band : list of Bubble
    obstacles : list of (center, radius)
    max_insertions_per_pass : int
        Safety limit to prevent infinite insertion loops.

    Returns
    -------
    band : list of Bubble
    n_inserted : int
    n_deleted : int
    """
    n_inserted = 0
    n_deleted_before = len(band)

    # Pass 1: insert where overlap is broken
    i = 0
    while i < len(band) - 1 and n_inserted < max_insertions_per_pass:
        if not bubbles_overlap(band[i], band[i + 1]):
            band = insert_bubble(band, i, obstacles)
            n_inserted += 1
            # Don't advance i — check the newly inserted bubble next
        else:
            i += 1

    # Pass 2: remove redundant bubbles
    band = remove_redundant(band)
    n_deleted = n_deleted_before + n_inserted - len(band)

    return band, n_inserted, n_deleted


def resample_band(band, n_output):
    """
    Resample a sparse band of Bubbles into a dense, uniformly-spaced
    trajectory by linear interpolation in C-space.

    Parameters
    ----------
    band : list of Bubble
        The optimized (potentially sparse) band.
    n_output : int
        Desired number of output waypoints.

    Returns
    -------
    trajectory : np.ndarray, shape (n_output, 6)
    """
    configs = np.array([b.config for b in band])
    n_band = len(configs)

    if n_band >= n_output:
        # Subsample uniformly
        indices = np.linspace(0, n_band - 1, n_output).astype(int)
        return configs[indices]

    # Compute cumulative arc length in C-space
    arc = np.zeros(n_band)
    for i in range(1, n_band):
        arc[i] = arc[i - 1] + np.linalg.norm(configs[i] - configs[i - 1])

    if arc[-1] < 1e-12:
        return np.tile(configs[0], (n_output, 1))

    # Interpolate at uniform arc-length spacing
    target_arc = np.linspace(0, arc[-1], n_output)
    trajectory = np.zeros((n_output, 6))
    for k in range(n_output):
        s = target_arc[k]
        # Find the segment containing s
        idx = np.searchsorted(arc, s, side='right') - 1
        idx = np.clip(idx, 0, n_band - 2)
        seg_len = arc[idx + 1] - arc[idx]
        if seg_len < 1e-12:
            t = 0.0
        else:
            t = (s - arc[idx]) / seg_len
        trajectory[k] = (1 - t) * configs[idx] + t * configs[idx + 1]

    return trajectory


# ═══════════════════════════════════════════════════════════════════
#  Force Calculations (Quinlan & Khatib 1993, Section 6)
# ═══════════════════════════════════════════════════════════════════

def contraction_force(band, i, k_c=1.0):
    """
    Normalized spring contraction force (Eq. from Section 6, p.806):
      f_c = k_c * (unit(q_{i-1} - q_i) + unit(q_{i+1} - q_i))

    Unlike the raw Laplacian, this produces uniform tension regardless
    of waypoint spacing.
    """
    q_prev = band[i - 1].config
    q_curr = band[i].config
    q_next = band[i + 1].config

    d_prev = q_prev - q_curr
    d_next = q_next - q_curr

    norm_prev = np.linalg.norm(d_prev)
    norm_next = np.linalg.norm(d_next)

    f = np.zeros(6)
    if norm_prev > 1e-10:
        f += d_prev / norm_prev
    if norm_next > 1e-10:
        f += d_next / norm_next

    return k_c * f


def repulsion_force(q, obstacles, rho, rho_0, k_r=1.0, h=1e-4):
    """
    C-space repulsive force (Eq. from Section 6, p.806):
      f_r = k_r * (rho_0 - rho) * d_rho/dq    if rho < rho_0
      f_r = 0                                    if rho >= rho_0

    Parameters
    ----------
    q : np.ndarray (6,)
    obstacles : list of (center, radius)
    rho : float
        Current clearance (already computed for the bubble).
    rho_0 : float
        Distance threshold — repulsion activates below this.
    k_r : float
        Repulsion gain.
    h : float
        Step size for finite-difference gradient.
    """
    if rho >= rho_0:
        return np.zeros(6)

    grad = clearance_gradient(q, obstacles, h)
    return k_r * (rho_0 - rho) * grad


def remove_tangential(force, q_prev, q_next):
    """
    Remove the tangential component of the force along the band direction
    (Section 6, p.806). This prevents oscillatory migration of bubbles
    along the elastic band.

      t = q_{i-1} - q_{i+1}
      f* = f - (f . t) * t / ||t||^2
    """
    t = q_prev - q_next
    t_norm_sq = np.dot(t, t)
    if t_norm_sq < 1e-16:
        return force
    return force - (np.dot(force, t) / t_norm_sq) * t


# ═══════════════════════════════════════════════════════════════════
#  Main Optimizer
# ═══════════════════════════════════════════════════════════════════

def bubble_strip_deform(
    trajectory,
    obstacles,
    joint_limits=None,
    n_iterations=100,
    k_contraction=0.5,
    k_repulsion=30.0,
    rho_0=0.25,
    damping=0.85,
    verbose=True,
):
    """
    Deform a trajectory using elastic bands with bubbles.

    Parameters
    ----------
    trajectory : np.ndarray, shape (N, 6)
        Initial trajectory (e.g. from STOMP or C-space LERP).
    obstacles : list of (center_xyz, radius)
        Spherical obstacles to avoid.
    joint_limits : np.ndarray, shape (6, 2), optional
        Joint limits. Defaults to KUKA KR6 R700.
    n_iterations : int
        Maximum number of deformation iterations.
    k_contraction : float
        Normalized spring stiffness.
    k_repulsion : float
        Repulsion gain for C-space gradient force.
    rho_0 : float
        Repulsion activation distance (meters).
    damping : float
        Velocity damping factor (0-1).
    verbose : bool
        Print progress every 20 iterations.

    Returns
    -------
    deformed : np.ndarray, shape (M, 6)
        Deformed trajectory. M may differ from N due to insertion/deletion.
    history : dict
        'min_clearance': list of float per iteration
        'band_length': list of int per iteration
    stats : dict
        Summary statistics (insertions, deletions, clearance calls, etc.)
    """
    if joint_limits is None:
        joint_limits = JOINT_LIMITS

    # ── Initialize band of bubbles ──
    band = [compute_bubble(trajectory[i], obstacles)
            for i in range(len(trajectory))]

    # Pin start and end
    q_start = trajectory[0].copy()
    q_goal = trajectory[-1].copy()

    # Velocity buffer (keyed by id, reset on insert/delete)
    velocities = {id(b): np.zeros(6) for b in band}

    history = {'min_clearance': [], 'band_length': []}
    total_insertions = 0
    total_deletions = 0
    total_clearance_calls = len(band)  # initial bubble computation

    for it in range(n_iterations):
        min_rho_iter = float('inf')

        # ── Deform interior bubbles ──
        for i in range(1, len(band) - 1):
            b = band[i]

            # Contraction force (always computed — cheap)
            f_c = contraction_force(band, i, k_contraction)

            # Repulsion force (uses cached rho; gradient needs FK calls)
            f_r = repulsion_force(
                b.config, obstacles, b.rho, rho_0, k_repulsion
            )
            if np.linalg.norm(f_r) > 0:
                total_clearance_calls += 12  # 12 FK evals for gradient

            # Remove tangential component
            f_total = remove_tangential(
                f_c + f_r,
                band[i - 1].config,
                band[i + 1].config,
            )

            # Adaptive step size: alpha proportional to rho, clamped
            alpha = np.clip(b.rho, 0.002, 0.015)

            # Physics integration with damping
            vel = velocities.get(id(b), np.zeros(6))
            vel = damping * vel + alpha * f_total
            new_config = b.config + vel

            # Joint limit clamping
            for j in range(6):
                new_config[j] = np.clip(
                    new_config[j], joint_limits[j, 0], joint_limits[j, 1])

            # Recompute bubble at new position
            new_bubble = compute_bubble(new_config, obstacles)
            total_clearance_calls += 1

            # Track min clearance
            if new_bubble.rho < min_rho_iter:
                min_rho_iter = new_bubble.rho

            # Update band
            band[i] = new_bubble
            velocities[id(new_bubble)] = vel

        # ── Re-pin endpoints ──
        band[0] = compute_bubble(q_start, obstacles)
        band[-1] = compute_bubble(q_goal, obstacles)

        # ── Maintain overlap invariant (every 10 iterations) ──
        n_ins, n_del = 0, 0
        if it % 10 == 0 or it == n_iterations - 1:
            max_band = max(3 * len(trajectory), 150)
            cap = max(max_band - len(band), 0)
            band, n_ins, n_del = maintain_overlap(
                band, obstacles, max_insertions_per_pass=min(15, cap))
            total_insertions += n_ins
            total_deletions += n_del
            total_clearance_calls += n_ins

        # Reset velocity map for new bubbles
        for b in band:
            if id(b) not in velocities:
                velocities[id(b)] = np.zeros(6)

        # ── Record history ──
        if min_rho_iter == float('inf'):
            # All interior bubbles were safe — use endpoint min
            min_rho_iter = min(b.rho for b in band)
        history['min_clearance'].append(min_rho_iter)
        history['band_length'].append(len(band))

        if verbose and (it % 20 == 0 or it == n_iterations - 1):
            print(f"  [Bubble iter {it:3d}/{n_iterations}] "
                  f"band={len(band):3d} wp, "
                  f"min_rho={min_rho_iter:.4f}m, "
                  f"ins={n_ins}, del={n_del}")

    # ── Build output trajectory ──
    # Resample to match input density (or at least original count)
    n_out = max(len(trajectory), len(band))
    deformed = resample_band(band, n_out)

    stats = {
        'initial_waypoints': len(trajectory),
        'final_waypoints': len(band),
        'output_waypoints': n_out,
        'total_insertions': total_insertions,
        'total_deletions': total_deletions,
        'total_clearance_calls': total_clearance_calls,
        'min_clearance': min(history['min_clearance']),
        'final_min_clearance': history['min_clearance'][-1],
    }

    return deformed, history, stats


# ═══════════════════════════════════════════════════════════════════
#  Visualization
# ═══════════════════════════════════════════════════════════════════

def plot_bubble_analysis(original_traj, deformed_traj, obstacles,
                         history, stats,
                         save_path="output_graphs/bubble_strips_analysis.png"):
    """Generate a 3-panel analysis of the bubble strip deformation."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(18, 5))

    # ── Panel 1: 3D EE Path ──
    ax = fig.add_subplot(1, 3, 1, projection='3d')

    orig_ee = []
    for q in original_traj:
        _, p = ik.fwd_kinematics(q)
        orig_ee.append(p)
    orig_ee = np.array(orig_ee)

    def_ee = []
    for q in deformed_traj:
        _, p = ik.fwd_kinematics(q)
        def_ee.append(p)
    def_ee = np.array(def_ee)

    ax.plot(orig_ee[:, 0], orig_ee[:, 1], orig_ee[:, 2],
            'r--', linewidth=1.5, alpha=0.5, label='Original')
    ax.plot(def_ee[:, 0], def_ee[:, 1], def_ee[:, 2],
            'b-', linewidth=2.5, label='Bubble Deformed')

    # Draw obstacle spheres
    for center, radius in obstacles:
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 15)
        x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
        y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
        z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
        ax.plot_surface(x, y, z, alpha=0.25, color='red')

    ax.set_title("3D EE Path", fontweight='bold')
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend(fontsize=8)

    # ── Panel 2: Min Clearance Convergence ──
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.plot(history['min_clearance'], 'b-', linewidth=2)
    ax2.axhline(0, color='r', linestyle=':', linewidth=1.5, label='Collision')
    ax2.fill_between(range(len(history['min_clearance'])),
                     history['min_clearance'], 0,
                     where=(np.array(history['min_clearance']) < 0),
                     color='red', alpha=0.3)
    ax2.set_title("Min Clearance Over Iterations", fontweight='bold')
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Clearance (m)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # ── Panel 3: Band Length (Waypoint Count) ──
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.plot(history['band_length'], 'g-', linewidth=2)
    ax3.axhline(stats['initial_waypoints'], color='gray', linestyle='--',
                alpha=0.5, label=f"Initial ({stats['initial_waypoints']})")
    ax3.set_title("Band Length (Dynamic Waypoints)", fontweight='bold')
    ax3.set_xlabel("Iteration")
    ax3.set_ylabel("Number of Waypoints")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    print(f"\n  Plot saved to {save_path}")


# ═══════════════════════════════════════════════════════════════════
#  Standalone Test
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("  Bubble Strips — Quinlan & Khatib (1993)")
    print("  KUKA KR6 R700 | C-Space Bubbles | Dynamic Waypoints")
    print("=" * 65)

    # Same scenario as elastic_strips.py for direct comparison
    Q_START = np.array([0.785, -0.94, 0.94, 0.0, 0.0, 0.0])
    Q_GOAL = np.array([-0.4911, -0.7409, 0.9101, -0.0578, 1.5099, -0.4844])

    N_WP = 30
    base_traj = np.zeros((N_WP, 6))
    for i in range(N_WP):
        t = i / (N_WP - 1)
        base_traj[i] = Q_START + t * (Q_GOAL - Q_START)

    obstacles = [
        (np.array([0.62, -0.05, 0.62]), 0.12),  # intersects elbow at midpoint
    ]

    print(f"\n  Lever arms (m/rad): {np.round(LEVER_ARMS, 4)}")
    print(f"  L_MAX = {L_MAX:.4f} m/rad")

    print(f"\n[Setup] Base trajectory: {N_WP} waypoints (C-Space LERP)")
    for k, (c, r) in enumerate(obstacles):
        print(f"[Setup] Obstacle {k}: center={c}, r={r}")

    # ── Check initial collisions ──
    print("\n[Pre-Deform] Checking initial trajectory...")
    initial_collisions = 0
    for i, q in enumerate(base_traj):
        rho = clearance(q, obstacles)
        if rho < 0:
            initial_collisions += 1
    print(f"  {initial_collisions}/{N_WP} waypoints penetrating obstacles")

    # ── Run Bubble Strips ──
    print("\n[Bubble Strips] Running deformation...")
    t0 = time.time()
    deformed, history, stats = bubble_strip_deform(
        base_traj,
        obstacles,
        n_iterations=150,
        k_contraction=0.5,
        k_repulsion=30.0,
        rho_0=0.25,
        damping=0.85,
        verbose=True,
    )
    elapsed = time.time() - t0

    # ── Check post-deform collisions ──
    print("\n[Post-Deform] Checking deformed trajectory...")
    post_collisions = 0
    for i, q in enumerate(deformed):
        rho = clearance(q, obstacles)
        if rho < 0:
            post_collisions += 1
    print(f"  {post_collisions}/{len(deformed)} waypoints penetrating obstacles")

    # ── Print stats ──
    print(f"\n[Stats]")
    print(f"  Time elapsed:      {elapsed:.2f}s")
    print(f"  Initial waypoints: {stats['initial_waypoints']}")
    print(f"  Final waypoints:   {stats['final_waypoints']}")
    print(f"  Total insertions:  {stats['total_insertions']}")
    print(f"  Total deletions:   {stats['total_deletions']}")
    print(f"  Clearance calls:   {stats['total_clearance_calls']}")
    print(f"  Min clearance:     {stats['min_clearance']:.4f}m")
    print(f"  Final min clear:   {stats['final_min_clearance']:.4f}m")

    # ── Plot ──
    plot_bubble_analysis(base_traj, deformed, obstacles, history, stats)

    print("\n" + "=" * 65)
    print("  Bubble Strips test complete!")
    print("=" * 65)

#!/usr/bin/env python3
"""
STOMP: Stochastic Trajectory Optimization for Motion Planning
=============================================================

Reference:
  Kalakrishnan et al., "STOMP: Stochastic Trajectory Optimization for
  Motion Planning", IEEE ICRA 2011.

Standalone implementation for 6-DOF robot arms. No GPU, no training,
no external dependencies beyond NumPy.

Usage:
    from stomp_planner import stomp_optimize
    trajectory = stomp_optimize(q_start, q_goal, joint_limits)
"""
import numpy as np


def _smoothness_matrix(n: int) -> np.ndarray:
    """
    Finite-difference approximation of the acceleration operator (A = D^T D).
    Minimizing q^T A q minimizes the sum of squared accelerations.
    """
    # Second-order finite difference matrix D (n-2 x n)
    D = np.zeros((n - 2, n))
    for i in range(n - 2):
        D[i, i] = 1.0
        D[i, i + 1] = -2.0
        D[i, i + 2] = 1.0
    # A = D^T @ D  →  (n x n) positive semi-definite
    A = D.T @ D
    return A


def _joint_limit_cost(trajectory: np.ndarray, limits: np.ndarray,
                       margin: float = 0.1) -> np.ndarray:
    """
    Soft barrier cost that increases as joints approach their limits.
    Returns per-waypoint cost (N,).
    """
    N, ndof = trajectory.shape
    cost = np.zeros(N)
    for j in range(ndof):
        lo, hi = limits[j]
        rng = hi - lo
        # Normalised distance from centre (0 = centre, 1 = at limit)
        centre = (lo + hi) / 2.0
        half = rng / 2.0
        d = np.abs(trajectory[:, j] - centre) / half  # in [0, 1+]
        # Quadratic penalty when within `margin` of the limit
        thresh = 1.0 - margin
        violations = np.maximum(d - thresh, 0.0)
        cost += (violations / margin) ** 2
    return cost


def _smoothness_cost(trajectory: np.ndarray, A: np.ndarray) -> float:
    """Total smoothness cost: sum over DOFs of q_j^T A q_j."""
    cost = 0.0
    for j in range(trajectory.shape[1]):
        q = trajectory[:, j]
        cost += q @ A @ q
    return cost


def _velocity_cost(trajectory: np.ndarray) -> np.ndarray:
    """Per-waypoint velocity penalty (large joint jumps)."""
    diffs = np.diff(trajectory, axis=0)
    # Sum of squared joint velocities at each interior waypoint
    vel = np.sum(diffs ** 2, axis=1)
    # Pad to match trajectory length
    return np.concatenate([[0.0], vel])


def stomp_optimize(
    q_start: np.ndarray,
    q_goal: np.ndarray,
    joint_limits: np.ndarray,
    n_waypoints: int = 30,
    n_iterations: int = 80,
    n_rollouts: int = 10,
    noise_stddev: float = 0.08,
    noise_decay: float = 0.97,
    w_smooth: float = 10.0,
    w_limit: float = 50.0,
    w_vel: float = 5.0,
    verbose: bool = True,
) -> np.ndarray:
    """
    Run STOMP to find a smooth, joint-limit-respecting trajectory.

    Parameters
    ----------
    q_start : (6,)   Start joint configuration (e.g. HOME).
    q_goal  : (6,)   Goal joint configuration (from IK-Geo).
    joint_limits : (6, 2)  [[lo, hi], ...] per joint.
    n_waypoints  : Number of trajectory waypoints (incl. start & goal).
    n_iterations : STOMP iterations.
    n_rollouts   : Noisy candidates per iteration.
    noise_stddev : Initial noise std dev (radians).
    noise_decay  : Multiplicative decay of noise each iteration.
    w_smooth     : Weight for smoothness cost.
    w_limit      : Weight for joint limit cost.
    w_vel        : Weight for velocity cost.
    verbose      : Print progress.

    Returns
    -------
    trajectory : (n_waypoints, 6)  Optimized joint trajectory.
    """
    ndof = len(q_start)
    assert len(q_goal) == ndof == joint_limits.shape[0]

    # ── Seed: linear interpolation ───────────────────────────────
    trajectory = np.zeros((n_waypoints, ndof))
    for i in range(n_waypoints):
        alpha = i / (n_waypoints - 1)
        trajectory[i] = (1 - alpha) * q_start + alpha * q_goal

    # Smoothness matrix (only for interior points, but we compute for all)
    A = _smoothness_matrix(n_waypoints)

    # Interior indices (we never modify start & goal)
    interior = slice(1, n_waypoints - 1)
    n_interior = n_waypoints - 2

    if verbose:
        print(f"  STOMP: {n_waypoints} waypoints, {n_iterations} iters, "
              f"{n_rollouts} rollouts, noise={noise_stddev:.3f}")

    best_cost = float("inf")
    noise = noise_stddev

    for it in range(n_iterations):
        # ── Generate noisy rollouts ──────────────────────────────
        candidates = []
        costs = []

        for _ in range(n_rollouts):
            # Add Gaussian noise to interior waypoints only
            delta = np.random.randn(n_interior, ndof) * noise
            candidate = trajectory.copy()
            candidate[interior] += delta

            # Clamp to joint limits
            for j in range(ndof):
                candidate[:, j] = np.clip(
                    candidate[:, j], joint_limits[j, 0], joint_limits[j, 1]
                )

            # Fix endpoints
            candidate[0] = q_start
            candidate[-1] = q_goal

            # ── Evaluate cost ────────────────────────────────────
            c_smooth = w_smooth * _smoothness_cost(candidate, A)
            c_limit = w_limit * np.sum(_joint_limit_cost(candidate, joint_limits))
            c_vel = w_vel * np.sum(_velocity_cost(candidate))
            total = c_smooth + c_limit + c_vel

            candidates.append(candidate)
            costs.append(total)

        # Also evaluate current trajectory
        c_smooth = w_smooth * _smoothness_cost(trajectory, A)
        c_limit = w_limit * np.sum(_joint_limit_cost(trajectory, joint_limits))
        c_vel = w_vel * np.sum(_velocity_cost(trajectory))
        current_cost = c_smooth + c_limit + c_vel

        candidates.append(trajectory.copy())
        costs.append(current_cost)

        # ── Probability-weighted update ──────────────────────────
        costs = np.array(costs)
        min_cost = np.min(costs)

        # Exponentiated cost (lower cost → higher probability)
        # Using temperature scaling for numerical stability
        h = 10.0  # inverse temperature
        exp_costs = np.exp(-h * (costs - min_cost) / (np.max(costs) - min_cost + 1e-10))
        probs = exp_costs / (np.sum(exp_costs) + 1e-10)

        # Weighted average of all candidates
        new_traj = np.zeros_like(trajectory)
        for k, cand in enumerate(candidates):
            new_traj += probs[k] * cand

        # Fix endpoints
        new_traj[0] = q_start
        new_traj[-1] = q_goal

        trajectory = new_traj

        # Track best
        if min_cost < best_cost:
            best_cost = min_cost

        # Decay noise
        noise *= noise_decay

        if verbose and (it % 20 == 0 or it == n_iterations - 1):
            print(f"    iter {it:3d}/{n_iterations} | cost: {current_cost:.2f} "
                  f"(smooth={c_smooth:.1f} limit={c_limit:.1f} vel={c_vel:.1f}) "
                  f"| noise: {noise:.4f}")

    # ── Final clamp ──────────────────────────────────────────────
    for j in range(ndof):
        trajectory[:, j] = np.clip(
            trajectory[:, j], joint_limits[j, 0], joint_limits[j, 1]
        )
    trajectory[0] = q_start
    trajectory[-1] = q_goal

    if verbose:
        # Final stats
        diffs = np.diff(trajectory, axis=0)
        max_jump = np.max(np.abs(diffs))
        mean_jump = np.mean(np.abs(diffs))
        print(f"  STOMP result: max_jump={max_jump:.4f} rad "
              f"({np.degrees(max_jump):.1f}°), "
              f"mean_jump={mean_jump:.4f} rad ({np.degrees(mean_jump):.1f}°)")

    return trajectory


# ── Quick test ───────────────────────────────────────────────────
if __name__ == "__main__":
    print("STOMP standalone test")
    print("=" * 50)

    # KUKA KR6 R700 joint limits
    limits = np.array([
        [-2.967059725,  2.967059725],
        [-3.316125575,  0.785398163],
        [-2.094395100,  2.722713630],
        [-3.228859113,  3.228859113],
        [-2.094395100,  2.094395100],
        [-6.108652375,  6.108652375],
    ])

    q_home = np.zeros(6)
    q_goal = np.array([0.5, -0.3, 0.8, -0.2, 0.4, 0.1])

    traj = stomp_optimize(q_home, q_goal, limits, n_waypoints=30)
    print(f"\nTrajectory shape: {traj.shape}")
    print(f"Start: {np.round(traj[0], 4)}")
    print(f"End:   {np.round(traj[-1], 4)}")

    # Verify limits
    for i in range(len(traj)):
        for j in range(6):
            assert limits[j, 0] <= traj[i, j] <= limits[j, 1], \
                f"Limit violation at wp{i}, joint {j}"
    print("✅ All waypoints within joint limits!")

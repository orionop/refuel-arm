#!/usr/bin/env python3
"""
Standalone IK-Geo test — runs on macOS CPU, no ROS or Gazebo needed.
Verifies the exact algebraic IK solver for the KUKA KR6 R700.
"""
import sys
import numpy as np
sys.path.insert(0, "kuka_refuel_ws/src/kuka_kr6_gazebo/scripts")
from ik_geometric import IK_spherical_2_parallel, fwd_kinematics, KIN_KR6_R700

def main():
    print("=" * 60)
    print("KUKA KR6 R700 — IK-Geo Algebraic Solver Test (Local Mac)")
    print("=" * 60)

    # ── Test 1: Round-trip validation ────────────────────────────
    q_test = np.array([0.3, -0.5, 0.7, 0.1, -0.4, 0.2])
    print(f"\n[Test 1] Round-trip FK → IK validation")
    print(f"  Input joints:  {np.round(q_test, 4)}")

    R_target, p_target = fwd_kinematics(q_test)
    print(f"  FK position:   {np.round(p_target, 6)}")

    solutions = IK_spherical_2_parallel(R_target, p_target)

    if solutions.size == 0:
        print("  ❌ No solutions found!")
        return

    n_solutions = solutions.shape[1]
    print(f"  Found {n_solutions} IK solutions")

    for i in range(n_solutions):
        q_sol = solutions[:, i]
        R_check, p_check = fwd_kinematics(q_sol)
        pos_err = np.linalg.norm(p_check - p_target)
        rot_err = np.linalg.norm(R_check - R_target, 'fro')
        total_err = pos_err + rot_err
        status = "✅" if total_err < 1e-6 else "⚠️"
        print(f"  Sol {i+1}: q={np.round(q_sol, 4)}  pos_err={pos_err:.2e}  rot_err={rot_err:.2e} {status}")

    # ── Test 2: 50-trial random round-trip ───────────────────────
    print(f"\n[Test 2] 50-trial random FK → IK round-trip")
    n_trials = 50
    max_err = 0
    n_fail = 0
    for t in range(n_trials):
        q_rand = np.random.uniform(-np.pi, np.pi, 6)
        R_t, p_t = fwd_kinematics(q_rand)
        Q = IK_spherical_2_parallel(R_t, p_t)
        if Q.size == 0:
            n_fail += 1
            continue
        errs = []
        for i in range(Q.shape[1]):
            R_c, p_c = fwd_kinematics(Q[:, i])
            errs.append(np.linalg.norm(p_c - p_t) + np.linalg.norm(R_c - R_t, 'fro'))
        best = min(errs)
        max_err = max(max_err, best)

    print(f"  Trials: {n_trials} | Failed: {n_fail} | Max best-error: {max_err:.2e}")
    if max_err < 1e-6:
        print("  ✅ ALL TESTS PASSED")
    else:
        print(f"  ⚠️  Max error is {max_err:.2e}")

    # ── Test 3: Refueling target ────────────────────────────────
    print(f"\n[Test 3] Refueling inlet target: [0.3, 0.4, 0.25]")
    target_xyz = np.array([0.3, 0.4, 0.25])
    R_06 = np.eye(3)
    solutions2 = IK_spherical_2_parallel(R_06, target_xyz)
    if solutions2.size == 0:
        print("  ⚠️  No exact solutions for identity orientation (expected for some targets)")
    else:
        n2 = solutions2.shape[1]
        print(f"  Found {n2} IK solutions!")
        for i in range(n2):
            q_sol = solutions2[:, i]
            _, p_check = fwd_kinematics(q_sol)
            pos_err = np.linalg.norm(p_check - target_xyz)
            print(f"  Sol {i+1}: joints={np.round(q_sol, 4)}, pos_err={pos_err:.2e}")

    print(f"\n{'=' * 60}")
    print("Done!")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    main()

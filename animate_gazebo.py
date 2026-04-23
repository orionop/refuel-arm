import scipy.optimize

import scipy.optimize

def IK_numerical(R_target, p_target, kin, q0):
    def error_func(q):
        R, p = fwd_kinematics(q, kin)
        pos_err = np.linalg.norm(p - p_target)
        # Orientation error (simplified)
        rot_err = np.linalg.norm(R - R_target)
        return pos_err + 0.1 * rot_err

    res = scipy.optimize.minimize(error_func, q0, bounds=kin['joint_limits'] * (np.pi/180.0))
    if res.fun < 1e-2:
        return res.x.reshape(-1, 1)
    return np.array([])

def IK_solve_robust(R_target, p_target, robot="kr6"):
    kin = KIN_KR6_R700 if robot == "kr6" else KIN_KR8_R2100
    # Try geometric first
    Q = IK_solve(R_target, p_target, robot=robot)
    if Q.size > 0: return Q
    
    # Fallback to numerical
    print(f"  [IK] Geometric failed, trying numerical fallback...")
    q_num = IK_numerical(R_target, p_target, kin, np.zeros(6))
    return q_num

#!/usr/bin/env python3
"""
KUKA KR 8 R2100 (Cybertech nano) — Gazebo Animation Script
============================================================

Replays the exact 8-phase refueling mission from refuel_mission.py
directly inside Gazebo using native gz-transport topics.
No ROS 2 required.

Usage:
  Terminal 1 (server):  gz sim -s worlds/refuel_gas_station_demo.sdf
  Terminal 2 (GUI):     gz sim -g
  Terminal 3 (animate): python3 animate_gazebo.py
"""
import sys
import os
import time
import subprocess
import numpy as np

# ── Path setup (same as refuel_mission.py) ────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(
    PROJECT_ROOT, 'kuka_refuel_ws', 'src',
    'kuka_kr6_gazebo', 'scripts'))
sys.path.insert(0, PROJECT_ROOT)

from ik_geometric import (
    IK_spherical_2_parallel, fwd_kinematics, rot,
    IK_solve, KIN_UR5, KIN_KR6_R700, KIN_KR210_R3100, KIN_KR8_R2100
)
from stomp_collision import stomp_optimize
from bubble_strips import bubble_strip_deform, set_kinematics as bs_set_kinematics
from car_model import (
    get_inlet_pose, get_preapproach_pose, spawn_target_marker,
    TARGET_XYZ_DEFAULT,
)

# Import planning functions from refuel_mission (does NOT call main)
from refuel_mission import (
    Q_HOME, Q_REST, JOINT_LIMITS_DEFAULT, DWELL_TIME,
    limits_deg_to_rad, filter_solutions,
    plan_stomp, plan_cartesian,
    plot_trajectory_3d, plot_joint_angles,
)

# ── Gazebo Transport ──────────────────────────────────────────────
# Auto-detect gz path (Mac homebrew vs Ubuntu standard)
GZ_BIN = "/opt/homebrew/bin/gz" if os.path.exists("/opt/homebrew/bin/gz") else "gz"
JOINT_TOPICS = [f"/kuka_cmd/j{i}" for i in range(1, 7)]
PUBLISH_RATE = 3  # Hz — PID controller smooths the motion between updates



def gz_set_joints(q):
    """Publish all 6 joints in parallel via a single shell call for max speed."""
    cmds = []
    for i, val in enumerate(q):
        cmds.append(f"{GZ_BIN} topic -t {JOINT_TOPICS[i]} -m gz.msgs.Double -p 'data: {val}'")
    full_cmd = " & ".join(cmds) + " & wait"
    os.system(f"({full_cmd}) >/dev/null 2>&1")


def animate_segment(traj, duration, rate=PUBLISH_RATE):
    dt = duration / len(traj)
    for q in traj:
        gz_set_joints(q)
        time.sleep(dt)


# ── Main ──────────────────────────────────────────────────────────
def main():
    active_robot = "kr8"
    kin_params = KIN_KR8_R2100

    joint_limits_deg = np.asarray(
        kin_params.get('joint_limits', JOINT_LIMITS_DEFAULT), dtype=float
    )
    joint_limits_rad = limits_deg_to_rad(joint_limits_deg)

    # Inject active platform parameters into bubble strips
    bs_set_kinematics(kin_params, joint_limits_rad)

    n_wp = 30
    rng = np.random.default_rng(42)

    print("=" * 65)
    print(f"  {active_robot.upper()} — Autonomous Refueling Mission")
    print("  IK-Geo + STOMP + Elastic Strips")
    print("=" * 65)

    # KR8 hardcoded target (Inlet Local)
    target_xyz = np.array([1.296, 0.242, 0.715])
    print(f"\n[Target]       [{target_xyz[0]:.3f}, {target_xyz[1]:.3f}, {target_xyz[2]:.3f}]")

    
    inlet_xyz, inlet_R = get_inlet_pose(target_xyz, robot=active_robot)
    # Neutral orientation pointing towards target for both robots
    yaw = np.arctan2(target_xyz[1], target_xyz[0])
    inlet_R = rot(np.array([0, 0, 1]), yaw) @ rot(np.array([0, 1, 0]), 0.1) # slight pitch down

    # Pre-approach: pull back 10cm along approach direction
    pre_xyz, _ = get_preapproach_pose(inlet_xyz, inlet_R, standoff=0.10, robot=active_robot)

    print(f"[Inlet]        [{inlet_xyz[0]:.3f}, {inlet_xyz[1]:.3f}, {inlet_xyz[2]:.3f}]")
    print(f"[Pre-approach] [{pre_xyz[0]:.3f}, {pre_xyz[1]:.3f}, {pre_xyz[2]:.3f}]")

    print(f"\n[IK-Geo] Solving for {active_robot.upper()} target pose...")
    Q_target = IK_solve_robust(inlet_R, inlet_xyz, robot=active_robot)
    Q_v_target = filter_solutions(Q_target, Q_HOME, limits=joint_limits_deg)
    if Q_v_target.size == 0:
        print(f"  No valid IK solution for {active_robot.upper()} Target!")
        return
    q_target = Q_v_target[:, 0]

    print(f"\n[IK-Geo] Solving for {active_robot.upper()} pre-approach pose...")
    Q_pre = IK_solve_robust(inlet_R, pre_xyz, robot=active_robot)
    Q_v_pre = filter_solutions(Q_pre, Q_HOME, limits=joint_limits_deg)
    if Q_v_pre.size == 0:
        print(f"  No valid IK solution for {active_robot.upper()} Pre-approach!")
        return
    q_pre = Q_v_pre[:, 0]

    # --- Dispenser Pose ---
    DISPENSER_XYZ = np.array([0.181, 1.330, 0.925])
    
    disp_yaw = np.arctan2(DISPENSER_XYZ[1], DISPENSER_XYZ[0])
    if active_robot == "kr6":
        R_dispenser = rot(np.array([0, 0, 1]), disp_yaw)
    else:
        R_dispenser = rot(np.array([0, 0, 1]), disp_yaw) @ rot(np.array([0, 1, 0]), -0.35)

    print(f"\n[IK-Geo] Solving for {active_robot.upper()} dispenser pose at {DISPENSER_XYZ}...")
    Q_disp = IK_solve_robust(R_dispenser, DISPENSER_XYZ, robot=active_robot)
    Q_v_disp = filter_solutions(Q_disp, Q_HOME, limits=joint_limits_deg)
    if Q_v_disp.size == 0:
        print(f"  No valid IK solution for {active_robot.upper()} Dispenser!")
        return
    q_disp = Q_v_disp[:, 0]
    print(f"     Dispenser Joints: {np.round(np.degrees(q_disp), 1)} deg")

    _, p_chk = fwd_kinematics(q_target, kin=kin_params)
    print(f"     Target Joints:    {np.round(np.degrees(q_target), 1)} deg")
    print(f"     Target FK error:  {np.linalg.norm(p_chk - inlet_xyz):.2e} m")

    # ── Obstacle Sync ─────────────────────────────────────────────
    print("\n[Obstacles] Syncing spherical collision envelopes for static Gazebo meshes...")
    obs_list = [
        (np.array([0.990, 1.218, 0.20]), 0.35),
        (np.array([1.40, -0.50, 0.50]), 0.60),
    ]

    # ── 8-Phase Mission Pipeline ──────────────────────────────────
    seg_p1 = plan_stomp(Q_HOME, q_disp, obs_list,
                        "Phase 1: REST -> Dispenser", n_wp, limits=joint_limits_deg, kin=kin_params)

    seg_p3 = plan_stomp(q_disp, q_pre, obs_list,
                        "Phase 3: Dispenser -> Pre-approach", n_wp, limits=joint_limits_deg, kin=kin_params)

    seg_p4 = plan_cartesian(pre_xyz, inlet_xyz, inlet_R,
                            q_pre, "Phase 4: Pre-approach -> Target", n_wp=15, kin=kin_params)

    seg_p6 = plan_cartesian(inlet_xyz, pre_xyz, inlet_R,
                            q_target, "Phase 6: Target -> Pre-approach", n_wp=15, kin=kin_params)

    seg_p7 = plan_stomp(q_pre, q_disp, obs_list,
                        "Phase 7: Pre-approach -> Dispenser", n_wp, limits=joint_limits_deg, kin=kin_params)

    seg_p8 = plan_stomp(q_disp, Q_HOME, obs_list,
                        "Phase 8: Dispenser -> REST", n_wp, limits=joint_limits_deg, kin=kin_params)

    full_traj = np.vstack([seg_p1, seg_p3, seg_p4, seg_p6, seg_p7, seg_p8])

    segments = [
        ("Phase 1: REST -> Dispenser",     seg_p1,  7.0),
        ("Phase 2: Dwell (Fetch Nozzle)",   None,    3.0),
        ("Phase 3: Dispenser -> Pre-app",   seg_p3,  7.0),
        ("Phase 4: Pre-app -> Inlet",       seg_p4,  4.0),
        ("Phase 5: Dwell (Refueling)",      None,    6.0),
        ("Phase 6: Inlet -> Pre-app",       seg_p6,  4.0),
        ("Phase 7: Pre-app -> Dispenser",   seg_p7,  7.0),
        ("Phase 8: Dispenser -> REST",      seg_p8,  7.0),
    ]

    # ── Generate graphs ───────────────────────────────────────────
    print("\n[Graphs]")
    os.makedirs("output_graphs", exist_ok=True)
    plot_trajectory_3d(full_traj, target_xyz, obs_list,
                       "output_graphs/ee_trajectory_3d.png", kin=kin_params)
    plot_joint_angles(full_traj, "output_graphs/joint_angle_trajectories.png")

    # ── Preview (same output as refuel_mission.py dry run) ────────
    print(f"\n[Preview]")
    total_wp_list = []
    for i, (label, traj, dt) in enumerate(segments, 1):
        if traj is None:
            print(f"  Step {i}: {label} ({dt:.0f}s dwell)")
        else:
            seg_len = int(len(traj))
            total_wp_list.append(seg_len)
            print(f"  Step {i}: {label}  ({seg_len} wp, dt={dt}s)")
            if seg_len > 0:
                print(f"           start={np.round(np.degrees(traj[0]), 1)} deg")
                print(f"           end  ={np.round(np.degrees(traj[-1]), 1)} deg")
    print(f"\n  Total waypoints: {sum(total_wp_list)}")

    # ── Gazebo Animation ──────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print("  Starting Gazebo Animation...")
    print("  (Make sure Gazebo is running and PLAYING)")
    print(f"{'=' * 65}")

    # Set initial pose
    print("\n  Settling at REST pose...")
    for _ in range(5):  # Send multiple times to ensure it registers
        gz_set_joints(Q_HOME)
        time.sleep(0.2)
    time.sleep(2.0)

    for i, (label, traj, dt) in enumerate(segments, 1):
        print(f"\n  Step {i}/{len(segments)}: {label}")
        if traj is None:
            print(f"     Holding for {dt:.0f}s...")
            time.sleep(dt)
            print(f"     Dwell complete")
        else:
            print(f"     Animating {len(traj)} waypoints over {dt:.1f}s...")
            animate_segment(traj, dt)
            print(f"     done")

    print(f"\n{'=' * 65}")
    print("  Mission complete!")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()

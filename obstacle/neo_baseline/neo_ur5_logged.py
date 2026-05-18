#!/usr/bin/env python
"""
Logged variant of NEO on UR5.
Adds per-step logging, post-run plots (matplotlib), and CSV export.
Algorithm unchanged from neo_ur5.py.
"""

import swift
import spatialgeometry as sg
import roboticstoolbox as rtb
import spatialmath as sm
import numpy as np
import qpsolvers as qp
import time
import os
import csv
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, write PNG files
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ---------- Setup ----------

env = swift.Swift()
env.launch()

panda = rtb.models.UR5()
panda.q = np.array([0.0, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0.0])
n = 6

s0 = sg.Sphere(radius=0.05, pose=sm.SE3(0.35, 0.5, 0.5))
s0.v = [0, -0.15, 0, 0, 0, 0]
s1 = sg.Sphere(radius=0.05, pose=sm.SE3(0.1, 0.5, 0.7))
s1.v = [0, -0.15, 0, 0, 0, 0]
collisions = [s0, s1]
obstacle_names = ["s0", "s1"]

target = sg.Sphere(radius=0.02, pose=sm.SE3(0.4, -0.4, 0.4))

env.add(panda)
env.add(s0)
env.add(s1)
env.add(target)

Tep = panda.fkine(panda.q)
Tep.A[:3, 3] = target.T[:3, 3]

# ---------- Logging buffers ----------

log = {
    "t": [],
    "ee_pos": [],          # (x, y, z)
    "ee_err": [],          # scalar error magnitude
    "q": [],               # joint angles (6,)
    "qd": [],              # joint velocities (6,)
    "manipulability": [],
    "qp_solve_ms": [],     # ms per QP
    "qp_status": [],       # "ok" or "infeasible"
    "dist_to_obstacle": {name: [] for name in obstacle_names},  # min dist arm-to-obstacle
}


def min_link_distance(robot, obstacle, q, link_start, link_end):
    """Compute minimum distance from any robot link (start..end) to the obstacle.
    Uses spatialgeometry's closest_point queries via link collision shapes.
    """
    min_d = np.inf
    # Iterate links between start and end inclusive
    in_range = False
    for link in robot.links:
        if link == link_start:
            in_range = True
        if in_range:
            for col_shape in link.collision:
                try:
                    d, _, _ = col_shape.closest_point(obstacle, np.inf)
                    if d is not None and d < min_d:
                        min_d = float(d)
                except Exception:
                    pass
        if link == link_end:
            in_range = False
    return min_d if min_d != np.inf else np.nan


t_sim = 0.0
dt = 0.01


def step():
    global t_sim

    Te = panda.fkine(panda.q)
    eTep = Te.inv() * Tep
    e = np.sum(np.abs(np.r_[eTep.t, eTep.rpy() * np.pi / 180]))

    v, arrived = rtb.p_servo(Te, Tep, 0.5, 0.01)

    Y = 0.01
    Q = np.eye(n + 6)
    Q[:n, :n] *= Y
    Q[n:, n:] = (1 / e) * np.eye(6)

    Aeq = np.c_[panda.jacobe(panda.q), np.eye(6)]
    beq = v.reshape((6,))

    Ain = np.zeros((n + 6, n + 6))
    bin = np.zeros(n + 6)

    ps = 0.05
    pi_inf = 0.9
    Ain[:n, :n], bin[:n] = panda.joint_velocity_damper(ps, pi_inf, n)

    for collision in collisions:
        c_Ain, c_bin = panda.link_collision_damper(
            collision,
            panda.q[:n],
            0.3,
            0.05,
            1.0,
            start=panda.link_dict["shoulder_link"],
            end=panda.link_dict["tool0"],
        )
        if c_Ain is not None and c_bin is not None:
            c_Ain = c_Ain[:, :n]
            c_Ain = np.c_[c_Ain, np.zeros((c_Ain.shape[0], 6))]
            Ain = np.r_[Ain, c_Ain]
            bin = np.r_[bin, c_bin]

    c = np.r_[-panda.jacobm(panda.q).reshape((n,)), np.zeros(6)]

    qdlim = np.pi * np.ones(n)
    lb = -np.r_[qdlim, 10 * np.ones(6)]
    ub = np.r_[qdlim, 10 * np.ones(6)]

    # --- Solve QP and time it ---
    t0 = time.perf_counter()
    qd = qp.solve_qp(Q, c, Ain, bin, Aeq, beq, lb=lb, ub=ub, solver="quadprog")
    solve_ms = (time.perf_counter() - t0) * 1000.0

    status = "ok"
    if qd is None:
        status = "infeasible"
        panda.qd[:n] = np.zeros(n)
        qd_log = np.zeros(n)
    else:
        panda.qd[:n] = qd[:n]
        qd_log = qd[:n].copy()

    # --- Log everything ---
    log["t"].append(t_sim)
    log["ee_pos"].append(Te.t.copy())
    log["ee_err"].append(float(e))
    log["q"].append(panda.q[:n].copy())
    log["qd"].append(qd_log)
    log["manipulability"].append(float(panda.manipulability(panda.q)))
    log["qp_solve_ms"].append(solve_ms)
    log["qp_status"].append(status)
    for name, obs in zip(obstacle_names, collisions):
        d = min_link_distance(
            panda, obs, panda.q,
            panda.link_dict["shoulder_link"], panda.link_dict["tool0"]
        )
        log["dist_to_obstacle"][name].append(d)

    env.step(dt)
    t_sim += dt
    # No time.sleep — we want fast logging run, not slow visual

    return arrived


# ---------- Run ----------

print("Running NEO on UR5, logging...")
arrived = False
max_steps = 5000  # safety cap (~50 sec sim time)
step_count = 0
while not arrived and step_count < max_steps:
    arrived = step()
    step_count += 1

print(f"Run finished. arrived={arrived}, steps={step_count}, sim_time={t_sim:.2f}s")

# ---------- Save CSV ----------

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(out_dir, exist_ok=True)
csv_path = os.path.join(out_dir, "neo_ur5_run.csv")

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    header = ["t", "ee_x", "ee_y", "ee_z", "ee_err", "manipulability",
              "qp_solve_ms", "qp_status"]
    header += [f"q{i}" for i in range(n)]
    header += [f"qd{i}" for i in range(n)]
    header += [f"dist_{name}" for name in obstacle_names]
    writer.writerow(header)
    for i in range(len(log["t"])):
        row = [
            log["t"][i],
            log["ee_pos"][i][0], log["ee_pos"][i][1], log["ee_pos"][i][2],
            log["ee_err"][i],
            log["manipulability"][i],
            log["qp_solve_ms"][i],
            log["qp_status"][i],
        ]
        row += list(log["q"][i])
        row += list(log["qd"][i])
        for name in obstacle_names:
            row.append(log["dist_to_obstacle"][name][i])
        writer.writerow(row)
print(f"CSV saved: {csv_path}")

# ---------- Plots ----------

t = np.array(log["t"])

# Plot 1: distance to obstacles + EE error (paper's Fig 4 a/b/c style)
fig, ax = plt.subplots(figsize=(10, 5))
for name in obstacle_names:
    d_arr = np.array(log["dist_to_obstacle"][name])
    ax.plot(t, d_arr, label=f"Distance to {name}")
ax.plot(t, log["ee_err"], label="EE error", linestyle="--", color="gray")
ax.axhline(0.05, color="r", linestyle=":", label="Stop dist (5cm)")
ax.axhline(0.30, color="orange", linestyle=":", label="Influence dist (30cm)")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Distance (m)")
ax.set_title("NEO on UR5 — Distances and EE error over time")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
plot1 = os.path.join(out_dir, "plot1_distances.png")
fig.savefig(plot1, dpi=140)
plt.close(fig)

# Plot 2: manipulability (paper's Fig 4d style)
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(t, log["manipulability"], color="C2")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Manipulability (Yoshikawa)")
ax.set_title("NEO on UR5 — Manipulability over time")
ax.grid(True, alpha=0.3)
fig.tight_layout()
plot2 = os.path.join(out_dir, "plot2_manipulability.png")
fig.savefig(plot2, dpi=140)
plt.close(fig)

# Plot 3: QP solve time histogram + over-time
fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4))
axL.plot(t, log["qp_solve_ms"], lw=0.8)
axL.set_xlabel("Time (s)")
axL.set_ylabel("QP solve (ms)")
axL.set_title("QP solve time per step")
axL.grid(True, alpha=0.3)
axR.hist(log["qp_solve_ms"], bins=40, edgecolor="black")
axR.set_xlabel("QP solve (ms)")
axR.set_ylabel("Count")
axR.set_title(f"Histogram (mean={np.mean(log['qp_solve_ms']):.2f} ms, "
              f"max={np.max(log['qp_solve_ms']):.2f} ms)")
axR.grid(True, alpha=0.3)
fig.tight_layout()
plot3 = os.path.join(out_dir, "plot3_qp_solve_time.png")
fig.savefig(plot3, dpi=140)
plt.close(fig)

# Plot 4: 3D end-effector trajectory
ee = np.array(log["ee_pos"])
fig = plt.figure(figsize=(7, 6))
ax3 = fig.add_subplot(111, projection="3d")
ax3.plot(ee[:, 0], ee[:, 1], ee[:, 2], lw=1.5, label="EE trajectory")
ax3.scatter(*ee[0], color="green", s=60, label="Start")
ax3.scatter(*Tep.t, color="red", s=60, label="Goal")
ax3.set_xlabel("X (m)")
ax3.set_ylabel("Y (m)")
ax3.set_zlabel("Z (m)")
ax3.set_title("End-effector trajectory")
ax3.legend()
fig.tight_layout()
plot4 = os.path.join(out_dir, "plot4_ee_trajectory.png")
fig.savefig(plot4, dpi=140)
plt.close(fig)

# Plot 5: joint velocities over time
fig, ax = plt.subplots(figsize=(10, 5))
qd_arr = np.array(log["qd"])
for i in range(n):
    ax.plot(t, qd_arr[:, i], label=f"q̇{i}")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Joint velocity (rad/s)")
ax.set_title("Joint velocities over time")
ax.legend(ncol=3)
ax.grid(True, alpha=0.3)
fig.tight_layout()
plot5 = os.path.join(out_dir, "plot5_joint_velocities.png")
fig.savefig(plot5, dpi=140)
plt.close(fig)

print("Plots saved:")
for p in [plot1, plot2, plot3, plot4, plot5]:
    print(f"  - {p}")

# ---------- Summary stats ----------

print("\n--- Run summary ---")
print(f"Steps:                {step_count}")
print(f"Sim time:             {t_sim:.2f} s")
print(f"Arrived at goal:      {arrived}")
print(f"Mean QP solve:        {np.mean(log['qp_solve_ms']):.2f} ms")
print(f"Max QP solve:         {np.max(log['qp_solve_ms']):.2f} ms")
print(f"QP infeasible count:  {sum(1 for s in log['qp_status'] if s == 'infeasible')}")
for name in obstacle_names:
    d = np.array(log["dist_to_obstacle"][name])
    d = d[~np.isnan(d)]
    if len(d) > 0:
        print(f"Min dist {name}:         {np.min(d):.4f} m")
print(f"Initial manipulability: {log['manipulability'][0]:.4f}")
print(f"Final manipulability:   {log['manipulability'][-1]:.4f}")
print(f"Min manipulability:     {np.min(log['manipulability']):.4f}")

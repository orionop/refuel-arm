#!/usr/bin/env python
"""
Independent sanity check for the penetration finding from neo_ur5_logged.py.

Uses a completely different distance computation:
  - Forward kinematics on logged q to get each link's coordinate frame origin
  - Obstacle sphere center reconstructed from initial pose + velocity * t
  - Distance = ||link_origin - sphere_center|| - sphere_radius
  - Min over all link origins = naive clearance estimate per timestep

This is geometrically simpler than spatialgeometry's closest_point queries
(which use full collision shapes). If even this conservative point-based
estimate reports penetration, the contact is real.

Note: point-based check UNDERESTIMATES clearance only when the link's
collision shape extends beyond its origin. For a link modeled as a capsule
along its length, the true clearance is LOWER than the origin-based estimate.
So: if naive check says "penetration," real geometry also says penetration.
                                       ^^^^ that's the logic for sanity.
"""

import pandas as pd
import numpy as np
import roboticstoolbox as rtb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# ---------- Load logged data ----------

csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "neo_ur5_run.csv")
df = pd.read_csv(csv_path)
print(f"Loaded {len(df)} steps from {csv_path}")

# ---------- Reconstruct obstacle positions ----------

# Match the scene definition in neo_ur5_logged.py
SPHERE_RADIUS = 0.05
s0_init = np.array([0.35, 0.5, 0.5])
s0_vel  = np.array([0.0, -0.15, 0.0])
s1_init = np.array([0.1, 0.5, 0.7])
s1_vel  = np.array([0.0, -0.15, 0.0])

def sphere_pos(init, vel, t):
    return init + vel * t

# ---------- Build robot for FK ----------

robot = rtb.models.UR5()
n = 6

# ---------- Recompute distance independently ----------

# We get distance from each link's coordinate frame origin to each sphere
# center, then take the min (across links) for that sphere.

# Get the list of link frames we want to check. Match the chain used by NEO:
# shoulder_link -> tool0.
chain_link_names = [
    "shoulder_link",
    "upper_arm_link",
    "forearm_link",
    "wrist_1_link",
    "wrist_2_link",
    "wrist_3_link",
    "tool0",
]
print(f"Checking these link origins: {chain_link_names}")

times = df["t"].values
indep_dist_s0 = np.zeros(len(df))
indep_dist_s1 = np.zeros(len(df))

for i in range(len(df)):
    t = times[i]
    q = df[[f"q{j}" for j in range(n)]].iloc[i].values

    # Forward kinematics: get all link transforms
    # robot.fkine_all(q) returns SE3 objects for each link
    Ts = robot.fkine_all(q)

    # Sphere centers at this t
    c0 = sphere_pos(s0_init, s0_vel, t)
    c1 = sphere_pos(s1_init, s1_vel, t)

    # Min distance from any chain link origin to each sphere
    min_d0 = np.inf
    min_d1 = np.inf
    for link_name in chain_link_names:
        if link_name not in robot.link_dict:
            continue
        link = robot.link_dict[link_name]
        # find index of link in robot.links
        try:
            idx = robot.links.index(link)
            origin = Ts[idx].t  # 3-vector
        except (ValueError, IndexError):
            continue
        d0 = np.linalg.norm(origin - c0) - SPHERE_RADIUS
        d1 = np.linalg.norm(origin - c1) - SPHERE_RADIUS
        if d0 < min_d0:
            min_d0 = d0
        if d1 < min_d1:
            min_d1 = d1

    indep_dist_s0[i] = min_d0
    indep_dist_s1[i] = min_d1

# ---------- Stats ----------

logged_s0 = df["dist_s0"].values
logged_s1 = df["dist_s1"].values

def stats(name, logged, indep):
    print(f"\n=== {name} ===")
    print(f"  Logged   min: {np.nanmin(logged):+.4f} m")
    print(f"  Indep    min: {np.nanmin(indep):+.4f} m  (origin-based, conservative)")
    pen_logged = np.sum(logged < 0)
    pen_indep  = np.sum(indep < 0)
    print(f"  Logged   penetration steps: {pen_logged}")
    print(f"  Indep    penetration steps: {pen_indep}")
    if pen_logged > 0:
        # Find the most-penetrating step in the logged data
        i = int(np.argmin(logged))
        print(f"  At step of max logged penetration (t={times[i]:.3f}s):")
        print(f"    logged: {logged[i]:+.4f} m  |  indep: {indep[i]:+.4f} m")

stats("s0 (left)", logged_s0, indep_dist_s0)
stats("s1 (right)", logged_s1, indep_dist_s1)

# ---------- Verdict ----------

print("\n=== Verdict ===")
if np.nanmin(indep_dist_s0) < 0:
    print("s0: independent origin-based check ALSO reports penetration.")
    print("    → Real contact (conservative metric still negative).")
else:
    print("s0: independent origin-based check is positive (no penetration).")
    print("    → Logged 'penetration' is geometry-dependent.")
    print("    → Possible that NEO's collision shape inflation differs from real geometry.")

if np.nanmin(indep_dist_s1) < 0:
    print("s1: independent origin-based check ALSO reports penetration.")
else:
    print("s1: independent origin-based check is positive.")
    print("    → s1 contact was a measurement quirk of collision-shape geometry.")

# ---------- Plot overlay ----------

fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
for ax, name, log_d, ind_d in zip(axes,
                                   ["s0 (left)", "s1 (right)"],
                                   [logged_s0, logged_s1],
                                   [indep_dist_s0, indep_dist_s1]):
    ax.plot(times, log_d, label="Logged (spatialgeometry closest_point)", lw=1)
    ax.plot(times, ind_d, label="Independent (link origin to sphere center)", lw=1, ls="--")
    ax.axhline(0, color="r", lw=1, ls=":", label="Contact threshold")
    ax.axhline(0.05, color="orange", lw=1, ls=":", alpha=0.5, label="Stop dist (5 cm)")
    ax.set_ylabel(f"Distance to {name} (m)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
axes[-1].set_xlabel("Time (s)")
fig.suptitle("Sanity check: logged distance vs independent recomputation")
fig.tight_layout()
out_png = os.path.join(os.path.dirname(csv_path), "verify_penetration.png")
fig.savefig(out_png, dpi=140)
plt.close(fig)
print(f"\nPlot saved: {out_png}")

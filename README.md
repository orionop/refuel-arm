# Design of Robotic Arm for Autonomous Refueling Applications

<p align="center">
  <b>Systems and Control Engineering (SysCon), IIT Bombay</b><br>
  Maintainer: <a href="https://github.com/orionop">Anurag Shetye</a>
</p>

---

## Overview

This repository implements an end-to-end simulation pipeline for autonomous car refueling using a **KUKA KR6 R700** 6-DOF industrial manipulator. The robot approaches a car model on an elevated platform, aligns with the fuel inlet, performs a slow precision insertion, dwells for 5 seconds (refueling), then retracts and returns home — all while dynamically avoiding obstacles discovered mid-execution. The mission runs in a **ROS Noetic / Gazebo** simulation environment.

**Mission sequence:** `HOME` → `Pre-approach` (coarse, obstacle-aware) → `Inlet` (slow insertion) → `Dwell 5s` → `Withdraw` → `HOME`

### Three-Layer Motion Planning Stack

| Layer | Component | Role |
|-------|-----------|------|
| 1 | **IK-Geo** | Exact algebraic closed-form IK for terminal target poses (~10⁻¹⁶ rad) |
| 2 | **STOMP** | Stochastic Trajectory Optimization — obstacle-informed smooth path planning |
| 3 | **Elastic Strips** | Reactive trajectory deformation for dynamically discovered obstacles |

---

## Technical Approach

### Terminal Pose — IK-Geo (Exact Algebraic Solver)

The KR6 R700 belongs to the `IK_spherical_2_parallel` kinematic family. IK-Geo decomposes the 6-DOF inverse kinematics into a sequence of canonical subproblems (Paden–Kahan), yielding **up to 8 closed-form solutions** per pose. The nearest valid solution (by Euclidean distance in joint space) is selected. Two IK targets are solved: a pre-approach pose (8 cm standoff from the fuel inlet) and the final insertion pose, using branch-consistent selection to ensure smooth continuity.

### Trajectory Planning — STOMP Optimizer

STOMP (Stochastic Trajectory Optimization for Motion Planning) generates smooth, collision-free trajectories between joint configurations. Detected obstacles are fed to STOMP's `simple_obstacles` parameter for informed planning with Euclidean Distance Transform penalties. Coarse segments (HOME ↔ pre-approach) use STOMP with 30 waypoints.

### Reactive Obstacle Avoidance — Elastic Strips

Based on Brock & Khatib (2002). Treats the STOMP trajectory as a physical rubber band: internal spring forces maintain smoothness while external repulsion forces (via Jacobian transpose) push waypoints away from obstacles. Handles obstacles discovered after initial STOMP planning.

### Two-Phase Approach

The final approach uses a coarse-to-fine strategy: STOMP handles the long-range path to 8 cm standoff, then C-space linear interpolation with 3x slower timing (dt=0.40s) handles the precision insertion into the fuel inlet.

### Obstacle Detection

Obstacles are simulated via Gazebo's `/model_states` topic with range-gated detection — obstacles only become "known" when an arm FK checkpoint is within sensor range (0.6 m). A default demo obstacle is spawned at startup; additional obstacles can be introduced mid-mission via `rosservice call /gazebo/spawn_sdf_model`.

---

## Quick Start

For full installation and execution instructions, please see **[SETUP.md](SETUP.md)**.

### Primary Commands

```bash
# Full autonomous refueling mission in Gazebo
python3 refuel_mission.py --ros

# Dry-run preview (no ROS needed)
python3 refuel_mission.py

# RViz-only visualization
python3 refuel_mission.py --rviz

# Generate analysis comparison graphs
python3 refuel_mission.py --analyze

# Custom car placement
python3 refuel_mission.py --ros --car-x 0.55 --car-y 0.35 --car-yaw 0.0
```

### Additional Tools

```bash
# C-Space vs. Workspace motion strategy comparison
python3 ik_trajectories/compare_cspace_workspace.py

# Topological 6-DOF manifold tracking (Mobius strip)
python3 ik_trajectories/test_ik_mobius.py --ros

# IK-Geo mathematical accuracy benchmark (3,830 roots)
python3 ik_trajectories/analyze_ik_accuracy.py
```

---

## Repository Structure

```
refuel-arm/
├── refuel_mission.py                # Main refueling mission orchestrator (consolidated)
├── car_model.py                     # Car SDF, platform, inlet pose computation
├── obstacle_detector.py             # Simulated sensor via /gazebo/model_states
├── stomp_collision.py               # STOMP trajectory optimizer with collision avoidance
├── elastic_strips.py                # Real-time reactive obstacle avoidance (Brock & Khatib 2002)
├── visualize_ik_solutions.py        # IK-Geo multimodal global solver and Gazebo visualizer
├── analyze_pipeline.py              # STOMP 4-panel analysis graph generator
├── ik_trajectories/                 # 6-DOF Topological Tracking & Comparison
│   ├── compare_cspace_workspace.py  # Dual-strategy comparison and visualization
│   ├── analyze_ik_accuracy.py       # Empirical mathematical precision benchmark
│   ├── test_joint_line.py           # Configuration Space linear interpolation
│   ├── test_ik_line.py              # Pure algebraic IK Cartesian line tracker
│   ├── test_ik_mobius.py            # 4π Möbius strip topological tracker
│   └── test_ik_pringle.py           # 3D hyperbolic paraboloid (saddle) tracking
│
├── ik-geo/                          # Algebraic IK library submodule
├── output_graphs/                   # Generated analysis plots
├── kuka_refuel_ws/                  # ROS Noetic catkin workspace
├── deprecated/                      # Previous approaches (test_full_pipeline.py, legacy IK)
├── report/                          # 8-page LaTeX report
└── README.md
```

---

## Shared Lab Machine Protocol

See [`SAFE_DEV_RULES.md`](SAFE_DEV_RULES.md) for the original lab protocol guidelines used during the development of this repository. Key historical rules included restricted `sudo` access, strict workspace isolation, and NVIDIA GPU sharing courtesy.

---

## License

IK-Geo: MIT License (RPI Robotics)
STOMP: Derived from Kalakrishnan et al. (2011)
Elastic Strips: Derived from Brock & Khatib (2002)
KUKA Robot Descriptions: Apache 2.0 (KUKA)

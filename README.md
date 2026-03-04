# Design of Robotic Arm for Autonomous Refueling Applications

<p align="center">
  <b>Systems and Control Engineering (SysCon), IIT Bombay</b><br>
  Maintainer: <a href="https://github.com/orionop">Anurag Shetye</a>
</p>

---

## Overview

This repository implements an end-to-end simulation pipeline for autonomous car refueling using a **KUKA KR6 R700** 6-DOF industrial manipulator *(Expansion to UR5 architecture coming soon!)*. The robot executes a precision autonomous sequence: `HOME` (Straight Up) → `YELLOW` (Grab Nozzle) → `RED` (Refuel 10s) → `YELLOW` (Return Nozzle) → `HOME`. The mission is executed within a physically accurate **ROS Noetic / Gazebo** environment.

The core trajectory generation combines two complementary kinematics strategies:

| Component | Role | Precision |
|-----------|------|-----------|
| **IK-Geo** | Exact algebraic closed-form IK for terminal target poses | ~10⁻¹⁶ rad |
| **STOMP** | Stochastic Trajectory Optimization for smooth motion generation | Collision-free paths |

---

---

## Technical Approach

### Terminal Pose — IK-Geo (Exact Algebraic Solver)

The KR6 R700 belongs to the `IK_spherical_2_parallel` kinematic family. IK-Geo decomposes the 6-DOF inverse kinematics into a sequence of canonical subproblems (Paden–Kahan), yielding **up to 8 closed-form solutions** per pose. This ensures exact spatial accuracy for critical refueling targets without the convergence risks of iterative solvers.

### Trajectory Planning — Collision-Aware STOMP (Sensor Fusion)

We utilize a **Gradient-Free Sensor Fusion** approach for trajectory generation. By integrating STOMP (Stochastic Trajectory Optimization) with a **2.5D Euclidean Distance Transform (EDT)** generated from workspace point clouds, the robot generates smooth, collision-free paths. Unlike standard RRT planners, this method:
- Guarantees constant joint velocities and mechanical smoothness.
- Avoids singularities by utilizing algebraic IK foundations.
- Operates natively over 2.5D heightmaps for efficient unstructured environment navigation.

### Configuration Space (C-Space) Interpolation

For high-speed transitions where a straight Cartesian line is not required, we implement purely linear joint-space interpolation. This eliminates the need for per-waypoint IK and provides the most mechanically efficient motion profile possible for a 6-DOF arm.

---

## Quick Start

For full installation and execution instructions, please see **[SETUP.md](SETUP.md)**.

### Primary Execution Previews

1. **Autonomous Refueling Pipeline (STOMP + 2.5D Collision Avoidance)**:
```bash
python3 test_full_pipeline.py --ros
```

2. **C-Space vs. Workspace Motion Strategy Comparison**:
```bash
python3 ik_trajectories/compare_cspace_workspace.py
```

3. **Topological 6-DOF Manifold Tracking (e.g. Möbius Strip)**:
```bash
python3 ik_trajectories/test_ik_mobius.py --ros
```

4. **IK-Geo Mathematical Accuracy Benchmark (3,830 Roots)**:
```bash
python3 ik_trajectories/analyze_ik_accuracy.py
```

---

## Repository Structure

```
refuel-arm/
├── test_full_pipeline.py            # Main Refueling execution orchestrator
├── stomp_collision.py               # 2.5D Collision-Aware STOMP Optimizer (NEW)
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
├── deprecated/                      # Previous approaches (stomp_planner.py, legacy IK)
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
KUKA Robot Descriptions: Apache 2.0 (KUKA)

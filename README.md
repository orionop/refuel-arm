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

## Technical Approach

### Terminal Pose — IK-Geo (Exact Algebraic Solver)

The KR6 R700 belongs to the `IK_spherical_2_parallel` kinematic family. IK-Geo decomposes the 6-DOF inverse kinematics into a sequence of canonical subproblems (Paden–Kahan), yielding **up to 8 closed-form solutions** per pose — no iterative methods, no local minima, no convergence failures. It successfully finds exact configurations for the tall Yellow Nozzle dock and the realistic Red car inlet target.

### Trajectory Planning — STOMP Optimizer

STOMP (Stochastic Trajectory Optimization for Motion Planning) is used to generate smooth, collision-free waypoints along the Cartesian paths between the 5 mission stages. Instead of instantly jumping between algebraic IK solutions, STOMP optimizes 30 smooth intermediate waypoints per segment, ensuring acceleration limits and joint limits are respected before execution via ROS's `JointTrajectoryController`.

---

## Quick Start

For full installation and execution instructions across Ubuntu, ROS Noetic, and Python environments, please see the **[SETUP.md](SETUP.md)** file!

### Basic Execution Previews

1. **Full Robotic Refueling Pipeline (`HOME → YELLOW → RED → YELLOW → HOME`)**:
```bash
python3 test_full_pipeline.py --ros
```

2. **Topological 6-DOF Trajectory Tests (e.g. Möbius Strip)**:
```bash
python3 ik_trajectories/test_ik_mobius.py --ros
```

3. **STOMP Pipeline Analysis Visualization (4-Panel Graph)**:
```bash
python3 analyze_pipeline.py
```

---

## Repository Structure

```
refuel-arm/
├── test_full_pipeline.py            # Main STOMP Refueling execution orchestrator
├── analyze_pipeline.py              # STOMP 4-panel analysis graph generator
├── stomp_planner.py                 # Core standalone STOMP trajectory optimizer
├── ik_trajectories/                 # 6-DOF Topological Tracking Scripts
│   ├── test_ik_line.py              # Pure algebraic IK Cartesian line tracker
│   ├── test_ik_wave.py              # Multi-cycle audio-wave with dynamic pitch
│   ├── test_ik_pringle.py           # 3D hyperbolic paraboloid (saddle) tracking
│   └── test_ik_mobius.py            # 4π Möbius strip topological inversion tracker
│
├── ik-geo/                          # Algebraic IK library submodule
├── output_graphs/                   # Generated analysis plots
├── kuka_refuel_ws/                  # ROS Noetic catkin workspace
│   └── src/kuka_kr6_gazebo/
│       ├── urdf/                    #   URDF with accurate physical inertials
│       ├── config/                  #   gazebo_ros_controllers & RViz config
│       ├── launch/                  #   gazebo.launch and rviz.launch
│       ├── worlds/                  #   Gazebo world with Yellow & Red markers
│       └── scripts/
│           ├── ik_geometric.py      #   Python port of IK_spherical_2_parallel
│
├── deprecated/                      # Previous approaches (IKFlow, MATLAB, instructions)
├── report/                          # LaTeX files for empirical mathematical tracking analysis
├── SETUP.md                         # Easy Quick-Start guide for execution & installation
├── SAFE_DEV_RULES.md                # Shared lab machine safety protocol
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

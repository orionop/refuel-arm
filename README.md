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

### Configuration Space (C-Space) Interpolation

For high-speed, mechanically smooth motions where a perfectly straight Cartesian line is not required, the repository supports purely linear joint-space interpolation. By solving IK only at the start and end of a segment, this method ensures constant joint velocities and eliminates the risk of IK failures or singularities in the middle of a motion.

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

3. **IK-Geo Mathematical Accuracy Benchmark**:
```bash
python3 ik_trajectories/analyze_ik_accuracy.py
```

4. **STOMP Pipeline Analysis Visualization (4-Panel Graph)**:
```bash
python3 analyze_pipeline.py
```

5. **Configuration Space (Joint Space) vs Workspace Demonstration**:
```bash
python3 ik_trajectories/test_joint_line.py
```

6. **Detailed C-Space vs Workspace Motion Strategy Comparison**:
```bash
python3 ik_trajectories/compare_cspace_workspace.py
```

---

## Repository Structure

```
refuel-arm/
├── test_full_pipeline.py            # Main STOMP Refueling execution orchestrator
├── analyze_pipeline.py              # STOMP 4-panel analysis graph generator
├── stomp_planner.py                 # Core standalone STOMP trajectory optimizer
├── ik_trajectories/                 # 6-DOF Topological Tracking Scripts
│   ├── analyze_ik_accuracy.py       # Empirical 3,830-root mathematical precision benchmark
│   ├── test_ik_line.py              # Pure algebraic IK Cartesian line tracker
│   ├── test_joint_line.py           # Configuration Space (Joint Space) linear tracker
│   ├── compare_cspace_workspace.py  # Dual-strategy comparison and visualization
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
│       └── Validation/
│           └── program_flowchart.md #   Step-by-step logic for validation pipeline
│
├── deprecated/                      # Previous approaches (IKFlow, MATLAB, instructions)
├── report/                          # 8-page LaTeX report (Mathematical Benchmarking & Step-by-Step Analysis)
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

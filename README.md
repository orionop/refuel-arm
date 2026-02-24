# KUKA KR6 R700 — Autonomous Refueling Simulation

<p align="center">
  <b>Systems and Control Engineering (SysCon), IIT Bombay</b><br>
  Maintainer: <a href="https://github.com/orionop">Anurag Shetye</a>
</p>

---

## Overview

This repository implements an end-to-end simulation pipeline for autonomous car refueling using a **KUKA KR6 R700** 6-DOF industrial manipulator. The robot executes a precision autonomous sequence: `HOME` (Straight Up) → `YELLOW` (Grab Nozzle) → `RED` (Refuel 10s) → `YELLOW` (Return Nozzle) → `HOME`. The mission is executed within a physically accurate **ROS Noetic / Gazebo** environment.

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

> **Prerequisites:** Ubuntu 20.04 with ROS Noetic, Gazebo, and an NVIDIA GPU.

```bash
# 1. Clone
cd /home/armslab-exp4/Desktop/anurag_ws/src
git clone https://github.com/orionop/refuel-arm.git
cd refuel-arm

# 2. Build ROS Noetic workspace
source /opt/ros/noetic/setup.bash
cd kuka_refuel_ws
catkin_make
source devel/setup.bash
cd ..

# 3. Launch purely in RViz (No Physics)
python3 test_full_pipeline.py --rviz

# OR: Launch in Gazebo (Full Physics)
# Terminal 1:
roslaunch kuka_kr6_gazebo gazebo.launch
# Terminal 2:
python3 test_full_pipeline.py --ros
```

---

## Repository Structure

```
refuel-arm/
├── test_full_pipeline.py            # Main mission execution logic
├── stomp_planner.py                 # Core STOMP trajectory optimizer
├── ik-geo/                          # Algebraic IK library (MATLAB/Python/C++/Rust)
│
├── deprecated/                      # Previous ML approaches (IKFlow, Kaggle, JRL)
│
├── kuka_refuel_ws/                  # ROS Noetic catkin workspace
│   └── src/kuka_kr6_gazebo/
│       ├── urdf/                    #   URDF with accurate physical inertials
│       ├── config/                  #   gazebo_ros_controllers & RViz config
│       ├── launch/                  #   gazebo.launch and rviz.launch
│       ├── worlds/                  #   Gazebo world with Yellow & Red markers
│       └── scripts/
│           ├── ik_geometric.py      #   Python port of IK_spherical_2_parallel
│
├── SAFE_DEV_RULES.md                # Shared lab machine safety protocol
└── README.md
```

---

## Shared Lab Machine Protocol

This project runs on a shared research machine. See [`SAFE_DEV_RULES.md`](SAFE_DEV_RULES.md) for the complete rulebook. Key rules:

- **No `sudo`** — no global package installations or removals
- **No modifications** to `/opt/ros/humble`, `~/.bashrc`, or system Python
- **Workspace isolation** — all work inside `/home/admin/Desktop/anurag_ws`
- **GPU courtesy** — run `nvidia-smi` before launching training; never kill others' processes

---

## License

IK-Geo: MIT License (RPI Robotics)  
STOMP: Derived from Kalakrishnan et al. (2011)  
KUKA Robot Descriptions: Apache 2.0 (KUKA)

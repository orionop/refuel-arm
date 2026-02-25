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

> **Prerequisites:** Ubuntu 20.04 with ROS Noetic, Gazebo, and an NVIDIA GPU (optional for simulation).

```bash
# 1. Clone into your ROS workspace
mkdir -p ~/kuka_ws/src
cd ~/kuka_ws/src
git clone https://github.com/orionop/refuel-arm.git
cd refuel-arm

# 2. Build ROS Noetic workspace
source /opt/ros/noetic/setup.bash
cd kuka_refuel_ws
catkin_make
source devel/setup.bash
cd ..
```

### Scenario A: Full Refueling Mission (IK + STOMP)
Executes the complex `HOME → YELLOW → RED → YELLOW → HOME` mission.

```bash
# Launch purely in RViz (No Physics)
python3 test_full_pipeline.py --rviz

# OR: Launch in Gazebo (Full Physics)
# Terminal 1:
roslaunch kuka_kr6_gazebo gazebo.launch
# Terminal 2:
python3 test_full_pipeline.py --ros
```

### Scenario B: Pure IK-Geo 6-DOF Cartesian Line Tracking
Demonstrates exact mathematical line tracking without a trajectory optimizer (generates 60 dense waypoints dynamically). Supports full 6-DOF control including dynamic end-effector orientation (wrist twisting) via SLERP axis-angle interpolation.

```bash
# Run the default straight line sweep with a 45° twist
python3 test_ik_line.py --ros

# Test specific dynamic coordinates and a custom 90° twist
python3 test_ik_line.py --ros --start 0.3 0.4 0.5 --end 0.65 -0.25 0.45 --twist 90
```

### Scenario C: Dynamic Orientation Wave Tracking (Human Painting Motion)
Demonstrates a multi-cycle Cartesian sine wave where the end-effector dynamically adjusts its pitch to stay tangent to the curve's analytical derivative (resembling a "painting" motion) while simultaneously applying a continuous wrist twist.

```bash
# Run the wave trajectory in Gazebo
python3 test_ik_wave.py --ros
```

### Scenario D: Hyperbolic Paraboloid (Pringle) Tracking
Demonstrates navigating a purely 3D multi-axis curve (a circle in XY with a saddle-like variation in Z). The orientation dynamically computes the 3D cross-product normal to tilt the tool completely tangent to the swoop.

```bash
python3 test_ik_pringle.py --ros
```

### Scenario E: Non-Orientable Topological Tracking (Möbius Strip)
Acts as the ultimate algebraic stress test. The robot traces the continuous edge of a Möbius strip, sweeping a massive $4\pi$ (720°) loop. Computes the instantaneous topological tangent to safely handle the topological inversion without an elbow flip.

```bash
python3 test_ik_mobius.py --ros
```

---

## Repository Structure

```
refuel-arm/
├── test_full_pipeline.py            # STOMP Refueling mission execution
├── test_ik_line.py                  # Pure algebraic IK Cartesian line tracker
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

See [`SAFE_DEV_RULES.md`](SAFE_DEV_RULES.md) for the original lab protocol guidelines used during the development of this repository. Key historical rules included restricted `sudo` access, strict workspace isolation, and NVIDIA GPU sharing courtesy.

---

## License

IK-Geo: MIT License (RPI Robotics)  
STOMP: Derived from Kalakrishnan et al. (2011)  
KUKA Robot Descriptions: Apache 2.0 (KUKA)

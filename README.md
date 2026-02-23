# KUKA KR6 R700 — Autonomous Refueling Simulation

<p align="center">
  <b>Systems and Control Engineering (SysCon), IIT Bombay</b><br>
  Maintainer: <a href="https://github.com/orionop">Anurag Shetye</a>
</p>

---

## Overview

This repository implements an end-to-end simulation pipeline for autonomous car refueling using a **KUKA KR6 R700** 6-DOF industrial manipulator. The robot executes a precision refueling sequence: departing from a rest configuration (`HOME`), navigating to a vehicle's refueling inlet (`TARGET` at `[0.3, 0.4, 0.25]` m), holding for 5–7 seconds, and returning to rest — all within a physically accurate **ROS 2 Humble / Gazebo** environment.

The core innovation combines two complementary inverse kinematics strategies:

| Component | Role | Precision |
|-----------|------|-----------|
| **IK-Geo** | Exact algebraic closed-form IK for terminal pose | ~10⁻¹⁶ rad |
| **IKFlow** | Neural network trajectory generation via Normalizing Flows | Collision-free paths |

---

## Technical Approach

### Terminal Pose — IK-Geo (Exact Algebraic Solver)

The KR6 R700 belongs to the `IK_spherical_2_parallel` kinematic family. IK-Geo decomposes the 6-DOF inverse kinematics into a sequence of canonical subproblems (Paden–Kahan), yielding **up to 8 closed-form solutions** per pose — no iterative methods, no local minima, no convergence failures.

### Trajectory Planning — IKFlow (Neural Network)

[IKFlow](https://github.com/jstmn/ikflow) is a Normalizing Flow network trained on 25M valid joint configurations. It generates diverse, collision-free waypoints along the Cartesian path from `HOME` to the IK-Geo terminal solution, providing smooth seed states for the `JointTrajectoryController`.

---

## Quick Start

> **Prerequisites:** Ubuntu 22.04 with ROS 2 Humble, Gazebo, and an NVIDIA GPU.

```bash
# 1. Clone
cd /home/admin/Desktop/anurag_ws
git clone https://github.com/orionop/refuel-arm.git
cd refuel-arm

# 2. Python virtual environment
python3 -m venv venv
source venv/bin/activate
pip install linearSubproblemSltns

# 3. Train IKFlow (check GPU availability first: nvidia-smi)
cd ikflow
pip install -e .
python scripts/build_dataset.py --robot_name=kr6_r700 \
    --training_set_size=25000000 --only_non_self_colliding
python scripts/train.py --robot_name=kr6_r700 \
    --nb_nodes=12 --batch_size=128 --learning_rate=0.0005
cd ..

# 4. Build ROS 2 workspace
source /opt/ros/humble/setup.bash
cd kuka_refuel_ws
colcon build
source install/setup.bash

# 5. Launch simulation
ros2 launch kuka_kr6_gazebo refuel_sim.launch.py
```

---

## Repository Structure

```
refuel-arm/
├── matlab/                          # MATLAB IK-Geo verification scripts
│   ├── +hardcoded_IK/KR6_R700.m    #   Hardcoded IK solver
│   ├── robot_examples/KR6_R700.m   #   URDF-to-POE conversion & 3D visualizer
│   ├── kr6_r700_2_clean.urdf       #   Clean URDF (no mesh dependencies)
│   └── solve_specific_pose.m       #   IK validation for user-defined poses
│
├── ik-geo/                          # Algebraic IK library (MATLAB/Python/C++/Rust)
│   ├── python/                      #   Python canonical subproblems (sp1–sp6)
│   └── matlab/+IK/                  #   IK_spherical_2_parallel solver
│
├── ikflow/                          # IKFlow neural network training pipeline
│   ├── scripts/train.py             #   Training entry point
│   ├── scripts/build_dataset.py     #   Dataset generator
│   └── ikflow/model.py              #   Normalizing Flow architecture
│
├── kuka_refuel_ws/                  # ROS 2 Humble colcon workspace
│   └── src/kuka_kr6_gazebo/
│       ├── urdf/                    #   URDF + ros2_control Xacro
│       ├── config/                  #   ros2_controllers.yaml
│       ├── launch/                  #   refuel_sim.launch.py
│       ├── worlds/                  #   Gazebo world with target inlet
│       └── scripts/
│           ├── ik_geometric.py      #   Python port of IK_spherical_2_parallel
│           └── refuel_mission_commander.py  # ROS 2 action client node
│
├── kuka_robot_descriptions/         # Official KUKA URDF + meshes (KR6 R700 only)
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
IKFlow: MIT License  
KUKA Robot Descriptions: Apache 2.0 (KUKA)

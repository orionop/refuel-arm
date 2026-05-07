# Autonomous Robotic Refueling Architecture

**Author:** Anurag Shetye  
**Affiliation:** Systems and Control Engineering, Indian Institute of Technology Bombay (IIT Bombay)

This repository contains the source code for a novel autonomous motion planning architecture designed for unstructured environments. It overcomes the limitations of traditional iterative Inverse Kinematics (IK) solvers and heavy sampling-based global planners by integrating three deeply specialized algorithms into a single continuous pipeline.

Currently targets **ROS 2 Jazzy + Gazebo Harmonic (Gz Sim)**.

## System Architecture

The motion planning stack seamlessly transitions between Configuration Space (C-Space) and Workspace (Cartesian) tracking:

1. **Algebraic Inversion (IK-Geo)**: Constant-time, exact spatial tracking based on Paden-Kahan subproblems.
2. **Derivative-Based Selector**: A custom physics filter evaluating Velocity, Acceleration, and Jerk over a rolling history buffer.
3. **Stochastic Optimization (STOMP)**: Gradient-free global planner that generates smooth, collision-free gross motion trajectories avoiding Cartesian obstacles.
4. **Geometric Reactivity (Bubble Strips)**: Real-time topological deflection via continuous collision evasion using Jacobian Transpose mapping.
5. **Force-Compliant Admittance Control**: Closed-loop admittance layer ($M\ddot{x} + D\dot{x} + Kx = F_{ext}$) enabling safe nozzle insertion via force-to-joint mapping.

## Directory Structure

- `kuka_refuel_ws/`: The main ROS 2 workspace containing the simulation, robot descriptions, and controllers for KUKA KR6 R700 and UR5.
- Core Motion Planning (Pure Python):
  - `ik_geometric.py`
  - `stomp_collision.py`
  - `bubble_strips.py`
  - `admittance_controller.py`
- Mission Execution:
  - `refuel_mission.py`
  - `admittance_node.py`
  - `animate_gazebo.py`
- `deprecated/`: Old iterations, unused scripts, ROS 1 Noetic code, and temporary scratchpads. Stored here to maintain a clean root workspace.

## Getting Started

### Prerequisites
- Ubuntu 24.04
- ROS 2 Jazzy
- Gazebo Harmonic (Gz Sim)
- Python 3.12+

### Build the ROS 2 Workspace

```bash
# Source ROS 2
source /opt/ros/jazzy/setup.bash

# Build the workspace
cd kuka_refuel_ws
colcon build --symlink-install
source install/setup.bash
```

### Launch the Simulation

```bash
# For KUKA KR6 R700
ros2 launch kuka_kr6_gazebo gazebo.launch.py

# For UR5
ros2 launch ur5_gazebo ur5_refuel.launch.py
```

### Run the Refueling Mission

Once the simulation is running, execute the 4-phase hybrid mission orchestrator:

```bash
# KUKA Execution
python3 refuel_mission.py --ros --robot kuka

# UR5 Execution (with Force-Compliant insertion)
python3 refuel_mission.py --ros --robot ur5 --compliant

# Dry-run / Visualization only
python3 refuel_mission.py --robot kuka
```

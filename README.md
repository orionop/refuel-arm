# Tri-Layered Hybrid Kinematic Architecture for 6-DOF Manipulators

**Author:** Anurag Shetye  
**Affiliation:** Systems and Control Engineering, Indian Institute of Technology Bombay (IIT Bombay)

This repository contains the source code for a novel autonomous motion planning architecture designed for unstructured environments. It overcomes the limitations of traditional iterative Inverse Kinematics (IK) solvers and heavy sampling-based global planners by integrating three deeply specialized algorithms into a single continuous pipeline.

> **Status:** ROS 2 Jazzy migration complete. All simulation infrastructure now targets **ROS 2 Jazzy + Gazebo Harmonic (Gz Sim)**. ROS 1 Noetic code preserved in `deprecated/`.

---

## System Architecture

The overarching pipeline seamlessly transitions between Configuration Space (C-Space) and Workspace (Cartesian) tracking:

1. **Algebraic Inversion (IK-Geo)**: Guarantees constant-time, exact spatial tracking ($10^{-16}\text{m}$ precision) based on Paden-Kahan subproblems. Gracefully handles non-orientable topological manifolds (e.g., Möbius strips).
2. **Derivative-Based Selector**: A custom physics filter that evaluates Velocity, Acceleration, and Jerk over a rolling trajectory history buffer, mathematically bridging algebraic roots with hardware safety limits.
3. **Stochastic Optimization (STOMP)**: An uncoupled, pure-NumPy gradient-free global planner that generates inherently smooth, collision-free gross motion trajectories avoiding identified Cartesian obstacles using 2.5D segment-based bounding cylinders.
4. **Geometric Reactivity (Bubble Strips)**: A completely custom 6-DOF reactive layer (based on Quinlan & Khatib) executing continuous collision evasion via Jacobian Transpose ($J^T$) mapping, enabling real-time topological deflection.
5. **Force-Compliant Admittance Control**: A closed-loop admittance layer ($M\ddot{x} + D\dot{x} + Kx = F_{ext}$) enabling safe nozzle insertion via $J^T$ force-to-joint mapping, with three operating modes: RIGID / COMPLIANT / ABORT.

---

## ROS Stack (ROS 2 Jazzy + Gazebo Harmonic)

**Requires:** Ubuntu 24.04, ROS 2 Jazzy, Gazebo Harmonic (Gz Sim), Python 3.12+

```bash
# Source ROS 2 and workspace
source /opt/ros/jazzy/setup.bash
cd ~/Desktop/anurag_ws/refuel-arm/kuka_refuel_ws
colcon build --symlink-install
source install/setup.bash

# Launch KUKA KR6 R700 in Gz Sim
ros2 launch kuka_kr6_gazebo gazebo.launch.py

# Launch UR5 in Gz Sim
ros2 launch ur5_gazebo ur5_refuel.launch.py

# Launch RViz (KUKA)
ros2 launch kuka_kr6_gazebo rviz.launch.py

# Launch RViz (UR5)
ros2 launch ur5_gazebo ur5_rviz.launch.py
```

### ROS 2 Package Structure

```
kuka_refuel_ws/src/
├── kuka_kr6_gazebo/       # KUKA KR6 R700 — description, worlds, launch, config
│   ├── urdf/              # kr6_r700.gazebo.xacro  (gz_ros2_control hardware interface)
│   ├── launch/            # gazebo.launch.py, rviz.launch.py, refuel_sim.launch.py
│   ├── config/            # ros2_controllers.yaml
│   └── worlds/            # refuel_world.sdf (Gz Sim SDF 1.9)
├── ur5_gazebo/            # UR5 — launch, config
│   ├── launch/            # ur5_refuel.launch.py, ur5_rviz.launch.py
│   └── config/            # ur5_ros2_controllers.yaml
└── ur_description/        # UR5 URDF (gz_ros2_control hardware interface + F/T sensor)
```

---

## Motion Planning Stack (Pure Python — ROS-independent)

These modules have **no ROS dependency** and run standalone:

| Module | Role |
|--------|------|
| `ik_geometric.py` | Algebraic closed-form IK (IK-Geo) for KUKA KR6 + UR5 |
| `stomp_collision.py` | Pure-NumPy STOMP optimizer with sphere-sweep collision checks |
| `bubble_strips.py` | Reactive Elastic Strips via Jacobian Transpose |
| `admittance_controller.py` | Admittance math ($M\ddot{x}+D\dot{x}+Kx=F_{ext}$) — no ROS |

---

## Mission Orchestrator

**`refuel_mission.py`** — 4-phase hybrid mission (HOME → Pre-approach → Target → Pre-approach → HOME):

```bash
# Dry-run (no ROS — planning + graphs only)
python3 refuel_mission.py --robot kuka
python3 refuel_mission.py --robot ur5

# Execute in Gz Sim (ROS 2)
python3 refuel_mission.py --ros --robot kuka
python3 refuel_mission.py --ros --robot ur5

# Force-compliant insertion/extraction (UR5 + F/T sensor)
python3 refuel_mission.py --ros --robot ur5 --compliant

# RViz visualization only
python3 refuel_mission.py --rviz --robot kuka
```

**`admittance_node.py`** — Standalone ROS 2 force-compliant execution node:
```bash
python3 admittance_node.py
```

---

## Empirical Benchmarks

```bash
# 15-seed Bubble Strips robustness benchmark
python3 test_15seed_bubbles.py
```

---

## Deprecation Policy

All ROS 1 Noetic files (launch XML, catkin build files, rospy nodes) are preserved in `deprecated/` and never deleted:

```
deprecated/
├── ros1_build/    # ROS1 package.xml, CMakeLists.txt, xacro, URDF, world
├── ros1_launch/   # ROS1 XML launch files
├── *.ros1.py      # ROS1 Python nodes (rospy)
└── ...            # Legacy pipeline variants
```

---

*This repository is actively maintained for academic publication and hardware deployment. Not licensed for commercial use without explicit permission.*

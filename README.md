# Tri-Layered Hybrid Kinematic Architecture for 6-DOF Manipulators

This repository contains the source code for a novel autonomous motion planning architecture designed for unstructured environments. It overcomes the limitations of traditional iterative Inverse Kinematics (IK) solvers and heavy sampling-based global planners by integrating three deeply specialized algorithms into a single continuous pipeline.

## System Architecture

The overarching pipeline seamlessly transitions between Configuration Space (C-Space) and Workspace (Cartesian) tracking:

1. **Algebraic Inversion (IK-Geo)**: Guarantees constant-time, exact spatial tracking ($10^{-16}\text{m}$ precision) and gracefully handles non-orientable topological manifolds (e.g., Möbius strips) without singularity collapse.
2. **Derivative-Based Selector**: A custom physics filter that evaluates Velocity, Acceleration, and Jerk over a rolling trajectory history buffer, mathematically bridging algebraic roots with physical hardware safety limits for standard 6-DOF manipulators.
3. **Stochastic Optimization (STOMP)**: An uncoupled, pure-NumPy gradient-free global planner that generates inherently smooth, collision-free gross motion trajectories avoiding identified Cartesian obstacles.
4. **Newtonian Reactivity (Elastic Strips)**: A 6-DOF reactive layer executing localized obstacle evasion via Jacobian Transpose ($J^T$) mapping, enabling real-time topological deflection during rigid Cartesian insertion tasks.

## Core Components

- `refuel_mission.py`: The primary orchestrator executing the 4-phase hybrid mission. Contains the definitive Velocity/Acceleration/Jerk derivative filtering logic and Cartesian obstacle detection.
- `stomp_collision.py`: The standalone NumPy stochastic optimizer with high-speed bounding sphere penalty checks.
- `elastic_strips.py`: The reactive physics engine computing instantaneous repulsive joint-torques for collision eviction.
- `ik_trajectories/`: Topological stress tests (e.g., $4\pi$ multi-axis boundary tracking, multi-namespace visualizers).

---
*This repository is actively maintained for academic publication drafting. Do not use for commercial deployment without license clarification.*

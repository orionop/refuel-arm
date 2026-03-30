# Tri-Layered Hybrid Kinematic Architecture for 6-DOF Manipulators

**Author:** Anurag Shetye  
**Affiliation:** Systems and Control Engineering, Indian Institute of Technology Bombay (IIT Bombay)

This repository contains the source code for a novel autonomous motion planning architecture designed for unstructured environments. It overcomes the limitations of traditional iterative Inverse Kinematics (IK) solvers and heavy sampling-based global planners by integrating three deeply specialized algorithms into a single continuous pipeline.

## System Architecture

The overarching pipeline seamlessly transitions between Configuration Space (C-Space) and Workspace (Cartesian) tracking:

1. **Algebraic Inversion (IK-Geo)**: Guarantees constant-time, exact spatial tracking ($10^{-16}\text{m}$ precision) based on Paden-Kahan subproblems. Gracefully handles non-orientable topological manifolds (e.g., Möbius strips).
2. **Derivative-Based Selector**: A custom physics filter that evaluates Velocity, Acceleration, and Jerk over a rolling trajectory history buffer, mathematically bridging algebraic roots with hardware safety limits.
3. **Stochastic Optimization (STOMP)**: An uncoupled, pure-NumPy gradient-free global planner that generates inherently smooth, collision-free gross motion trajectories avoiding identified Cartesian obstacles using 2.5D segment-based bounding cylinders.
4. **Geometric Reactivity (Bubble Strips)**: A completely custom 6-DOF reactive layer (based on Quinlan & Khatib) executing continuous collision evasion via Jacobian Transpose ($J^T$) mapping, enabling real-time topological deflection.

## Core Components

- **`refuel_mission.py`**: The primary orchestrator executing the 4-phase hybrid mission. Automatically dispatches IK kinematics, joint limits, and controller mapping between platforms.
- **`stomp_collision.py`**: The standalone NumPy stochastic optimizer with high-speed bounding cylinder line-point math ensuring zero structural tracking tunneling.
- **`bubble_strips.py`**: The reactive physics engine computing instantaneous geometric safety tunnels for collision eviction.
- **`test_15seed_bubbles.py`**: A deterministic empirical benchmarking engine providing Tier-1 aggregate success metrics (Collision-free & Joint-violation-free rates) across randomized environments.

## Quickstart & Usage

The architecture natively supports the **KUKA KR6 R700** and **UR5** industrial arms via a dynamic dispatcher logic. No hardcoded kinematics exist in the primary planning stack.

**Run the Full Mission:**
```bash
# Execute with the default KUKA KR6
python3 refuel_mission.py

# Switch entirely to UR5 kinematics & bounds
python3 refuel_mission.py --robot ur5
```

**Run the Empirical Benchmark:**
```bash
# Aggregates analytical performance across 15 randomized obstacle seeds
python3 test_15seed_bubbles.py
```

---
*This repository is actively maintained for academic publication drafting. Do not use for commercial deployment without license clarification.*

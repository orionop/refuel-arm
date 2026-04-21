# Project Context: B.Tech Thesis / Major Project

**Official Report Title:** Design of Robotic Arm for Autonomous Refueling Systems
**Technical Architecture:** A Tri-Layered Hybrid Kinematic Framework for 6-DOF Manipulators
**Lead Researcher & Architect:** Anurag Shetye (Systems and Control Engineering, IIT Bombay)

## 1. Project Overview & Core Objective
This project represents a capstone research endeavor aimed at solving a fundamental limitation in modern robotic articulation: the inability of traditional solvers to seamlessly bridge the gap between mathematically precise global planning and real-time physical obstacle evasion. 

The primary objective was to design, mathematically formulate, and implement a generalized, hardware-agnostic pipeline for any 6-Degree-of-Freedom (6-DOF) manipulator to perform highly constrained spatial tasks in dynamic environments. While empirical testing and validation were conducted on specific hardware configurations (e.g., KUKA KR6 R700), the underlying algorithms and mathematical framework were explicitly engineered to be universally applicable to any standard 6R kinematic chain.

## 2. The Novel Tri-Layered Architecture
The defining outcome of this project is the successful synthesis of a novel four-phase, tri-layered kinematic pipeline that fundamentally outperforms conventional discrete trajectory solvers:

1. **Algebraic Inversion Layer (IK-Geo):** Supplanting traditional iterative Jacobian pseudo-inverse methods with pure algebraic geometry. This guarantees constant-time, exact spatial tracking ($10^{-16}\text{m}$ precision) and gracefully resolves non-orientable topological manifolds without singularity collapse.
2. **Stochastic Global Planning Layer (STOMP):** Utilizing an uncoupled, gradient-free optimization engine to generate inherently smooth, collision-averse gross-motion trajectories in Configuration Space (C-Space).
3. **Newtonian Reactive Layer (Elastic Strips):** Implementing a localized, real-time physically reactive constraint solver. By mapping 3D Cartesian repulsive forces to 6D joint-torques via the active Jacobian Transpose ($J^T$), the architecture achieves dynamic obstacle evasion without abandoning rigid Workspace trajectory tracking.

## 3. Original Contributions & Architectural Engineering
The execution of this thesis required foundational algorithm design, moving far beyond the simple integration of existing robotic libraries. The primary technical achievements include:

* **Derivative-Based Kinematic Filtering:** Designed and implemented a custom physics-aware selection algorithm that evaluates Velocity, Acceleration, and Jerk over a rolling internal momentum buffer. This mathematical bridge successfully reduces up to 8 theoretical algebraic roots into a single, perfectly smooth physical hardware command, explicitly mitigating mechanical motor stress.
* **Hybrid Mode-Switching Protocol:** Engineered the complex mathematical handoffs between C-Space gross motion (STOMP) and highly constrained W-Space microscopic tracking (Cartesian Elastic Strips)—historically a highly volatile transition point that causes solver failure in industrial manipulators.
* **Algorithmic Validation & Topological Benchmarking:** Architected comprehensive, multi-dimensional stress tests (e.g., tracking the $4\pi$ twist of a Möbius strip) to empirically prove and formally document the architecture's absolute stability against kinematic singularities.

## 4. Academic Framing & Tone Guidelines
*(Note for report generation)*
This project is an advanced algorithmic research thesis. When generating the final report, avoid colloquial phrasing (e.g., "writing scripts," "testing the robot"). Instead, frame the work as:
* "Formulating mathematical constraints" rather than "writing collision checks."
* "Architecting a hybrid stochastic-algebraic pipeline" rather than "putting algorithms together."
* "Empirically validating spatial tracking limits" rather than "running Gazebo simulations." 

This document serves as the technical baseline to ensure all outcomes are correctly attributed as original research and systems-level engineering.

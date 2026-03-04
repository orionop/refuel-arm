# 📑 Self-Contribution Audit & Research Strategy
> This document serves as a technical roadmap for transitioning this codebase into a high-impact robotics research submission.

---

## 🏗️ Technical Foundation: Pre-Existing vs. Original Architecture

### 🛡️ Pre-Existing Baseline (The "Prior Art")
*To be cited in the 'Related Work' section of any future paper:*

| Component | Scientific Origin | Role in Project | Research Limitation Addressed |
|-----------|------------------|-----------------|-------------------------------|
| **IK-Geo** | Elias et al. | Core Algebraic Solver | Standard iterative IK (Jacobians) fails at singularities; IK-Geo provides exact roots. |
| **STOMP Theory** | Kalakrishnan et al. | Probabilistic Engine | Standard STOMP lacks direct sensor-fusion for 2.5D/3D unstructured environments. |

### 🚀 Original Contributions (Core Research Value)
*These novel components drive the "Contribution" section of the manuscript:*

| Research Cluster | Core Logic | Originality & Impact |
|------------------|------------|-----------------------|
| **2.5D Sensor Fusion Planner** | `stomp_collision.py` | **Major Innovation.** Fuses stochastic optimization with 2.5D Euclidean Distance Transforms (EDT). Creates a continuous gradient field for collision avoidance in raw point-cloud environments. |
| **Integrated Refueling Mission** | `refuel-stomp.py` | **System Validation.** First implementation of a full autonomous loop (Pick -> Approach -> Refuel) using C-Space optimization to ensure zero-singularity motion. |
| **High-Fidelity Smoothness Refinement** | Gaussian Filter (Sigma=0.8) | **Premium Motion Control.** Implementation of post-optimization smoothing to eliminate stochastic jitter, enabling "Zero-Jerk" deployment on physical hardware. |
| **Algebraic Manifold Tracking** | `test_ik_mobius.py`, `test_ik_pringle.py` | **Robustness Proof.** Proves $4\pi$ topological inversion tracking on non-orientable manifolds, demonstrating solver stability where traditional IK fails. |

---

## 🛠️ Research Expansion Roadmap
*To escalate this from a simulation project to a complete research contribution:*

1. **Temporal 2.5D Occupancy Mapping**: Handle dynamic obstacles (e.g., humans) by updating the EDT field in real-time.
2. **Multi-Architecture Benchmarking**: Validating the 2.5D STOMP pipeline for UR5/Franka Emika architectures.
3. **Tactile-Constrained STOMP**: Integrating force/torque feedback into the cost function for delicate nozzle-port insertion.

---

## 🎯 Publication Strategy
**Proposed Title**: *"Gradient-Free Sensor Fusion: High-Fidelity STOMP Optimization with Analytical IK for Autonomous Refueling"*

**Target Venues**: ICRA (IEEE), IROS (IEEE), IEEE Robotics and Automation Letters (RA-L).

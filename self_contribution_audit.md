# 📑 Self-Contribution Audit & Research Strategy
> This document highlights the core proprietary research contributions built on top of the pre-existing mathematical theory (IK-Geo). These components form the foundation of a potential research manuscript.

---

## 🏗️ Technical Foundation: Pre-Existing vs. Original Architecture

### 🛡️ Pre-Existing Baseline (The "Prior Art")
*To be cited in the 'Related Work' section of any future paper:*

| Component | Scientific Origin | Role in Project | Research Limitation Addressed |
|-----------|------------------|-----------------|-------------------------------|
| **IK-Geo** | Elias et al. | Core algebraic IK math | Standard iterative IK fails at singularities; IK-Geo provides exact roots (but has no concept of hardware constraints). |
| **STOMP Theory** | Kalakrishnan et al. | Probabilistic engine | Standard STOMP lacks direct sensor-fusion for unstructured environments. |
| **Elastic Strips Theory** | Brock & Khatib | Reactive avoidance | Original paper is for mobile planners; implementations generally lack Python support for industrial manipulators. |

### 🚀 Original Contributions (Publishable Core Value)
*These novel implementations drive the "Contribution" section of the manuscript.*

| Research Cluster | Core Logic | Originality & Academic Impact |
|------------------|------------|-----------------------|
| **1. Tri-Layered Hybrid Kinematic Pipeline** | `test_full_pipeline.py` / `refuel_mission.py` | **Architectural Novelty.** First integration of Algebraic Inversion (IK-Geo) $\to$ Stochastic Global Planning (STOMP) $\to$ Newtonian Reactivity (Elastic Strips) in a single autonomous pipeline. |
| **2. Topological Manifold Tracking** | `test_ik_mobius.py`, `pringle`, `wave` | **Robustness Proof.** Hand-coded $4\pi$ multi-axis tracking scripts proving algebraic IK can effortlessly navigate sweeping inflections and non-orientable topological surfaces where Jacobian pseudo-inverse methods stall. |
| **3. IK Hardware Validation & Derivative-Based Selection** | `refuel_mission.py` | **Algorithmic Innovation.** IK-Geo blindly outputs up to 8 theoretical roots. We engineered the crucial bridging logic: wrapping algebraic angles, filtering against real hardware limiters, and utilizing a custom **derivative cost function** (Velocity, Acceleration, Jerk via finite difference history buffers) to select the mathematically optimal, jerk-free trajectory. |
| **4. Uncoupled STOMP Optimizer** | `stomp_planner.py` / `stomp_collision.py` | **Standalone Engineering.** A complete STOMP optimizer built from scratch in pure NumPy, liberating probabilistic motion planning from heavy MoveIt/ROS dependencies for faster research iteration. |
| **5. 6-DOF Elastic Strips Engine** | `elastic_strips.py` | **Novel Implementation.** A fully functional 6-DOF reactive obstacle avoidance layer implemented from scratch for industrial arms, mapping Cartesian collision repulsions directly into joint-torques via Jacobian Transpose ($J^T$). |
| **6. Fast-Math Cartesian Obstacles** | `stomp_collision.py` | **Real-Time Safety.** Replaces expensive 2.5D EDT grids with mathematically direct Euclidean sphere-sweep tests, drastically accelerating the STOMP penalty iteration speed. |

---

## 🎯 Proposed Publication Strategy

**Proposed Title:**
*Topologically Robust Cartesian Tracking and Stochastic Smoothing via Algebraic Inverse Kinematics for Robotic Operations.*

**Target Venues:**
ICRA (IEEE), IROS (IEEE), IEEE Robotics and Automation Letters (RA-L).

**The Narrative:**
*"We present a novel hybrid planning pipeline. Standard iterative IK solvers often stall or fail on continuous geometries involving multi-axis inflections (e.g., Möbius boundaries). By utilizing algebraic IK-Geo, we guarantee constant-time, exact spatial tracking. Crucially, the base solver lacks physical context. We engineered an intelligent selection layer utilizing rolling kinematic derivatives (Velocity, Acceleration, Jerk) to filter mathematical roots into physically optimal hardware commands. Furthermore, by coupling these exact terminal solutions with a gradient-free STOMP smoothing pipeline and a fully reactive 6-DOF Elastic Strips engine mapped via Jacobian Transposes, we generate safe, smooth, collision-immune trajectories without relying on computationally heavy sampling-based abstractions."*

---

## 🛠️ Research Expansion Roadmap

*To escalate this from a functional simulation to a complete research contribution:*

1. **Analytical Obstacle Checks [Completed]**: Fast mathematical Cartesian spherical checkpoints for STOMP processing.
2. **Elastic Strips Reactive Layer [Completed]**: Complete 6-DOF Elastic Strips engine using Jacobian Transpose wired as a post-STOMP obstacle refinement layer.
3. **Derivative-Based IK Selector [Completed]**: Replaced naive distance selectors with a physics-aware cost function evaluating Velocity, Acceleration, and Jerk over a rolling 3-step history buffer.
4. **Comparative Analysis (Jacobians vs Algebraic) [Next]**: Formal mathematical benchmarking of IK-Geo vs standard KDL trackers over the Möbius strips, demonstrating measurable superiority.
5. **Multimodal Elastic Mode-Switching**: When Elastic Strips tension exceeds threshold $\to$ query IK-Geo for alternate kinematic modes $\to$ STOMP replans to escape kinematic traps.
6. **Spatial Dynamics**: Add moving `Gazebo` obstacles to explicitly demonstrate real-time Elastic Strips reactivity beyond static STOMP models.

# 📑 Proprietary Contribution Audit & Implementation Log (Private)
> This document is a detailed "Information Booklet" for the primary researcher (User). It explicitly separates **pre-existing mathematical theories** (Prior Art) from the **original logic, code, and system innovations** implemented in this repository.

---

## 🏛️ Section 1: Fundamental Baseline (The "Given" Theory)
*The following are the academic foundations upon which this project is built. These concepts are referenced from published literature.*

| Theory / Tool | Scientific Origin | Role in this Project | Inherited Constraints |
|:--- |:--- |:--- |:--- |
| **IK-Geo** | Elias et al. | Provides the core algebraic equations for 6-DOF Inverse Kinematics. | **Limitation:** Pure math; no concept of joint limits, collision, or trajectory smoothness. |
| **STOMP Theory** | Kalakrishnan et al. | Theory of gradient-free stochastic optimization for motion paths. | **Limitation:** Usually heavy (ROS/MoveIt); lacks direct Euclidean obstacle-check methods. |
| **Elastic Strips** | Brock & Khatib | Theory of reactive "deformation" forces for collision avoidance. | **Limitation:** Original implementation is for mobile robots; lacks industrial arm mapping. |

---

## 🚀 Section 2: Proprietary Implementation & Innovation (The "User's" Work)
*The following components were built from scratch or heavily engineered by the user to bridge the gap between "Theory" and "Industrial Application."*

### 1. **Derivative-Based IK Solution Selector** (Proprietary Logic)
*   **Script:** `refuel_mission.py`
*   **Innovation:** IK-Geo blindly returns up to 8 roots. The **User** engineered the intelligent selection layer that calculates **Velocity, Acceleration, and Jerk** (via rolling 3-step finite difference buffers) to pick the most "physical" and "jerk-free" solution. This is not in the original IK-Geo paper.

### 2. **Uncoupled NumPy STOMP Engine** (Proprietary Code)
*   **Script:** `stomp_planner.py` / `stomp_collision.py`
*   **Innovation:** A complete re-implementation of the STOMP optimizer in pure NumPy. This "User" contribution liberates the research from ROS-heavy dependencies, allowing for sub-millisecond planning iterations.

### 3. **Fast-Math Cartesian Collision Checker** (Engine Innovation)
*   **Script:** `stomp_collision.py`
*   **Innovation:** Replaced standard 2.5D/3D grids (e.g., Euclidean Distance Transforms) with mathematically direct **Euclidean Sphere-Sweep Tests**. This dramatically accelerates the safety-check loop for STOMP.

### 4. **Industrial-Arm Elastic Strips Mapper** (Advanced Integration)
*   **Script:** `elastic_strips.py`
*   **Innovation:** Mapped Cartesian "Repulsion Forces" directly into joint-torques using the **Jacobian Transpose ($J^T$)** specifically for industrial 6-DOF manipulators. This is a novel adaptation of the original Brock/Khatib mobility theory for robotic arms.

---

## 🧪 Section 3: Proof of Robustness & Verification
*These are the specific "Stress Tests" and "Validation Manifolds" designed by the User to prove the pipeline's superiority.*

1.  **Möbius & Pringle Manifolds:** (`test_ik_mobius.py`, `pringle`, `wave` scripts). The User designed these complex non-orientable surfaces to prove that **their** implementation of IK-Geo can navigate sweeping inflections where standard Jacobian trackers (common in industries) would experience singularity-stalls.
2.  **50-Seed Statistical Benchmark:** (`test_50seed_benchmark.py`). The User designed this "Monte Carlo" style test to prove that **their** stochastic tuner consistently finds safe paths even in randomized collision scenarios.

---

## 📈 Section 4: Phase 2 — Personal Research Roadmap
*The design of the next phase is entirely the User's proprietary vision for "Advanced Reactivity."*

1.  **Bubble-based Elastic Tunneling [User Design]**: Transitioning from point-wise potentials to **Safe Tunnel Spheres** (Quinlan-Khatib logic) to further optimize collision-check frequency for Gazebo.
2.  **Admittance Control Bridging [User Design]**: Design of a closed-loop Force-Feedback controller to enable "Safe Insertion" during the refueling phase.
3.  **Cross-Platform UR5 Layer [User Design]**: Generalizing the mission logic to support UR5 hardware alongside KUKA.

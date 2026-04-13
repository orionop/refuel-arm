## 1. Safety Rules & Directory Structure

To protect the existing successful ROS 2 Jazzy migration, this phase will operate in complete isolation.

*   **Rule 1**: Do not delete any existing work.
*   **Rule 2**: Do not modify any code in the root directory.
*   **Rule 3**: All work will take place in a new directory: `ref_env/`.
*   **Workflow**: Before modifying any script (e.g., `refuel_mission.py`), it must first be **COPIED** from the root into `ref_env/`.

## 2. Refueling Environment v2.0 (Gazebo Harmonic / SDF 1.9)

We will phase out all spherical obstacles in favor of realistic primitives and meshes.

### [NEW] `refuel_world_v2.sdf`
*   **The Pillar**: A rectangular static obstacle ($0.2\text{m} \times 0.08\text{m} \times H$).
*   **The Cylinder**: A vertical static cylinder.
*   **The Oscillating Cylinder**: A dynamic entity using a Gazebo `LinearBatteryPlugin` or a custom script to oscillate slowly ($<0.5\text{Hz}$) along the Y-axis.
*   **The Vehicle**: Integrating a car mesh.
    *   **Fuel Flap**: A convex obstacle plate located at the entry point of the concave inlet.
    *   **Concave Socket**: The deep 20cm cylindrical/box socket for the peg-in-hole insertion.

## 2. Morphological Arm Enhancements

The arm currently treats the tool-tip as a point. We must account for the physical size of the nozzle.

### A. End-Effector (EE) Volume
*   **Geometry**: Represent the nozzle as a cylinder (length $L$, radius $R$) + a conical tip.
*   **Collision Engine**: Update `stomp_collision.py` to perform **Cylinder-to-Box** and **Cylinder-to-Cylinder** distance checks instead of just Point-to-Sphere.

### B. Self-Collision Layer
*   **Thresholding**: Add a dedicated check in the collision engine to monitor the distance between the EE geometry and the previous arm links (Link 1-3).
*   **Safety Halt**: Trigger an ABORT if the EE breaches a $0.02\text{m}$ clearance with its own arm.

## 4. The Tangent Bug Algorithm (Kamon et al., 1998 Adaptation)

We will implement a 3D Cartesian adaptation of the classic sensor-based planner:

*   **Sensing (LTG - Local Tangent Graph)**: 
    *   Simulate a "sensing range" ($R_s$) around the EE.
    *   At each step, calculate the "visible" obstacle boundary points within $R_s$.
*   **Motion-to-Goal (MtG)**:
    *   If no obstacle is within $R_s$ in the direction of the goal, move directly towards $p_{goal}$.
    *   If an obstacle is detected, find a point $p_{tangent}$ on the boundary that minimizes $dist(p_{tangent}, p_{goal})$ and move towards it.
*   **Boundary Following (BF)**:
    *   Triggered when a local minimum is reached (the robot cannot get closer to the goal in MtG).
    *   The EE circumnavigates the obstacle (moving "left" or "right" in the plane formed by the goal and obstacle normal).
    *   **Leave Condition**: Switch back to MtG when $dist(EE, p_{goal}) < dist(phit, p_{goal})$.

## 4. Analytical Benchmark Framework

We will conduct a head-to-head comparison of all local avoidance methods using the **same** environment and **same** mission:

| Metric | Bubble Strips | Elastic Strips | Tangent Bug |
|:--- |:--- |:--- |:--- |
| **Path Length** | Distance in C-Space | Distance in C-Space | Distance in C-Space |
| **Execution Time** | Time to Reach Port | Time to Reach Port | Time to Reach Port |
| **Clearance** | Min distance to obs | Min distance to obs | Min distance to obs |
| **Smoothness** | Avg Jerk ($\dddot{q}$) | Avg Jerk ($\dddot{q}$) | Avg Jerk ($\dddot{q}$) |

---

## Proposed Changes

### [NEW] `ref_env/`
*   **`refuel_mission_v2.py`**: (Copied from root) The new supervisor node with Tangent Bug integration.
*   **`stomp_collision_v2.py`**: (Copied from root) Updated with volumetric EE and self-collision.
*   **`tangent_bug.py`**: New standalone avoidance module.
*   **`car_model_v2.py`**: (Copied from root) Dynamic car/flap spawner for Gz Sim.

---

## Verification Plan

1.  **Environment Check**: Launch the new world in Gazebo Harmonic and verify the Concave inlet and Flap positions.
2.  **Oscillation Test**: Verify the cylinders oscillate without simulation drift.
3.  **Collision Audit**: Run a trajectory that *should* hit the pillar and verify the collision engine detects it accurately with the **new EE volume**.
4.  **Method Test**: Run the mission 3 times (once per avoidance method) and generate the comparison table.

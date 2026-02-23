# ur5_motion_planning — Path planning and trajectory analysis

Joint-space path planning and time-parameterized trajectory analysis for UR5. Uses MoveIt’s planning interface; **no execution, no GUI**. After planning, runs a **post-planning analysis** (waypoint cohesion, workspace containment, singularity detection) and prints diagnostics.

## Requirements

- **move_group** must be running (e.g. `ros2 launch ur5_minimal_moveit ur5_moveit_minimal.launch.py`).
- Planning group: `ur_manipulator`; end-effector: `tool0`.
- For **full analysis** (workspace + singularity), run the node via the launch file so it receives `robot_description` and `robot_description_semantic`.

## Build

```bash
cd /workspace/ur5_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select ur5_motion_planning --symlink-install
source install/setup.bash
```

## Run

**Terminal 1:** Start move_group (and joint_state_publisher / robot_state_publisher):

```bash
source /workspace/ur5_ws/install/setup.bash
ros2 launch ur5_minimal_moveit ur5_moveit_minimal.launch.py
```

**Terminal 2:** Run the planning + analysis node.

- **Full analysis** (waypoint cohesion + workspace + singularity): use the launch file so the node gets the robot model:

```bash
source /workspace/ur5_ws/install/setup.bash
ros2 launch ur5_motion_planning run_plan_and_analyze.launch.py
```

- **Planning + waypoint cohesion only** (no workspace/singularity): run the executable directly (robot model not loaded, so B and C are skipped):

```bash
source /workspace/ur5_ws/install/setup.bash
ros2 run ur5_motion_planning ur5_plan_and_analyze_node
```

## Target pose (same as IK baseline)

- Position: (0.4, 0.1, 0.4) m  
- Orientation: identity quaternion  

## Post-planning analysis

1. **Waypoint cohesion (joint-space continuity)**  
   For each consecutive pair of waypoints: per-joint absolute delta and total L2 delta.  
   - Reports: max delta per joint over the trajectory; max total delta between any two consecutive waypoints.  
   - Logs a **warning** (no abort) if any joint delta > 0.5 rad or total delta norm > 1.0 rad.

2. **Workspace containment**  
   Workspace: x ∈ [0.2, 0.8], y ∈ [-0.4, 0.4], z ∈ [0.1, 0.7] (m).  
   For each waypoint, FK is used to compute the end-effector pose; reports waypoint indices and (x, y, z) that violate the bounds.

3. **Singularity detection**  
   For each waypoint: Jacobian at tip, manipulability w(q) = √det(J Jᵀ). Threshold ε = 1e-3.  
   Reports minimum manipulability over the trajectory and waypoint indices where w(q) < ε.

Analysis is **report-only**; the trajectory is not modified or rejected.

## Example output (with full analysis)

```
[INFO] [ur5_plan_and_analyze]: Planning to pose (0.400, 0.100, 0.400), identity orientation.
[INFO] [ur5_plan_and_analyze]: Planning SUCCEEDED.
[INFO] [ur5_plan_and_analyze]: Number of trajectory waypoints: 17
[INFO] [ur5_plan_and_analyze]: Total trajectory duration: 2.3456 s
[INFO] [ur5_plan_and_analyze]: First waypoint (rad):
[INFO] [ur5_plan_and_analyze]:   shoulder_pan_joint = 0.000000
...
[INFO] [ur5_plan_and_analyze]: Last waypoint (rad):
...
[INFO] [ur5_plan_and_analyze]: --- Trajectory analysis: waypoint cohesion ---
[INFO] [ur5_plan_and_analyze]: Max joint delta per joint (rad):
[INFO] [ur5_plan_and_analyze]:   shoulder_pan_joint = 0.234567
...
[INFO] [ur5_plan_and_analyze]: Max total joint-space delta (L2) between consecutive waypoints: 0.456789 rad
[INFO] [ur5_plan_and_analyze]: --- Trajectory analysis: workspace containment ---
[INFO] [ur5_plan_and_analyze]: Workspace: x=[0.20, 0.80], y=[-0.40, 0.40], z=[0.10, 0.70]
[INFO] [ur5_plan_and_analyze]: All waypoints inside workspace.
[INFO] [ur5_plan_and_analyze]: --- Trajectory analysis: singularity (manipulability) ---
[INFO] [ur5_plan_and_analyze]: Minimum manipulability w(q) over trajectory: 1.234567e-02
[INFO] [ur5_plan_and_analyze]: Threshold epsilon: 1.000000e-03
[INFO] [ur5_plan_and_analyze]: No waypoints with w(q) < epsilon.
[INFO] [ur5_plan_and_analyze]: --- Analysis complete. Exiting. ---
```

Start and end joint states differ; no execution occurs.

## Validation

- Node runs while move_group is active.
- Planning succeeds for the target pose.
- Trajectory has more than one waypoint.
- Analysis runs on the resulting trajectory (A always; B and C when launched with robot description).
- Diagnostics are printed clearly; node exits without errors.
- No execution; process exits cleanly.

# ur5_kdl_ik — Baseline IK with MoveIt 2 KDL

Research baseline: inverse kinematics for UR5 using MoveIt 2’s built-in **KDL** kinematics plugin. No motion planning, no execution, no RViz/Gazebo.

**IDE:** If the editor shows “cannot open source file” for `rclcpp`, `moveit`, etc., that’s because ROS 2 / MoveIt headers are in the Docker image, not on the host. The code builds and runs correctly in the container. For IntelliSense, open the project inside the container (e.g. VS Code/Cursor **Remote - Containers**) and use the **Linux (Docker / ROS 2 Humble)** C++ config.

## Active kinematics solver

- **Plugin:** `kdl_kinematics_plugin/KDLKinematicsPlugin` (from `ur_moveit_config/config/kinematics.yaml`)
- **Group:** `ur_manipulator` (base_link → tool0)

## Package creation (already done)

```bash
cd /workspace/ur5_ws/src
# Package is already at ur5_kdl_ik/
```

## Build

```bash
cd /workspace/ur5_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select ur5_kdl_ik --symlink-install
source install/setup.bash
```

Requires `ur_description` and `ur_moveit_config` (built or installed).

## Run

**Option A — With move_group running (recommended)**  
Terminal 1:

```bash
ros2 launch ur5_minimal_moveit ur5_moveit_minimal.launch.py
```

Terminal 2:

```bash
source /workspace/ur5_ws/install/setup.bash
ros2 launch ur5_kdl_ik run_ik_baseline.launch.py
```

**Option B — Standalone (launch passes robot_description)**  
No move_group needed; the launch file loads URDF/SRDF and passes them to the node:

```bash
source /workspace/ur5_ws/install/setup.bash
ros2 launch ur5_kdl_ik run_ik_baseline.launch.py
```

## Fixed target pose

- **Position:** (0.4, 0.1, 0.4) m  
- **Orientation:** identity quaternion (w=1, x=y=z=0)

## Example expected output

```
[INFO] [ur5_kdl_ik_baseline]: Kinematics solver: using default (KDL per kinematics.yaml).
[INFO] [ur5_kdl_ik_baseline]: Target pose: position (0.400, 0.100, 0.400), orientation identity.
[INFO] [ur5_kdl_ik_baseline]: IK SUCCESS.
[INFO] [ur5_kdl_ik_baseline]: Joint angles (rad):
[INFO] [ur5_kdl_ik_baseline]:   shoulder_pan_joint = -0.022943
[INFO] [ur5_kdl_ik_baseline]:   shoulder_lift_joint = 0.366566
[INFO] [ur5_kdl_ik_baseline]:   elbow_joint = -2.178872
[INFO] [ur5_kdl_ik_baseline]:   wrist_1_joint = 0.241509
[INFO] [ur5_kdl_ik_baseline]:   wrist_2_joint = 1.570796
[INFO] [ur5_kdl_ik_baseline]:   wrist_3_joint = -1.547854
```

Exact joint values depend on KDL’s solution (and seed). Re-runs with the same seed are deterministic.

## Validation

- Node runs (with or without move_group).
- IK succeeds for the fixed pose.
- Joint angles printed in radians.
- Re-runs give consistent results (same seed).
- On failure: “IK FAILED” or “outside joint bounds”, no segfault.

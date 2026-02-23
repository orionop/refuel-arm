# UR5 Minimal MoveIt — Build and Launch

Planning-only pipeline (no ros2_control, no hardware, no UR vendor launch). For IK-Geo / research.

## Prerequisites

- **Docker:** Use the project Dockerfile so the image has xacro, joint_state_publisher, move_group, etc.:
  ```bash
  cd /path/to/ros2_docker_ws
  docker build -t ros2_humble_ur5 .
  docker run -it --name ros2_humble --platform=linux/arm64 -v ~/ros2_docker_ws:/workspace ros2_humble_ur5 bash
  ```
- `ur_description` and `ur_moveit_config` must be available at runtime:
  - **Option A (binary):** Install in container: `apt-get update && apt-get install -y ros-humble-ur-description ros-humble-ur-moveit-config`
  - **Option B (from workspace):** Build them in this workspace (see commands below).

## Build (inside container, workspace root `/workspace/ur5_ws`)

```bash
cd /workspace/ur5_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select ur_description ur_moveit_config ur5_minimal_moveit --symlink-install
source install/setup.bash
```

If `ur_description` and `ur_moveit_config` are already installed (binary), build only the custom package:

```bash
cd /workspace/ur5_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select ur5_minimal_moveit --symlink-install
source install/setup.bash
```

## Launch

```bash
ros2 launch ur5_minimal_moveit ur5_moveit_minimal.launch.py
```

Leave this running (headless; no RViz).

## Verification (in another terminal, same container)

```bash
source /workspace/ur5_ws/install/setup.bash
ros2 node list
# Expect: /joint_state_publisher, /robot_state_publisher, /move_group

ros2 service list | grep plan
# Expect: /plan_kinematic_path (and possibly /plan_kinematic_path_async_*)
```

## Checklist

- [ ] `ros2 launch ur5_minimal_moveit ur5_moveit_minimal.launch.py` runs and stays alive
- [ ] `ros2 node list` shows `/robot_state_publisher` and `/move_group`
- [ ] `ros2 service list` includes `/plan_kinematic_path`

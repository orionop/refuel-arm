# Obstacle Workspace

Dynamic-obstacle safety system for the KUKA KR6 R700, fully independent from
the autonomous refueling pipeline in `../refueling/`.

## Stack

- **OS:** Ubuntu 24.04
- **ROS2:** Jazzy
- **Simulator:** gz sim Harmonic
- **Robot description + sim launch:** [kroshu/kuka_robot_descriptions](https://github.com/kroshu/kuka_robot_descriptions) (master branch, Jazzy + Harmonic supported)
- **Control:** `gz_ros2_control` (no raw `gz topic` hacks)

## Setup (Ubuntu 24 + ROS2 Jazzy)

```bash
# system deps
sudo apt install ros-jazzy-gz-ros2-control ros-jazzy-ros-gz* \
                 ros-jazzy-ros2-control ros-jazzy-ros2-controllers \
                 ros-jazzy-xacro

# fresh ROS2 workspace (separate from this repo)
mkdir -p ~/kuka_ws/src && cd ~/kuka_ws/src
git clone -b master https://github.com/kroshu/kuka_robot_descriptions.git

cd ~/kuka_ws
rosdep install --from-paths src -y --ignore-src
colcon build --symlink-install
source install/setup.bash
```

## Phase 0 — bring-up smoke test

```bash
source ~/kuka_ws/install/setup.bash
ros2 launch kuka_resources gazebo_startup.launch.py \
     robot_model:=kr6_r700_sixx robot_family:=kr_agilus
```

Verify:
- Arm comes up stable, no tipping
- `ros2 topic list | grep controller` shows controller topics
- `ros2 control list_controllers` shows running controllers

Once Phase 0 passes, the obstacle/ scripts will be written to talk to those
controllers (joint trajectory action / forward position controller).

## Layout

```
obstacle/
├── kuka_robot_descriptions/   # gitignored, cloned from kroshu
└── README.md
```

Scripts, worlds, and models will be added once the kroshu launch is verified
working on the Ubuntu box. Everything from the earlier refueling-derived
attempt has been moved to `../deprecated/obstacle_phase0_from_refueling/`.

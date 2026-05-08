# Obstacle Workspace

Reactive obstacle-avoidance safety system for a UR5 manipulator, fully
independent from the autonomous refueling pipeline in `../refueling/`.

## Stack

- **OS:** Ubuntu 24.04
- **ROS2:** Jazzy
- **Simulator:** gz sim Harmonic
- **Robot description + sim launch:** [Universal_Robots_ROS2_GZ_Simulation](https://github.com/UniversalRobots/Universal_Robots_ROS2_GZ_Simulation) (`ros2` branch — Jazzy + Harmonic)
- **Control bridge:** `gz_ros2_control`
- **QP solver:** `qpsolvers` (OSQP backend)

## Layout

```
obstacle/
├── safety/                          # core algorithms (pure python, runs anywhere)
│   ├── types.py                     # RobotState, Obstacle, ControlOutput
│   ├── kinematics.py                # UR5 FK + analytical Jacobian
│   ├── methods/
│   │   ├── base.py                  # SafetyMethod ABC
│   │   ├── threshold.py             # B: distance threshold
│   │   ├── apf.py                   # 1: APF + Circular Fields
│   │   ├── neo.py                   # 2: NEO velocity damper (QP)
│   │   ├── hocbf.py                 # 3: HOCBF safety filter (QP)
│   │   └── _qp.py                   # qpsolvers wrapper + scipy fallback
│   └── harness/
│       ├── scenarios.py             # head_on, oblique, passing, fast_dash, ...
│       ├── metrics.py               # min_sep, reaction_time, deviation_l2, jerk
│       └── runner.py                # pure-python sim loop + benchmark sweep
├── ros2_node/
│   ├── safety_node.py               # ros2 node, subscribes joints + obstacle, publishes qdot
│   └── obstacle_simulator.py        # spawns + animates a cylinder, publishes /obstacle_pose
├── scripts/
│   └── benchmark.py                 # CLI: offline 4-method × N-scenario sweep
├── Universal_Robots_ROS2_GZ_Simulation/   # cloned, gitignored
└── README.md
```

## Method set (locked)

| # | Method | Type | Reference |
|---|---|---|---|
| B | Distance threshold + e-stop | naive baseline | — |
| 1 | APF + Informed Circular Fields | reactive vector field | Becker et al., Frontiers 2024 |
| 2 | NEO velocity damper | reactive QP | Haviland & Corke, RA-L 2021 |
| 3 | HOCBF safety filter | reactive QP w/ slack | Singletary et al., RA-L 2022 |

All four implement the same `SafetyMethod.step(state, obstacles, qdot_nominal)`
interface so the benchmark and ROS2 node can swap them by name.

## Quickstart — local benchmark (any OS, no Gazebo needed)

```bash
pip install qpsolvers osqp quadprog scipy numpy
python3 obstacle/scripts/benchmark.py
```

## Quickstart — Ubuntu 24 + ROS2 Jazzy + gz sim

### One-time setup

```bash
sudo apt install ros-jazzy-gz-ros2-control ros-jazzy-ros-gz* \
                 ros-jazzy-ros2-control ros-jazzy-ros2-controllers \
                 ros-jazzy-xacro
pip install qpsolvers osqp

# fetch UR sim into a fresh ROS2 workspace
mkdir -p ~/ur_ws/src && cd ~/ur_ws/src
git clone -b ros2 https://github.com/UniversalRobots/Universal_Robots_ROS2_GZ_Simulation.git
vcs import . < Universal_Robots_ROS2_GZ_Simulation/ur_simulation_gz.jazzy.repos
cd ~/ur_ws
rosdep install --from-paths src -y --ignore-src
colcon build --symlink-install
source install/setup.bash
```

### Run

```bash
# terminal 1 — UR5 + gz sim (publishes /joint_states, accepts /forward_velocity_controller/commands)
ros2 launch ur_simulation_gz ur_sim_control.launch.py ur_type:=ur5

# terminal 2 — moving obstacle
PYTHONPATH=$PWD/obstacle python3 obstacle/ros2_node/obstacle_simulator.py \
    --ros-args -p world:=ur_simulation_gz \
               -p start_xyz:=[1.5,0.0,0.5] \
               -p end_xyz:=[-0.5,0.0,0.5] \
               -p duration:=3.0

# terminal 3 — safety supervisor (pick method: threshold | apf | neo | hocbf)
PYTHONPATH=$PWD/obstacle python3 obstacle/ros2_node/safety_node.py \
    --ros-args -p method:=hocbf -p control_rate:=50.0
```

### Verify

```bash
ros2 topic list | grep -E "joint_states|obstacle_pose|controller"
ros2 control list_controllers
```

You should see:
- `/joint_states` publishing at ~125 Hz
- `/obstacle_pose` publishing at 50 Hz
- `forward_velocity_controller` active
- The arm jogs along nominal qdot until the cylinder enters the danger zone,
  then the chosen safety method takes over

## Where novelty plugs in later

The `safety/methods/` directory is structured so a new method drops in as a
fifth file with the same `SafetyMethod` interface — the benchmark, harness,
and ROS2 node pick it up automatically.

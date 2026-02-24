# Setup Instructions — KUKA KR6 R700 Refueling Pipeline

## Prerequisites

- Ubuntu 20.04 with ROS Noetic
- NVIDIA GPU (for Gazebo)
- Git, Python 3

---

## Step 1: Check GPU (SAFE_DEV_RULES Rule 5)

```bash
nvidia-smi
```

If GPU is heavily used by others, **do not proceed** with Gazebo.

---

## Step 2: Create Workspace

```bash
mkdir -p ~/kuka_ws/src
cd ~/kuka_ws/src
```

---

## Step 3: Clone Repository

```bash
git clone https://github.com/orionop/refuel-arm.git
```

---

## Step 4: Python Virtual Environment (SAFE_DEV_RULES Rule 4)

```bash
cd ~/kuka_ws/src/refuel-arm
python3 -m venv venv
source venv/bin/activate
pip install numpy linearsubproblemsltns
```

Verify isolation:
```bash
pwd        # Must show: .../kuka_ws/src/refuel-arm
which python  # Must show: .../venv/bin/python
```

---

## Step 5: Test Pipeline Locally (No ROS Needed)

```bash
cd ~/kuka_ws/src/refuel-arm
python3 test_full_pipeline.py
```

Expected output:
```
[Setup] Verifying mission waypoints...
[Planning] STOMP trajectory optimization for 4 motion segments
[Preview] Mission trajectory summary...
```

---

## Step 6: Build ROS Workspace

```bash
source /opt/ros/noetic/setup.bash
cd ~/kuka_ws/src/refuel-arm/kuka_refuel_ws
catkin_make
source devel/setup.bash
```

---

## Step 7: Launch Gazebo (Terminal 1)

```bash
source /opt/ros/noetic/setup.bash
cd ~/kuka_ws/src/refuel-arm/kuka_refuel_ws
source devel/setup.bash
roslaunch kuka_kr6_gazebo gazebo.launch
```

*(You can also use `roslaunch kuka_kr6_gazebo rviz.launch` to visualize purely in RViz.)*

---

## Step 8: Run Refueling Mission (Terminal 2)

```bash
source /opt/ros/noetic/setup.bash
cd ~/kuka_ws/src/refuel-arm/kuka_refuel_ws
source devel/setup.bash
cd ~/kuka_ws/src/refuel-arm
# NOTE: Do NOT use the venv when running ROS commands here
python3 test_full_pipeline.py --ros

# Or, if running RViz instead of Gazebo:
# python3 test_full_pipeline.py --rviz
```

---

## Step 9: Run IK-Geo Line Tracking (Optional Scenario B)

Demonstrate exact mathematical 60-waypoint Cartesian line tracking dynamically.

```bash
cd ~/kuka_ws/src/refuel-arm

# Draw default line
python3 test_ik_line.py --ros

# Draw custom length line dynamically
python3 test_ik_line.py --ros --start 0.3 0.4 0.5 --end 0.65 -0.25 0.45
```

---

## Pipeline Overview

| Phase | Component | What It Does |
|-------|-----------|--------------|
| 1 | **IK-Geo** | Algebraic IK solver to precisely determine valid joint poses for Real-world Refuel Inlet and Nozzle Station. |
| 2 | **STOMP** | Optimises 30-waypoint trajectory across the 5 stage continuous sequence (REST → YELLOW → RED (10s refuel) → YELLOW → REST). |
| 3 | **ROS & RViz** | Broadcasts directly to ROS via `/joint_states` (`--rviz`) or `JointTrajectoryController` (`--ros`). |

## References

- **IK-Geo:** Elias et al., "IK-Geo: Unified Robot Inverse Kinematics Using Subproblem Decomposition", 2022
- **STOMP:** Kalakrishnan et al., "STOMP: Stochastic Trajectory Optimization for Motion Planning", IEEE ICRA 2011

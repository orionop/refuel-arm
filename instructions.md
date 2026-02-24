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
cd /home/armslab-exp4/Desktop
mkdir -p anurag_ws/src
cd anurag_ws/src
```

---

## Step 3: Clone Repository

```bash
git clone https://github.com/orionop/refuel-arm.git
```

---

## Step 4: Python Virtual Environment (SAFE_DEV_RULES Rule 4)

```bash
cd /home/armslab-exp4/Desktop/anurag_ws
python3 -m venv venv
source venv/bin/activate
pip install numpy
```

Verify isolation:
```bash
pwd        # Must show: /home/armslab-exp4/Desktop/anurag_ws
which python  # Must show: /home/armslab-exp4/Desktop/anurag_ws/venv/bin/python
```

---

## Step 5: Test Pipeline Locally (No ROS Needed)

```bash
cd /home/armslab-exp4/Desktop/anurag_ws/src/refuel-arm
python3 test_full_pipeline.py
```

Expected output:
```
[Phase 1] IK-Geo → 8 solutions, 1 valid, FK error: 1.24e-16 m ✅
[Phase 2] STOMP → 30 waypoints, max jump 3.8°, all within limits ✅
[Phase 3] Local mode — trajectory preview ✅
```

---

## Step 6: Build ROS Workspace

```bash
source /opt/ros/noetic/setup.bash
cd /home/armslab-exp4/Desktop/anurag_ws
catkin_make
source devel/setup.bash
```

---

## Step 7: Launch Gazebo (Terminal 1)

```bash
source /opt/ros/noetic/setup.bash
cd /home/armslab-exp4/Desktop/anurag_ws
source devel/setup.bash
roslaunch kuka_kr6_gazebo gazebo.launch
```

---

## Step 8: Run Pipeline with ROS (Terminal 2)

```bash
source /opt/ros/noetic/setup.bash
cd /home/armslab-exp4/Desktop/anurag_ws
source devel/setup.bash
source venv/bin/activate
cd src/refuel-arm
python3 test_full_pipeline.py --ros
```

---

## Pipeline Overview

| Phase | Component | What It Does |
|-------|-----------|--------------|
| 1 | **IK-Geo** | Computes exact joint angles for target pose (8 solutions, filters by limits) |
| 2 | **STOMP** | Optimises 30-waypoint trajectory (smooth, within joint limits) |
| 3 | **ROS** | Sends trajectory to Gazebo `JointTrajectoryController` |

## References

- **IK-Geo:** Elias et al., "IK-Geo: Unified Robot Inverse Kinematics Using Subproblem Decomposition", 2022
- **STOMP:** Kalakrishnan et al., "STOMP: Stochastic Trajectory Optimization for Motion Planning", IEEE ICRA 2011

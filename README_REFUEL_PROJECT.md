# KUKA KR6 R700 Autonomous Refueling Simulation

**Institution:** Systems and Control Engineering (SysCon), IIT Bombay  
**Maintainer:** Anurag Shetye  

## 🚀 Project Overview

This repository contains the simulation pipeline for the **Autonomous Car Refueling Arm** project. The objective is to simulate a KUKA KR6 R700 industrial robot starting from a rest (`HOME`) configuration, autonomously identifying a vehicle's refueling inlet (`TARGET`), navigating to it, performing a 5-7 second refueling hold, and safely returning to rest. 

This project integrates three distinct technologies to achieve perfectly accurate, collision-free autonomous movement:
1. **IK-Geo:** The exact algebraic geometric solver library.
2. **IKFlow:** A Normalizing Flow Neural Network for diverse path generation.
3. **ROS 2 Humble / Gazebo:** The physical simulator and hardware interface.

---

## 🧠 The Approach (IK-Geo + IKFlow)

Traditional inverse kinematics (IK) solvers like KDL or TRAC-IK (used by standard MoveIt!) rely on iterative/numerical methods (Jacobian pseudo-inverses) which can be slow, get stuck in local minima, or fail to find all possible solutions for a pose.

Our approach splits the problem into two parts to guarantee both **absolute precision** and **smooth navigation**:

### 1. Terminal Pose Accuracy (IK-Geo)
The exact $XYZ$ + $RPY$ coordinate of the car's refueling inlet is passed to **IK-Geo**. 
- The KR6 R700 belongs to the `IK_spherical_2_parallel` kinematic family. 
- IK-Geo instantly computes all **8 mathematically exact joint configurations** to reach the target with $\sim 10^{-16}$ precision.
- This guarantees the robot flange perfectly aligns with the inlet without numerical slipping.

### 2. Trajectory Generation (IKFlow)
With the exact terminal joint state chosen from IK-Geo, we need a path to get there from `HOME`.
- We use **IKFlow**, a trained Normalizing Flow Neural Network, to generate diverse, collision-free IK solutions along the Cartesian path to the inlet. 
- These intermediate NN-generated joint states are used as high-quality seed states or direct waypoints for the ROS trajectory planner (MoveIt/OMPL), ensuring the arm smoothly dodges self-collisions and singularities while executing the refueling movement.

---

## 💻 Simulation Pipeline & Usage

*Note: You must execute these steps on the shared Ubuntu GPU machine. These instructions strictly follow the Safe Development Rulebook for this machine.*

### Step 1: Clone and Environment Setup
SSH or log directly into the Ubuntu PC. All work must be strictly isolated to the `/home/admin/Desktop/anurag_ws` directory.

```bash
# 1. Enter the designated safe workspace and verify location
cd /home/admin/Desktop/anurag_ws
pwd # MUST return /home/admin/Desktop/anurag_ws. Abort if not.

# 2. Clone the repository
git clone https://github.com/orionop/refuel-arm.git
cd refuel-arm

# 3. Create and enter a Python 3 virtual environment INSIDE the workspace
python3 -m venv venv
source venv/bin/activate

# 4. Verify Python isolation before proceeding
which python # MUST return /home/admin/Desktop/anurag_ws/refuel-arm/venv/bin/python
```

### Step 2: Train IKFlow for the KUKA KR6
IKFlow ships with pre-trained models for Franka/Fetch, so we map out the specific state-space of the KR6 first.

```bash
# Ensure you are in the venv (source venv/bin/activate)
cd ikflow

# 1. Install IKFlow into the venv
uv pip install -e .

# 2. Generate 25 million valid KR6 joint poses
uv run python scripts/build_dataset.py --robot_name=kr6_r700 --training_set_size=25000000 --only_non_self_colliding

# 3. VERIFY GPU LOAD BEFORE TRAINING
nvidia-smi  # Do NOT start training if GPU memory is heavily used by other PhD students
top         # Check running processes if unsure

# 4. Train the Neural Network on the GPU
uv run python scripts/train.py --robot_name=kr6_r700 --nb_nodes=12 --batch_size=128 --learning_rate=0.0005
# return to repo root
cd ..
```

### Step 3: Build the ROS 2 Workspace
```bash
# Note: Never run global apt, sudo, or modify /opt/ros/humble
# 1. Source the global ROS Humble installation
source /opt/ros/humble/setup.bash

# 2. Compile the package
cd kuka_refuel_ws
colcon build

# 3. Source the local overlay
source install/setup.bash
```

### Step 4: Run the Refueling Simulation
*Remember the correct sourcing order before testing: global setup $\rightarrow$ local setup $\rightarrow$ venv activate*
```bash
# Launch Gazebo, spawn the robot, run ros2_control, and spawn the mock red "Car Inlet"
ros2 launch kuka_kr6_gazebo refuel_sim.launch.py
```

Once Gazebo is running, the Python State Machine Commander (`refuel_task_node.py` - *WIP*) is executed. It will automatically load the IKFlow model weights, query IK-Geo for the exact terminal geometry, and command the `JointTrajectoryController` through the `HOME` $\rightarrow$ `REFUEl` (Wait 7s) $\rightarrow$ `HOME` sequence.

---

## 📁 Repository Structure
* `matlab/` - Contains the foundational algebraic IK-Geo derivations and the pure MATLAB 3D mathematical visualizer (`KR6_R700.m`).
* `kuka_refuel_ws/` - The complete ROS 2 Humble `colcon` workspace containing:
  * `kuka_kr6_gazebo/urdf/`: The KUKA URDF injected with `gazebo_ros2_control` transmissions.
  * `kuka_kr6_gazebo/worlds/`: The Gazebo physics environment containing the target inlet.
  * `kuka_kr6_gazebo/launch/`: Master `.launch.py` files bridging controllers and physics.
* `ikflow/` - Submodule/implementation of the Normalizing Flow training infrastructure.

---

## ⚠️ Shared Lab Machine Safety Protocol

This project is deployed on a shared IITB research machine. Strict adherence to the **Safe Development Rulebook** is required:

### Forbidden Actions
- **NO `sudo` commands.** 
- **NO global `apt` installations or removals.**
- **NO directory deletions outside of `/home/admin/Desktop/anurag_ws`.**
- **NO modifying `~/.bashrc`, `/opt/ros/`, `/usr/`, `/etc/`, or system Python.**
- **NO restarting or reinstalling CUDA/NVIDIA drivers.**

### Final Operating Principles
1. **Always verify path:** Run `pwd` before deleting anything.
2. **Always verify isolation:** Run `which python` to ensure you are in the local `venv`.
3. **Always respect peers:** Check `nvidia-smi` and `top` before launching simulations. Do not kill jobs you do not own.

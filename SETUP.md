# Quick Start Guide: KUKA Refuel Arm Simulation

Welcome! This guide will walk you through setting up the autonomous car refueling pipeline, running the exact mathematical Cartesian line tracking programs, and testing out the STOMP trajectory optimizer.

Everything runs on a standard Ubuntu/ROS Noetic environment.

---

## 🛠️ Prerequisites

To run this repository, you must have the following system requirements compiled and available:
- **OS**: Ubuntu 20.04
- **ROS**: ROS Noetic
- **Python**: Python 3.8+
- **Simulation**: Gazebo (preferably backed by a physical NVIDIA GPU, as high-frequency 6-DOF solving benefits immensely from hardware acceleration).

---

## 🚀 Installation & Setup

### 1. Create a ROS Workspace
You need a fresh place to build the ROS packages required for the KUKA arm.
```bash
mkdir -p ~/kuka_ws/src
cd ~/kuka_ws/src
```

### 2. Clone the Repository
Inside your `src` folder, clone this repo:
```bash
git clone https://github.com/orionop/refuel-arm.git
```

### 3. Setup Python Virtual Environment
To protect your global site-packages (especially important on shared machines), create a dedicated Python environment:
```bash
cd ~/kuka_ws/src/refuel-arm
python3 -m venv venv
source venv/bin/activate

# Install the necessary mathematical computing dependencies:
pip install numpy linearsubproblemsltns matplotlib
```

To verify the installation was local, run:
```bash
which python
# Expected output: /home/<user>/kuka_ws/src/refuel-arm/venv/bin/python
```

### 4. Build the ROS Workspace
With the environment sourced, compile the specific KR6 robot descriptions and controllers:
```bash
source /opt/ros/noetic/setup.bash
cd ~/kuka_ws/src/refuel-arm/kuka_refuel_ws
catkin_make
source devel/setup.bash
```

---

## 🎮 Running the Pipeline

Now that everything is built, you can run the primary refueling scenario or the highly specialized IK topological trajectories.

### Scenario A: Full Refueling Mission (IK + STOMP)
This runs the full orchestration: `HOME → YELLOW Station → RED Port (Refuels for 10s) → YELLOW Station → HOME`.

**Terminal 1 (Launch Gazebo):**
```bash
source /opt/ros/noetic/setup.bash
cd ~/kuka_ws/src/refuel-arm/kuka_refuel_ws
source devel/setup.bash

roslaunch kuka_kr6_gazebo gazebo.launch
# NOTE: If your laptop is slow or has no GPU, launch rviz.launch instead!
```

**Terminal 2 (Run Python Orchestrator):**
```bash
source /opt/ros/noetic/setup.bash
cd ~/kuka_ws/src/refuel-arm/kuka_refuel_ws
source devel/setup.bash

cd ~/kuka_ws/src/refuel-arm
# WARNING: Do NOT 'source venv' when running ROS commands! Let ROS use the system python.
python3 test_full_pipeline.py --ros
```

---

### Scenario B: Testing 6-DOF Topological Tracking
We built 4 distinct trajectory stress-tests to prove our algebraic IK-Geo solver correctly processes continuous geometry without triggering chaotic elbow-flips at $2\pi$ wrap-arounds.

These files live in the `ik_trajectories/` directory. With Gazebo running in Terminal 1, execute these in Terminal 2:

```bash
cd ~/kuka_ws/src/refuel-arm

# 1. 3D Straight Line with dynamic 6-DOF wrist twist
python3 ik_trajectories/test_ik_line.py --ros

# 2. Multi-Cycle Wave (Pitch dynamically adjusts tangent to exact dx/dy/dz)
python3 ik_trajectories/test_ik_wave.py --ros

# 3. 3D Hyperbolic Paraboloid (Saddle/Pringle Shape)
python3 ik_trajectories/test_ik_pringle.py --ros

# 4. Topological Edge 4π Sweep (Möbius Strip Inversion Tracker)
python3 ik_trajectories/test_ik_mobius.py --ros
```

---

## 📊 Performance Analysis & Visualization

If you want to view the quantitative metrics (Cartesian Step Distance, $\Delta Q$ Jump Norm, and joint orientation gradients) instead of running the physical robot arm, you can execute the analysis scripts natively in Python.

*(You MUST activate the `venv` for this, as it relies on `matplotlib`)*
```bash
cd ~/kuka_ws/src/refuel-arm
source venv/bin/activate

# Analyze the full STOMP hybrid pipeline:
python3 analyze_pipeline.py

# Your 4-panel graph will be automatically generated and saved directly to:
# refuel-arm/output_graphs/
```

Happy tracking! If you encounter issues, please check `SAFE_DEV_RULES.md` to ensure your shared lab environment hasn't encountered hardware GPU locking issues.

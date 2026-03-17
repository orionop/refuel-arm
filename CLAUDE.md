# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

End-to-end simulation pipeline for autonomous car refueling using a KUKA KR6 R700 6-DOF industrial arm. The full system runs in ROS Noetic + Gazebo. Mission: HOME → RED (refuel port, 5s dwell) → HOME.

## Environment Setup

**Requires:** Ubuntu 20.04, ROS Noetic, Python 3.8+, Gazebo. All work must be inside the designated ROS workspace (`~/kuka_ws`).

```bash
# Source ROS and workspace (required before any ROS commands)
source /opt/ros/noetic/setup.bash
source ~/kuka_ws/devel/setup.bash

# Activate Python virtual environment
source ~/kuka_ws/venv/bin/activate

# Install Python dependencies
pip install numpy linearsubproblemsltns matplotlib

# Build ROS workspace
cd ~/kuka_ws && catkin_make
```

## Running the Pipeline

```bash
# Full refueling mission in Gazebo (requires running ROS master + Gazebo)
python3 test_full_pipeline.py --ros

# Motion planning strategy comparison (no ROS needed)
python3 ik_trajectories/compare_cspace_workspace.py

# IK accuracy benchmark (no ROS needed)
python3 ik_trajectories/analyze_ik_accuracy.py

# Topological 6-DOF tracking tests
python3 ik_trajectories/test_ik_mobius.py --ros
python3 ik_trajectories/test_ik_line.py --ros
python3 ik_trajectories/test_ik_pringle.py --ros

# Launch Gazebo simulation
roslaunch kuka_kr6_gazebo gazebo.launch

# Launch RViz visualization
roslaunch kuka_kr6_gazebo rviz.launch
```

## Architecture: Three-Layer Motion Planning Stack

### Layer 1 — Inverse Kinematics (IK-Geo)
**`kuka_refuel_ws/src/kuka_kr6_gazebo/scripts/ik_geometric.py`**

Exact algebraic closed-form IK for the KUKA KR6 R700 (`IK_spherical_2_parallel` kinematic family). Returns up to 8 joint configurations per target pose with ~10⁻¹⁶ rad precision. Uses the `linearsubproblemsltns` package to solve canonical subproblems (SP1, SP3, SP4, SP6). Also provides `fwd_kinematics(q)` for FK validation.

The `ik-geo/` directory is the upstream algebraic IK library (submodule); the ROS scripts use the Python bindings from `ik-geo/python/`.

### Layer 2 — Trajectory Optimization (STOMP)
**`stomp_collision.py`**

Stochastic Trajectory Optimization for Motion Planning. Takes start/goal joint configs and produces a smooth, collision-free joint trajectory. Key internals:
- `Grid3D`: Builds a 3D point cloud collision penalty grid from obstacle definitions
- Euclidean Distance Transform (EDT) for smooth workspace gradients
- Smoothness cost via second-derivative (acceleration) matrix
- Joint limit costs and collision penalties jointly minimized

### Layer 3 — Reactive Avoidance (Elastic Strips)
**`elastic_strips.py`**

Post-processes the STOMP trajectory to handle dynamic/unexpected obstacles. Treats the trajectory as a physical rubber band:
- Internal spring forces maintain trajectory smoothness
- External repulsion forces push waypoints away from obstacles
- Workspace repulsion → joint space via Jacobian transpose (numerical Jacobian computed by FK perturbation)
- Operates on full 6-DOF FK checkpoints along the trajectory

### Orchestrator
**`test_full_pipeline.py`**

Runs the complete mission: calls IK-Geo for target poses, STOMP for trajectory segments, Elastic Strips for reactive deformation, then publishes `JointTrajectoryController` commands via ROS actionlib. The `--ros` flag enables actual robot execution; omitting it runs in analysis/dry-run mode.

## Key Conventions

- **No deletions:** Move old code to `deprecated/` instead of deleting.
- **No system modifications:** Never `sudo apt`, never touch `/opt/ros/noetic`, never modify `/usr`.
- **Python isolation:** Always use the `venv/` inside the workspace, not system Python.
- **GPU protocol:** Check `nvidia-smi` before any GPU-intensive work; never kill others' processes.
- Analysis output graphs go to `output_graphs/`; the LaTeX report is in `report/`.
- The `ros2-claude-code-template/` directory is a third-party template; ignore it for this project's logic.

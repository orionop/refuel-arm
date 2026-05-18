# NEO + Gazebo Classic + ROS1 Noetic — Lab Run Instructions

Phase 2 deliverable. Take what we validated in Swift (kinematic) and run it under
real physics in Gazebo. Same NEO algorithm, ROS1 wrapper around it.

**Default robot: Franka Emika Panda** (matches the original NEO paper exactly).
UR5 variant kept as a fallback in case Panda setup isn't available.

---

## Files

| File | Robot | Role |
|---|---|---|
| `neo_ros_node_panda.py` | Panda (7-DOF) | NEO QP wrapped as ROS1 node — **primary** |
| `obstacle_spawner_panda.py` | — | Spawns the original paper's two moving spheres in Gazebo |
| `neo_ros_node.py` | UR5 (6-DOF) | Fallback if Panda unavailable |
| `obstacle_spawner.py` | — | UR5 scene obstacle spawner |
| `README_lab.md` | — | This file |

---

## Pre-flight (~5 min)

On the lab PC:

```bash
# 1. Pull the repo to wherever your catkin_ws lives
cd ~/catkin_ws/src/<your_repo>/obstacle/ros1_neo
git pull

# 2. Make scripts executable
chmod +x neo_ros_node_panda.py obstacle_spawner_panda.py

# 3. Confirm Python deps in the system Python 3.8 (Noetic's default rospy)
python3 -c "import rospy, roboticstoolbox as rtb, spatialgeometry, spatialmath, qpsolvers; print('ok')"
```

If any import fails, install in the system Python (NOT in a venv — rospy needs to match):

```bash
pip3 install --user roboticstoolbox-python spatialgeometry spatialmath-python qpsolvers quadprog "numpy<2"
```

---

## Terminal layout (3 terminals)

### Terminal 1 — Gazebo + Panda + controllers

Launch whatever Panda Gazebo setup your lab PC has. Typical options:

```bash
# Option A: franka_gazebo (if franka_ros is installed)
roslaunch franka_gazebo panda.launch

# Option B: a custom launch from your repo
roslaunch <your_pkg> panda_gazebo.launch
```

Verify:
- Gazebo Classic opens with Panda spawned in its ready pose
- `rostopic list` shows `/joint_states` publishing
- `rostopic list` shows a controller command topic (often `/panda_arm_controller/command`
  or `/position_joint_trajectory_controller/command`)

### Terminal 2 — Spawn moving obstacles

```bash
cd ~/catkin_ws/src/<your_repo>/obstacle/ros1_neo
python3 obstacle_spawner_panda.py
```

You should see two red spheres spawn in Gazebo and start translating in the -y direction.
Verify with:

```bash
rostopic echo /gazebo/model_states | head
# should list obs_0 and obs_1
```

### Terminal 3 — Run NEO controller

```bash
cd ~/catkin_ws/src/<your_repo>/obstacle/ros1_neo
python3 neo_ros_node_panda.py
```

If the controller topic name doesn't match your config, override it:

```bash
python3 neo_ros_node_panda.py _controller_topic:=/your_actual_controller/command
```

Other useful overrides:

```bash
python3 neo_ros_node_panda.py \
    _controller_topic:=/panda_arm_controller/command \
    _target:=[0.6,-0.2,0.0] \
    _control_rate:=100.0 \
    _influence_dist:=0.30 \
    _stop_dist:=0.05
```

---

## What success looks like

1. Panda arm starts moving toward the target after first /joint_states arrives
2. As obstacles approach, NEO bends the arm to dodge them
3. Panda reaches target with no collision in Gazebo
4. `neo_ros_node_panda.py` does **not** log "QP infeasible" repeatedly

Based on our Swift validation, **the Panda 7-DOF scene should run cleanly** —
zero penetrations were observed in the kinematic version. Gazebo physics adds
~10-30 ms of controller latency, so expect slightly more rough motion but
no actual contact under default parameters.

---

## What partial success looks like (still informative)

1. Periodic "QP infeasible" warnings when both obstacles are simultaneously
   inside the influence distance → matches what we saw in Swift for tight scenes
2. Slight latency in tracking when obstacles move fast → expected with the
   position-trajectory controller; NEO is computing at 100 Hz but the controller
   acts at the controller's update rate

---

## Tunables to play with

If the arm collides too much, raise the influence distance:

```bash
python3 neo_ros_node_panda.py _influence_dist:=0.40 _stop_dist:=0.08
```

If the arm is too cautious / stalls:

```bash
python3 neo_ros_node_panda.py _influence_dist:=0.20 _stop_dist:=0.03
```

If the obstacles move too fast for the controller, edit `obstacle_spawner_panda.py`
and reduce the `velocity` field for each obstacle.

---

## Falling back to UR5

If franka_gazebo / Panda setup isn't available, run the UR5 variant:

```bash
# Terminal 1 — UR5 launch (your existing setup from deprecated/ros1_launch/)
roslaunch <your_pkg> ur5.ros1.launch

# Terminal 2 — UR5 obstacle scene
python3 obstacle_spawner.py

# Terminal 3 — NEO on UR5
python3 neo_ros_node.py
```

Note: based on our Swift testing, UR5 6-DOF + dynamic obstacles **will** produce
penetrations and infeasible-QP windows. This is interesting empirically (it
matches the paper's untested case) but isn't the "clean success" demo.

---

## Known assumptions / things to verify on the lab PC

1. **Controller type:** Code assumes a JointTrajectoryController on the configured topic.
   If your Panda Gazebo config uses a velocity controller (Float64MultiArray), the message
   construction in the publisher will need to change — ping me.
2. **Joint names:** Default Panda joint names `panda_joint1`..`panda_joint7` hardcoded.
   If your URDF renames them, override via `_joint_names:=[...]`.
3. **NumPy 2.x incompatibility:** If pybullet imports fail, `pip3 install --user "numpy<2"`.
4. **Panda Gazebo plugin:** Different distributions ship slightly different plugins. The
   important property is that `/gazebo/model_states` publishes obstacle poses — verify
   with `rostopic hz /gazebo/model_states`.

---

## If it doesn't work — quick diagnostic checklist

```bash
# Joint states publishing?
rostopic hz /joint_states

# Are obstacle model states visible?
rostopic echo /gazebo/model_states -n 1 | grep obs_

# Is the controller alive?
rosservice list | grep controller_manager

# Test command message acceptance manually (Panda):
rostopic pub -1 /panda_arm_controller/command trajectory_msgs/JointTrajectory \
    "{joint_names: ['panda_joint1','panda_joint2','panda_joint3','panda_joint4','panda_joint5','panda_joint6','panda_joint7'],
      points: [{positions: [0, -0.785, 0, -2.356, 0, 1.571, 0.785], time_from_start: {secs: 2}}]}"
```

If `rostopic pub` moves the arm but `neo_ros_node_panda.py` doesn't — issue is in the node.
If `rostopic pub` doesn't move the arm — issue is in the controller setup.

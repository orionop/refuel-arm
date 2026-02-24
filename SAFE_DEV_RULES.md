TITLE: SAFE DEVELOPMENT RULEBOOK — ROS1 NOETIC — SHARED LAB MACHINE

CONTEXT:
Working on shared Ubuntu research machines.
ROS version: ROS1 Noetic / ROS1 Noetic
Workspace:
`~/kuka_ws` (or your chosen path)
PhD students have existing work on this system.
Goal: Zero interference with system, ROS installation, CUDA, GPU jobs, or other research.

------------------------------------------------------------

1) STRICT WORKSPACE ISOLATION

All work must remain inside your designated workspace:
`~/kuka_ws`

Before running any delete/move command:
`pwd`

If output does NOT match your designated isolated workspace, Abort immediately.

Never run:
`rm -rf *`
without confirming location using:
`pwd`

------------------------------------------------------------

2) ROS1 NOETIC SAFETY RULES

Never modify global ROS installation:
`/opt/ros/noetic`

Never run:
`sudo apt remove ros-*`
`sudo apt upgrade`
`sudo apt install ros-*`

Never edit:
`/opt/ros/noetic/setup.bash`

Only source globally installed ROS:
`source /opt/ros/noetic/setup.bash`

Do NOT reinstall ROS1 Noetic.

------------------------------------------------------------

3) SAFE ROS WORKSPACE STRUCTURE

Inside your isolated workspace:
`~/kuka_ws`

Structure must be (ROS 2 example):

`kuka_ws/`
  `src/`
  `build/`
  `install/`
  `log/`

Create workspace:

`mkdir -p ~/kuka_ws/src`
`cd ~/kuka_ws`

Build only inside workspace:

`catkin_make`

After build:

`source devel/setup.bash`

Never build inside:
`/opt/ros/`
or any other user directory.

------------------------------------------------------------

4) PYTHON ISOLATION (MANDATORY)

Create virtual environment inside your workspace:

`cd ~/kuka_ws`
`python3 -m venv venv`
`source venv/bin/activate`

Verify isolation:

`which python`

Must return the path to your internal `venv/bin/python`.

Install packages only after activation:

`pip install -r requirements.txt`

Never use:
`sudo pip install`
Never upgrade system Python.

------------------------------------------------------------

5) GPU USAGE PROTOCOL

Before training or launching heavy simulation:

`nvidia-smi`

If GPU memory is heavily used:
Do NOT start training.

Never kill processes you did not start.

Check running processes:

`top`
`htop`

Only terminate your own processes.

------------------------------------------------------------

6) SAFE ENVIRONMENT SOURCING ORDER

Correct order (ROS 2 example):

`source /opt/ros/noetic/setup.bash`
`cd ~/kuka_ws`
`source devel/setup.bash`
`source venv/bin/activate`

Never modify:
`~/.bashrc` permanently without permission.

------------------------------------------------------------

7) FORBIDDEN ACTIONS

Do NOT:

`sudo apt install`
`sudo apt remove`
`sudo rm`
`pip install --upgrade` (global)
Modify `/usr/`
Modify `/etc/`
Modify `/opt/`
Reinstall CUDA
Reinstall NVIDIA drivers
Upgrade kernel

------------------------------------------------------------

8) DATA SAFETY

Do not delete unknown folders.
Do not modify other researchers’ datasets.
Store all outputs inside your designated workspace bounds:

`~/kuka_ws`

------------------------------------------------------------

9) FINAL OPERATING PRINCIPLES

- No `sudo`
- No global modifications
- No deletion outside workspace
- Always verify directory with:
  `pwd`
- Always verify Python path with:
  `which python`
- Always check GPU usage with:
  `nvidia-smi`

If uncertain, do not execute the command.

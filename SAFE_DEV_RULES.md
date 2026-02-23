TITLE: SAFE DEVELOPMENT RULEBOOK — ROS2 HUMBLE — SHARED LAB MACHINE

CONTEXT:
Working on shared Ubuntu research machine.
ROS version: ROS2 Humble
Workspace:
`/home/admin/Desktop/anurag_ws`
PhD students have existing work on this system.
Goal: Zero interference with system, ROS installation, CUDA, GPU jobs, or other research.

------------------------------------------------------------

1) STRICT WORKSPACE ISOLATION

All work must remain inside:
`/home/admin/Desktop/anurag_ws`

Before running any delete/move command:
`pwd`

If output is NOT:
`/home/admin/Desktop/anurag_ws`
Abort immediately.

Never run:
`rm -rf *`
without confirming location using:
`pwd`

------------------------------------------------------------

2) ROS2 HUMBLE SAFETY RULES

Never modify global ROS installation:
`/opt/ros/humble`

Never run:
`sudo apt remove ros-*`
`sudo apt upgrade`
`sudo apt install ros-*`

Never edit:
`/opt/ros/humble/setup.bash`

Only source globally installed ROS:
`source /opt/ros/humble/setup.bash`

Do NOT reinstall ROS2 Humble.

------------------------------------------------------------

3) SAFE ROS2 WORKSPACE STRUCTURE

Inside:
`/home/admin/Desktop/anurag_ws`

Structure must be:

`anurag_ws/`
  `src/`
  `build/`
  `install/`
  `log/`

Create workspace:

`cd /home/admin/Desktop/anurag_ws`
`mkdir -p src`

Build only inside workspace:

`colcon build`

After build:

`source install/setup.bash`

Never build inside:
`/opt/ros/`
or any other user directory.

------------------------------------------------------------

4) PYTHON ISOLATION (MANDATORY)

Create virtual environment inside workspace:

`cd /home/admin/Desktop/anurag_ws`
`python3 -m venv venv`
`source venv/bin/activate`

Verify isolation:

`which python`

Must return:
`/home/admin/Desktop/anurag_ws/venv/bin/python`

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

Correct order:

`source /opt/ros/humble/setup.bash`
`cd /home/admin/Desktop/anurag_ws`
`source install/setup.bash`
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
Store all outputs inside:

`/home/admin/Desktop/anurag_ws`

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

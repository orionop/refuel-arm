#!/bin/bash
#
# Run the obstacle spawner and the NEO controller together.
#
# Prerequisites (must already be running before launching this):
#   Terminal 1:  roslaunch franka_gazebo panda.launch
#   Terminal 2:  rosrun controller_manager spawner effort_joint_trajectory_controller
#
# Then in a third terminal:
#   bash ~/Desktop/anurag_ws/refuel-arm/obstacle/ros1_neo/neo_demo.sh
#
# Ctrl+C once cleans up both background processes.

set -e

# Source ROS + workspace
source /opt/ros/noetic/setup.bash
if [ -f "$HOME/Desktop/anurag_ws/franka_ws/devel/setup.bash" ]; then
    source "$HOME/Desktop/anurag_ws/franka_ws/devel/setup.bash"
fi

# Resolve script directory (the directory this .sh file lives in)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Sanity checks
if ! command -v python3 >/dev/null 2>&1; then
    echo "[neo_demo] ERROR: python3 not on PATH"
    exit 1
fi

if ! rostopic list >/dev/null 2>&1; then
    echo "[neo_demo] ERROR: roscore / Gazebo is not running."
    echo "          Start Terminal 1 first: roslaunch franka_gazebo panda.launch"
    exit 1
fi

# Verify the controller is up
if ! rostopic list | grep -q "/effort_joint_trajectory_controller/command"; then
    echo "[neo_demo] ERROR: effort_joint_trajectory_controller is not running."
    echo "          Start Terminal 2 first:"
    echo "          rosrun controller_manager spawner effort_joint_trajectory_controller"
    exit 1
fi

echo "[neo_demo] Starting obstacle spawner in background..."
python3 obstacle_spawner_panda.py &
OBS_PID=$!
echo "[neo_demo] Obstacle spawner PID: $OBS_PID"

# Give the spawner a moment to register obstacles before NEO starts
sleep 1.0

echo "[neo_demo] Starting NEO controller (foreground)..."
echo "[neo_demo] Press Ctrl+C once to stop both."

# Clean up obstacle spawner if NEO exits or we get Ctrl+C
cleanup() {
    echo ""
    echo "[neo_demo] Shutting down obstacle spawner (PID $OBS_PID)..."
    kill "$OBS_PID" 2>/dev/null || true
    wait "$OBS_PID" 2>/dev/null || true
    echo "[neo_demo] Done."
}
trap cleanup EXIT INT TERM

# Run NEO in foreground. When it exits or you Ctrl+C, the trap fires.
python3 neo_ros_node_panda.py _controller_topic:=/effort_joint_trajectory_controller/command

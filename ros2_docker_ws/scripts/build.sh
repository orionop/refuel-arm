#!/bin/bash
# ============================================================
# KUKA KR6 R700 Refueling Simulation — Docker Helper
# ============================================================
# Usage:
#   ./build.sh        Build the Docker image
#   ./build.sh run    Build + Run interactively
#   ./build.sh sim    Build + Launch Gazebo simulation
# ============================================================

IMAGE_NAME="kuka_refuel"

echo "=== Building Docker Image: ${IMAGE_NAME} ==="
docker build -t ${IMAGE_NAME} .

if [ "$1" == "run" ]; then
    echo "=== Starting Interactive Shell ==="
    docker run -it --rm \
        --net=host \
        -e DISPLAY=$DISPLAY \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        ${IMAGE_NAME}

elif [ "$1" == "sim" ]; then
    echo "=== Launching Gazebo Refueling Simulation ==="
    docker run -it --rm \
        --net=host \
        -e DISPLAY=$DISPLAY \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        ${IMAGE_NAME} \
        bash -c "source /opt/ros/humble/setup.bash && \
                 source /home/admin/Desktop/anurag_ws/refuel-arm/kuka_refuel_ws/install/setup.bash && \
                 ros2 launch kuka_kr6_gazebo refuel_sim.launch.py"
fi

# ============================================================
# ROS 2 Humble + Gazebo for KUKA KR6 R700 Refueling Simulation
# ============================================================
# Build:  docker build -t kuka_refuel .
# Run:    docker run -it --rm --net=host \
#           -e DISPLAY=$DISPLAY \
#           -v /tmp/.X11-unix:/tmp/.X11-unix \
#           kuka_refuel
# ============================================================
FROM ros:humble-ros-base

# ---- System dependencies (ROS 2 + Gazebo + ros2_control) ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-xacro \
    ros-humble-joint-state-publisher \
    ros-humble-robot-state-publisher \
    ros-humble-gazebo-ros \
    ros-humble-gazebo-ros2-control \
    ros-humble-gazebo-ros-pkgs \
    ros-humble-ros2-control \
    ros-humble-controller-manager \
    ros-humble-controller-interface \
    ros-humble-hardware-interface \
    ros-humble-joint-state-broadcaster \
    ros-humble-joint-trajectory-controller \
    ros-humble-control-msgs \
    ros-humble-action-msgs \
    ros-humble-trajectory-msgs \
    python3-pip \
    python3-venv \
    python3-yaml \
    python3-numpy \
    && rm -rf /var/lib/apt/lists/*

# ---- Python IK dependencies (inside the container) ----
RUN pip3 install --no-cache-dir linearSubproblemSltns scipy

# ---- Copy workspace into container ----
WORKDIR /home/admin/Desktop/anurag_ws/refuel-arm
COPY kuka_ws/src kuka_refuel_ws/src

# ---- Build the ROS 2 workspace ----
SHELL ["/bin/bash", "-c"]
RUN source /opt/ros/humble/setup.bash && \
    cd kuka_refuel_ws && \
    colcon build && \
    echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc && \
    echo "source /home/admin/Desktop/anurag_ws/refuel-arm/kuka_refuel_ws/install/setup.bash" >> /root/.bashrc

# ---- Default: open a sourced bash shell ----
CMD ["/bin/bash"]

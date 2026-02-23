# ROS 2 Humble + deps for ur5_minimal_moveit and real controllers in sim (ros2_control + mock hardware)
FROM ros:humble-ros-base
RUN apt-get update && apt-get install -y -qq \
    ros-humble-xacro \
    ros-humble-joint-state-publisher \
    ros-humble-robot-state-publisher \
    ros-humble-moveit-ros-move-group \
    ros-humble-moveit-ros-planning-interface \
    ros-humble-moveit-planners-ompl \
    ros-humble-moveit-simple-controller-manager \
    ros-humble-control-msgs \
    ros-humble-ros2-control \
    ros-humble-controller-manager \
    ros-humble-controller-interface \
    ros-humble-hardware-interface \
    ros-humble-joint-trajectory-controller \
    ros-humble-ur-msgs \
    ros-humble-ur-client-library \
    ros-humble-ur \
    ros-humble-ur-moveit-config \
    python3-yaml \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace

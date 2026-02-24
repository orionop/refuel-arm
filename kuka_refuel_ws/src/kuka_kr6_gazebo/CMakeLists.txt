cmake_minimum_required(VERSION 3.0.2)
project(kuka_kr6_gazebo)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  trajectory_msgs
  control_msgs
  actionlib
  gazebo_ros
)

catkin_package()

# Install launch, urdf, worlds, config directories
install(DIRECTORY launch urdf worlds config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

# Install Python scripts
catkin_install_python(PROGRAMS
  scripts/ik_geometric.py
  scripts/refuel_mission_commander.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

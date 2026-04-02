cmake_minimum_required(VERSION 3.0.2)
project(ur5_gazebo)

find_package(catkin REQUIRED COMPONENTS
  gazebo_ros
  robot_state_publisher
  controller_manager
  ur_description
)

catkin_package()

install(DIRECTORY launch config worlds
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

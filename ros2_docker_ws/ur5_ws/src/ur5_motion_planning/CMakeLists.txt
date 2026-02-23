cmake_minimum_required(VERSION 3.8)
project(ur5_motion_planning)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(control_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(shape_msgs REQUIRED)

add_executable(ur5_plan_and_analyze_node src/ur5_plan_and_analyze_node.cpp)
ament_target_dependencies(ur5_plan_and_analyze_node
  rclcpp
  rclcpp_action
  control_msgs
  geometry_msgs
  trajectory_msgs
  tf2_geometry_msgs
  moveit_core
  moveit_ros_planning_interface
  moveit_msgs
  shape_msgs
)

install(TARGETS ur5_plan_and_analyze_node
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL
)

ament_package()

cmake_minimum_required(VERSION 3.8)
project(ur5_kdl_ik)

# Export compile_commands.json for IDE (e.g. when building inside container)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(moveit_kinematics REQUIRED)

add_executable(ur5_kdl_ik_baseline src/ur5_kdl_ik_node.cpp)
ament_target_dependencies(ur5_kdl_ik_baseline
  rclcpp
  geometry_msgs
  tf2_geometry_msgs
  moveit_core
  moveit_ros_planning_interface
  moveit_kinematics
)

install(TARGETS ur5_kdl_ik_baseline
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()

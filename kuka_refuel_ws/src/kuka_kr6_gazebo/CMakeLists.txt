cmake_minimum_required(VERSION 3.8)
project(kuka_kr6_gazebo)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(gazebo_ros REQUIRED)

# Install directories
install(
  DIRECTORY launch urdf worlds config
  DESTINATION share/${PROJECT_NAME}
)

# Install Python Nodes
install(PROGRAMS
  scripts/ik_geometric.py
  scripts/refuel_mission_commander.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()

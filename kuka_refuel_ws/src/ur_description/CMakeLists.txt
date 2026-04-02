cmake_minimum_required(VERSION 3.8)
project(ur_description)

find_package(ament_cmake REQUIRED)

# Install URDF directory
install(DIRECTORY
  urdf
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()

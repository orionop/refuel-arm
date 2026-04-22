cmake_minimum_required(VERSION 3.8)
project(kuka_kr6_gazebo)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# Generate model.sdf for KR8 with the absolute installed path to
# ros2_controllers.yaml baked in. gz_ros2_control passes <parameters>
# as a literal --params-file argument to rcl, so it must be absolute.
set(KR8_CONFIG_YAML
    "${CMAKE_INSTALL_PREFIX}/share/${PROJECT_NAME}/models/kr8_r2100/config/ros2_controllers.yaml")
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/models/kr8_r2100/model.sdf.in"
    "${CMAKE_CURRENT_BINARY_DIR}/kr8_model.sdf"
    @ONLY)

# Install all resource directories. model.sdf.in is harmless in the install —
# gz sim only looks for model.sdf, which is installed separately below.
install(DIRECTORY
  launch
  urdf
  worlds
  config
  meshes
  models
  DESTINATION share/${PROJECT_NAME}/
)

# Install the generated model.sdf (with absolute config path) over the .in file.
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/kr8_model.sdf"
    DESTINATION "share/${PROJECT_NAME}/models/kr8_r2100/"
    RENAME "model.sdf")

# Install Python scripts
install(PROGRAMS
  scripts/ik_geometric.py
  scripts/refuel_mission_commander.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()

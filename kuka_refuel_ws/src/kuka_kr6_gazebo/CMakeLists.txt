cmake_minimum_required(VERSION 3.8)
project(kuka_kr6_gazebo)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# Install directories (launch, urdf, worlds, config, models)
install(DIRECTORY
  launch
  urdf
  worlds
  config
  DESTINATION share/${PROJECT_NAME}/
)

# Install Python scripts
install(PROGRAMS
  scripts/ik_geometric.py
  scripts/refuel_mission_commander.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()

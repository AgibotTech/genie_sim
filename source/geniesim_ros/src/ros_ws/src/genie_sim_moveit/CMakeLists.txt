cmake_minimum_required(VERSION 3.22)
project(genie_sim_moveit)

find_package(genie_sim_common REQUIRED)
genie_sim_package()

find_package(ament_cmake REQUIRED)

install(
  PROGRAMS
    scripts/moveit_joint_states_bridge.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/launch")
  install(
    DIRECTORY launch
    DESTINATION share/${PROJECT_NAME}
    PATTERN "setup_assistant.launch" EXCLUDE)
endif()

install(DIRECTORY config DESTINATION share/${PROJECT_NAME})
install(FILES .setup_assistant DESTINATION share/${PROJECT_NAME})

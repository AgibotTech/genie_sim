cmake_minimum_required(VERSION 3.10)
project(genie_sim_common NONE)

# Consume CMAKE_EXPORT_COMPILE_COMMANDS (colcon passes it via -D) so
# NONE-language packages don't warn about an unused CLI variable.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL "Export compile_commands.json")

find_package(ament_cmake_core REQUIRED)

ament_package(
  CONFIG_EXTRAS "genie_sim_common-extras.cmake"
)

install(
  DIRECTORY cmake
  DESTINATION share/${PROJECT_NAME}
)

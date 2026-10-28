cmake_minimum_required(VERSION 3.8)
project(genie_sim_controllers)

if(CMAKE_EXPORT_COMPILE_COMMANDS)
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -Werror
                      -Wno-unused-parameter -Wno-reorder
                      -Wno-sign-compare -Wno-unused-variable)
endif()

set(CMAKE_CXX_STANDARD 20)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(controller_interface REQUIRED)
find_package(pluginlib REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(osqp REQUIRED)
set(OSQP_TARGET osqp::osqpstatic)

# ---------------------------------------------------------------------------
# Solver library (used by the servo plugins).
# ---------------------------------------------------------------------------
add_library(genie_sim_controllers_solvers_lib STATIC
  src/solvers/scalar_function.cpp
  src/solvers/quadratic_function.cpp
  src/solvers/relative_function.cpp
  src/solvers/problem.cpp
  src/solvers/osqp_solver.cpp
)
target_include_directories(genie_sim_controllers_solvers_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(genie_sim_controllers_solvers_lib PUBLIC Eigen3::Eigen ${OSQP_TARGET})
set_target_properties(genie_sim_controllers_solvers_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------------------
# Servo plugins (Parking / General / Optimal / Selftest / Spin).
# Instantiated directly (not loaded via pluginlib at runtime).
# ---------------------------------------------------------------------------
add_library(genie_sim_controllers_servo_plugins SHARED
  src/plugins/parking_servo.cpp
  src/plugins/general_servo.cpp
  src/plugins/optimal_servo.cpp
  src/plugins/selftest_servo.cpp
  src/plugins/spin_servo.cpp
)
target_include_directories(genie_sim_controllers_servo_plugins PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(genie_sim_controllers_servo_plugins genie_sim_controllers_solvers_lib Eigen3::Eigen)

# ---------------------------------------------------------------------------
# ServoCore — pure control law, no rclcpp dependency. Linked into both
# the controller plugin and the standalone Node below.
# ---------------------------------------------------------------------------
add_library(genie_sim_controllers_servo_core SHARED
  src/servo_core.cpp
)
target_include_directories(genie_sim_controllers_servo_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(genie_sim_controllers_servo_core genie_sim_controllers_servo_plugins Eigen3::Eigen)

# ---------------------------------------------------------------------------
# Controller plugin — ros2_control entrypoint. Thin wrapper around
# ServoCore: lifecycle + topic plumbing.
# ---------------------------------------------------------------------------
add_library(genie_sim_controllers_chassis_servo_controller SHARED
  src/chassis_servo_controller.cpp
)
target_include_directories(genie_sim_controllers_chassis_servo_controller PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(genie_sim_controllers_chassis_servo_controller
  genie_sim_controllers_servo_core
  Eigen3::Eigen
)
ament_target_dependencies(genie_sim_controllers_chassis_servo_controller
  rclcpp
  rclcpp_lifecycle
  controller_interface
  pluginlib
  std_msgs
  sensor_msgs
  geometry_msgs
)

pluginlib_export_plugin_description_file(
  controller_interface plugin_chassis_servo_controller.xml)

# ---------------------------------------------------------------------------
# Standalone servo_node — rclcpp::Node form. Wall-timer-driven. Same
# ServoCore as the controller above; use this when there's no
# controller_manager in the bringup graph.
# ---------------------------------------------------------------------------
add_library(genie_sim_controllers_servo_node_component SHARED
  src/servo_node.cpp
)
target_include_directories(genie_sim_controllers_servo_node_component PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(genie_sim_controllers_servo_node_component genie_sim_controllers_servo_core Eigen3::Eigen)
ament_target_dependencies(genie_sim_controllers_servo_node_component
  rclcpp
  rclcpp_components
  std_msgs
  sensor_msgs
  geometry_msgs
)

add_executable(genie_sim_chassis_servo_node src/main.cpp)
target_link_libraries(genie_sim_chassis_servo_node genie_sim_controllers_servo_node_component)
ament_target_dependencies(genie_sim_chassis_servo_node rclcpp)

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------
install(TARGETS
  genie_sim_controllers_solvers_lib
  genie_sim_controllers_servo_plugins
  genie_sim_controllers_servo_core
  genie_sim_controllers_servo_node_component
  genie_sim_controllers_chassis_servo_controller
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(TARGETS genie_sim_chassis_servo_node DESTINATION lib/${PROJECT_NAME})

install(
  PROGRAMS
    scripts/teleop_joy.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config/ DESTINATION share/${PROJECT_NAME}/config)
install(DIRECTORY launch/ DESTINATION share/${PROJECT_NAME}/launch)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_chassis_kinematics test/test_chassis_kinematics.cpp)
  target_link_libraries(test_chassis_kinematics
    genie_sim_controllers_servo_plugins Eigen3::Eigen)

  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

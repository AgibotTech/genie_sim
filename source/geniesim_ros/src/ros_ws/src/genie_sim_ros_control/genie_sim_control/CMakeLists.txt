cmake_minimum_required(VERSION 3.22)
project(genie_sim_control)

find_package(genie_sim_common REQUIRED)
genie_sim_package(WERROR)

# ROS 2 Jazzy (hardware_interface 4.x) deprecated the
# ``SystemInterface::on_init(const HardwareInfo &)`` signature in
# favour of ``on_init(const HardwareComponentInterfaceParams &)``.
# The new params struct wraps the HardwareInfo plus clock/logger
# handles.  We support BOTH APIs via a compile-time switch keyed on
# ``$ENV{ROS_DISTRO}`` — Humble keeps the legacy signature, Jazzy
# and newer use the params variant.  When the macro is set, the
# .hpp + .cpp both expose the new override; when unset, they keep
# the old override and ``-Werror`` will (correctly) reject the
# Jazzy deprecation if you build on Jazzy without setting the env.
#
# Env-var detection (not hardware_interface_VERSION) because
# colcon always exports ROS_DISTRO before invoking CMake, and it's
# the same signal ros2_control's own internal headers use to gate
# Humble-vs-Jazzy code paths.
set(_ros_distro "$ENV{ROS_DISTRO}")
if(_ros_distro STREQUAL "jazzy" OR
   _ros_distro STREQUAL "kilted" OR
   _ros_distro STREQUAL "rolling")
  add_compile_definitions(GENIE_HW_USE_PARAMS_API)
  message(STATUS "genie_sim_control: ROS_DISTRO='${_ros_distro}' "
                 "→ using HardwareComponentInterfaceParams on_init signature")
else()
  message(STATUS "genie_sim_control: ROS_DISTRO='${_ros_distro}' "
                 "→ using legacy HardwareInfo on_init signature")
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(controller_interface REQUIRED)
find_package(control_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(pluginlib REQUIRED)

add_library(genie_sim_robot_interface SHARED
  src/genie_sim_robot_interface.cpp
)
target_include_directories(genie_sim_robot_interface PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(genie_sim_robot_interface
  rclcpp
  rclcpp_lifecycle
  hardware_interface
  sensor_msgs
  pluginlib
)

pluginlib_export_plugin_description_file(hardware_interface plugin_genie_sim_robot_interface.xml)

add_library(genie_sim_planar_base_controller SHARED
  src/planar_base_controller.cpp
)
target_include_directories(genie_sim_planar_base_controller PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(genie_sim_planar_base_controller
  rclcpp
  rclcpp_action
  rclcpp_lifecycle
  controller_interface
  control_msgs
  geometry_msgs
  nav_msgs
  trajectory_msgs
  tf2
  tf2_ros
  pluginlib
)

pluginlib_export_plugin_description_file(controller_interface plugin_planar_base_controller.xml)

install(
  TARGETS genie_sim_robot_interface genie_sim_planar_base_controller
  EXPORT  ${PROJECT_NAME}Targets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/ DESTINATION include)

ament_package()

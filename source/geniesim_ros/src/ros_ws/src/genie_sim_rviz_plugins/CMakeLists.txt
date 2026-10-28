cmake_minimum_required(VERSION 3.15)
project(genie_sim_rviz_plugins)

find_package(genie_sim_common REQUIRED)
genie_sim_package()

set(CMAKE_AUTOMOC ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rviz_common REQUIRED)
find_package(rviz_rendering REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(pluginlib REQUIRED)
find_package(Qt5 REQUIRED COMPONENTS Core Widgets)
find_package(OpenSSL REQUIRED)

set(${PROJECT_NAME}_HEADERS_TO_MOC
  include/${PROJECT_NAME}/view_camera_pose_publisher_display.hpp
)

add_library(${PROJECT_NAME} SHARED
  src/view_camera_pose_publisher_display.cpp
  ${${PROJECT_NAME}_HEADERS_TO_MOC}
)

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(${PROJECT_NAME}
  Qt5::Core
  Qt5::Widgets
)

ament_target_dependencies(${PROJECT_NAME}
  rclcpp
  rviz_common
  rviz_rendering
  geometry_msgs
  tf2_ros
  pluginlib
)

target_compile_definitions(${PROJECT_NAME} PRIVATE
  "GENIE_SIM_RVIZ_PLUGINS_BUILDING_LIBRARY"
)

pluginlib_export_plugin_description_file(rviz_common plugin_view_camera_pose.xml)

install(
  TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(
  DIRECTORY include/
  DESTINATION include
)

install(
  FILES plugin_view_camera_pose.xml
  DESTINATION share/${PROJECT_NAME}
)

ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME})
ament_export_dependencies(rclcpp rviz_common rviz_rendering geometry_msgs pluginlib)

ament_package()

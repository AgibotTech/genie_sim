cmake_minimum_required(VERSION 3.15)
project(genie_sim_render)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -Werror)
endif()

if(CMAKE_EXPORT_COMPILE_COMMANDS)
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(std_msgs REQUIRED)
find_package(image_transport REQUIRED)
find_package(pluginlib REQUIRED)
find_package(nlohmann_json 3.10 REQUIRED)

# --- ovrtx SDK (from pip-installed ovrtx package) ---
execute_process(
  COMMAND python3 -c "import ovrtx, os; print(os.path.dirname(ovrtx.__file__))"
  OUTPUT_VARIABLE OVRTX_PKG_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE _ovrtx_result
)
if(NOT _ovrtx_result EQUAL 0 OR NOT EXISTS "${OVRTX_PKG_DIR}/include/ovrtx/ovrtx.h")
  message(FATAL_ERROR
    "ovrtx not found via pip. Install with: python3 -m pip install ovrtx")
endif()
message(STATUS "Found ovrtx at: ${OVRTX_PKG_DIR}")

add_library(ovrtx::ovrtx SHARED IMPORTED)
set_target_properties(ovrtx::ovrtx PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${OVRTX_PKG_DIR}/include"
  IMPORTED_LOCATION "${OVRTX_PKG_DIR}/bin/libovrtx-dynamic.so"
  IMPORTED_NO_SONAME TRUE
)

# --- Renderer Node ---
add_executable(genie_sim_render_node
  src/main.cpp
  src/render_node.cpp
)

target_include_directories(genie_sim_render_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(genie_sim_render_node
  ovrtx::ovrtx
  nlohmann_json::nlohmann_json
)

ament_target_dependencies(genie_sim_render_node
  rclcpp
  geometry_msgs
  tf2_msgs
  std_msgs
  pluginlib
)

# Set RPATH so the node can find libovrtx-dynamic.so at runtime
set_target_properties(genie_sim_render_node PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  INSTALL_RPATH "${OVRTX_PKG_DIR}/bin"
)

install(TARGETS genie_sim_render_node
  DESTINATION lib/${PROJECT_NAME}
)

# --- ROS image publisher plugin (default) ---
add_library(genie_sim_render_ros_image_publisher_plugin SHARED
  src/plugins/ros_image_publisher_plugin.cpp
)
target_include_directories(genie_sim_render_ros_image_publisher_plugin PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
ament_target_dependencies(genie_sim_render_ros_image_publisher_plugin
  rclcpp pluginlib sensor_msgs geometry_msgs tf2_ros image_transport
)

pluginlib_export_plugin_description_file(genie_sim_render plugin_ros_image_publisher.xml)

install(TARGETS genie_sim_render_ros_image_publisher_plugin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(
  DIRECTORY include/genie_sim_render/
  DESTINATION include/genie_sim_render
)

ament_export_include_directories(include)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

install(PROGRAMS
  scripts/isaacsim_render.py
  scripts/_isaac_render_common.py
  scripts/_instance_materials.py
  scripts/_render_randomize.py
  scripts/_fused_point_instancers.py
  scripts/_fused_background_reset.py
  scripts/_fused_mimic_pose.py
  scripts/_render_perf.py
  scripts/isaacsim_tiled_render.py
  scripts/isaacsim_fused_sim.py
  DESTINATION lib/${PROJECT_NAME})

ament_package()

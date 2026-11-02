cmake_minimum_required(VERSION 3.15)
project(genie_sim_render)

find_package(genie_sim_common REQUIRED)
genie_sim_package(WERROR)

find_package(ament_cmake REQUIRED)
option(GENIE_SIM_RENDER_BUILD_OVRTX "Build the OVRTX C++ node, plugin and tests" ON)

if(GENIE_SIM_RENDER_BUILD_OVRTX)
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

# The retained 0.3 SDK uses a bool; 0.4 exposes a motion-BVH mode enum.
# Detect the declaration from the installed header rather than Isaac version.
file(READ "${OVRTX_PKG_DIR}/include/ovrtx/ovrtx_config.h" _ovrtx_config_header)
if(_ovrtx_config_header MATCHES "ovrtx_config_entry_motion_bvh\\(")
  set(GENIE_OVRTX_HAS_MOTION_BVH_MODE TRUE)
endif()

# --- Renderer Node ---
add_executable(genie_sim_render_node
  src/main.cpp
  src/render_node.cpp
  src/manifest_schema.cpp
)

if(GENIE_OVRTX_HAS_MOTION_BVH_MODE)
  target_compile_definitions(genie_sim_render_node PRIVATE GENIE_OVRTX_HAS_MOTION_BVH_MODE=1)
endif()

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
  tf2_ros
  std_msgs
  pluginlib
)

# Keep the build-tree binary runnable, but do not bake the build machine's
# pip installation path into the installed ELF. physics_backend.launch.py
# supplies the active OVRTX library path to this process at runtime.
set_target_properties(genie_sim_render_node PROPERTIES
  BUILD_RPATH "${OVRTX_PKG_DIR}/bin"
  INSTALL_RPATH ""
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
endif()

install(PROGRAMS
  scripts/isaacsim_render.py
  scripts/isaacsim_tiled_render.py
  scripts/isaacsim_fused_sim.py
  scripts/_isaac_render_common.py
  scripts/_sync_capture.py
  scripts/_material_generalization.py
  scripts/_instance_materials.py
  scripts/_render_randomize.py
  scripts/_fused_point_instancers.py
  scripts/_fused_background_reset.py
  scripts/_fused_mimic_pose.py
  scripts/_render_perf.py
  scripts/isaacsim_snapshot_render.py
  scripts/_isaac_snapshot_scene.py
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY docs/
  DESTINATION share/${PROJECT_NAME}/docs)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(isaacsim_transport_tests test/test_isaacsim_transport.py)
  ament_add_pytest_test(sync_initialization_tests test/test_sync_initialization.py)
  ament_add_pytest_test(isaac_snapshot_tests test/test_isaac_snapshot.py TIMEOUT 30)
  if(GENIE_SIM_RENDER_BUILD_OVRTX)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_manifest_schema
    test/test_manifest_schema.cpp
    src/manifest_schema.cpp
  )
  target_include_directories(test_manifest_schema PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  target_link_libraries(test_manifest_schema
    nlohmann_json::nlohmann_json
  )
  endif()
endif()

ament_package()

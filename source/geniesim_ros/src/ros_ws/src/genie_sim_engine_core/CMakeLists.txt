cmake_minimum_required(VERSION 3.16)
project(genie_sim_engine_core CXX)

if(NOT UNIX OR APPLE)
  message(FATAL_ERROR "genie_sim_engine_core is Linux-only")
endif()

find_package(genie_sim_common REQUIRED)
genie_sim_package()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(python_cmake_module REQUIRED)
find_package(PythonExtra MODULE REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(rosgraph_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(gazebo_msgs REQUIRED)
find_package(lifecycle_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(realtime_tools REQUIRED)
find_package(image_transport REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11_vendor QUIET)
find_package(pybind11 REQUIRED)

# ---------------------------------------------------------------------------
# Python package (PhysicsEngine ABC, EngineSession base, run loop, params).
# ---------------------------------------------------------------------------
ament_python_install_package(${PROJECT_NAME})

# ---------------------------------------------------------------------------
# genie_sim_engine_py — pybind11 ROS bridge.
#
# A C++ extension module that publishes /joint_states, /tf_scene, /odom,
# /clock for every backend, and exposes the realtime scheduler / stats ring
# to Python. All concrete backends use this module via
# ``import genie_sim_engine_py as _core`` inside EngineSession.__init__.
#
# Installed to ${PYTHON_INSTALL_DIR} so the module resolves from any
# process on the workspace's Python path — no reliance on the
# entry-point script's directory.
# ---------------------------------------------------------------------------
pybind11_add_module(genie_sim_engine_py MODULE
  src/pybinding.cpp
  src/realtime_ros_node.cpp
  src/realtime_scheduler.cpp
  src/stats.cpp
  src/realtime_buffer.cpp
)

target_include_directories(genie_sim_engine_py PRIVATE src)

ament_target_dependencies(genie_sim_engine_py PUBLIC
  rclcpp
  sensor_msgs
  nav_msgs
  geometry_msgs
  tf2_msgs
  tf2_ros
  rosgraph_msgs
  std_msgs
  builtin_interfaces
  gazebo_msgs
  lifecycle_msgs
  std_srvs
  realtime_tools
  image_transport
)

target_compile_options(genie_sim_engine_py PRIVATE
  -fvisibility=hidden
  -fvisibility-inlines-hidden
  -ffunction-sections
  -fdata-sections
  -fno-math-errno
  -Wall
  -Wextra
  -Wno-unused-parameter
)

# Obfuscation / stripping only on Release builds.
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  target_compile_options(genie_sim_engine_py PRIVATE -O3 -fno-ident)
  target_link_options(genie_sim_engine_py PRIVATE
    -Wl,--gc-sections
    -Wl,--build-id=none
    -Wl,--strip-all
    -s
  )
endif()

install(TARGETS genie_sim_engine_py
  LIBRARY DESTINATION "${PYTHON_INSTALL_DIR}"
)

# ---------------------------------------------------------------------------
# Shared configuration consumed by EngineSession (physics_params.yaml).
# ---------------------------------------------------------------------------
install(
  DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
  # Schema round-trip / version guard tests. Runs the module without
  # ROS runtime — pure dataclass + YAML/JSON. ament_cmake_pytest picks
  # up tests/ automatically once a pytest target is declared.
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(api_tests tests/api
    TIMEOUT 60
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )
  ament_add_pytest_test(assemble_tests tests/assemble
    TIMEOUT 60
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )
  ament_add_pytest_test(schema_tests tests/schema
    TIMEOUT 60
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )
  # Loop pacing (step_mode) tests. Same shape as schema_tests: no ROS
  # runtime, no GPU. Drives a SCRIPTED clock, so a 30 s simulated pause
  # costs no wall time and the suite cannot flake on machine load —
  # keep it that way; a real sleep() here would be a CI time bomb.
  ament_add_pytest_test(pacing_tests tests/pacing
    TIMEOUT 60
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )
  # SimulationControl / ActivitySignal contract tests, and the ROS reset
  # path. Every ABC default is asserted callable on a minimal backend, so a
  # member added to the Protocol without a default breaks here rather than
  # at a downstream consumer's first call.
  ament_add_pytest_test(lifecycle_tests tests/lifecycle
    TIMEOUT 60
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )
  # /joint_states_scene channel selection via PublishPlan.
  ament_add_pytest_test(publish_tests tests/publish
    TIMEOUT 60
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )
  ament_add_pytest_test(params_tests tests/params
    TIMEOUT 60
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )
  ament_add_pytest_test(session_tests tests/session
    TIMEOUT 60
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )
endif()

ament_package()

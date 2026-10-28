cmake_minimum_required(VERSION 3.22)
project(genie_sim_moveit_plugins)

find_package(genie_sim_common REQUIRED)
genie_sim_package(WERROR)

# Detect ROS 2 distro and toggle the MoveIt planning_interface API
# compatibility macros — see ``include/.../moveit_compat.hpp`` for
# the full list of renames Humble → Jazzy.  Setting this macro
# routes the GENIE_MOVEIT_FIELD / GENIE_MOVEIT_SOLVE_* macros to the
# Jazzy-flavoured names; leaving it undefined keeps the legacy
# Humble forms.
set(_ros_distro "$ENV{ROS_DISTRO}")
if(_ros_distro STREQUAL "jazzy" OR
   _ros_distro STREQUAL "kilted" OR
   _ros_distro STREQUAL "rolling")
  add_compile_definitions(GENIE_MOVEIT_USE_NEW_API)
  message(STATUS "genie_sim_moveit_plugins: ROS_DISTRO='${_ros_distro}' "
                 "→ using new MoveIt planning_interface API "
                 "(void solve, no-underscore fields)")
else()
  message(STATUS "genie_sim_moveit_plugins: ROS_DISTRO='${_ros_distro}' "
                 "→ using legacy MoveIt planning_interface API "
                 "(bool solve, underscore-suffixed fields)")
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(moveit_core REQUIRED)
find_package(pluginlib REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(moveit_kinematics REQUIRED)
find_package(bio_ik REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(OpenMP)

set(DEPS
  rclcpp
  moveit_core
  pluginlib
  tf2_eigen
  geometry_msgs
  moveit_msgs
)

# ===========================================================================
# IK Plugin — Relaxed IK (NLopt-based with collision / singularity costs)
# ===========================================================================
add_library(genie_sim_moveit_plugins_relaxed_ik_plugin SHARED
  src/relaxed_ik/relaxed_ik_plugin.cpp
)
target_include_directories(genie_sim_moveit_plugins_relaxed_ik_plugin PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(genie_sim_moveit_plugins_relaxed_ik_plugin
  Eigen3::Eigen
  yaml-cpp
)
ament_target_dependencies(genie_sim_moveit_plugins_relaxed_ik_plugin ${DEPS})

# ===========================================================================
# Planner Plugin — RRT-Connect + TOPP-RA time parameterization
# ===========================================================================
add_library(genie_sim_moveit_plugins_planner_plugin SHARED
  src/genie_planner/genie_planner_manager.cpp
  src/genie_planner/rrt_connect_planner.cpp
  src/genie_planner/topp_ra.cpp
)
target_include_directories(genie_sim_moveit_plugins_planner_plugin PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(genie_sim_moveit_plugins_planner_plugin
  Eigen3::Eigen
)
ament_target_dependencies(genie_sim_moveit_plugins_planner_plugin ${DEPS})

# ===========================================================================
# KDL + Coupled Constraints Plugin
# ===========================================================================
add_library(genie_sim_moveit_plugins_kdl_coupled_plugin SHARED
  src/kdl_coupled/kdl_coupled_plugin.cpp
)
target_include_directories(genie_sim_moveit_plugins_kdl_coupled_plugin PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
# Locate the upstream KDL plugin shared lib that defines
# ``kdl_kinematics_plugin::KDLKinematicsPlugin``'s vtable / typeinfo.
# ``moveit_kinematics`` is the meta-package that ships this .so but
# does NOT expose it as a CMake-imported target — only the
# kdl_kinematics_parameters helper.  find_library searches
# CMAKE_PREFIX_PATH (which includes /opt/ros/<distro>/lib via the
# ROS environment) so this works on any ROS install.
#
# Without this link, dlopen of the plugin fails at runtime with::
#
#   undefined symbol: _ZTIN21kdl_kinematics_plugin19KDLKinematicsPluginE
find_library(MOVEIT_KDL_KINEMATICS_PLUGIN_LIB
  NAMES moveit_kdl_kinematics_plugin
  REQUIRED
)
target_link_libraries(genie_sim_moveit_plugins_kdl_coupled_plugin
  Eigen3::Eigen
  yaml-cpp
  ${MOVEIT_KDL_KINEMATICS_PLUGIN_LIB}
)
# Disable ``--as-needed`` for this target so the linker doesn't strip
# ``libmoveit_kdl_kinematics_plugin.so`` from DT_NEEDED.  Pluginlib
# needs the typeinfo / vtable for ``kdl_kinematics_plugin::KDLKinematicsPlugin``
# at dlopen, but the .cpp doesn't reference any symbol from it at
# link time — modern ``ld --as-needed`` drops the dep on that basis.
#
# ``BEFORE`` prepends the flag to the link command so it covers
# every library that follows.  Side effect: other libs in
# target_link_libraries above (Eigen, yaml-cpp) end up unconditionally
# in DT_NEEDED too — fine, since this plugin uses them.
target_link_options(genie_sim_moveit_plugins_kdl_coupled_plugin BEFORE PRIVATE
  "LINKER:--no-as-needed")
ament_target_dependencies(genie_sim_moveit_plugins_kdl_coupled_plugin ${DEPS} moveit_kinematics)

# ===========================================================================
# BioIK + Coupled Constraints Plugin
# ===========================================================================
add_library(genie_sim_moveit_plugins_bio_ik_coupled_plugin SHARED
  src/bio_ik_coupled/bio_ik_coupled_plugin.cpp
)
target_include_directories(genie_sim_moveit_plugins_bio_ik_coupled_plugin PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(genie_sim_moveit_plugins_bio_ik_coupled_plugin
  Eigen3::Eigen
  yaml-cpp
  bio_ik::bio_ik
)
ament_target_dependencies(genie_sim_moveit_plugins_bio_ik_coupled_plugin ${DEPS} bio_ik ament_index_cpp)

# ===========================================================================
# Plugin description exports
# ===========================================================================
pluginlib_export_plugin_description_file(moveit_core plugin_relaxed_ik.xml)
pluginlib_export_plugin_description_file(moveit_core plugin_genie_planner.xml)
pluginlib_export_plugin_description_file(moveit_core plugin_kdl_coupled.xml)
pluginlib_export_plugin_description_file(moveit_core plugin_bio_ik_coupled.xml)

# ===========================================================================
# Install
# ===========================================================================
install(
  TARGETS
    genie_sim_moveit_plugins_relaxed_ik_plugin
    genie_sim_moveit_plugins_planner_plugin
    genie_sim_moveit_plugins_kdl_coupled_plugin
    genie_sim_moveit_plugins_bio_ik_coupled_plugin
  EXPORT  ${PROJECT_NAME}Targets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config  DESTINATION share/${PROJECT_NAME})

ament_package()

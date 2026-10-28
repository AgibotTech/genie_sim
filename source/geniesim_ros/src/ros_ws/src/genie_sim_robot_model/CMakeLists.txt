cmake_minimum_required(VERSION 3.10)
project(genie_sim_robot_model)

if(CMAKE_EXPORT_COMPILE_COMMANDS)
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

set(CMAKE_INSTALL_MESSAGE "LAZY")

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(xacro REQUIRED)

file(MAKE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/urdf")

# xacro_add_xacro_file runs ``xacro --deps`` at *configure* time without wrapping COMMAND;
# it needs the same fake prefix as the build rule (see share/xacro/cmake/xacro-extras.cmake).
set(_GENIE_SIM_AMENT_IDX "${PROJECT_BINARY_DIR}/ament_cmake_index")
set(_GENIE_SIM_PKG_SHARE "${_GENIE_SIM_AMENT_IDX}/share/${PROJECT_NAME}")
file(MAKE_DIRECTORY "${_GENIE_SIM_AMENT_IDX}/share")
if(EXISTS "${_GENIE_SIM_PKG_SHARE}")
  file(REMOVE "${_GENIE_SIM_PKG_SHARE}")
endif()
execute_process(
  COMMAND ${CMAKE_COMMAND} -E create_symlink
          "${CMAKE_CURRENT_SOURCE_DIR}" "${_GENIE_SIM_PKG_SHARE}"
  RESULT_VARIABLE _genie_sim_symlink_rc
)
if(_genie_sim_symlink_rc)
  message(FATAL_ERROR "genie_sim_robot_model: failed to create ${_GENIE_SIM_PKG_SHARE}")
endif()
# ament_index_python requires this marker (symlink to source alone is not enough).
file(MAKE_DIRECTORY "${_GENIE_SIM_AMENT_IDX}/share/ament_index/resource_index/packages")
file(WRITE "${_GENIE_SIM_AMENT_IDX}/share/ament_index/resource_index/packages/${PROJECT_NAME}" "")
set(_GENIE_SIM_SAVED_AMENT_PREFIX_PATH "$ENV{AMENT_PREFIX_PATH}")
if(WIN32)
  set(_GENIE_SIM_AMENT_SEP ";")
else()
  set(_GENIE_SIM_AMENT_SEP ":")
endif()
set(ENV{AMENT_PREFIX_PATH} "${_GENIE_SIM_AMENT_IDX}${_GENIE_SIM_AMENT_SEP}${_GENIE_SIM_SAVED_AMENT_PREFIX_PATH}")

# Official ROS 2 xacro CMake helpers (xacro-extras.cmake): dependency scan + ament_cmake_index for $(find pkg).
set(GENIE_SIM_URDF_OUTPUTS "")

macro(genie_sim_xacro_robot _stem)
  xacro_add_xacro_file(
    xacro/robot.xacro
    "${CMAKE_CURRENT_SOURCE_DIR}/urdf/${_stem}.urdf"
    REMAP ${ARGN}
    OUTPUT GENIE_SIM_XACRO_OUT_${_stem})
  list(APPEND GENIE_SIM_URDF_OUTPUTS ${GENIE_SIM_XACRO_OUT_${_stem}})
endmacro()

# Entries mirror robots/*/robot.yml (add REMAP lines when adding robots).
# Gripper selector arg name is ``gripper`` consistently across all
# provider scripts (agilex.xacro, genie.xacro, universal_robots.xacro).
genie_sim_xacro_robot(agilex_piper robot_model:=agilex arm:=piper gripper:=with_gripper)
genie_sim_xacro_robot(agilex_piper_no_gripper robot_model:=agilex arm:=piper gripper:=no_gripper)
genie_sim_xacro_robot(arx_x5 robot_model:=arx arm:=x5)
genie_sim_xacro_robot(arx_acone robot_model:=arx arm:=acone)
# Identified drive/passive profiles; the stock descriptions remain selectable.
genie_sim_xacro_robot(arx_x5_sysid robot_model:=arx arm:=x5 param_source:=sysid)
genie_sim_xacro_robot(arx_acone_sysid robot_model:=arx arm:=acone param_source:=sysid)
genie_sim_xacro_robot(agilex_aloha robot_model:=agilex arm:=aloha)
genie_sim_xacro_robot(franka_fr3 robot_model:=franka arm:=fr3)
genie_sim_xacro_robot(genie_g2_crs_omnipicker robot_model:=genie body:=g2 arm:=crs gripper:=omnipicker)
genie_sim_xacro_robot(genie_g2_crs_swiftpicker robot_model:=genie body:=g2 arm:=crs gripper:=swiftpicker)
genie_sim_xacro_robot(genie_g2_crsB_omnipicker robot_model:=genie body:=g2 arm:=crsB gripper:=omnipicker)
genie_sim_xacro_robot(genie_g2_crsB_swiftpicker robot_model:=genie body:=g2 arm:=crsB gripper:=swiftpicker)
# Identified link inertials and passive joint parameters; kinematics are shared.
genie_sim_xacro_robot(genie_g2_crsB_omnipicker_sysid robot_model:=genie body:=g2 arm:=crsB gripper:=omnipicker inertia_source:=sysid)
genie_sim_xacro_robot(genie_g2_crsB_swiftpicker_sysid robot_model:=genie body:=g2 arm:=crsB gripper:=swiftpicker inertia_source:=sysid)
genie_sim_xacro_robot(ur5_robotiq_85 robot_model:=universal_robots arm:=ur5 gripper:=robotiq_85)
genie_sim_xacro_robot(ur5_robotiq_140 robot_model:=universal_robots arm:=ur5 gripper:=robotiq_140)
genie_sim_xacro_robot(ur5_no_gripper robot_model:=universal_robots arm:=ur5 gripper:=no_gripper)

set(ENV{AMENT_PREFIX_PATH} "${_GENIE_SIM_SAVED_AMENT_PREFIX_PATH}")

add_custom_target(generate_urdf ALL DEPENDS ${GENIE_SIM_URDF_OUTPUTS})

install(
  DIRECTORY xacro robots launch rviz urdf
  DESTINATION share/${PROJECT_NAME}
)

ament_python_install_package(${PROJECT_NAME})

install(
  PROGRAMS
    scripts/normalize_obj_names.py
    scripts/diagnose_urdf.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  FILES LICENSE
  DESTINATION share/${PROJECT_NAME}
)

ament_package()

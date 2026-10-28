# Copyright (c) AgiBot Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

#
# Standard genie_sim project setup.
#
# Applies the conventions shared by every genie_sim_* package:
#   * default CMAKE_BUILD_TYPE=Release when not otherwise set
#   * CMAKE_EXPORT_COMPILE_COMMANDS honored if the caller enabled it
#   * C++ standard defaults to 20 (override by setting CMAKE_CXX_STANDARD
#     before calling, or by passing CXX_STANDARD <n>)
#   * warning flags on gcc/clang: -Wall -Wextra -Wpedantic
#     (add -Werror by passing WERROR)
#
# Usage:
#   find_package(genie_sim_common REQUIRED)
#   genie_sim_package()
#   genie_sim_package(CXX_STANDARD 17 WERROR)
#
macro(genie_sim_package)
  set(_gs_options WERROR)
  set(_gs_one_value CXX_STANDARD)
  set(_gs_multi_value "")
  cmake_parse_arguments(_GS "${_gs_options}" "${_gs_one_value}"
                        "${_gs_multi_value}" ${ARGN})

  if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "genie_sim_package: no build type set, defaulting to Release")
    set(CMAKE_BUILD_TYPE "Release" CACHE
        STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
      "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
  endif()

  if(NOT DEFINED CMAKE_EXPORT_COMPILE_COMMANDS)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
  endif()
  # Reference the variable via `set(... CACHE ...)` so CMake's
  # unused-vars tracker counts it as consumed on NONE-language /
  # non-compiling packages that inherit the CLI -D from colcon.
  set(CMAKE_EXPORT_COMPILE_COMMANDS
      "${CMAKE_EXPORT_COMPILE_COMMANDS}"
      CACHE BOOL "Export compile_commands.json")

  if(NOT CMAKE_CXX_STANDARD)
    if(_GS_CXX_STANDARD)
      set(CMAKE_CXX_STANDARD ${_GS_CXX_STANDARD})
    else()
      set(CMAKE_CXX_STANDARD 20)
    endif()
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
  endif()

  set(CMAKE_POSITION_INDEPENDENT_CODE ON)

  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
    if(_GS_WERROR)
      add_compile_options(-Werror)
    endif()
  endif()

  option(GENIE_SIM_COVERAGE_ENABLED "Enable code coverage" FALSE)
  if(GENIE_SIM_COVERAGE_ENABLED)
    add_compile_options(--coverage)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} --coverage")
  endif()
endmacro()

#
# Isolated-test wrappers. Prefer the ament_add_ros_isolated_* variants
# when available (Kilted+), otherwise fall back to the classic
# ament_add_* forms. Toggled by USE_ISOLATED_TESTS (default OFF).
#
option(USE_ISOLATED_TESTS "Enable ros_isolated_test" OFF)

function(genie_sim_add_test target)
  if(COMMAND ament_add_ros_isolated_test AND USE_ISOLATED_TESTS)
    ament_add_ros_isolated_test(${target} ${ARGN})
  else()
    ament_add_test(${target} ${ARGN})
  endif()
endfunction()

function(genie_sim_add_gtest target)
  if(COMMAND ament_add_ros_isolated_gtest AND USE_ISOLATED_TESTS)
    ament_add_ros_isolated_gtest(${target} ${ARGN})
  else()
    ament_add_gtest(${target} ${ARGN})
  endif()
endfunction()

function(genie_sim_add_gmock target)
  if(COMMAND ament_add_ros_isolated_gmock AND USE_ISOLATED_TESTS)
    ament_add_ros_isolated_gmock(${target} ${ARGN})
  else()
    ament_add_gmock(${target} ${ARGN})
  endif()
endfunction()

function(genie_sim_add_pytest_test target)
  if(COMMAND ament_add_ros_isolated_pytest_test AND USE_ISOLATED_TESTS)
    ament_add_ros_isolated_pytest_test(${target} ${ARGN})
  else()
    ament_add_pytest_test(${target} ${ARGN})
  endif()
endfunction()

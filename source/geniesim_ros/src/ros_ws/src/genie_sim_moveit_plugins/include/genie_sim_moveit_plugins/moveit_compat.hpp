// Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
// Author: Genie Sim Team
// License: Mozilla Public License Version 2.0

#pragma once

// MoveIt planning_interface compatibility shim — Humble ↔ Jazzy.
//
// MoveIt's ``planning_interface`` API changed between ROS 2 Humble
// (moveit_core 2.5.x) and ROS 2 Jazzy (moveit_core 2.10+):
//
//   1. ``PlanningContext::solve(...)`` return type
//      Humble:   virtual bool solve(MotionPlanResponse & res) = 0;
//      Jazzy:    virtual void solve(MotionPlanResponse & res) = 0;
//      A Humble override returning ``bool`` is rejected by Jazzy's
//      ``-Werror=overloaded-virtual`` (different return type).
//
//   2. ``MotionPlanResponse`` / ``MotionPlanDetailedResponse`` field
//      names lost their trailing underscores
//      Humble:   res.error_code_, res.trajectory_, res.planning_time_,
//                res.processing_time_, res.description_
//      Jazzy:    res.error_code,  res.trajectory,  res.planning_time,
//                res.processing_time,  res.description
//
//   3. ``MoveGroupInterface::Plan::trajectory_`` → ``trajectory``
//      Same trailing-underscore strip as the response structs.
//
// The macro ``GENIE_MOVEIT_USE_NEW_API`` is defined by the package's
// CMakeLists.txt when ``$ENV{ROS_DISTRO}`` is jazzy/kilted/rolling.
// Humble builds leave it undefined and fall through to the
// underscore variant.  Add a new distro to the CMake check when
// upstream ROS bumps the moveit API again.
//
// Usage in .cpp:
//
//   #include "genie_sim_moveit_plugins/moveit_compat.hpp"
//   ...
//   res.GENIE_MOVEIT_FIELD(error_code).val = …;
//   plan.GENIE_MOVEIT_FIELD(trajectory).joint_trajectory;
//   GENIE_MOVEIT_SOLVE_OK;        // bool ``return true`` on Humble,
//                                 // void ``return``      on Jazzy
//   GENIE_MOVEIT_SOLVE_ERR;       // same dual ``return``
//
// Usage in .hpp for solve() declarations:
//
//   GENIE_MOVEIT_SOLVE_RT solve(MotionPlanResponse & res) override;
//
// Why preprocessor and not a template/wrapper class: the field
// rename is at struct-member-access level and can't be abstracted
// without a wrapper struct that proxies every field, which is far
// more invasive than 6 token-pasting macros.

#ifdef GENIE_MOVEIT_USE_NEW_API

// Jazzy + later: field names without trailing underscore, void solve().
#define GENIE_MOVEIT_FIELD(name)  name
#define GENIE_MOVEIT_SOLVE_RT     void
#define GENIE_MOVEIT_SOLVE_OK     return        // void function, bare return
#define GENIE_MOVEIT_SOLVE_ERR    return

#else

// Humble + earlier: trailing-underscore field names, bool solve().
#define GENIE_MOVEIT_FIELD(name)  name ## _
#define GENIE_MOVEIT_SOLVE_RT     bool
#define GENIE_MOVEIT_SOLVE_OK     return true
#define GENIE_MOVEIT_SOLVE_ERR    return false

#endif  // GENIE_MOVEIT_USE_NEW_API

// ===========================================================================
// MoveIt header path routing (Humble ``.h`` ↔ Jazzy ``.hpp``).
//
// Jazzy renamed every public moveit_core header from ``.h`` to ``.hpp``
// and emits a ``#pragma message`` warning on every ``.h`` include. The
// ``.hpp`` form does not exist on Humble. To keep both distros buildable
// without #ifdef-ing every include site, route each header through one
// of the ``MOVEIT_H_*`` macros and use the computed-include form::
//
//   #include "genie_sim_moveit_plugins/moveit_compat.hpp"
//   #include MOVEIT_H_KINEMATICS_BASE
//
// Add a new ``MOVEIT_H_*`` entry whenever a new moveit_core header is
// pulled in by this package.
// ===========================================================================

#ifdef GENIE_MOVEIT_USE_NEW_API
#define MOVEIT_H_COLLISION_COMMON          <moveit/collision_detection/collision_common.hpp>
#define MOVEIT_H_COLLISION_ENV_FCL         <moveit/collision_detection_fcl/collision_env_fcl.hpp>
#define MOVEIT_H_KDL_KINEMATICS_PLUGIN     <moveit/kdl_kinematics_plugin/kdl_kinematics_plugin.hpp>
#define MOVEIT_H_KINEMATICS_BASE           <moveit/kinematics_base/kinematics_base.hpp>
#define MOVEIT_H_MOVE_GROUP_INTERFACE      <moveit/move_group_interface/move_group_interface.hpp>
#define MOVEIT_H_PLANNING_INTERFACE        <moveit/planning_interface/planning_interface.hpp>
#define MOVEIT_H_PLANNING_REQUEST_ADAPTER  <moveit/planning_interface/planning_request_adapter.hpp>
#define MOVEIT_H_PLANNING_SCENE            <moveit/planning_scene/planning_scene.hpp>
#define MOVEIT_H_PLANNING_SCENE_INTERFACE \
  "moveit/planning_scene_interface/planning_scene_interface.hpp"
#define MOVEIT_H_PLANNING_SCENE_MONITOR \
  "moveit/planning_scene_monitor/planning_scene_monitor.hpp"
#define MOVEIT_H_JOINT_MODEL_GROUP         <moveit/robot_model/joint_model_group.hpp>
#define MOVEIT_H_LINK_MODEL                <moveit/robot_model/link_model.hpp>
#define MOVEIT_H_ROBOT_MODEL_LOADER        <moveit/robot_model_loader/robot_model_loader.hpp>
#define MOVEIT_H_CONVERSIONS               <moveit/robot_state/conversions.hpp>
#define MOVEIT_H_ROBOT_STATE               <moveit/robot_state/robot_state.hpp>
#define MOVEIT_H_ROBOT_TRAJECTORY          <moveit/robot_trajectory/robot_trajectory.hpp>
#else
#define MOVEIT_H_COLLISION_COMMON          <moveit/collision_detection/collision_common.h>
#define MOVEIT_H_COLLISION_ENV_FCL         <moveit/collision_detection_fcl/collision_env_fcl.h>
#define MOVEIT_H_KDL_KINEMATICS_PLUGIN     <moveit/kdl_kinematics_plugin/kdl_kinematics_plugin.h>
#define MOVEIT_H_KINEMATICS_BASE           <moveit/kinematics_base/kinematics_base.h>
#define MOVEIT_H_MOVE_GROUP_INTERFACE      <moveit/move_group_interface/move_group_interface.h>
#define MOVEIT_H_PLANNING_INTERFACE        <moveit/planning_interface/planning_interface.h>
#define MOVEIT_H_PLANNING_REQUEST_ADAPTER \
  "moveit/planning_request_adapter/planning_request_adapter.h"
#define MOVEIT_H_PLANNING_SCENE            <moveit/planning_scene/planning_scene.h>
#define MOVEIT_H_PLANNING_SCENE_INTERFACE \
  "moveit/planning_scene_interface/planning_scene_interface.h"
#define MOVEIT_H_PLANNING_SCENE_MONITOR    <moveit/planning_scene_monitor/planning_scene_monitor.h>
#define MOVEIT_H_JOINT_MODEL_GROUP         <moveit/robot_model/joint_model_group.h>
#define MOVEIT_H_LINK_MODEL                <moveit/robot_model/link_model.h>
#define MOVEIT_H_ROBOT_MODEL_LOADER        <moveit/robot_model_loader/robot_model_loader.h>
#define MOVEIT_H_CONVERSIONS               <moveit/robot_state/conversions.h>
#define MOVEIT_H_ROBOT_STATE               <moveit/robot_state/robot_state.h>
#define MOVEIT_H_ROBOT_TRAJECTORY          <moveit/robot_trajectory/robot_trajectory.h>
#endif

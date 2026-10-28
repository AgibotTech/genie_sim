// Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
// Author: Genie Sim Team
// Proprietary and confidential.

#pragma once

#include <chrono>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "realtime_tools/realtime_buffer.hpp"

namespace gsi
{

// Alias for the canonical ros_controls/realtime_tools primitive — a generic
// single-T latest-value buffer using the try_lock-on-reader / unique_ptr-swap
// pattern. Aligns with ros_control idioms.
template<typename T>
using LockFreeBuffer = realtime_tools::RealtimeBuffer<T>;

// ---------------------------------------------------------------------------
// RealtimeCommandBuffer
//
// Domain-specific accumulator for joint / 4WS commands. Unlike a plain
// realtime_tools::RealtimeBuffer<CommandSnapshot>, this class *merges*
// incoming partial commands keyed by joint name across many ROS messages —
// required because different controllers publish disjoint subsets of joints
// at different rates and the simulator must hold the union as a sticky
// setpoint.
//
// Concurrency: protected by a short-lived std::mutex. Producers (ROS
// subscription callbacks) and the consumer (the physics tick) all take the
// same lock for the duration of a map copy. Critical sections are O(N) in
// the number of joints — small and bounded — and never call into Python.
// ---------------------------------------------------------------------------
class RealtimeCommandBuffer
{
public:
  RealtimeCommandBuffer() = default;

  void on_joint_command(
    const std::vector<std::string> & names,
    const std::vector<double> & positions,
    const std::vector<double> & efforts);

  void on_cmd_4ws(
    const std::vector<std::string> & names,
    const std::vector<double> & positions,
    const std::vector<double> & velocities);

  // Snapshot the accumulated dictionaries to the consumer. The internal
  // maps are *not* cleared — joint setpoints stick until a new ROS message
  // overrides them (matching the ros_control "latest command" semantics).
  // The 4WS stamp is the wall-clock seconds (steady_clock) of the last
  // /cmd_4ws message at receive time.
  void swap_out(
    std::unordered_map<std::string, double> & cmd_positions_out,
    std::unordered_map<std::string, double> & cmd_efforts_out,
    std::unordered_map<std::string, double> & cmd_4ws_steer_pos_out,
    std::unordered_map<std::string, double> & cmd_4ws_drive_vel_out,
    double & cmd_4ws_stamp_out);

private:
  static double monotonic_now()
  {
    using clock = std::chrono::steady_clock;
    auto t = clock::now().time_since_epoch();
    return std::chrono::duration<double>(t).count();
  }

  std::mutex mu_;
  std::unordered_map<std::string, double> cmd_positions_;
  std::unordered_map<std::string, double> cmd_efforts_;
  std::unordered_map<std::string, double> cmd_4ws_steer_pos_;
  std::unordered_map<std::string, double> cmd_4ws_drive_vel_;
  double cmd_4ws_stamp_ = 0.0;
};

}  // namespace gsi

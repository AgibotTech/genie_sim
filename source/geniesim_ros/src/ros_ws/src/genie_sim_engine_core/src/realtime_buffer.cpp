// Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
// Author: Genie Sim Team
// Proprietary and confidential.

#include "realtime_buffer.hpp"

#include <vector>

namespace gsi
{

void RealtimeCommandBuffer::on_joint_command(
  const std::vector<std::string> & names,
  const std::vector<double> & positions,
  const std::vector<double> & efforts)
{
  std::lock_guard<std::mutex> lk(mu_);
  std::size_t n = names.size();
  for (std::size_t i = 0; i < n; ++i) {
    // Position: accept every value the publisher sent, including exactly 0.0.
    // Filtering zeros here would silently drop legitimate setpoints (e.g. the
    // gripper "open" position is canonically 0.0, see interaction_tools.py),
    // making such joints latch on the first non-zero command and ignore every
    // subsequent return-to-zero. Publishers are responsible for only sending
    // names of joints they actually want to drive.
    if (i < positions.size()) {
      cmd_positions_[names[i]] = positions[i];
    }
    // Effort: the "0 means position-controlled" sentinel — upstream nodes pad
    // effort with zeros for joints they want to leave under pure position
    // control, so dropping zeros here is intentional.
    if (i < efforts.size() && efforts[i] != 0.0) {
      cmd_efforts_[names[i]] = efforts[i];
    }
  }
}

void RealtimeCommandBuffer::on_cmd_4ws(
  const std::vector<std::string> & names,
  const std::vector<double> & positions,
  const std::vector<double> & velocities)
{
  std::lock_guard<std::mutex> lk(mu_);
  cmd_4ws_stamp_ = monotonic_now();
  std::size_t n = names.size();
  for (std::size_t i = 0; i < n; ++i) {
    const std::string & name = names[i];
    bool is_steer = name.size() >= 6 && name.compare(name.size() - 6, 6, "joint1") == 0;
    bool is_drive = name.size() >= 6 && name.compare(name.size() - 6, 6, "joint2") == 0;
    if (is_steer && i < positions.size()) {
      cmd_4ws_steer_pos_[name] = positions[i];
    }
    if (is_drive && i < velocities.size()) {
      cmd_4ws_drive_vel_[name] = velocities[i];
    }
  }
}

void RealtimeCommandBuffer::swap_out(
  std::unordered_map<std::string, double> & cmd_positions_out,
  std::unordered_map<std::string, double> & cmd_efforts_out,
  std::unordered_map<std::string, double> & cmd_4ws_steer_pos_out,
  std::unordered_map<std::string, double> & cmd_4ws_drive_vel_out,
  double & cmd_4ws_stamp_out)
{
  std::lock_guard<std::mutex> lk(mu_);
  cmd_positions_out = cmd_positions_;
  cmd_efforts_out = cmd_efforts_;
  cmd_4ws_steer_pos_out = cmd_4ws_steer_pos_;
  cmd_4ws_drive_vel_out = cmd_4ws_drive_vel_;
  cmd_4ws_stamp_out = cmd_4ws_stamp_;
}

}  // namespace gsi

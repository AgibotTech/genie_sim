// Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
// Author: Genie Sim Team
// Proprietary and confidential. Unauthorized copying, distribution,
// modification, reverse engineering, or use of this file, via any medium,
// is strictly prohibited without prior written permission.

#pragma once

#include <atomic>
#include <cstdint>

namespace gsi
{

class RenderScheduler
{
public:
  RenderScheduler(double target_hz, double safety_ms, double physics_dt);

  bool should_render(double now, double budget_s);
  void mark_rendered(double t);

  std::uint64_t rendered() const {return rendered_.load(std::memory_order_relaxed);}
  std::uint64_t skipped_period() const {return skipped_period_.load(std::memory_order_relaxed);}
  std::uint64_t skipped_budget() const {return skipped_budget_.load(std::memory_order_relaxed);}
  std::uint64_t skipped_back2back() const
  {
    return skipped_back2back_.load(std::memory_order_relaxed);
  }

private:
  double period_;
  double safety_s_;
  double min_gap_s_;
  double next_due_;
  double last_render_;

  std::atomic<std::uint64_t> rendered_;
  std::atomic<std::uint64_t> skipped_period_;
  std::atomic<std::uint64_t> skipped_budget_;
  std::atomic<std::uint64_t> skipped_back2back_;
};

}  // namespace gsi

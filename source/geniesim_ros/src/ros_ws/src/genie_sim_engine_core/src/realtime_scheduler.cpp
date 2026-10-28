// Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
// Author: Genie Sim Team
// Proprietary and confidential.

#include "realtime_scheduler.hpp"

#include <algorithm>

namespace gsi
{

RenderScheduler::RenderScheduler(double target_hz, double safety_ms, double physics_dt)
: period_(1.0 / std::max(target_hz, 1.0)),
  safety_s_(std::max(safety_ms, 0.0) / 1000.0),
  min_gap_s_(std::max(physics_dt - 0.0005, 0.0)),
  next_due_(0.0),
  last_render_(-1.0e9),
  rendered_(0),
  skipped_period_(0),
  skipped_budget_(0),
  skipped_back2back_(0)
{}

bool RenderScheduler::should_render(double now, double budget_s)
{
  if (next_due_ == 0.0) {
    next_due_ = now;
  }
  if (now < next_due_) {
    skipped_period_.fetch_add(1, std::memory_order_relaxed);
    return false;
  }
  if ((now - last_render_) < min_gap_s_) {
    skipped_back2back_.fetch_add(1, std::memory_order_relaxed);
    return false;
  }
  if (budget_s < -safety_s_) {
    skipped_budget_.fetch_add(1, std::memory_order_relaxed);
    return false;
  }
  return true;
}

void RenderScheduler::mark_rendered(double t)
{
  next_due_ += period_;
  if (next_due_ < t - period_) {
    next_due_ = t + period_;
  }
  last_render_ = t;
  rendered_.fetch_add(1, std::memory_order_relaxed);
}

}  // namespace gsi

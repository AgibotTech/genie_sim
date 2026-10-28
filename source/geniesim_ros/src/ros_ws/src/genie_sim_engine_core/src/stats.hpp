// Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
// Author: Genie Sim Team
// Proprietary and confidential.

#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>

namespace gsi
{

struct TimingSample
{
  double interval_ms;
  double step_ms;
  // Time spent inside ``sim.step()`` — the physics solver call itself,
  // regardless of which backend implements it (PhysX, Newton-in-Isaac,
  // mjwarp, Featherstone, ...). The user-facing label is "Solver".
  double solver_ms;
  double render_ms;
  double publish_ms;
  double spin_ms;
  bool did_render;
};

class StatsRing
{
public:
  explicit StatsRing(std::size_t capacity);

  void push(
    double step_ms, double solver_ms, double render_ms,
    double publish_ms, double spin_ms, bool did_render, double now_s);

  std::string format(
    std::int64_t step_count, double physics_hz, double rtf,
    std::uint64_t rendered, std::uint64_t skipped_period,
    std::uint64_t skipped_budget) const;

  std::size_t size() const;

private:
  std::size_t capacity_;
  mutable std::mutex mu_;
  std::deque<TimingSample> samples_;
  double last_step_time_s_;
};

}  // namespace gsi

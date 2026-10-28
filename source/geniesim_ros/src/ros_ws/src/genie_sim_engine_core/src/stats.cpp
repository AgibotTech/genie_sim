// Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
// Author: Genie Sim Team
// Proprietary and confidential.

#include "stats.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <vector>

namespace gsi
{

StatsRing::StatsRing(std::size_t capacity)
: capacity_(capacity == 0 ? 500 : capacity), last_step_time_s_(0.0)
{}

void StatsRing::push(
  double step_ms, double solver_ms, double render_ms,
  double publish_ms, double spin_ms, bool did_render, double now_s)
{
  std::lock_guard<std::mutex> lk(mu_);
  if (last_step_time_s_ > 0.0) {
    TimingSample s;
    s.interval_ms = (now_s - last_step_time_s_) * 1000.0;
    s.step_ms = step_ms;
    s.solver_ms = solver_ms;
    s.render_ms = render_ms;
    s.publish_ms = publish_ms;
    s.spin_ms = spin_ms;
    s.did_render = did_render;
    samples_.push_back(s);
    while (samples_.size() > capacity_) {
      samples_.pop_front();
    }
  }
  last_step_time_s_ = now_s;
}

std::size_t StatsRing::size() const
{
  std::lock_guard<std::mutex> lk(mu_);
  return samples_.size();
}

namespace
{

struct Stats5
{
  double mean, mn, mx, std, p95, p99;
};

Stats5 compute(const std::vector<double> & v)
{
  Stats5 r{0, 0, 0, 0, 0, 0};
  if (v.empty()) {return r;}
  r.mn = v.front();
  r.mx = v.front();
  double sum = 0.0;
  for (double x : v) {
    r.mn = std::min(r.mn, x);
    r.mx = std::max(r.mx, x);
    sum += x;
  }
  r.mean = sum / static_cast<double>(v.size());
  double sq = 0.0;
  for (double x : v) {
    double d = x - r.mean;
    sq += d * d;
  }
  r.std = std::sqrt(sq / static_cast<double>(v.size()));
  std::vector<double> sv = v;
  std::sort(sv.begin(), sv.end());
  std::size_t n = sv.size();
  std::size_t p95i = std::min(n - 1, static_cast<std::size_t>(0.95 * n));
  std::size_t p99i = std::min(n - 1, static_cast<std::size_t>(0.99 * n));
  r.p95 = sv[p95i];
  r.p99 = sv[p99i];
  return r;
}

}  // namespace

std::string StatsRing::format(
  std::int64_t step_count, double physics_hz, double rtf,
  std::uint64_t rendered, std::uint64_t skipped_period,
  std::uint64_t skipped_budget) const
{
  std::lock_guard<std::mutex> lk(mu_);
  if (samples_.empty()) {
    return {};
  }
  std::vector<double> intervals, steps, solver, render, publish, spin;
  std::vector<double> render_steps, nonrender_steps;
  intervals.reserve(samples_.size());
  steps.reserve(samples_.size());
  solver.reserve(samples_.size());
  render.reserve(samples_.size());
  publish.reserve(samples_.size());
  spin.reserve(samples_.size());
  for (const auto & s : samples_) {
    intervals.push_back(s.interval_ms);
    steps.push_back(s.step_ms);
    solver.push_back(s.solver_ms);
    render.push_back(s.render_ms);
    publish.push_back(s.publish_ms);
    spin.push_back(s.spin_ms);
    if (s.did_render) {
      render_steps.push_back(s.step_ms);
    } else {
      nonrender_steps.push_back(s.step_ms);
    }
  }
  Stats5 i = compute(intervals);
  Stats5 s = compute(steps);
  Stats5 sv = compute(solver);
  Stats5 rd = compute(render);
  Stats5 pb = compute(publish);
  Stats5 sp = compute(spin);

  double rtf_safe = rtf > 0.0 ? rtf : 1.0;
  // actual_hz: convert wall-clock tick interval to sim-time Hz so it matches
  // the user-configured physics_hz target directly. With rtf=0.5 a 33ms wall
  // interval = 1 sim-step / (33ms × 0.5) = 60Hz sim. The wall-clock rate is
  // half that — shown as "rtf=0.5" so users can derive it.
  double actual_hz = (i.mean > 0.0) ? (1000.0 / i.mean / rtf_safe) : 0.0;
  // target_ms: wall-clock period between ticks. (1/physics_hz) is sim dt;
  // divide by rtf to get wall dt. Overrun count uses this wall target.
  double target_ms = 1000.0 / (physics_hz * rtf_safe);
  std::size_t n_render = render_steps.size();
  std::size_t n_nonrender = nonrender_steps.size();
  double rs_mean = 0.0, ns_mean = 0.0;
  if (n_render) {
    double sum = 0.0;
    for (double v : render_steps) {
      sum += v;
    }
    rs_mean = sum / static_cast<double>(n_render);
  }
  if (n_nonrender) {
    double sum = 0.0;
    for (double v : nonrender_steps) {
      sum += v;
    }
    ns_mean = sum / static_cast<double>(n_nonrender);
  }
  std::size_t overruns = 0;
  for (double v : steps) {
    if (v > target_ms) {++overruns;}
  }
  std::size_t n = samples_.size();

  std::ostringstream oss;
  oss.setf(std::ios::fixed);
  oss.precision(2);
  oss << "--- Physics Stats (last " << n << " steps) ---\n"
      << "  Steps: " << step_count << "  Actual Hz: " << actual_hz
      << " (target " << static_cast<int>(physics_hz) << ", rtf=" << rtf_safe
      << ")  overruns: " << overruns << "/" << n << "\n"
      << "  Step:     mean=" << s.mean << "  min=" << s.mn << "  max=" << s.mx
      << "  std=" << s.std << "  p95=" << s.p95 << "  p99=" << s.p99 << " ms\n"
      << "  Interval: mean=" << i.mean << "  min=" << i.mn << "  max=" << i.mx
      << "  jitter=" << i.std << "  p95=" << i.p95 << "  p99=" << i.p99
      << " ms (target=" << target_ms << " ms)\n"
      << "  Solver:   mean=" << sv.mean << "  max=" << sv.mx
      << "  p99=" << sv.p99 << " ms  (sim.step — backend-agnostic)\n"
      << "  Viewport(in-loop): mean=" << rd.mean << "  max=" << rd.mx
      << " ms  (" << n_render << " of " << n << " ticks; "
      << "0 is normal for ovrtx — see [ovrtx-viz] N frames/s)\n"
      << "  Publish:  mean=" << pb.mean << "  max=" << pb.mx
      << "  p99=" << pb.p99 << " ms\n"
      << "  Spin:     mean=" << sp.mean << "  max=" << sp.mx
      << "  p99=" << sp.p99 << " ms\n"
      << "  Step by tick type: render-tick mean=" << rs_mean
      << " ms  non-render mean=" << ns_mean << " ms";
  if (rendered != 0 || skipped_period != 0 || skipped_budget != 0) {
    oss << "\n  Scheduler: rendered=" << rendered
        << "  skipped(period)=" << skipped_period
        << "  skipped(budget)=" << skipped_budget;
  }
  return oss.str();
}

}  // namespace gsi

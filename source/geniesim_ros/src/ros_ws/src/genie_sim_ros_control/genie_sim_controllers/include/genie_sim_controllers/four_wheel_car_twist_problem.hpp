#pragma once

#include <iostream>
#include "genie_sim_controllers/solvers/constraint.hpp"
#include "genie_sim_controllers/solvers/problem.hpp"
#include "genie_sim_controllers/solvers/weighted_squared_function.hpp"

namespace genie_sim_controllers
{

inline void getDirectCmd(
  const Eigen::Vector3d & u_ref, const Eigen::Vector4d & steer_angles,
  const Eigen::Vector4d & drive_speeds, const std::array<double, 4> & rxs,
  const std::array<double, 4> & rys, double max_steer_angle, double max_drive_speed,
  const std::array<bool, 4> & active_wheels,
  Eigen::Vector<double, 8> & cmd_ref)
{
  static const auto dir_thresh = std::cos(5e-2 * M_PI / 180.0);
  cmd_ref.setZero();

  const double vx = u_ref(0);
  const double vy = u_ref(1);
  const double wz = u_ref(2);
  const double wz_abs = std::abs(wz);

  double alpha = 1.0;
  for (size_t i = 0; i < 4; ++i) {
    double vxi = wz_abs < 1e-9 ? vx : vx - wz * rys[i];
    double vyi = wz_abs < 1e-9 ? vy : vy + wz * rxs[i];
    double vi = std::hypot(vxi, vyi);

    double theta_fwd = std::atan2(vyi, vxi);
    double theta_bck = std::atan2(-vyi, -vxi);
    double theta_prev = steer_angles[i];
    double d_fwd = std::abs(theta_fwd - theta_prev);
    double d_bck = std::abs(theta_bck - theta_prev);

    double theta, v;
    if (theta_bck > max_steer_angle || theta_bck < -max_steer_angle) {
      theta = theta_fwd;
      v = vi;
    } else if (theta_fwd > max_steer_angle || theta_fwd < -max_steer_angle) {
      theta = theta_bck;
      v = -vi;
    } else if (d_fwd <= d_bck) {
      theta = theta_fwd;
      v = vi;
    } else {
      theta = theta_bck;
      v = -vi;
    }

    v = std::clamp(v, -max_drive_speed, max_drive_speed);

    cmd_ref(i) = theta;
    cmd_ref(4 + i) = v;

    // Passive-caster corners never contribute to the alpha gate: their
    // steer feedback (empty joint name -> no /joint_states match) is
    // pinned at 0, so cos(target - 0) collapses alpha to 0 for pure yaw
    // and zeroes every drive command. Physically they don't apply
    // tangential force anyway, so their "alignment" is irrelevant.
    if (!active_wheels[i]) {
      continue;
    }

    if (wz_abs < 1e-2 || std::abs(drive_speeds(i)) > 1e-2) {
      continue;
    }

    Eigen::Vector2d d{std::cos(theta_prev), std::sin(theta_prev)};
    Eigen::Vector2d t{std::cos(theta), std::sin(theta)};

    double alpha_i = std::clamp(d.dot(t), 0.0, 1.0);
    alpha_i = alpha_i > dir_thresh ? alpha_i : 0.0;
    alpha = std::min(alpha, alpha_i);
  }

  for (size_t i = 0; i < 4; ++i) {
    cmd_ref(4 + i) *= alpha;
  }
}

class FourWheelCarCmdCost : public solvers::FunctionBase
{
public:
  FourWheelCarCmdCost()
  : FunctionBase(8, 8, true, false) {}
  ~FourWheelCarCmdCost() override = default;

  void setParameters(
    double wheel_distance, double axis_distance, double max_steer_angle,
    double max_drive_speed,
    const std::array<bool, 4> & active_wheels = {{true, true, true, true}})
  {
    wheel_distance_ = wheel_distance;
    axis_distance_ = axis_distance;
    max_steer_angle_ = max_steer_angle;
    max_drive_speed_ = max_drive_speed;
    active_wheels_ = active_wheels;
    const double rx = 0.5 * axis_distance_;
    const double ry = 0.5 * wheel_distance_;
    rxs_ = std::array<double, 4>({rx, rx, -rx, -rx});
    rys_ = std::array<double, 4>({ry, -ry, ry, -ry});
  }

  void updateState(
    const Eigen::Vector3d & u_ref, const Eigen::Vector4d & steer_angles,
    const Eigen::Vector4d & drive_speeds)
  {
    u_ref_ = u_ref;
    steer_angles_ = steer_angles;
    drive_speeds_ = drive_speeds;
    getDirectCmd(
      u_ref_, steer_angles_, drive_speeds_, rxs_, rys_, max_steer_angle_,
      max_drive_speed_, active_wheels_, cmd_ref_);
  }

  [[nodiscard]] std::shared_ptr<FunctionBase> deep_copy() const override
  {
    return std::make_shared<FourWheelCarCmdCost>(*this);
  }

protected:
  void doCompute(
    FunctionBase::InputType input, FunctionBase::OutputTypeOpt output_opt,
    FunctionBase::JacobianTypeOpt jacobian_opt, FunctionBase::HessianArrayTypeOpt) override
  {
    auto & output = output_opt.value();
    output.setZero();
    output = input - cmd_ref_;

    if (!jacobian_opt.has_value()) {return;}
    auto & jacobian = jacobian_opt.value();
    jacobian.setZero();
    for (size_t i = 0; i < 8; ++i) {
      jacobian(i, i) = 1.0;
    }
  }

private:
  Eigen::Vector3d u_ref_{Eigen::Vector3d::Zero()};
  Eigen::Vector<double, 8> cmd_ref_{Eigen::Vector<double, 8>::Zero(8)};
  Eigen::Vector4d steer_angles_{Eigen::Vector4d::Zero()};
  Eigen::Vector4d drive_speeds_{Eigen::Vector4d::Zero()};
  double wheel_distance_{0.0};
  double axis_distance_{0.0};
  std::array<double, 4> rxs_{0.0};
  std::array<double, 4> rys_{0.0};
  double max_steer_angle_{0.0};
  double max_drive_speed_{0.0};
  std::array<bool, 4> active_wheels_{{true, true, true, true}};
};

class FourWheelCarKinematicConstraint : public solvers::EqualityConstraint
{
public:
  FourWheelCarKinematicConstraint()
  : EqualityConstraint(11, 8, Eigen::VectorXd::Zero(8)) {}
  ~FourWheelCarKinematicConstraint() override = default;

  void setParameters(
    double wheel_distance, double axis_distance,
    const std::array<bool, 4> & active_wheels = {{true, true, true, true}})
  {
    wheel_distance_ = wheel_distance;
    axis_distance_ = axis_distance;
    active_wheels_ = active_wheels;
    const double rx = 0.5 * axis_distance_;
    const double ry = 0.5 * wheel_distance_;
    rxs_ = std::array<double, 4>({rx, rx, -rx, -rx});
    rys_ = std::array<double, 4>({ry, -ry, ry, -ry});
  }

  void updateState(const Eigen::Vector4d & steer_angles, const Eigen::Vector4d & drive_speeds)
  {
    steer_angles_ = steer_angles;
    drive_speeds_ = drive_speeds;
  }

  [[nodiscard]] std::shared_ptr<FunctionBase> deep_copy() const override
  {
    return std::make_shared<FourWheelCarKinematicConstraint>(*this);
  }

protected:
  void doCompute(
    FunctionBase::InputType input, FunctionBase::OutputTypeOpt output_opt,
    FunctionBase::JacobianTypeOpt jacobian_opt) override
  {
    Eigen::Vector4d angles = input.segment<4>(0);
    Eigen::Vector4d speeds = input.segment<4>(4);
    Eigen::Vector4d sin_angles = angles.array().sin();
    Eigen::Vector4d cos_angles = angles.array().cos();
    double vx = input(8);
    double vy = input(9);
    double wz = input(10);

    auto & output = output_opt.value();
    output.setZero();
    for (size_t i = 0; i < 4; ++i) {
      // Passive corners are NOT no-slip contacts — a hard
      // no-slip equality here forces the QP to satisfy
      // v*cos(θ) = vx − wz*y AND v*sin(θ) = vy + wz*x at a corner
      // whose θ is delta-capped near 0. That pins the reachable twist
      // to a tiny box near 0 (and every drive to near-0), which is
      // exactly the "linear.x works, crab/yaw collapse" symptom on
      // a diagonal 2WD chassis. Leave those rows as 0 — the equality
      // "0 = 0" is trivially satisfied, and (v, θ) at this corner
      // remain free.
      if (!active_wheels_[i]) {continue;}
      const size_t row = 2 * i;
      const double x = rxs_[i];
      const double y = rys_[i];
      const double c = cos_angles(i);
      const double s = sin_angles(i);
      const double v = speeds(i);
      output(row) = v * c - (vx - wz * y);
      output(row + 1) = v * s - (vy + wz * x);
    }

    if (!jacobian_opt.has_value()) {return;}
    auto & jacobian = jacobian_opt.value();
    jacobian.setZero();
    for (size_t i = 0; i < 4; ++i) {
      if (!active_wheels_[i]) {continue;}
      const size_t row = 2 * i;
      const double x = rxs_[i];
      const double y = rys_[i];
      const double c = cos_angles(i);
      const double s = sin_angles(i);
      const double v = speeds(i);
      jacobian(row, i) = -v * s;
      jacobian(row + 1, i) = v * c;
      jacobian(row, 4 + i) = c;
      jacobian(row + 1, 4 + i) = s;
      jacobian(row, 8) = -1.0;
      jacobian(row, 10) = y;
      jacobian(row + 1, 9) = -1.0;
      jacobian(row + 1, 10) = -x;
    }
  }

private:
  Eigen::Vector4d steer_angles_{Eigen::Vector4d::Zero()};
  Eigen::Vector4d drive_speeds_{Eigen::Vector4d::Zero()};
  double wheel_distance_{0.0};
  double axis_distance_{0.0};
  std::array<double, 4> rxs_{0.0};
  std::array<double, 4> rys_{0.0};
  std::array<bool, 4> active_wheels_{{true, true, true, true}};
};

class FourWheelAngleConstraint : public solvers::InequalityConstraint
{
public:
  FourWheelAngleConstraint()
  : InequalityConstraint(4, 16, Eigen::VectorXd::Zero(16)) {}
  ~FourWheelAngleConstraint() override = default;

  void setParameters(double max_steer_angle, double max_delta_angle)
  {
    max_steer_angle_ = max_steer_angle;
    max_delta_angle_ = max_delta_angle;
  }

  void updateState(const Eigen::Vector4d & steer_angles) {steer_angles_ = steer_angles;}

  [[nodiscard]] std::shared_ptr<FunctionBase> deep_copy() const override
  {
    return std::make_shared<FourWheelAngleConstraint>(*this);
  }

protected:
  void doCompute(
    FunctionBase::InputType input, FunctionBase::OutputTypeOpt output_opt,
    FunctionBase::JacobianTypeOpt jacobian_opt) override
  {
    auto & output = output_opt.value();
    output.setZero();
    for (size_t i = 0; i < 4; ++i) {
      output(4 * i) = input(i) - max_steer_angle_;
      output(4 * i + 1) = -input(i) - max_steer_angle_;
      output(4 * i + 2) = input(i) - steer_angles_(i) - max_delta_angle_;
      output(4 * i + 3) = -input(i) + steer_angles_(i) - max_delta_angle_;
    }
    if (!jacobian_opt.has_value()) {return;}
    auto & jacobian = jacobian_opt.value();
    jacobian.setZero();
    for (size_t i = 0; i < 4; ++i) {
      jacobian(4 * i, i) = 1.0;
      jacobian(4 * i + 1, i) = -1.0;
      jacobian(4 * i + 2, i) = 1.0;
      jacobian(4 * i + 3, i) = -1.0;
    }
  }

private:
  Eigen::Vector4d steer_angles_{Eigen::Vector4d::Zero()};
  double max_steer_angle_{0.0};
  double max_delta_angle_{0.0};
};

class FourWheelVelConstraint : public solvers::InequalityConstraint
{
public:
  FourWheelVelConstraint()
  : InequalityConstraint(4, 16, Eigen::VectorXd::Zero(16)) {}
  ~FourWheelVelConstraint() override = default;

  void setParameters(double max_drive_speed, double max_delta_speed)
  {
    max_drive_speed_ = max_drive_speed;
    max_delta_speed_ = max_delta_speed;
  }

  void updateState(const Eigen::Vector4d & drive_speeds) {drive_speeds_ = drive_speeds;}

  [[nodiscard]] std::shared_ptr<FunctionBase> deep_copy() const override
  {
    return std::make_shared<FourWheelVelConstraint>(*this);
  }

protected:
  void doCompute(
    FunctionBase::InputType input, FunctionBase::OutputTypeOpt output_opt,
    FunctionBase::JacobianTypeOpt jacobian_opt) override
  {
    auto & output = output_opt.value();
    output.setZero();
    for (size_t i = 0; i < 4; ++i) {
      output(4 * i) = input(i) - max_drive_speed_;
      output(4 * i + 1) = -input(i) - max_drive_speed_;
      output(4 * i + 2) = input(i) - drive_speeds_(i) - max_delta_speed_;
      output(4 * i + 3) = -input(i) + drive_speeds_(i) - max_delta_speed_;
    }
    if (!jacobian_opt.has_value()) {return;}
    auto & jacobian = jacobian_opt.value();
    jacobian.setZero();
    for (size_t i = 0; i < 4; ++i) {
      jacobian(4 * i, i) = 1.0;
      jacobian(4 * i + 1, i) = -1.0;
      jacobian(4 * i + 2, i) = 1.0;
      jacobian(4 * i + 3, i) = -1.0;
    }
  }

private:
  Eigen::Vector4d drive_speeds_{Eigen::Vector4d::Zero()};
  double max_drive_speed_{0.0};
  double max_delta_speed_{0.0};
};

class TwistTrackingCost : public solvers::FunctionBase
{
public:
  TwistTrackingCost()
  : FunctionBase(3, 3, true, false) {}
  ~TwistTrackingCost() override = default;

  void updateState(const Eigen::Vector3d & u_ref) {u_ref_ = u_ref;}

  [[nodiscard]] std::shared_ptr<FunctionBase> deep_copy() const override
  {
    return std::make_shared<TwistTrackingCost>(*this);
  }

protected:
  void doCompute(
    FunctionBase::InputType input, FunctionBase::OutputTypeOpt output_opt,
    FunctionBase::JacobianTypeOpt jacobian_opt, FunctionBase::HessianArrayTypeOpt) override
  {
    auto & output = output_opt.value();
    output = input - u_ref_;

    if (!jacobian_opt.has_value()) {return;}
    auto & jacobian = jacobian_opt.value();
    jacobian.setIdentity();
  }

private:
  Eigen::Vector3d u_ref_{Eigen::Vector3d::Zero()};
};

class FourWheelCmdProblem final : public solvers::Problem
{
public:
  FourWheelCmdProblem()
  : Problem(11) {}
  ~FourWheelCmdProblem() override = default;

  void setParameters(
    double wheel_distance, double axis_distance, double max_drive_speed, double max_delta_speed,
    double max_steer_angle, double max_delta_angle,
    const std::array<bool, 4> & active_wheels = {{true, true, true, true}})
  {
    wheel_distance_ = wheel_distance;
    axis_distance_ = axis_distance;
    max_drive_speed_ = max_drive_speed;
    max_delta_speed_ = max_delta_speed;
    max_steer_angle_ = max_steer_angle;
    max_delta_angle_ = max_delta_angle;
    active_wheels_ = active_wheels;
  }

  void updateState(
    const Eigen::Vector3d & u_ref, const Eigen::Vector4d & steer_angles,
    const Eigen::Vector4d & drive_speeds)
  {
    u_ref_ = u_ref;
    steer_angles_ = steer_angles;
    drive_speeds_ = drive_speeds;
  }

protected:
  bool doMakeProblem() override
  {
    init_guess_.setZero();
    init_guess_.segment<4>(0) = steer_angles_;
    init_guess_.segment<4>(4) = drive_speeds_;
    init_guess_.segment<3>(8) = u_ref_;

    FourWheelCarCmdCost cmd_cost;
    cmd_cost.setParameters(
      wheel_distance_, axis_distance_, max_steer_angle_, max_drive_speed_, active_wheels_);
    cmd_cost.updateState(u_ref_, steer_angles_, drive_speeds_);
    solvers::WeightedSquaredFunction weighted_cmd_cost(cmd_cost, Eigen::Vector<double, 8>::Ones());
    solvers::RelativeFunctionWrapper<solvers::WeightedSquaredFunction> relative_cmd_cost(
      weighted_cmd_cost, {std::pair<size_t, size_t>(0, 8)});
    addCostFunction(relative_cmd_cost);

    TwistTrackingCost twist_cost;
    twist_cost.updateState(u_ref_);
    Eigen::Vector3d twist_weights(10.0, 10.0, 10.0);
    solvers::WeightedSquaredFunction weighted_twist_cost(twist_cost, twist_weights);
    solvers::RelativeFunctionWrapper<solvers::WeightedSquaredFunction> relative_twist_cost(
      weighted_twist_cost, {std::pair<size_t, size_t>(8, 11)});
    addCostFunction(relative_twist_cost);

    FourWheelCarKinematicConstraint kinematic_constraint;
    kinematic_constraint.setParameters(wheel_distance_, axis_distance_, active_wheels_);
    kinematic_constraint.updateState(steer_angles_, drive_speeds_);
    solvers::RelativeFunctionWrapper<FourWheelCarKinematicConstraint> relative_kinematic_constraint(
      kinematic_constraint, {std::pair<size_t, size_t>(0, 11)});
    addConstraint(relative_kinematic_constraint);

    FourWheelAngleConstraint angle_constraint;
    angle_constraint.setParameters(max_steer_angle_, max_delta_angle_);
    angle_constraint.updateState(steer_angles_);
    solvers::RelativeFunctionWrapper<FourWheelAngleConstraint> relative_angle_constraint(
      angle_constraint, {std::pair<size_t, size_t>(0, 4)});
    addConstraint(relative_angle_constraint);

    FourWheelVelConstraint vel_constraint;
    vel_constraint.setParameters(max_drive_speed_, max_delta_speed_);
    vel_constraint.updateState(drive_speeds_);
    solvers::RelativeFunctionWrapper<FourWheelVelConstraint> relative_vel_constraint(vel_constraint,
      {std::pair<size_t, size_t>(4, 8)});
    addConstraint(relative_vel_constraint);

    return true;
  }

private:
  Eigen::Vector3d u_ref_{Eigen::Vector3d::Zero()};
  Eigen::Vector4d steer_angles_{Eigen::Vector4d::Zero()};
  Eigen::Vector4d drive_speeds_{Eigen::Vector4d::Zero()};
  double wheel_distance_{0.0};
  double axis_distance_{0.0};
  double max_drive_speed_{0.0};
  double max_delta_speed_{0.0};
  double max_steer_angle_{0.0};
  double max_delta_angle_{0.0};
  std::array<bool, 4> active_wheels_{{true, true, true, true}};
};

}  // namespace genie_sim_controllers

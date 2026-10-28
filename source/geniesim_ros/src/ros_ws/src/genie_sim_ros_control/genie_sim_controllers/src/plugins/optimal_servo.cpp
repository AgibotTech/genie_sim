#include "genie_sim_controllers/plugins/optimal_servo.hpp"
#include "genie_sim_controllers/four_wheel_car_twist_problem.hpp"
#include "genie_sim_controllers/solvers/osqp_solver.hpp"
#include <cmath>

namespace genie_sim_controllers
{

void OptimalServo::initialize(const CommonParams & params)
{
  setCommonParams(params);
  recover_steer_angles_ = true;
}

void OptimalServo::reset() {inner_state_ = InnerState::INIT;}

bool OptimalServo::ready() const {return inner_state_ == InnerState::IDLE;}

std::optional<WheelCommand> OptimalServo::update(
  const std::optional<TwistCmd> & twist_opt,
  const std::optional<WheelState> & wheel_state_opt,
  const WheelCommand & wheel_cmd_prev)
{
  if (inner_state_ == InnerState::INIT) {
    inner_state_ = InnerState::IDLE;
  }

  if (!twist_opt.has_value()) {
    return std::nullopt;
  }

  const auto & twist = twist_opt.value();
  Eigen::Vector3d u_ref(twist.vx, twist.vy, twist.wz);

  Eigen::Vector4d steer_angles;
  Eigen::Vector4d drive_speeds;
  for (size_t i = 0; i < 4; ++i) {
    steer_angles(i) = wheel_cmd_prev.steer_angles[i];
    drive_speeds(i) = wheel_cmd_prev.drive_speeds[i];
  }
  drive_speeds *= common_params_.wheel_radius;

  if (recover_steer_angles_ && wheel_state_opt.has_value()) {
    for (size_t i = 0; i < 4; ++i) {
      // Skip passive-caster corners: their /joint_states slot never
      // matches (empty joint name) so wheel_state_opt->steer_angles[i]
      // stays pinned at 0. Overwriting with 0 would collapse the
      // OptimalServo alpha gate for pure yaw. Keep the previous cmd
      // value (which the QP evolved on its own, unconstrained by a
      // physical joint).
      if (!common_params_.active_wheels[i]) {continue;}
      steer_angles(i) = wheel_state_opt->steer_angles[i];
    }
  }

  auto problem = std::make_shared<FourWheelCmdProblem>();
  problem->setParameters(
    common_params_.wheel_distance, common_params_.axis_distance,
    common_params_.max_drive_speed,
    common_params_.max_drive_accel * common_params_.wheel_radius * common_params_.dt,
    common_params_.max_steer_angle, common_params_.max_steer_speed * common_params_.dt,
    common_params_.active_wheels);
  problem->updateState(u_ref, steer_angles, drive_speeds);
  problem->makeProblem();

  solvers::osqp_solver::OSQPSolver solver(problem);
  solver.setX(problem->init_guess());
  solver.solve();

  if (solver.status() != solvers::Solver::Status::Solved) {
    return std::nullopt;
  }

  WheelCommand wheel_cmd{};
  auto x = solver.x();
  for (size_t i = 0; i < 4; ++i) {
    wheel_cmd.steer_angles[i] = static_cast<float>(x(i));
    wheel_cmd.drive_speeds[i] = static_cast<float>(x(4 + i) / common_params_.wheel_radius);
  }

  return wheel_cmd;
}

}  // namespace genie_sim_controllers

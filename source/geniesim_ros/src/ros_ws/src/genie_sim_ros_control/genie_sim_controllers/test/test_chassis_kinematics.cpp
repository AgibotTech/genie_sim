// Copyright (c) 2026, AgiBot Inc. All Rights Reserved.
// License: Mozilla Public License Version 2.0
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <tuple>

#include "genie_sim_controllers/four_wheel_car_twist_problem.hpp"
#include "genie_sim_controllers/plugins/optimal_servo.hpp"

namespace genie_sim_controllers
{
namespace
{
constexpr std::array<bool, 4> diagonal{{false, true, true, false}};
constexpr std::array<double, 4> xs{{0.3075, 0.3075, -0.3075, -0.3075}};
constexpr std::array<double, 4> ys{{0.1925, -0.1925, 0.1925, -0.1925}};

void alignWheel(double vx, double vy, double & angle, double & speed)
{
  angle = std::atan2(vy, vx);
  speed = std::hypot(vx, vy);
  if (angle > M_PI_2) {
    angle -= M_PI;
    speed = -speed;
  } else if (angle < -M_PI_2) {
    angle += M_PI;
    speed = -speed;
  }
}

TEST(ChassisKinematics, PassiveCastersDoNotImposeNoSlipOnCrabOrYaw)
{
  for (const Eigen::Vector3d & twist : {Eigen::Vector3d(0.0, 0.2, 0.0),
      Eigen::Vector3d(0.0, 0.0, 0.3)})
  {
    Eigen::VectorXd input = Eigen::VectorXd::Zero(11);
    input.tail<3>() = twist;
    for (size_t i = 0; i < 4; ++i) {
      if (diagonal[i]) {
        alignWheel(twist(0) - twist(2) * ys[i], twist(1) + twist(2) * xs[i],
          input(i), input(4 + i));
      }
    }
    FourWheelCarKinematicConstraint constraint;
    constraint.setParameters(0.385, 0.615, diagonal);
    Eigen::VectorXd residual(8);
    constraint.evaluate(input, residual);
    EXPECT_LT(residual.norm(), 1e-12);

    // Treating the stationary casters as driven wheels would reject this
    // physically valid chassis motion. All four wheels remain active by default.
    constraint.setParameters(0.385, 0.615);
    constraint.evaluate(input, residual);
    EXPECT_GT(residual.norm(), 0.1);
  }
}

TEST(ChassisKinematics, UnalignedPassiveCastersDoNotStopAlignedDrives)
{
  const Eigen::Vector3d twist(0.0, 0.0, 0.3);
  Eigen::Vector4d angles = Eigen::Vector4d::Zero();
  Eigen::Vector4d expected = Eigen::Vector4d::Zero();
  for (size_t i = 0; i < 4; ++i) {
    if (diagonal[i]) {
      alignWheel(-twist(2) * ys[i], twist(2) * xs[i], angles(i), expected(i));
    }
  }
  Eigen::Vector<double, 8> cmd;
  getDirectCmd(twist, angles, Eigen::Vector4d::Zero(), xs, ys, M_PI_2, 1.0,
    diagonal, cmd);
  for (size_t i = 0; i < 4; ++i) {
    if (diagonal[i]) {
      EXPECT_NEAR(cmd(4 + i), expected(i), 1e-12);
      EXPECT_GT(std::abs(cmd(4 + i)), 0.1);
    }
  }
}

class OptimalChassis : public testing::TestWithParam<std::tuple<bool, int>> {};

TEST_P(OptimalChassis, TracksChassisTwistWithMeasuredActiveWheels)
{
  const bool two_wheel = std::get<0>(GetParam());
  const std::array<TwistCmd, 6> commands{{
    {0.2f, 0.0f, 0.0f}, {-0.2f, 0.0f, 0.0f}, {0.0f, 0.2f, 0.0f},
    {0.0f, 0.0f, 0.3f}, {0.0f, 0.0f, -0.3f}, {0.1f, 0.1f, 0.2f}}};
  const auto twist = commands[std::get<1>(GetParam())];
  CommonParams params;
  params.wheel_distance = two_wheel ? 0.385f : 0.44f;
  params.axis_distance = two_wheel ? 0.615f : 0.46f;
  params.wheel_radius = two_wheel ? 0.075f : 0.07f;
  params.max_steer_angle = M_PI_2;
  params.max_steer_speed = 12.0f;
  params.max_drive_accel = 20.0f;
  params.dt = 0.01f;
  if (two_wheel) {params.active_wheels = diagonal;}

  OptimalServo servo;
  servo.initialize(params);
  WheelCommand previous{};
  WheelState measured{};
  for (int tick = 0; tick < 250; ++tick) {
    // Ideal plant follows commanded active joints. Empty passive joint slots
    // remain zero, just as in the ROS wrappers' joint-state mapping.
    for (size_t i = 0; i < 4; ++i) {
      if (params.active_wheels[i]) {
        measured.steer_angles[i] = previous.steer_angles[i];
        measured.drive_speeds[i] = previous.drive_speeds[i];
      }
    }
    const auto result = servo.update(twist, measured, previous);
    ASSERT_TRUE(result.has_value()) << "tick " << tick;
    previous = *result;
    for (size_t i = 0; i < 4; ++i) {
      ASSERT_TRUE(std::isfinite(previous.steer_angles[i]));
      ASSERT_TRUE(std::isfinite(previous.drive_speeds[i]));
      EXPECT_LE(std::abs(previous.steer_angles[i]), M_PI_2 + 0.01);
      EXPECT_LE(std::abs(previous.drive_speeds[i] * params.wheel_radius), 1.01);
    }
  }

  Eigen::MatrixXd contact(2 * (two_wheel ? 2 : 4), 3);
  Eigen::VectorXd velocity(contact.rows());
  int row = 0;
  for (size_t i = 0; i < 4; ++i) {
    if (!params.active_wheels[i]) {continue;}
    const double x = (i < 2 ? 0.5 : -0.5) * params.axis_distance;
    const double y = (i % 2 == 0 ? 0.5 : -0.5) * params.wheel_distance;
    contact.row(row) << 1.0, 0.0, -y;
    contact.row(row + 1) << 0.0, 1.0, x;
    const double speed = previous.drive_speeds[i] * params.wheel_radius;
    velocity(row) = speed * std::cos(previous.steer_angles[i]);
    velocity(row + 1) = speed * std::sin(previous.steer_angles[i]);
    row += 2;
  }
  const Eigen::Vector3d actual = contact.colPivHouseholderQr().solve(velocity);
  EXPECT_NEAR(actual(0), twist.vx, 0.02);
  EXPECT_NEAR(actual(1), twist.vy, 0.02);
  EXPECT_NEAR(actual(2), twist.wz, 0.02);
}

INSTANTIATE_TEST_SUITE_P(TwoAndFourWheel, OptimalChassis,
  testing::Combine(testing::Bool(), testing::Range(0, 6)));
}  // namespace
}  // namespace genie_sim_controllers

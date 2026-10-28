#include "genie_sim_controllers/chassis_servo_controller.hpp"

#include <utility>

#include <pluginlib/class_list_macros.hpp>

namespace genie_sim_controllers
{

controller_interface::InterfaceConfiguration
ChassisServoController::command_interface_configuration() const
{
  return {controller_interface::interface_configuration_type::NONE, {}};
}

controller_interface::InterfaceConfiguration
ChassisServoController::state_interface_configuration() const
{
  return {controller_interface::interface_configuration_type::NONE, {}};
}

controller_interface::CallbackReturn ChassisServoController::on_init()
{
  try {
    auto_declare<double>("control_rate", 200.0);
    auto_declare<double>("wheel_distance", 1.0);
    auto_declare<double>("axis_distance", 1.0);
    auto_declare<double>("wheel_radius", 0.1);
    auto_declare<double>("max_steer_angle", 0.523599);
    auto_declare<double>("max_steer_speed", 0.5);
    auto_declare<double>("max_drive_speed", 1.0);
    auto_declare<double>("max_drive_accel", 20.0);
    auto_declare<double>("init_wait_time_ms", 1000.0);
    auto_declare<double>("set_cmd_timeout_ms", 5000.0);
    auto_declare<double>("twist_timeout_ms", 1000.0);
    auto_declare<double>("state_timeout_ms", 1000.0);
    auto_declare<bool>("check_state_cmd_diff", false);
    auto_declare<double>("max_steer_state_cmd_diff", 0.1);
    auto_declare<bool>("use_steer_ltd", false);
    auto_declare<double>("steer_ltd_r", 30.0);
    auto_declare<bool>("use_drive_ltd", false);
    auto_declare<double>("drive_ltd_r", 30.0);
    auto_declare<std::string>("default_servo", std::string("ParkingServo"));
    auto_declare<std::vector<std::string>>(
      "steer_joint_names",
      std::vector<std::string>{
        "idx111_chassis_lwheel_front_joint1",
        "idx131_chassis_rwheel_front_joint1",
        "idx121_chassis_lwheel_rear_joint1",
        "idx141_chassis_rwheel_rear_joint1"});
    auto_declare<std::vector<std::string>>(
      "drive_joint_names",
      std::vector<std::string>{
        "idx112_chassis_lwheel_front_joint2",
        "idx132_chassis_rwheel_front_joint2",
        "idx122_chassis_lwheel_rear_joint2",
        "idx142_chassis_rwheel_rear_joint2"});
  } catch (const std::exception & e) {
    RCLCPP_ERROR(
      rclcpp::get_logger("ChassisServoController"),
      "Exception in on_init(): %s", e.what());
    return controller_interface::CallbackReturn::ERROR;
  }
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn ChassisServoController::on_configure(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  auto node = get_node();

  ServoParams params;
  const double control_rate = node->get_parameter("control_rate").as_double();
  params.common.wheel_distance =
    static_cast<float>(node->get_parameter("wheel_distance").as_double());
  params.common.axis_distance =
    static_cast<float>(node->get_parameter("axis_distance").as_double());
  params.common.wheel_radius =
    static_cast<float>(node->get_parameter("wheel_radius").as_double());
  params.common.max_steer_angle =
    static_cast<float>(node->get_parameter("max_steer_angle").as_double());
  params.common.max_steer_speed =
    static_cast<float>(node->get_parameter("max_steer_speed").as_double());
  params.common.max_drive_speed =
    static_cast<float>(node->get_parameter("max_drive_speed").as_double());
  params.common.max_drive_accel =
    static_cast<float>(node->get_parameter("max_drive_accel").as_double());
  params.common.dt = static_cast<float>(1.0 / control_rate);

  params.init_wait_time_ms = node->get_parameter("init_wait_time_ms").as_double();
  params.set_cmd_timeout_ms = node->get_parameter("set_cmd_timeout_ms").as_double();
  params.twist_timeout_ms = node->get_parameter("twist_timeout_ms").as_double();
  params.state_timeout_ms = node->get_parameter("state_timeout_ms").as_double();
  params.check_state_cmd_diff = node->get_parameter("check_state_cmd_diff").as_bool();
  params.max_steer_state_cmd_diff = node->get_parameter("max_steer_state_cmd_diff").as_double();
  params.use_steer_ltd = node->get_parameter("use_steer_ltd").as_bool();
  params.steer_ltd_r = node->get_parameter("steer_ltd_r").as_double();
  params.use_drive_ltd = node->get_parameter("use_drive_ltd").as_bool();
  params.drive_ltd_r = node->get_parameter("drive_ltd_r").as_double();
  params.default_servo = node->get_parameter("default_servo").as_string();

  steer_joint_names_ = node->get_parameter("steer_joint_names").as_string_array();
  drive_joint_names_ = node->get_parameter("drive_joint_names").as_string_array();
  if (steer_joint_names_.size() != 4 || drive_joint_names_.size() != 4) {
    RCLCPP_ERROR(
      node->get_logger(),
      "steer_joint_names and drive_joint_names must each have exactly 4 entries");
    return controller_interface::CallbackReturn::ERROR;
  }
  // Corners whose steer OR drive joint name is empty are passive
  // (e.g. a diagonal 2WD chassis where the two off-diagonal wheels
  // are frictionless caster balls). Mark them inactive so OptimalServo
  // drops the no-slip kinematic equality and the per-wheel alpha gate
  // at those slots.
  for (size_t i = 0; i < 4; ++i) {
    params.common.active_wheels[i] =
      !steer_joint_names_[i].empty() && !drive_joint_names_[i].empty();
  }

  core_.configure(params);

  cmd_twist_sub_ = node->create_subscription<geometry_msgs::msg::Twist>(
    "cmd_twist", rclcpp::QoS(1),
    [this, node](geometry_msgs::msg::Twist::ConstSharedPtr msg) {
      core_.setTwist(
        TwistCmd{
        static_cast<float>(msg->linear.x),
        static_cast<float>(msg->linear.y),
        static_cast<float>(msg->angular.z)},
        node->now().seconds());
    });

  joint_states_sub_ = node->create_subscription<sensor_msgs::msg::JointState>(
    "joint_states", rclcpp::SensorDataQoS(),
    [this, node](sensor_msgs::msg::JointState::ConstSharedPtr msg) {
      WheelState ws{};
      for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < msg->name.size(); ++j) {
          if (msg->name[j] == steer_joint_names_[i]) {
            if (j < msg->position.size()) {
              ws.steer_angles[i] = static_cast<float>(msg->position[j]);
            }
            if (j < msg->velocity.size()) {
              ws.steer_speeds[i] = static_cast<float>(msg->velocity[j]);
            }
          }
          if (msg->name[j] == drive_joint_names_[i]) {
            if (j < msg->velocity.size()) {
              ws.drive_speeds[i] = static_cast<float>(msg->velocity[j]);
            }
          }
        }
      }
      core_.setWheelState(ws, node->now().seconds());
    });

  set_mode_sub_ = node->create_subscription<std_msgs::msg::String>(
    "set_servo_mode", rclcpp::QoS(1),
    [this, node](std_msgs::msg::String::ConstSharedPtr msg) {
      if (!core_.requestMode(msg->data)) {
        RCLCPP_WARN(
          node->get_logger(),
          "Unknown servo mode: '%s', ignoring", msg->data.c_str());
      }
    });

  cmd_4ws_pub_ = node->create_publisher<sensor_msgs::msg::JointState>(
    "cmd_4ws", rclcpp::QoS(1));

  RCLCPP_INFO(
    node->get_logger(),
    "ChassisServoController configured (default servo: %s, control rate %.1f Hz)",
    params.default_servo.c_str(), control_rate);

  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn ChassisServoController::on_activate(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  core_.reset();
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn ChassisServoController::on_deactivate(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::return_type ChassisServoController::update(
  const rclcpp::Time & now, const rclcpp::Duration & /*period*/)
{
  auto cmd_opt = core_.tick(now.seconds());
  drainEventsToLogger();

  if (!cmd_opt.has_value()) {return controller_interface::return_type::OK;}

  const auto & cmd = cmd_opt.value();
  sensor_msgs::msg::JointState out_msg;
  out_msg.header.stamp = now;
  out_msg.name.resize(8);
  out_msg.position.resize(8, 0.0);
  out_msg.velocity.resize(8, 0.0);
  out_msg.effort.resize(8, 0.0);
  for (size_t i = 0; i < 4; ++i) {
    out_msg.name[i * 2] = steer_joint_names_[i];
    out_msg.name[i * 2 + 1] = drive_joint_names_[i];
    out_msg.position[i * 2] = cmd.steer_angles[i];
    out_msg.velocity[i * 2 + 1] = cmd.drive_speeds[i];
  }
  cmd_4ws_pub_->publish(out_msg);

  return controller_interface::return_type::OK;
}

void ChassisServoController::drainEventsToLogger()
{
  auto events = core_.drainEvents();
  auto logger = get_node()->get_logger();
  for (const auto & ev : events) {
    if (ev.level == ServoCore::LogLevel::WARN) {
      RCLCPP_WARN(logger, "%s", ev.message.c_str());
    } else {
      RCLCPP_INFO(logger, "%s", ev.message.c_str());
    }
  }
}

}  // namespace genie_sim_controllers

PLUGINLIB_EXPORT_CLASS(
  genie_sim_controllers::ChassisServoController,
  controller_interface::ControllerInterface)

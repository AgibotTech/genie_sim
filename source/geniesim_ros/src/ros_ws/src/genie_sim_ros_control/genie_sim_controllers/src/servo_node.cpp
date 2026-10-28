#include "genie_sim_controllers/servo_node.hpp"

#include <chrono>
#include <utility>

#include <rclcpp_components/register_node_macro.hpp>

namespace genie_sim_controllers
{

namespace
{
template<typename NodeT, typename T>
T declareOrGet(NodeT * node, const std::string & name, const T & default_value)
{
  if (node->has_parameter(name)) {
    return node->get_parameter(name).template get_value<T>();
  }
  return node->template declare_parameter<T>(name, default_value);
}
}  // namespace

ServoNode::ServoNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("servo_node", options)
{
  ServoParams params;
  const double control_rate = declareOrGet<rclcpp::Node, double>(this, "control_rate", 200.0);
  params.common.wheel_distance =
    static_cast<float>(declareOrGet<rclcpp::Node, double>(this, "wheel_distance", 1.0));
  params.common.axis_distance =
    static_cast<float>(declareOrGet<rclcpp::Node, double>(this, "axis_distance", 1.0));
  params.common.wheel_radius =
    static_cast<float>(declareOrGet<rclcpp::Node, double>(this, "wheel_radius", 0.1));
  params.common.max_steer_angle =
    static_cast<float>(declareOrGet<rclcpp::Node, double>(this, "max_steer_angle", 0.523599));
  params.common.max_steer_speed =
    static_cast<float>(declareOrGet<rclcpp::Node, double>(this, "max_steer_speed", 0.5));
  params.common.max_drive_speed =
    static_cast<float>(declareOrGet<rclcpp::Node, double>(this, "max_drive_speed", 1.0));
  params.common.max_drive_accel =
    static_cast<float>(declareOrGet<rclcpp::Node, double>(this, "max_drive_accel", 20.0));
  params.common.dt = static_cast<float>(1.0 / control_rate);

  params.init_wait_time_ms = declareOrGet<rclcpp::Node, double>(this, "init_wait_time_ms", 1000.0);
  params.set_cmd_timeout_ms =
    declareOrGet<rclcpp::Node, double>(this, "set_cmd_timeout_ms", 5000.0);
  params.twist_timeout_ms = declareOrGet<rclcpp::Node, double>(this, "twist_timeout_ms", 1000.0);
  params.state_timeout_ms = declareOrGet<rclcpp::Node, double>(this, "state_timeout_ms", 1000.0);
  params.check_state_cmd_diff =
    declareOrGet<rclcpp::Node, bool>(this, "check_state_cmd_diff", false);
  params.max_steer_state_cmd_diff =
    declareOrGet<rclcpp::Node, double>(this, "max_steer_state_cmd_diff", 0.1);
  params.use_steer_ltd = declareOrGet<rclcpp::Node, bool>(this, "use_steer_ltd", false);
  params.steer_ltd_r = declareOrGet<rclcpp::Node, double>(this, "steer_ltd_r", 30.0);
  params.use_drive_ltd = declareOrGet<rclcpp::Node, bool>(this, "use_drive_ltd", false);
  params.drive_ltd_r = declareOrGet<rclcpp::Node, double>(this, "drive_ltd_r", 30.0);
  params.default_servo =
    declareOrGet<rclcpp::Node, std::string>(this, "default_servo", std::string("ParkingServo"));

  steer_joint_names_ = declareOrGet<rclcpp::Node, std::vector<std::string>>(
    this, "steer_joint_names",
    std::vector<std::string>{
      "idx111_chassis_lwheel_front_joint1",
      "idx131_chassis_rwheel_front_joint1",
      "idx121_chassis_lwheel_rear_joint1",
      "idx141_chassis_rwheel_rear_joint1"});
  drive_joint_names_ = declareOrGet<rclcpp::Node, std::vector<std::string>>(
    this, "drive_joint_names",
    std::vector<std::string>{
      "idx112_chassis_lwheel_front_joint2",
      "idx132_chassis_rwheel_front_joint2",
      "idx122_chassis_lwheel_rear_joint2",
      "idx142_chassis_rwheel_rear_joint2"});

  if (steer_joint_names_.size() != 4 || drive_joint_names_.size() != 4) {
    RCLCPP_FATAL(
      get_logger(),
      "steer_joint_names and drive_joint_names must each have exactly 4 entries");
    throw std::runtime_error("servo_node: bad joint name list size");
  }
  // Kept in lockstep with the ros2_control plugin path
  // (chassis_servo_controller.cpp). Empty joint name slots mark passive
  // corners; OptimalServo skips them in the QP kinematic constraint
  // and the alpha alignment gate.
  for (size_t i = 0; i < 4; ++i) {
    params.common.active_wheels[i] =
      !steer_joint_names_[i].empty() && !drive_joint_names_[i].empty();
  }

  core_.configure(params);

  cmd_twist_sub_ = create_subscription<geometry_msgs::msg::Twist>(
    "cmd_twist", rclcpp::QoS(1),
    [this](geometry_msgs::msg::Twist::ConstSharedPtr msg) {
      core_.setTwist(
        TwistCmd{
        static_cast<float>(msg->linear.x),
        static_cast<float>(msg->linear.y),
        static_cast<float>(msg->angular.z)},
        now().seconds());
    });

  joint_states_sub_ = create_subscription<sensor_msgs::msg::JointState>(
    "joint_states", rclcpp::SensorDataQoS(),
    [this](sensor_msgs::msg::JointState::ConstSharedPtr msg) {
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
      core_.setWheelState(ws, now().seconds());
    });

  set_mode_sub_ = create_subscription<std_msgs::msg::String>(
    "set_servo_mode", rclcpp::QoS(1),
    [this](std_msgs::msg::String::ConstSharedPtr msg) {
      if (!core_.requestMode(msg->data)) {
        RCLCPP_WARN(get_logger(), "Unknown servo mode: '%s', ignoring", msg->data.c_str());
      }
    });

  cmd_4ws_pub_ = create_publisher<sensor_msgs::msg::JointState>("cmd_4ws", rclcpp::QoS(1));

  const auto period = std::chrono::duration<double>(1.0 / control_rate);
  control_timer_ = create_wall_timer(
    std::chrono::duration_cast<std::chrono::nanoseconds>(period),
    std::bind(&ServoNode::onTimer, this));

  RCLCPP_INFO(
    get_logger(),
    "ServoNode initialised (default servo: %s, control rate %.1f Hz)",
    params.default_servo.c_str(), control_rate);
}

void ServoNode::onTimer()
{
  auto cmd_opt = core_.tick(now().seconds());
  drainEventsToLogger();

  if (!cmd_opt.has_value()) {return;}

  const auto & cmd = cmd_opt.value();
  sensor_msgs::msg::JointState out_msg;
  out_msg.header.stamp = now();
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
}

void ServoNode::drainEventsToLogger()
{
  auto events = core_.drainEvents();
  for (const auto & ev : events) {
    if (ev.level == ServoCore::LogLevel::WARN) {
      RCLCPP_WARN(get_logger(), "%s", ev.message.c_str());
    } else {
      RCLCPP_INFO(get_logger(), "%s", ev.message.c_str());
    }
  }
}

}  // namespace genie_sim_controllers

RCLCPP_COMPONENTS_REGISTER_NODE(genie_sim_controllers::ServoNode)

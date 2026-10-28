// Copyright 2026 GenieSim contributors. All rights reserved.
//
// View Camera Pose Publisher RViz Display.
//
// Reads the active rviz_common::ViewController's Ogre camera every render
// frame and publishes its world-space pose as geometry_msgs/PoseStamped.
// The header.frame_id is the RViz Fixed Frame, which matches the convention
// used by genie_sim_render's ``~/free_cam_pose`` subscriber.

#ifndef GENIE_SIM_RVIZ_PLUGINS__VIEW_CAMERA_POSE_PUBLISHER_DISPLAY_HPP_
#define GENIE_SIM_RVIZ_PLUGINS__VIEW_CAMERA_POSE_PUBLISHER_DISPLAY_HPP_

#include <memory>
#include <string>

#include <QObject>

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <std_msgs/msg/float64.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include <rviz_common/display.hpp>

namespace rviz_common
{
namespace properties
{
class Property;
class StringProperty;
class FloatProperty;
class BoolProperty;
}  // namespace properties
}  // namespace rviz_common

namespace genie_sim_rviz_plugins
{

class ViewCameraPosePublisherDisplay : public rviz_common::Display
{
  Q_OBJECT

public:
  ViewCameraPosePublisherDisplay();
  ~ViewCameraPosePublisherDisplay() override;

  void onInitialize() override;
  void update(float wall_dt, float ros_dt) override;
  void reset() override;

protected:
  void onEnable() override;
  void onDisable() override;

private Q_SLOTS:
  void updateTopic();
  void updateZoomTopic();

private:
  friend class ViewCameraPosePublisherDisplayTest;

  void ensurePublisher();
  void publishZoom();
  void updateZoomFromOrbit(
    rviz_common::properties::Property * view_controller, const QString & class_id);

  rviz_common::properties::StringProperty * topic_property_{nullptr};
  rviz_common::properties::StringProperty * zoom_topic_property_{nullptr};
  rviz_common::properties::FloatProperty * zoom_factor_property_{nullptr};
  rviz_common::properties::BoolProperty * auto_orbit_zoom_property_{nullptr};
  rviz_common::properties::FloatProperty * rate_property_{nullptr};
  rviz_common::properties::BoolProperty * only_on_change_property_{nullptr};
  rviz_common::properties::BoolProperty * publish_tf_property_{nullptr};
  rviz_common::properties::StringProperty * tf_child_frame_property_{nullptr};

  rclcpp::Node::SharedPtr node_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr zoom_pub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  std::string current_topic_;
  std::string current_zoom_topic_;

  float accumulator_{0.0f};
  geometry_msgs::msg::PoseStamped last_published_;
  bool has_last_{false};
  double last_zoom_{1.0};
  bool has_last_zoom_{false};
  rviz_common::properties::Property * last_view_controller_{nullptr};
  double last_orbit_distance_{0.0};
  bool has_last_orbit_distance_{false};
};

}  // namespace genie_sim_rviz_plugins

#endif  // GENIE_SIM_RVIZ_PLUGINS__VIEW_CAMERA_POSE_PUBLISHER_DISPLAY_HPP_

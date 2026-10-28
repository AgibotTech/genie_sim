// Copyright 2026 GenieSim contributors. All rights reserved.

#include "genie_sim_rviz_plugins/view_camera_pose_publisher_display.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>

#include <OgreCamera.h>
#include <OgreQuaternion.h>
#include <QVariant>
// OGRE renamed ``OgreVector{3,4}.h`` to a single ``OgreVector.h`` (the
// old headers are deprecation stubs that #include the new one and emit
// a #pragma warning). Humble ships OGRE 1.12 where the rename hadn't
// happened yet, so detect the new header rather than gating on
// ROS_DISTRO.
#if __has_include(<OgreVector.h>)
#  include <OgreVector.h>
#else
#  include <OgreVector3.h>
#endif

#include <geometry_msgs/msg/transform_stamped.hpp>

#include <pluginlib/class_list_macros.hpp>

#include <rviz_common/display_context.hpp>
#include <rviz_common/frame_manager_iface.hpp>
#include <rviz_common/properties/bool_property.hpp>
#include <rviz_common/properties/float_property.hpp>
#include <rviz_common/properties/status_property.hpp>
#include <rviz_common/properties/string_property.hpp>
#include <rviz_common/ros_integration/ros_node_abstraction_iface.hpp>
#include <rviz_common/view_controller.hpp>
#include <rviz_common/view_manager.hpp>

namespace genie_sim_rviz_plugins
{

ViewCameraPosePublisherDisplay::ViewCameraPosePublisherDisplay()
{
  topic_property_ = new rviz_common::properties::StringProperty(
    "Topic",
    "/genie_sim_engine/viewer/camera_pose",
    "geometry_msgs/PoseStamped topic. The genie_sim_render node remaps "
    "~/free_cam_pose to /<physics_node>/viewer/camera_pose by default.",
    this, SLOT(updateTopic()));

  zoom_topic_property_ = new rviz_common::properties::StringProperty(
    "OVRtx Zoom Topic",
    "",
    "Optional std_msgs/Float64 optical-zoom topic. Empty keeps the existing "
    "pose-only contract used by external renderers; Newton inline OVRtx can "
    "opt in through its launcher YAML.",
    this, SLOT(updateZoomTopic()));

  zoom_factor_property_ = new rviz_common::properties::FloatProperty(
    "OVRtx Zoom Factor",
    1.0f,
    "Optical focal-length multiplier published on OVRtx Zoom Topic. "
    "1.0 keeps the authored FreeCam field of view.",
    this);
  zoom_factor_property_->setMin(0.25f);
  zoom_factor_property_->setMax(8.0f);

  auto_orbit_zoom_property_ = new rviz_common::properties::BoolProperty(
    "OVRtx Auto Zoom from Orbit",
    true,
    "Convert changes to the active Orbit view's Distance property (including "
    "mouse-wheel motion) into OVRtx optical focal-length zoom.",
    this);

  rate_property_ = new rviz_common::properties::FloatProperty(
    "Publish Rate (Hz)",
    30.0f,
    "Maximum rate at which the camera pose is published. Set to 0 to "
    "publish on every RViz render frame.",
    this);
  rate_property_->setMin(0.0f);
  rate_property_->setMax(240.0f);

  only_on_change_property_ = new rviz_common::properties::BoolProperty(
    "Only on change",
    true,
    "Skip publishing when the camera pose hasn't changed since the last "
    "publish. Reduces idle traffic.",
    this);

  publish_tf_property_ = new rviz_common::properties::BoolProperty(
    "Publish TF",
    true,
    "Also broadcast a TransformStamped from the RViz Fixed Frame to the "
    "FreeCam child frame on /tf.  Required for RViz's Camera display to "
    "anchor the OVRtx-rendered image — without this TF the panel is "
    "blank because the image's frame_id can't be resolved.",
    this);

  tf_child_frame_property_ = new rviz_common::properties::StringProperty(
    "TF Child Frame",
    "FreeCam",
    "child_frame_id used when ``Publish TF`` is on.  Must match the "
    "header.frame_id the in-process OVRtx visualizer stamps onto its "
    "FreeCam image and camera_info — see CameraCfg.frame_id in "
    "the physics visualizer and "
    "render_node.cpp's CameraConfig.path on the cross-process side.",
    this);
}

ViewCameraPosePublisherDisplay::~ViewCameraPosePublisherDisplay() = default;

void ViewCameraPosePublisherDisplay::onInitialize()
{
  Display::onInitialize();

  auto ros_node_abstraction = context_->getRosNodeAbstraction().lock();
  if (!ros_node_abstraction) {
    setStatus(
      rviz_common::properties::StatusProperty::Error, "RosNode",
      "RViz did not provide a ROS node abstraction.");
    return;
  }
  node_ = ros_node_abstraction->get_raw_node();
  // Single broadcaster for the whole life of the display — reuses the
  // RViz node's /tf publisher under the hood.  Cheap to construct so we
  // don't gate it on the property; the publish itself in update() is
  // gated on publish_tf_property_.
  tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(node_);
  ensurePublisher();
}

void ViewCameraPosePublisherDisplay::onEnable()
{
  ensurePublisher();
}

void ViewCameraPosePublisherDisplay::onDisable()
{
  has_last_ = false;
  has_last_zoom_ = false;
  has_last_orbit_distance_ = false;
  last_view_controller_ = nullptr;
}

void ViewCameraPosePublisherDisplay::reset()
{
  Display::reset();
  has_last_ = false;
  has_last_zoom_ = false;
  has_last_orbit_distance_ = false;
  last_view_controller_ = nullptr;
}

void ViewCameraPosePublisherDisplay::updateTopic()
{
  pose_pub_.reset();
  ensurePublisher();
  has_last_ = false;
}

void ViewCameraPosePublisherDisplay::updateZoomTopic()
{
  zoom_pub_.reset();
  current_zoom_topic_.clear();
  has_last_zoom_ = false;
  ensurePublisher();
}

void ViewCameraPosePublisherDisplay::ensurePublisher()
{
  if (!node_) {
    return;
  }
  const std::string desired = topic_property_->getStdString();
  if (desired.empty()) {
    pose_pub_.reset();
    current_topic_.clear();
    setStatus(
      rviz_common::properties::StatusProperty::Warn, "Topic",
      "Topic is empty.");
  } else if (!pose_pub_ || desired != current_topic_) {
    try {
      // Latched QoS: KeepLast(1) + Reliable + TransientLocal —
      // new subscribers immediately receive the last camera state.
      rclcpp::QoS latched_qos(rclcpp::KeepLast(1));
      latched_qos.reliable();
      latched_qos.transient_local();
      pose_pub_ = node_->create_publisher<geometry_msgs::msg::PoseStamped>(
        desired, latched_qos);
      current_topic_ = desired;
      setStatus(
        rviz_common::properties::StatusProperty::Ok, "Topic",
        QString::fromStdString("Publishing on " + desired));
    } catch (const std::exception & e) {
      pose_pub_.reset();
      current_topic_.clear();
      setStatus(
        rviz_common::properties::StatusProperty::Error, "Topic",
        QString::fromStdString(std::string("Failed to create publisher: ") + e.what()));
    }
  }

  const std::string desired_zoom = zoom_topic_property_->getStdString();
  if (desired_zoom.empty()) {
    zoom_pub_.reset();
    current_zoom_topic_.clear();
  } else if (!zoom_pub_ || desired_zoom != current_zoom_topic_) {
    try {
      rclcpp::QoS latched_qos(rclcpp::KeepLast(1));
      latched_qos.reliable();
      latched_qos.transient_local();
      zoom_pub_ = node_->create_publisher<std_msgs::msg::Float64>(desired_zoom, latched_qos);
      current_zoom_topic_ = desired_zoom;
      has_last_zoom_ = false;
      setStatus(
        rviz_common::properties::StatusProperty::Ok, "OVRtx Zoom Topic",
        QString::fromStdString("Publishing on " + desired_zoom));
    } catch (const std::exception & e) {
      zoom_pub_.reset();
      current_zoom_topic_.clear();
      setStatus(
        rviz_common::properties::StatusProperty::Error, "OVRtx Zoom Topic",
        QString::fromStdString(std::string("Failed to create publisher: ") + e.what()));
    }
  }
}

namespace
{

bool poses_equal(
  const geometry_msgs::msg::Pose & a,
  const geometry_msgs::msg::Pose & b,
  double pos_eps,
  double quat_eps)
{
  if (std::fabs(a.position.x - b.position.x) > pos_eps) {return false;}
  if (std::fabs(a.position.y - b.position.y) > pos_eps) {return false;}
  if (std::fabs(a.position.z - b.position.z) > pos_eps) {return false;}
  if (std::fabs(a.orientation.x - b.orientation.x) > quat_eps) {return false;}
  if (std::fabs(a.orientation.y - b.orientation.y) > quat_eps) {return false;}
  if (std::fabs(a.orientation.z - b.orientation.z) > quat_eps) {return false;}
  if (std::fabs(a.orientation.w - b.orientation.w) > quat_eps) {return false;}
  return true;
}

}  // namespace

void ViewCameraPosePublisherDisplay::updateZoomFromOrbit(
  rviz_common::properties::Property * view_controller, const QString & class_id)
{
  if (!auto_orbit_zoom_property_->getBool() || !view_controller) {
    has_last_orbit_distance_ = false;
    last_view_controller_ = view_controller;
    return;
  }

  if (view_controller != last_view_controller_) {
    has_last_orbit_distance_ = false;
    last_view_controller_ = view_controller;
  }

  // Orbit's public property tree is the stable interface available to an
  // independent Display plugin. Reading it avoids a build/runtime dependency
  // on rviz_default_plugins and tracks wheel events after Orbit applies them.
  // Other view-controller types keep the manual zoom factor.
  if (!class_id.endsWith("/Orbit")) {
    has_last_orbit_distance_ = false;
    return;
  }
  bool ok = false;
  const double orbit_distance =
    view_controller->subProp("Distance")->getValue().toDouble(&ok);
  if (!ok || !std::isfinite(orbit_distance) || orbit_distance <= 1.0e-6) {
    has_last_orbit_distance_ = false;
    return;
  }
  if (!has_last_orbit_distance_) {
    last_orbit_distance_ = orbit_distance;
    has_last_orbit_distance_ = true;
    return;
  }
  if (std::fabs(orbit_distance - last_orbit_distance_) <= 1.0e-9) {
    return;
  }

  const double distance_ratio = last_orbit_distance_ / orbit_distance;
  const double current_zoom = static_cast<double>(zoom_factor_property_->getFloat());
  const double next_zoom = std::min(8.0, std::max(0.25, current_zoom * distance_ratio));
  last_orbit_distance_ = orbit_distance;
  zoom_factor_property_->setFloat(static_cast<float>(next_zoom));
}

void ViewCameraPosePublisherDisplay::publishZoom()
{
  if (zoom_pub_) {
    const double zoom = static_cast<double>(zoom_factor_property_->getFloat());
    if (!has_last_zoom_ || std::fabs(zoom - last_zoom_) > 1e-6) {
      std_msgs::msg::Float64 zoom_msg;
      zoom_msg.data = zoom;
      zoom_pub_->publish(zoom_msg);
      last_zoom_ = zoom;
      has_last_zoom_ = true;
    }
  }
}

void ViewCameraPosePublisherDisplay::update(float wall_dt, float /*ros_dt*/)
{
  if (!isEnabled()) {
    return;
  }

  // Rate-limit publishes.
  const float rate = rate_property_->getFloat();
  if (rate > 0.0f) {
    accumulator_ += wall_dt;
    const float period = 1.0f / rate;
    if (accumulator_ < period) {
      return;
    }
    accumulator_ = 0.0f;
  }

  auto * view_manager = context_->getViewManager();
  if (!view_manager) {return;}
  auto * vc = view_manager->getCurrent();
  if (!vc) {return;}
  updateZoomFromOrbit(vc, vc->getClassId());
  publishZoom();

  if (!pose_pub_) {
    return;
  }
  Ogre::Camera * cam = vc->getCamera();
  if (!cam) {return;}

  // Ogre cameras hold a derived (world-space) position/orientation that is
  // already in the RViz scene's coordinate frame, which the FrameManager
  // maintains aligned with the Fixed Frame. Both Ogre cameras and USD/OVRTX
  // cameras follow the convention of looking down -Z with +Y up, so the
  // orientation transfers without an axis swap.
  const Ogre::Vector3 pos = cam->getDerivedPosition();
  const Ogre::Quaternion quat = cam->getDerivedOrientation();

  geometry_msgs::msg::PoseStamped msg;
  msg.header.stamp = node_->now();
  msg.header.frame_id = context_->getFrameManager()->getFixedFrame();
  msg.pose.position.x = static_cast<double>(pos.x);
  msg.pose.position.y = static_cast<double>(pos.y);
  msg.pose.position.z = static_cast<double>(pos.z);
  msg.pose.orientation.x = static_cast<double>(quat.x);
  msg.pose.orientation.y = static_cast<double>(quat.y);
  msg.pose.orientation.z = static_cast<double>(quat.z);
  msg.pose.orientation.w = static_cast<double>(quat.w);

  if (only_on_change_property_->getBool() && has_last_) {
    if (poses_equal(msg.pose, last_published_.pose, 1e-5, 1e-6)) {
      return;
    }
  }

  pose_pub_->publish(msg);

  // TF broadcast for RViz's Camera display.  The Camera display anchors
  // the rendered image using the frame named in camera_info.header.frame_id;
  // without a TF chain from Fixed Frame to that name, the panel stays
  // blank.  Publish the same world-space pose we just sent so the OVRtx
  // image lines up with whatever camera_info the engine publishes for
  // its FreeCam.  Same QoS path as the rest of /tf (RealtimePublisher
  // owned by tf2_ros), so this is safe to call at the same rate as the
  // pose itself.
  if (publish_tf_property_->getBool() && tf_broadcaster_) {
    geometry_msgs::msg::TransformStamped tf;
    tf.header = msg.header;
    tf.child_frame_id = tf_child_frame_property_->getStdString();
    if (!tf.child_frame_id.empty()) {
      tf.transform.translation.x = msg.pose.position.x;
      tf.transform.translation.y = msg.pose.position.y;
      tf.transform.translation.z = msg.pose.position.z;
      tf.transform.rotation = msg.pose.orientation;
      tf_broadcaster_->sendTransform(tf);
    }
  }

  last_published_ = msg;
  has_last_ = true;
}

}  // namespace genie_sim_rviz_plugins

PLUGINLIB_EXPORT_CLASS(
  genie_sim_rviz_plugins::ViewCameraPosePublisherDisplay,
  rviz_common::Display)

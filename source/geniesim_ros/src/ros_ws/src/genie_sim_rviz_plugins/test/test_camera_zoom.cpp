// Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
// Author: Genie Sim Team
// License: Mozilla Public License Version 2.0

#include <QApplication>
#include <gtest/gtest.h>
#include <rviz_common/properties/float_property.hpp>
#include <rviz_common/properties/string_property.hpp>

#include <chrono>
#include <functional>
#include <limits>
#include <thread>
#include <vector>

#include "genie_sim_rviz_plugins/view_camera_pose_publisher_display.hpp"

namespace genie_sim_rviz_plugins
{

// Exercise the public Orbit property contract without an Ogre camera or GPU.
class TestView : public rviz_common::properties::Property
{
public:
  explicit TestView(const QString & id = "rviz_default_plugins/Orbit")
  : class_id(id)
  {
    distance = new rviz_common::properties::FloatProperty("Distance", 10.0f, "", this);
  }
  QString class_id;
  rviz_common::properties::FloatProperty * distance;
};

class ViewCameraPosePublisherDisplayTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    static unsigned int count = 0;
    topic_ = "/rviz_zoom_test_" + std::to_string(++count);
    node_ = std::make_shared<rclcpp::Node>("rviz_zoom_test_" + std::to_string(count));
    display_ = std::make_unique<ViewCameraPosePublisherDisplay>();
    display_->node_ = node_;
    display_->ensurePublisher();
  }

  void track(TestView * view)
  {
    display_->updateZoomFromOrbit(view, view ? view->class_id : QString());
  }
  float zoom() const {return display_->zoom_factor_property_->getFloat();}
  void zoom(float value) {display_->zoom_factor_property_->setFloat(value);}
  void autoZoom(bool value) {display_->auto_orbit_zoom_property_->setBool(value);}
  void topic(const std::string & value) {display_->zoom_topic_property_->setStdString(value);}
  void sample() {display_->publishZoom();}
  void disable() {display_->onDisable();}
  bool posePublisher() const {return static_cast<bool>(display_->pose_pub_);}
  bool zoomPublisher() const {return static_cast<bool>(display_->zoom_pub_);}

  void spinFor(std::chrono::milliseconds duration)
  {
    const auto end = std::chrono::steady_clock::now() + duration;
    do {
      rclcpp::spin_some(node_);
      QCoreApplication::processEvents();
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    } while (std::chrono::steady_clock::now() < end);
  }

  bool waitFor(const std::function<bool()> & predicate)
  {
    const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (!predicate() && std::chrono::steady_clock::now() < end) {
      spinFor(std::chrono::milliseconds(10));
    }
    return predicate();
  }

  rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr subscribe(
    const std::string & name, std::vector<double> & values)
  {
    return node_->create_subscription<std_msgs::msg::Float64>(
      name, rclcpp::QoS(1).reliable().transient_local(),
      [&values](std_msgs::msg::Float64::ConstSharedPtr msg) {values.push_back(msg->data);});
  }

  std::string topic_;
  rclcpp::Node::SharedPtr node_;
  std::unique_ptr<ViewCameraPosePublisherDisplay> display_;
};

TEST_F(ViewCameraPosePublisherDisplayTest, ZoomIsOptInAndIndependentOfPoseTopic)
{
  EXPECT_TRUE(posePublisher());
  EXPECT_FALSE(zoomPublisher());
  display_->subProp("Topic")->setValue("");
  topic(topic_);
  EXPECT_FALSE(posePublisher());
  EXPECT_TRUE(zoomPublisher());
  topic("");
  EXPECT_FALSE(zoomPublisher());
}

TEST_F(ViewCameraPosePublisherDisplayTest, OrbitRatiosClampAndRebaseline)
{
  TestView orbit;
  track(&orbit);
  orbit.distance->setFloat(5.0f);
  track(&orbit);
  EXPECT_FLOAT_EQ(zoom(), 2.0f);
  track(&orbit);
  EXPECT_FLOAT_EQ(zoom(), 2.0f);
  orbit.distance->setFloat(0.01f);
  track(&orbit);
  EXPECT_FLOAT_EQ(zoom(), 8.0f);
  orbit.distance->setFloat(100.0f);
  track(&orbit);
  EXPECT_FLOAT_EQ(zoom(), 0.25f);
  for (float invalid : {0.0f, -1.0f, std::numeric_limits<float>::infinity()}) {
    orbit.distance->rviz_common::properties::Property::setValue(invalid);
    track(&orbit);
    EXPECT_FLOAT_EQ(zoom(), 0.25f);
    orbit.distance->setFloat(10.0f);
    track(&orbit);
    EXPECT_FLOAT_EQ(zoom(), 0.25f);
  }
  TestView replacement;
  replacement.distance->setFloat(2.0f);
  track(&replacement);
  EXPECT_FLOAT_EQ(zoom(), 0.25f);
  track(nullptr);
  replacement.distance->setFloat(1.0f);
  track(&replacement);
  EXPECT_FLOAT_EQ(zoom(), 0.25f);
}

TEST_F(ViewCameraPosePublisherDisplayTest, ManualModeAndOtherControllersPreserveFactor)
{
  TestView orbit;
  track(&orbit);
  autoZoom(false);
  zoom(3.0f);
  orbit.distance->setFloat(5.0f);
  track(&orbit);
  EXPECT_FLOAT_EQ(zoom(), 3.0f);
  autoZoom(true);
  track(&orbit);
  EXPECT_FLOAT_EQ(zoom(), 3.0f);
  orbit.distance->setFloat(2.5f);
  track(&orbit);
  EXPECT_FLOAT_EQ(zoom(), 6.0f);
  TestView top_down("rviz_default_plugins/TopDownOrtho");
  track(&top_down);
  top_down.distance->setFloat(1.0f);
  track(&top_down);
  EXPECT_FLOAT_EQ(zoom(), 6.0f);
  zoom(100.0f);
  EXPECT_FLOAT_EQ(zoom(), 8.0f);
  zoom(-1.0f);
  EXPECT_FLOAT_EQ(zoom(), 0.25f);
}

TEST_F(ViewCameraPosePublisherDisplayTest, PublishesLatchedChangesAndResendsAfterReset)
{
  topic(topic_);
  zoom(2.0f);
  sample();  // Publish before subscription to verify transient-local delivery.
  std::vector<double> values;
  auto sub = subscribe(topic_, values);
  ASSERT_TRUE(waitFor([&]() {return values.size() == 1;}));
  EXPECT_DOUBLE_EQ(values.back(), 2.0);
  sample();
  spinFor(std::chrono::milliseconds(100));
  EXPECT_EQ(values.size(), 1u);
  zoom(3.0f);
  sample();
  ASSERT_TRUE(waitFor([&]() {return values.size() == 2;}));
  EXPECT_DOUBLE_EQ(values.back(), 3.0);
  display_->reset();
  sample();
  ASSERT_TRUE(waitFor([&]() {return values.size() == 3;}));
  disable();
  sample();
  ASSERT_TRUE(waitFor([&]() {return values.size() == 4;}));

  std::vector<double> moved_values;
  topic(topic_ + "_moved");
  auto moved_sub = subscribe(topic_ + "_moved", moved_values);
  sample();
  ASSERT_TRUE(waitFor([&]() {return moved_values.size() == 1;}));
  EXPECT_DOUBLE_EQ(moved_values.back(), 3.0);
  EXPECT_EQ(values.size(), 4u);
  topic("");
  zoom(4.0f);
  sample();
  spinFor(std::chrono::milliseconds(100));
  EXPECT_EQ(moved_values.size(), 1u);
}

TEST_F(ViewCameraPosePublisherDisplayTest, InvalidZoomTopicLeavesPosePublisherUsable)
{
  topic("invalid topic");
  EXPECT_FALSE(zoomPublisher());
  EXPECT_TRUE(posePublisher());
  topic(topic_);
  EXPECT_TRUE(zoomPublisher());
  EXPECT_TRUE(posePublisher());
}

}  // namespace genie_sim_rviz_plugins

int main(int argc, char ** argv)
{
  QApplication app(argc, argv);
  rclcpp::init(argc, argv);
  ::testing::InitGoogleTest(&argc, argv);
  const int result = RUN_ALL_TESTS();
  rclcpp::shutdown();
  return result;
}

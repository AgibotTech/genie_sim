#include "genie_sim_render/image_publisher_plugin.h"

#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <image_transport/image_transport.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/static_transform_broadcaster.h>

#include <map>

#include <pluginlib/class_list_macros.hpp>

namespace genie_sim_render
{

class RosImagePublisherPlugin : public IImagePublisherPlugin
{
public:
  RosImagePublisherPlugin() = default;
  ~RosImagePublisherPlugin() override = default;

  void initialize(rclcpp::Node * node) override {node_ = node;}

  void create_camera_publisher(
    const std::string & topic, uint32_t /*width*/, uint32_t /*height*/) override
  {
    img_pubs_[topic] = image_transport::create_publisher(node_, topic + "/image_raw");
    info_pubs_[topic] =
      node_->create_publisher<sensor_msgs::msg::CameraInfo>(topic + "/camera_info", 10);
  }

  void publish_camera_image(
    const std::string & topic, const CameraImageMsgConstPtr & msg_ptr) override
  {
    auto it = img_pubs_.find(topic);
    if (it == img_pubs_.end() || !msg_ptr) {
      return;
    }
    const CameraImageMsg & msg = *msg_ptr;
    auto ros_msg = std::make_unique<sensor_msgs::msg::Image>();
    ros_msg->header.stamp.sec = static_cast<int32_t>(msg.timestamp);
    ros_msg->header.stamp.nanosec =
      static_cast<uint32_t>((msg.timestamp - static_cast<int>(msg.timestamp)) * 1e9);
    ros_msg->header.frame_id = msg.frame_id;
    ros_msg->width = msg.width;
    ros_msg->height = msg.height;
    ros_msg->encoding = msg.encoding;
    ros_msg->is_bigendian = false;
    ros_msg->step = msg.step;
    // sensor_msgs::Image must own its data.
    ros_msg->data.assign(msg.data.begin(), msg.data.end());
    it->second.publish(std::move(ros_msg));
  }

  void publish_camera_info(const std::string & topic, const CameraInfoMsg & msg) override
  {
    auto it = info_pubs_.find(topic);
    if (it == info_pubs_.end()) {
      return;
    }
    auto ros_msg = std::make_unique<sensor_msgs::msg::CameraInfo>();
    ros_msg->header.stamp.sec = static_cast<int32_t>(msg.timestamp);
    ros_msg->header.stamp.nanosec =
      static_cast<uint32_t>((msg.timestamp - static_cast<int>(msg.timestamp)) * 1e9);
    ros_msg->header.frame_id = msg.frame_id;
    ros_msg->width = msg.width;
    ros_msg->height = msg.height;
    ros_msg->distortion_model = msg.distortion_model;
    ros_msg->d = msg.d;
    ros_msg->k = msg.k;
    ros_msg->r = msg.r;
    ros_msg->p = msg.p;
    it->second->publish(std::move(ros_msg));
  }

  void create_pointcloud_publisher(const std::string & topic) override
  {
    pc_pubs_[topic] =
      node_->create_publisher<sensor_msgs::msg::PointCloud2>(topic, 10);
  }

  void publish_pointcloud(const std::string & topic, const PointCloudMsg & msg) override
  {
    auto it = pc_pubs_.find(topic);
    if (it == pc_pubs_.end()) {
      return;
    }
    auto ros_msg = std::make_unique<sensor_msgs::msg::PointCloud2>();
    ros_msg->header.stamp.sec = static_cast<int32_t>(msg.timestamp);
    ros_msg->header.stamp.nanosec =
      static_cast<uint32_t>((msg.timestamp - static_cast<int>(msg.timestamp)) * 1e9);
    ros_msg->header.frame_id = msg.frame_id;
    ros_msg->height = msg.height;
    ros_msg->width = msg.width;
    ros_msg->is_bigendian = false;
    ros_msg->point_step = msg.point_step;
    ros_msg->row_step = msg.point_step * msg.width;
    ros_msg->is_dense = msg.is_dense;
    ros_msg->fields = pointcloud_fields();
    ros_msg->data = msg.data;
    it->second->publish(std::move(ros_msg));
  }

  void create_tf_publisher() override
  {
    if (!static_tf_pub_) {
      static_tf_pub_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(node_);
    }
    if (!dynamic_tf_pub_) {
      dynamic_tf_pub_ = std::make_shared<tf2_ros::TransformBroadcaster>(node_);
    }
  }

  void publish_transforms(
    const std::vector<TransformMsg> & transforms, bool is_static) override
  {
    std::vector<geometry_msgs::msg::TransformStamped> tfs;
    tfs.reserve(transforms.size());
    for (const auto & t : transforms) {
      geometry_msgs::msg::TransformStamped tf;
      tf.header.stamp.sec = static_cast<int32_t>(t.timestamp);
      tf.header.stamp.nanosec =
        static_cast<uint32_t>((t.timestamp - static_cast<int>(t.timestamp)) * 1e9);
      tf.header.frame_id = t.parent_frame;
      tf.child_frame_id = t.child_frame;
      tf.transform.translation.x = t.tx;
      tf.transform.translation.y = t.ty;
      tf.transform.translation.z = t.tz;
      tf.transform.rotation.x = t.qx;
      tf.transform.rotation.y = t.qy;
      tf.transform.rotation.z = t.qz;
      tf.transform.rotation.w = t.qw;
      tfs.push_back(tf);
    }
    if (is_static && static_tf_pub_) {
      static_tf_pub_->sendTransform(tfs);
    } else if (!is_static && dynamic_tf_pub_) {
      dynamic_tf_pub_->sendTransform(tfs);
    }
  }

private:
  // 22-byte layout: x,y,z FLOAT32 @0/4/8; reflectivity UINT8 @12; tag UINT8 @13; timestamp FLOAT64 @14.
  static std::vector<sensor_msgs::msg::PointField> pointcloud_fields()
  {
    using PF = sensor_msgs::msg::PointField;
    auto field = [](const char * name, uint32_t offset, uint8_t datatype) {
        PF f;
        f.name = name;
        f.offset = offset;
        f.datatype = datatype;
        f.count = 1;
        return f;
      };
    return {
      field("x", 0, PF::FLOAT32),
      field("y", 4, PF::FLOAT32),
      field("z", 8, PF::FLOAT32),
      field("reflectivity", 12, PF::UINT8),
      field("tag", 13, PF::UINT8),
      field("timestamp", 14, PF::FLOAT64),
    };
  }

private:
  rclcpp::Node * node_ {nullptr};
  std::map<std::string, image_transport::Publisher> img_pubs_;
  std::map<std::string, rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr> info_pubs_;
  std::map<std::string, rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr> pc_pubs_;
  std::shared_ptr<tf2_ros::StaticTransformBroadcaster> static_tf_pub_;
  std::shared_ptr<tf2_ros::TransformBroadcaster> dynamic_tf_pub_;
};

}  // namespace genie_sim_render

PLUGINLIB_EXPORT_CLASS(
  genie_sim_render::RosImagePublisherPlugin,
  genie_sim_render::IImagePublisherPlugin)

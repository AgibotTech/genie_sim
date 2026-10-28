#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>

namespace genie_sim_render
{

  struct CameraImageMsg
  {
    std::string frame_id;
    double timestamp {0.0};
    uint32_t width {0};
    uint32_t height {0};
    std::string encoding;
    uint32_t step {0};
    std::vector < uint8_t > data;
  };

  // Shared ownership handed to the plugin: an async plugin can hold the
  // frame directly instead of copying a full image. While held, the frame's
  // pixel memory is not released, so async plugins must use a bounded queue.
  using CameraImageMsgConstPtr = std::shared_ptr < const CameraImageMsg >;

  struct CameraInfoMsg
  {
    std::string frame_id;
    double timestamp {0.0};
    uint32_t width {0};
    uint32_t height {0};
    std::string distortion_model;
    std::vector < double > d;
    std::array < double, 9 > k;
    std::array < double, 9 > r;
    std::array < double, 12 > p;
  };

// Raw, already-packed point cloud. ``data`` holds ``width * point_step`` bytes in the
// rmagine-compatible 22-byte layout: x,y,z FLOAT32 @0/4/8; reflectivity UINT8 @12; pad @13;
// timestamp FLOAT64 @14. The plugin only wraps these bytes in a transport message.
  struct PointCloudMsg
  {
    std::string frame_id;
    double timestamp {0.0};
    uint32_t width {0};
    uint32_t height {1};
    uint32_t point_step {22};
    bool is_dense {true};
    std::vector < uint8_t > data;
  };

// A single TF edge (parent_frame -> child_frame). Rotation is a quaternion (x,y,z,w).
  struct TransformMsg
  {
    std::string parent_frame;
    std::string child_frame;
    double timestamp {0.0};
    double tx {0.0};
    double ty {0.0};
    double tz {0.0};
    double qx {0.0};
    double qy {0.0};
    double qz {0.0};
    double qw {1.0};
  };

  class IImagePublisherPlugin
  {
public:
    virtual ~IImagePublisherPlugin() = default;

    virtual void initialize(rclcpp::Node * node) = 0;

    virtual void create_camera_publisher(
      const std::string & topic, uint32_t width, uint32_t height) = 0;

    virtual void publish_camera_image(
      const std::string & topic, const CameraImageMsgConstPtr & msg) = 0;

    virtual void publish_camera_info(
      const std::string & topic, const CameraInfoMsg & msg) = 0;

    // Explicit ROS topic -> DDS topic mapping (forwarded from the dds_topic field in scene json).
    // Default no-op: only the DDS plugin overrides this; the ROS plugin ignores it.
    // Unregistered ROS topics are skipped by the DDS plugin (e.g. FreeCam).
    virtual void register_dds_mapping(
      const std::string & /*ros_topic*/,
      const std::string & /*dds_topic*/) {}

    // ROS depth topic -> DDS depth topic mapping (forwarded from the dds_depth_topic field
    // in scene json). Mirrors register_dds_mapping but is dispatched on the depth side so
    // the plugin can build a depth-typed publisher (e.g. Z16 raw) instead of the JPEG one.
    // Default no-op for the same reason.
    virtual void register_dds_depth_mapping(
      const std::string & /*ros_depth_topic*/,
      const std::string & /*dds_depth_topic*/) {}

    // ROS image topic -> DDS topic mapping for cameras whose DDS side must be a
    // vendor IMAGE packet carrying a JPEG payload (encoding=JPEG, color_format=RGB,
    // bit_depth=8) instead of the sensor_msgs CompressedImage used by register_dds_mapping.
    // Fisheye cameras use this to match the real robot's packet format. Default no-op.
    virtual void register_dds_image_packet_mapping(
      const std::string & /*ros_topic*/,
      const std::string & /*dds_topic*/) {}

    // Point cloud publishing. Default no-op so only plugins that support lidar (currently the
    // ROS plugin) implement them; the DDS plugin can opt in later. create_* is called once per
    // lidar topic at setup; publish_* is called per scan with an already-packed PointCloudMsg.
    virtual void create_pointcloud_publisher(const std::string & /*topic*/) {}

    virtual void publish_pointcloud(
      const std::string & /*topic*/, const PointCloudMsg & /*msg*/) {}

    // TF publishing (parity with the rmagine path). create_tf_publisher() sets up the static
    // (/tf_static) and dynamic (/tf) broadcasters once; publish_transforms() emits a batch on the
    // static or dynamic broadcaster. Default no-op so plugins opt in (ROS uses tf2 broadcasters,
    // the DDS plugin publishes vendor TF DDS messages; render-only setups ignore them).
    virtual void create_tf_publisher() {}

    virtual void publish_transforms(
      const std::vector < TransformMsg > & /*transforms*/, bool /*is_static*/) {}
  };

}  // namespace genie_sim_render

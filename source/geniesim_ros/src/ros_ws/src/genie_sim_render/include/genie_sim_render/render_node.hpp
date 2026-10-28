#pragma once

#include <rclcpp/rclcpp.hpp>
#include <tf2_msgs/msg/tf_message.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_msgs/msg/string.hpp>
#include <pluginlib/class_loader.hpp>

#include <ovrtx/ovrtx.h>
#include <ovrtx/ovrtx_config.h>
#include <ovrtx/ovrtx_types.h>

#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <deque>
#include <unordered_map>
#include <unordered_set>

namespace genie_sim_render
{

class IImagePublisherPlugin;

struct CameraConfig
{
  std::string path;
  std::string topic;
  // Depth ROS topic (from manifest's depth_topic field; scene yaml's topic.depth).
  // If empty, depth is not published and DistanceToImagePlaneSD render var is not authored on the render layer.
  std::string depth_topic;
  // DDS target topic (from scene json's dds_topic field). If empty, the DDS plugin skips this camera.
  std::string dds_topic;
  // DDS target topic for depth (from scene json's dds_depth_topic field). If empty, depth is not mirrored to DDS.
  std::string dds_depth_topic;
  int width{1280};
  int height{800};
  double fx{610.0};
  double fy{610.0};
  double cx{640.0};
  double cy{400.0};
  double k1{0.0};
  double k2{0.0};
  double p1{0.0};
  double p2{0.0};
  double k3{0.0};
  // 4th fisheye coefficient (OpenCV fisheye / equidistant uses k1..k4, no p1/p2). Only meaningful
  // when distortion_model is the fisheye model; ignored for plumb_bob.
  double k4{0.0};
  // CameraInfo distortion model string published to ROS. "plumb_bob" (default, Brown-Conrady) or
  // "equidistant" (OpenCV fisheye). Derived from the manifest "model" field by load_manifest.
  std::string distortion_model{"plumb_bob"};
  // Raw lens model string from the manifest ("opencvFisheye" for fisheye cams, empty otherwise).
  // Used to route the DDS image side: fisheye -> JPEG vendor IMAGE packet, else CompressedImage.
  std::string model;
  // Total field of view of a >180 deg fisheye, in degrees (manifest
  // "max_fov_deg", scene yaml "mask_fov_deg"). Only sizes the circular black
  // mask -- see fisheye_mask_radius_px in render_node.cpp. 0 disables masking.
  double max_fov_deg{0.0};
  double min_range{0.01};
  double max_range{10000.0};

  std::string render_product_path;
  bool is_free_cam{false};
};

struct LidarConfig
{
  std::string render_product_path;
  std::string topic;
  std::string frame_id;
  // Sensor frame the per-lidar cloud is published in (e.g. livox_front), derived from the topic
  // basename. Also the child frame of the static base_link->sensor TF. Matches the rmagine path.
  std::string sensor_frame;
  // Body link the lidar is parented to (e.g. base_link); its world TF is mirrored onto the
  // render-layer lidar copy so ovrtx raycasts from the right pose.
  std::string parent_body;
  // Static base_link <- sensor transform: points come out in the sensor frame and are
  // transformed into the body frame before publishing.
  double bl_T_s_xyz[3]{0.0, 0.0, 0.0};
  double bl_T_s_wxyz[4]{1.0, 0.0, 0.0, 0.0};
  // Wall-time of the last publish, for the 10 Hz safety throttle.
  std::chrono::steady_clock::time_point last_publish{};
};

// One deformable (cloth/tet) object as declared by the physics side's
// ``<topic>_topology`` latched message (deformable_publisher.py). Mirrors
// the rigid-body ``robot_body_paths_`` map's role but keyed by the small
// integer ``mesh_id`` the flat per-tick message uses instead of a string.
struct DeformableObject
{
  std::string name;
  std::string prim_path;
  size_t n_particles{0};
};

struct FrameTiming
{
  double step_ms{0.0};
  double wait_ms{0.0};
  double fetch_ms{0.0};
  double map_ms{0.0};
  double publish_ms{0.0};
  double total_ms{0.0};
  double interval_ms{0.0};
};

class RenderNode : public rclcpp::Node
{
public:
  using Clock = std::chrono::steady_clock;
  using TimePoint = Clock::time_point;

  explicit RenderNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~RenderNode() override;

private:
  void load_manifest(const std::string & manifest_path);
  void init_renderer();
  void setup_cameras();
  void setup_lidars();
  void load_plugins();
  void shutdown_renderer();

  void on_tf_scene(const tf2_msgs::msg::TFMessage::SharedPtr msg);
  void on_free_cam_pose(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
  void render_timer_callback();
  void stats_timer_callback();

  // Live deformable-particle channel (opt-in via the ``deformable_topic``
  // ROS param, wired from the scene yaml's ``newton.deformable_publish``
  // block through the launcher yaml's ``render_ovrtx`` params). See
  // engine/newton/deformable_publisher.py for the wire format.
  void on_deformable_topology(const std_msgs::msg::String::SharedPtr msg);
  void on_deformable_flat(const std_msgs::msg::Float32MultiArray::SharedPtr msg);
  // Writes one object's world-space points straight to its mesh prim's
  // ``points`` attribute via the generic ovrtx_write_attribute path (no
  // convenience wrapper exists for this, unlike xform/path/token). First
  // touch flags ``omni:resetXformStack=1`` + identity xform on the prim
  // (reusing reset_xform_done_) so the incoming points -- already
  // world-space, straight out of Newton's particle_q -- land in the
  // renderer without any local/world conversion.
  void write_deformable_points(
    const DeformableObject & obj, const float * xyz, size_t n_points);

  // v2 protocol: pull every URDF link's composed world pose out of the
  // ``tf2::Buffer`` (fed by ``/tf`` + ``/tf_static``) and write it to
  // OVRTX. Called once at the top of every ``render_timer_callback``.
  // The tf2 buffer composes the URDF chain from RSP for us, so this
  // side only does prim-path lookup + matrix conversion + OVRTX write.
  void apply_robot_tf();

  void publish_camera_images(const ovrtx_render_product_set_outputs_t & outputs);
  void publish_lidar_pointclouds(const ovrtx_render_product_set_outputs_t & outputs);
  void publish_lidar_tf();
  void publish_timing(const FrameTiming & t);

  ovrtx_render_var_output_handle_t find_output_for_product(
    const ovrtx_render_product_set_outputs_t & outputs,
    size_t product_index,
    const char * render_var_name);

  ovrtx_render_var_output_handle_t find_output_by_path(
    const ovrtx_render_product_set_outputs_t & outputs,
    const std::string & product_path,
    const char * render_var_name);

  std::string usd_path_;
  std::string robot_usda_;
  std::string render_layer_usda_;
  std::string robot_prefix_;
  //! ``manifest.robot["body_paths"]`` cached at scene-load time.
  //! Keyed by bare URDF link name (matches ``/tf`` ``child_frame_id``
  //! values), value is the absolute OVRTX prim path.
  //! ``apply_robot_tf()`` iterates this on every render tick.
  std::unordered_map<std::string, std::string> robot_body_paths_;
  double render_fps_;
  std::vector<std::string> prim_paths_;
  std::vector<CameraConfig> cameras_;
  std::vector<LidarConfig> lidars_;
  // ROS param 'lidar' (default false): lidar is opt-in for the ovrtx renderer. When false the
  // manifest's lidars are ignored, so no lidar render products are stepped and no PointCloud2
  // is published. Set 'lidar: true' in the launcher yaml's render_ovrtx params to enable.
  bool lidar_enabled_{false};
  std::string merged_lidar_topic_;

  ovrtx_renderer_t * renderer_{nullptr};
  ovrtx_usd_handle_t scene_handle_{0};
  ovrtx_usd_handle_t robot_handle_{0};
  ovrtx_usd_handle_t camera_layer_handle_{0};

  rclcpp::Subscription<tf2_msgs::msg::TFMessage>::SharedPtr tf_scene_sub_;
  //! ``/tf`` + ``/tf_static`` consumer for the robot channel.
  //! ``TransformListener`` spawns its own subscription pair; every
  //! transform lands in ``tf_buffer_`` where ``apply_robot_tf()``
  //! composes the URDF chain per render tick.
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr free_cam_sub_;
  std::string free_cam_prim_path_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr timing_pub_;
  rclcpp::TimerBase::SharedPtr render_timer_;
  rclcpp::TimerBase::SharedPtr stats_timer_;

  // Live deformable-particle channel state. ``deformable_topic_`` empty
  // (the default) means the feature is off -- no subscriptions created.
  std::string deformable_topic_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr deformable_topology_sub_;
  rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr deformable_flat_sub_;
  std::unordered_map<int, DeformableObject> deformable_objects_;
  int deformable_last_tick_{-1};
  uint64_t deformable_frames_received_{0};

  std::vector<ovrtx_xform_matrix44d_t> transforms_;
  std::unordered_map<std::string, std::string> cam_body_to_render_path_;
  std::unordered_map<std::string, std::string> lidar_body_to_render_path_;
  // Prims we've already authored ``omni:resetXformStack=1`` on. Newton's
  // in-process visualizer does this on bind; the cross-process render node
  // does it lazily — first time a /tf_scene entry names a prim, flag it
  // so the world-space matrix we then write isn't composed with the
  // parent xform stack. Resetting the stack is required for any prim
  // driven by /tf_scene whose USD parent carries a non-identity xform
  // (e.g. /World/objects/<bottle> with an authored xformOp:translate).
  std::unordered_set<std::string> reset_xform_done_;
  bool scene_loaded_{false};
  // Latest /tf_scene stamp (sim time). Published clouds/images use this so their timestamps
  // share the TF tree's time base, else RViz's tf2 message filter drops them.
  double latest_tf_time_{0.0};

  // base_link's latest world pose (from /tf_scene), used to publish the dynamic base_link->world
  // TF (= its inverse) so consumers can place the base_link/sensor-frame clouds as the robot moves.
  bool has_base_link_world_{false};
  bool tf_static_published_{false};
  double base_link_world_q_[4]{0.0, 0.0, 0.0, 1.0};  // x, y, z, w
  double base_link_world_t_[3]{0.0, 0.0, 0.0};

  TimePoint last_frame_time_{};
  uint64_t frame_count_{0};
  uint64_t drop_count_{0};
  // First-frame compile / shader-cache warmup can take 30-90s. Operators
  // see blank images and assume the renderer is hung. Track the first
  // wait so render_timer_callback can spawn a 1Hz heartbeat thread.
  bool first_frame_done_{false};
  std::deque<FrameTiming> timing_history_;
  static constexpr size_t kHistorySize = 300;

  std::shared_ptr<pluginlib::ClassLoader<IImagePublisherPlugin>> plugin_loader_;
  std::vector<std::shared_ptr<IImagePublisherPlugin>> image_plugins_;
};

}  // namespace genie_sim_render

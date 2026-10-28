// Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
// Author: Genie Sim Team
// Proprietary and confidential.

#pragma once

#include <array>
#include <atomic>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp/executors/multi_threaded_executor.hpp"

#include "image_transport/image_transport.hpp"
#include "realtime_tools/realtime_publisher.hpp"

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "gazebo_msgs/msg/contacts_state.hpp"
#include "lifecycle_msgs/msg/state.hpp"
#include "lifecycle_msgs/srv/change_state.hpp"
#include "lifecycle_msgs/srv/get_state.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "rosgraph_msgs/msg/clock.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "std_msgs/msg/float32.hpp"
#include "std_msgs/msg/float64.hpp"
#include "std_srvs/srv/empty.hpp"
#include "tf2_msgs/msg/tf_message.hpp"
#include "tf2_ros/static_transform_broadcaster.h"

#include "realtime_buffer.hpp"

namespace gsi
{

struct NodeOptions
{
  std::string node_name = "genie_sim_engine";
  std::string namespace_ = "";

  std::string topic_clock = "/clock";
  std::string topic_joint_states = "/joint_states";
  std::string topic_joint_states_scene = "/joint_states_scene";
  std::string topic_odom = "/odom";
  std::string topic_tf = "/tf";
  std::string topic_tf_scene = "/tf_scene";
  std::string topic_tf_render = "/tf_render";
  std::string topic_contact_states = "/contact_states";
  std::string topic_joint_command = "/joint_command";
  std::string topic_cmd_4ws = "/cmd_4ws";
  //! Out-of-band pose writes. ``tf2_msgs/TFMessage`` in, so the wire format
  //! is the same one ``/tf_scene`` publishes back out and a client can echo
  //! a pose, edit it, and send it — which is exactly how the downstream
  //! mujoco reference drives its teleports.
  std::string topic_teleport = "/genie_sim/teleport";

  std::string odom_frame = "odom";
  std::string base_frame = "base_link";
  std::string map_frame = "map";
  std::string world_frame = "world";

  bool fake_slam = false;

  //! Single-topic body-pose mode (v1, deprecated).
  //!
  //! When true, publishes every body pose — robot links included — on the
  //! single topic ``topic_tf_render`` (``/tf_render``) and suppresses
  //! ``/tf_scene`` entirely.  This is the **v1 protocol** and is kept only
  //! for deployments that run no ``robot_state_publisher`` and therefore
  //! cannot use the v2 split.
  //!
  //! **Remaining v1 consumer:** ``genie_sim_render/scripts/isaacsim_render.py``
  //! (single-camera Isaac renderer, no-RSP deployments only).  All other
  //! render paths have migrated to v2.
  //!
  //! **v2 protocol (preferred):** leave this false.  The bridge publishes
  //! scene-body poses on ``/tf_scene`` and robot-link poses reach consumers
  //! via ``/tf`` (``map → base_link``) + ``robot_state_publisher`` URDF FK.
  //! This is the default for all bringup-launch deployments.
  //!
  //! Exactly one publisher is ever created, so a topic that nobody writes
  //! never appears in ``ros2 topic list`` as a phantom.
  //!
  //! @deprecated  Use the v2 split (use_tf_render = false) for all new work.
  bool use_tf_render = false;

  int executor_threads = 2;
};

class RosBridge
{
public:
  RosBridge(int argc, char ** argv, const NodeOptions & opts);
  ~RosBridge();

  // Configure the topology of the per-tick state vectors. Must be called
  // before the first publish_*. After this, joint_state and body_tf vectors
  // must arrive with sizes that match these names.
  //
  // ``body_is_robot`` (v2 protocol): parallel to ``body_frames``,
  // marks each body as a robot link (``true``) or a scene body
  // (``false``). ``publish_body_tf_scene`` skips robot entries — those
  // reach consumers via ``/tf`` (physics-side ``map → base_link`` +
  // ``robot_state_publisher`` URDF FK). Passing an empty vector treats
  // every body as a scene body (v1 behaviour, retained for callers
  // that haven't been updated).
  //
  // ``publish_tf_render`` ignores this mask entirely — see its docs.
  void set_topology(
    const std::vector<std::string> & joint_names,
    const std::vector<std::string> & body_frames,
    const std::vector<bool> & body_is_robot = {});

  // === Publish path (called from Python's GIL-released hot path) ===========
  // sim_time is seconds (float64). joint_pos/joint_vel are deg-converted-to-rad
  // arrays, length = joint_names_.size(). body_xyzwxyz is a flat row-major
  // array (px,py,pz,qw,qx,qy,qz) per body, length = 7*body_frames_.size().
  // base_xyzwxyz is the 7-vector for /odom + odom->base_link.
  // base_twist is the 6-vector (vx, vy, vz, wx, wy, wz) expressed in the
  // base_link (child) frame, matching nav_msgs/Odometry.twist convention.
  // Pass nullptr to leave twist all-zero (e.g. on the very first tick).
  void publish_clock(double sim_time);
  void publish_rtf(float rtf);

  // ``frame_id`` carries the wrapper's reset_epoch causal credential
  // ("reset_epoch:<n>", or empty when the active backend doesn't own a
  // reset epoch) — see genie_sim_engine_core/common/loop.py::_publish_tick
  // (Python caller) and rlinf_geniesim's _ros_bridge.py js_reset_epoch()
  // (wrapper-side consumer). Not otherwise used by any /joint_states
  // consumer (RViz et al. only read header.stamp + name/position/velocity),
  // so repurposing it here is wire-compatible.
  //
  // NOTE — WORKAROUND: header.frame_id is a TF frame name field, not a
  // causal-credential channel. Encoding a monotonically-bumped integer
  // here as ``reset_epoch:<n>`` is a stop-gap because /joint_states is
  // the ONLY signal both the engine and the wrapper already share on
  // the hot path, and adding a sidecar topic would race the state
  // publish (the whole point of the credential is to be observed
  // ATOMICALLY with the post-reset state). Replace with a dedicated
  // channel once one exists — e.g. a service-response ack from
  // /reset_envs carrying the epoch, or a new /reset_epoch topic
  // published under the same trylock as /joint_states. Until then the
  // string prefix scheme is deliberately narrow: only the "reset_epoch:"
  // prefix is claimed, so if a future consumer needs frame_id for its
  // intended TF purpose the prefix can be dropped without touching the
  // wire type.
  void publish_joint_states(
    double sim_time,
    const double * joint_pos,
    const double * joint_vel,
    const double * joint_eff,
    const std::string & frame_id,
    std::size_t n);

  //! v2 protocol: publish an articulated-scene DoF stream on
  //! ``/joint_states_scene``. Names are MJCF joint names verbatim
  //! (per DESIGN.DATA_EXCHANGE_PROTOCOL.md §5). Free-joint bodies
  //! and robot URDF joints do not appear here. The common publish
  //! loop sources the set one of two ways, per backend: either by
  //! splitting ``sim.joint_names`` on manifest ``robot.joint_names``
  //! membership (Isaac Sim paths, whose joint collection spans the
  //! whole stage), or from a dedicated ``get_scene_joint_states``
  //! readback for backends that keep scene articulations off the
  //! controllable surface altogether (newton-standalone, where a
  //! joint-map entry would also acquire a PD drive). Either way what
  //! arrives here is the same wire contract.
  //!
  //! ``names.size() == n``. ``joint_pos`` / ``joint_vel`` are dense
  //! arrays of length ``n``; either pointer may be nullptr to omit
  //! that field (empty vector on the wire, standard sensor_msgs
  //! convention for "not measured"). ``joint_eff`` follows the same
  //! optional-nullptr convention — passive scene joints (springs,
  //! contact-only) still carry constraint / contact torque that
  //! benchmark checkers threshold on, so a backend that can measure
  //! it should pass it. Calls with ``n == 0`` return without
  //! publishing.
  void publish_joint_states_scene(
    double sim_time,
    const std::vector<std::string> & names,
    const double * joint_pos,
    const double * joint_vel,
    const double * joint_eff,
    std::size_t n);

  //! v2 protocol: scene bodies only, on ``/tf_scene``. Robot links are
  //! filtered out via ``body_is_robot_`` — they reach consumers over
  //! ``/tf`` instead (physics ``map → base_link`` + robot_state_publisher
  //! URDF FK). No-op unless the bridge was built with
  //! ``use_tf_render = false``.
  void publish_body_tf_scene(
    double sim_time,
    const double * body_xyzwxyz,
    std::size_t n_bodies);

  //! Single-topic mode: EVERY body — robot links included — on
  //! ``/tf_render``, for deployments with no robot_state_publisher to fan
  //! link poses onto ``/tf``.
  //!
  //! Ignores ``body_is_robot_`` by construction. That is the whole point of
  //! having a second entry point rather than a flag inside
  //! ``publish_body_tf_scene``: "render mode carries the robot" is then a
  //! property of the code, not of whatever mask the caller happened to pass.
  //!
  //! One message, one stamp, every body — consumers that difference a
  //! gripper against an object never straddle two ticks. No-op unless the
  //! bridge was built with ``use_tf_render = true``.
  void publish_tf_render(
    double sim_time,
    const double * body_xyzwxyz,
    std::size_t n_bodies);

  //! v2 protocol: publish per-tick contact state on ``/contact_states``.
  //! Wire type is ``gazebo_msgs/ContactsState``, aggregated per body
  //! pair — one ``ContactState`` entry per unique ``(body1, body2)``
  //! pair with every point that hit that pair listed inside
  //! ``contact_positions[]`` / ``contact_normals[]`` / ``depths[]`` /
  //! ``wrenches[]``, plus a summed ``total_wrench``. Grouping happens
  //! here so backends can hand over raw per-point data.
  //!
  //! Frame convention: ``header.frame_id = map`` for positions and
  //! normals (matches ``/tf_scene``). Wrench vectors are in the
  //! producer engine's native frame — MuJoCo publishes them in each
  //! contact's local frame (column 0 = normal, columns 1/2 = tangent
  //! friction axes) as returned by ``mj_contactForce``. Downstream
  //! consumers that want a world-frame force rotate using
  //! ``contact_normals[i]`` and the two tangent basis vectors
  //! implied by it. Other backends may pick a different native
  //! convention; per-backend documentation covers what is emitted.
  //!
  //! Parallel per-point arrays of length ``n_points``. Any array
  //! pointer may be nullptr, in which case that field publishes as
  //! all-zero (positions/normals) or empty (depths/wrenches). Passing
  //! ``n_points == 0`` still publishes an empty ``ContactsState``
  //! message — the "no contacts this frame" reading is a meaningful
  //! signal for grasp / collision checkers, distinct from a stalled
  //! publisher.
  void publish_contact_states(
    double sim_time,
    const std::vector<std::string> & body1_names,
    const std::vector<std::string> & body2_names,
    const double * positions,
    const double * normals,
    const double * depths,
    const double * wrenches,
    std::size_t n_points);

  void publish_odom(
    double sim_time,
    const double * base_xyzwxyz,
    const double * base_twist);

  // === Inline OVRtx visualizer support =====================================
  // Methods below are called from the OVRtx render thread (separate from the
  // physics tick thread). All publish paths go through realtime_tools::
  // RealtimePublisher just like the physics-thread publishers, so the
  // try_lock + unlockAndPublish pattern keeps the OVRtx thread non-blocking
  // and the actual DDS publish hops onto each RealtimePublisher's dedicated
  // thread. Subscriber counts use rclcpp::Publisher::get_subscription_count
  // which is thread-safe.

  // Latest sim_time stamped by publish_clock(). Atomic so the OVRtx thread
  // can read without a mutex while the physics thread writes.
  double last_sim_time() const;

  // Lazily construct (or look up) RGB / depth publishers per topic. Called
  // once per camera at startup; idempotent. width/height are recorded for
  // logging only — image dimensions are taken from each frame's DLTensor.
  void create_camera_publisher(
    const std::string & topic,
    std::uint32_t width,
    std::uint32_t height,
    bool is_depth);

  // Subscriber-count gate (matches render_node.cpp:313 pattern).
  // Returns false when the publisher does not exist or has zero subscribers.
  bool has_image_subscribers(const std::string & topic) const;

  // Hot-path publishers. Called from the OVRtx render thread.
  // Image data layout is the OVRtx convention used by render_node.cpp:
  //   * RGB:   shape (H, W, 4), uint8, encoding "rgba8", step = W*4
  //   * depth: shape (H, W),    float32, encoding "32FC1", step = W*4
  // The pointer must remain valid for the duration of this call; the
  // RealtimePublisher copies into its message buffer.
  void publish_camera_image_rgba8(
    double sim_time,
    const std::string & topic,
    const std::string & frame_id,
    std::uint32_t height,
    std::uint32_t width,
    const std::uint8_t * data);

  void publish_camera_image_depth32f(
    double sim_time,
    const std::string & topic,
    const std::string & frame_id,
    std::uint32_t height,
    std::uint32_t width,
    const float * data);

  // Camera info (intrinsics). 9-vector K (row-major), 12-vector P,
  // 9-vector R, distortion vector D (typically 5 plumb_bob coefs).
  void publish_camera_info(
    double sim_time,
    const std::string & topic,
    const std::string & frame_id,
    std::uint32_t height,
    std::uint32_t width,
    const std::array<double, 9> & K,
    const std::array<double, 12> & P,
    const std::array<double, 9> & R,
    const std::vector<double> & D,
    const std::string & distortion_model);

  // Free-camera pose subscription used by the inline OVRtx visualizer.
  // The subscription runs on the existing executor thread (rclcpp); the
  // callback converts the incoming ``geometry_msgs::PoseStamped`` directly
  // into a 4x4 row-major ``mat44d`` (USD row-vector convention, translation
  // in the last row) using the same formula as
  // ``genie_sim_render::on_free_cam_pose`` (render_node.cpp:531-540).  The
  // OVRtx render thread calls ``take_free_cam_xform(...)`` once per frame
  // to drain the slot.  Idempotent — ``subscribe_camera_pose`` only creates
  // a subscription on first call.
  void subscribe_camera_pose(const std::string & topic);

  // Drain the latest free-cam transform.  Returns ``true`` and fills
  // ``out_mat44d`` (length 16, row-major float64) when a fresh pose has
  // arrived since the last call; returns ``false`` otherwise.  This
  // matches the layout the OVRtx ``omni:xform`` writer expects (USD
  // row-vector convention; same as ``ovrtx_xform_matrix44d_t.v[]`` used
  // by ``ovrtx_set_xform_mat`` in the cross-process renderer).
  bool take_free_cam_xform(double * out_mat44d);

  // Optional optical-zoom channel for renderers that explicitly opt in.
  // It is separate from camera pose so existing external renderers keep
  // their current behaviour. The latest finite value is latched until the
  // render thread drains it.
  void subscribe_camera_zoom(const std::string & topic);
  bool take_free_cam_zoom(double * out_zoom);

  // === Command path =========================================================
  RealtimeCommandBuffer & commands() {return cmd_store_;}

  // === Simulation lifecycle command path ====================================
  // ROS callbacks only enqueue requests.  The Python physics loop drains this
  // queue at a tick boundary, so reset/pause/resume never mutate backend state
  // concurrently with a solver step.
  std::vector<std::string> pop_lifecycle_commands();
  void set_lifecycle_state(uint8_t id, const std::string & label);
  lifecycle_msgs::msg::State lifecycle_state() const;

  // === Teleport path ========================================================
  //! One requested pose write. ``body`` is the incoming ``child_frame_id``;
  //! the quaternion is stored WXYZ, reordered out of the ROS message's
  //! x/y/z/w field order by the subscription callback so that exactly one
  //! convention crosses into Python (see ``TeleportRequest`` in
  //! ``genie_sim_engine_core/engine/lifecycle.py``).
  struct TeleportRequest
  {
    std::string body;
    double x = 0.0, y = 0.0, z = 0.0;
    double qw = 1.0, qx = 0.0, qy = 0.0, qz = 0.0;
  };

  //! Drain every teleport received since the last call. Same discipline as
  //! the lifecycle queue: the ROS callback only enqueues, and the physics
  //! loop applies at a tick boundary, so a pose write never races a solver
  //! step. Unlike the command buffer this queue does NOT latch — a teleport
  //! is a one-shot event, and re-applying the last one every tick would pin
  //! the body in place.
  std::vector<TeleportRequest> pop_teleports();

  bool ok() const;
  void request_shutdown();

private:
  // rclcpp owned init/shutdown wrapper to be safe across reload.
  static std::atomic<bool> rcl_inited_;

  NodeOptions opts_;
  std::shared_ptr<rclcpp::Context> ctx_;
  std::shared_ptr<rclcpp::Node> node_;
  std::vector<std::string> joint_names_;
  std::vector<std::string> body_frames_;
  // Parallel to body_frames_. true = robot link (skip on /tf_scene,
  // handled by RSP on /tf). false = scene body. Empty ⇒ every body
  // is a scene body (v1 fallback).
  std::vector<bool> body_is_robot_;

  // Publishers are wrapped in realtime_tools::RealtimePublisher so that the
  // RT physics tick only does try_lock + msg_ assignment + unlockAndPublish;
  // the actual DDS publish happens on a dedicated non-RT thread owned by
  // each RealtimePublisher. This bounds the publish-side latency seen by
  // the physics loop and keeps allocations off the hot path.
  using RTPubClock = realtime_tools::RealtimePublisher<rosgraph_msgs::msg::Clock>;
  using RTPubJoint = realtime_tools::RealtimePublisher<sensor_msgs::msg::JointState>;
  using RTPubOdom = realtime_tools::RealtimePublisher<nav_msgs::msg::Odometry>;
  using RTPubTF = realtime_tools::RealtimePublisher<tf2_msgs::msg::TFMessage>;
  using RTPubCamInfo = realtime_tools::RealtimePublisher<sensor_msgs::msg::CameraInfo>;
  using RTPubFloat32 = realtime_tools::RealtimePublisher<std_msgs::msg::Float32>;
  using RTPubContacts = realtime_tools::RealtimePublisher<gazebo_msgs::msg::ContactsState>;

  //! Shared body-pose message builder behind ``publish_body_tf_scene`` and
  //! ``publish_tf_render``. Fills ``pub->msg_`` in place and publishes; the
  //! two differ only in ``skip_robot``. Returns without touching the wire if
  //! ``pub`` is null (the mode this build did not select) or if the realtime
  //! trylock fails.
  void publish_body_tf_on(
    const std::shared_ptr<RTPubTF> & pub,
    const double * body_xyzwxyz,
    std::size_t n_bodies,
    bool skip_robot);

  std::shared_ptr<RTPubClock> pub_clock_;
  std::shared_ptr<RTPubJoint> pub_joint_;
  //! v2 protocol: parallel publisher for the articulated-scene DoF
  //! stream. Same message type as ``pub_joint_``; distinct topic and
  //! variable ``name[]`` (MJCF joint names, per-scene set).
  std::shared_ptr<RTPubJoint> pub_joint_scene_;
  std::shared_ptr<RTPubOdom> pub_odom_;
  std::shared_ptr<RTPubTF> pub_tf_;
  //! Body-pose publishers. Exactly one is non-null, chosen by
  //! ``opts_.use_tf_render`` at construction.
  std::shared_ptr<RTPubTF> pub_tf_scene_;
  std::shared_ptr<RTPubTF> pub_tf_render_;
  std::shared_ptr<RTPubContacts> pub_contact_states_;
  std::shared_ptr<RTPubFloat32> pub_rtf_;

  // Camera publisher map. Built lazily by create_camera_publisher().
  // Reads (publish + has_image_subscribers) from the OVRtx render thread are
  // protected by cam_pubs_mu_; writes happen at startup (single-threaded).
  //
  // We use ``image_transport::Publisher`` for the image side so that
  // ``<topic>/image_raw/compressed`` (and ``compressedDepth`` for depth
  // topics) are advertised automatically — same pattern as
  // genie_sim_render's RosImagePublisherPlugin.  Subscriber-count gating
  // uses ``image_transport::Publisher::getNumSubscribers()`` which sums
  // across all advertised transports (raw + compressed + theora + …),
  // so a subscriber on /compressed alone still keeps the gate open.
  //
  // CameraInfo retains the realtime_tools fast-path because it is small
  // and emitted alongside every image at the same rate.
  mutable std::mutex cam_pubs_mu_;
  std::unordered_map<std::string, image_transport::Publisher> pub_images_it_;
  std::unordered_map<std::string, std::shared_ptr<RTPubCamInfo>> pub_caminfos_;

  // Atomic stamp for cross-thread sim-time reads from the OVRtx thread.
  // Written by publish_clock(); read by last_sim_time().
  std::atomic<double> last_sim_time_{0.0};

  std::shared_ptr<tf2_ros::StaticTransformBroadcaster> static_tf_broadcaster_;

  struct LifecycleQueue
  {
    mutable std::mutex mutex;
    std::deque<std::string> commands;
  } lifecycle_queue_;
  std::atomic<uint8_t> lifecycle_state_id_{lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED};
  mutable std::mutex lifecycle_state_mu_;
  std::string lifecycle_state_label_ = "unconfigured";

  rclcpp::Service<std_srvs::srv::Empty>::SharedPtr srv_pause_physics_;
  rclcpp::Service<std_srvs::srv::Empty>::SharedPtr srv_unpause_physics_;
  rclcpp::Service<std_srvs::srv::Empty>::SharedPtr srv_reset_simulation_;
  rclcpp::Service<std_srvs::srv::Empty>::SharedPtr srv_reset_world_;
  rclcpp::Service<std_srvs::srv::Empty>::SharedPtr srv_pause_;
  rclcpp::Service<std_srvs::srv::Empty>::SharedPtr srv_resume_;
  rclcpp::Service<std_srvs::srv::Empty>::SharedPtr srv_reset_;
  rclcpp::Service<lifecycle_msgs::srv::ChangeState>::SharedPtr srv_change_state_;
  rclcpp::Service<lifecycle_msgs::srv::GetState>::SharedPtr srv_get_state_;

  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr sub_joint_cmd_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr sub_cmd_4ws_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr sub_cam_pose_;
  rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr sub_cam_zoom_;
  rclcpp::Subscription<tf2_msgs::msg::TFMessage>::SharedPtr sub_teleport_;

  //! Producer: the teleport subscription on the executor thread. Consumer:
  //! the physics loop via ``pop_teleports``. Bounded so a client that spams
  //! the topic while the loop is blocked cannot grow it without limit; the
  //! oldest entries are dropped, with a throttled warning, because the
  //! newest pose is the one the client wants.
  struct TeleportQueue
  {
    mutable std::mutex mutex;
    std::deque<TeleportRequest> requests;
    std::size_t dropped = 0;
  } teleport_queue_;
  static constexpr std::size_t kMaxPendingTeleports = 512;

  // Free-cam mat44d latch.  The subscription callback (executor thread)
  // converts the incoming PoseStamped into a row-major 4x4 double matrix
  // (USD row-vector convention) and stores it here.  The OVRtx render
  // thread calls take_free_cam_xform() once per frame to drain.
  // ``cam_pose_dirty_`` lets the reader skip the mutex on the common
  // no-update path.
  mutable std::mutex cam_pose_mu_;
  std::atomic<bool> cam_pose_dirty_{false};
  std::array<double, 16> cam_pose_mat44d_{};

  std::atomic<bool> cam_zoom_dirty_{false};
  std::atomic<double> cam_zoom_{1.0};

  std::unique_ptr<rclcpp::executors::MultiThreadedExecutor> executor_;
  std::thread spin_thread_;

  RealtimeCommandBuffer cmd_store_;

  void enqueue_lifecycle(const std::string & command);
};

}  // namespace gsi

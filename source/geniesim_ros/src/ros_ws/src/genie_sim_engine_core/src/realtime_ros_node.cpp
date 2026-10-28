// Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
// Author: Genie Sim Team
// Proprietary and confidential.

#include "realtime_ros_node.hpp"

#include <chrono>
#include <cmath>
#include <cstring>
#include <utility>

namespace gsi
{

std::atomic<bool> RosBridge::rcl_inited_{false};

namespace
{

inline builtin_interfaces::msg::Time stamp_from_seconds(double t)
{
  builtin_interfaces::msg::Time s;
  if (t < 0.0) {t = 0.0;}
  s.sec = static_cast<int32_t>(t);
  double frac = t - static_cast<double>(s.sec);
  s.nanosec = static_cast<uint32_t>(frac * 1e9);
  return s;
}

}  // namespace

RosBridge::RosBridge(int argc, char ** argv, const NodeOptions & opts)
: opts_(opts)
{
  ctx_ = std::make_shared<rclcpp::Context>();
  rclcpp::InitOptions init_opts;
  ctx_->init(argc, argv, init_opts);

  rclcpp::NodeOptions node_opts;
  node_opts.context(ctx_);
  node_opts.use_intra_process_comms(false);

  node_ = std::make_shared<rclcpp::Node>(opts_.node_name, opts_.namespace_, node_opts);

  rclcpp::QoS qos_default(rclcpp::KeepLast(10));
  rclcpp::QoS qos_clock(rclcpp::KeepLast(10));
  const rclcpp::QoS qos_joint_states = rclcpp::SensorDataQoS();
  // /tf publisher must match tf2_ros::DynamicBroadcasterQoS — depth 100,
  // Reliable, Volatile. RViz's MoveIt MotionPlanning display + every other
  // tf2 consumer subscribes with depth 100 KEEP_LAST under Reliable QoS.
  // Publishing at 200 Hz with depth 10 only buffers ~50 ms; any momentary
  // DDS stall makes the publisher's reliable queue overflow and downstream
  // subscribers stop seeing fresh frames — the symptom is "RViz Frames list
  // only shows /tf_static frames" even though `ros2 topic echo /tf` works.
  const rclcpp::QoS qos_tf = rclcpp::QoS(rclcpp::KeepLast(100));

  pub_clock_ = std::make_shared<RTPubClock>(
    node_->create_publisher<rosgraph_msgs::msg::Clock>(opts_.topic_clock, qos_clock));
  pub_joint_ = std::make_shared<RTPubJoint>(
    node_->create_publisher<sensor_msgs::msg::JointState>(
      opts_.topic_joint_states, qos_joint_states));
  pub_joint_scene_ = std::make_shared<RTPubJoint>(
    node_->create_publisher<sensor_msgs::msg::JointState>(
      opts_.topic_joint_states_scene, qos_joint_states));
  pub_odom_ = std::make_shared<RTPubOdom>(
    node_->create_publisher<nav_msgs::msg::Odometry>(opts_.topic_odom, qos_default));
  pub_tf_ = std::make_shared<RTPubTF>(
    node_->create_publisher<tf2_msgs::msg::TFMessage>(opts_.topic_tf, qos_tf));
  // Body poses go on exactly ONE topic, picked here. Creating only the
  // selected publisher matters for diagnosis: an advertised-but-never-written
  // topic still shows up in `ros2 topic list`, so a phantom /tf_scene would
  // make "is anyone publishing?" unanswerable without `ros2 topic info`.
  if (opts_.use_tf_render) {
    pub_tf_render_ = std::make_shared<RTPubTF>(
      node_->create_publisher<tf2_msgs::msg::TFMessage>(opts_.topic_tf_render, qos_tf));
  } else {
    pub_tf_scene_ = std::make_shared<RTPubTF>(
      node_->create_publisher<tf2_msgs::msg::TFMessage>(opts_.topic_tf_scene, qos_tf));
  }
  // ``/contact_states`` uses SensorDataQoS (best-effort, depth 10) so the
  // physics-thread try_lock+publish never blocks on a slow consumer. Contact
  // topology can churn frame-to-frame (grasp make/break) — a reliable queue
  // would compound backpressure during a make-break burst; benchmark
  // consumers just want fresh state, dropped frames are recoverable.
  pub_contact_states_ = std::make_shared<RTPubContacts>(
    node_->create_publisher<gazebo_msgs::msg::ContactsState>(
      opts_.topic_contact_states, qos_joint_states));
  pub_rtf_ = std::make_shared<RTPubFloat32>(
    node_->create_publisher<std_msgs::msg::Float32>("/rtf", qos_default));

  // Publish a one-shot static `world -> map` transform so MoveIt (planning frame
  // = `world`) can resolve poses expressed in `map` / `odom`. The sim process
  // owns every world-rooted frame it produces.
  //
  // The `map -> odom` edge is intentionally NOT published here. It's owned by
  // the navigation / SLAM stack (real localization) when running, and by the
  // ``fake_slam`` dynamic publisher inside ``publish_odom`` (identity, every
  // tick) for sim-only setups — see ``opts_.fake_slam`` below.
  static_tf_broadcaster_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(node_);
  {
    geometry_msgs::msg::TransformStamped tf_w2m;
    tf_w2m.header.stamp = node_->now();
    tf_w2m.header.frame_id = opts_.world_frame;
    tf_w2m.child_frame_id = opts_.map_frame;
    tf_w2m.transform.rotation.w = 1.0;
    static_tf_broadcaster_->sendTransform(tf_w2m);
  }

  // /joint_command and /cmd_4ws are published by ros2_control hardware
  // interfaces (genie_sim_robot_interface) and mujoco_geniesim's command_controller
  // with rclcpp::SensorDataQoS() (BEST_EFFORT). Subscriptions must use the
  // same reliability or the QoS handshake fails silently and no messages are
  // delivered. See genie_sim_control/genie_sim_robot_interface.cpp:128
  // and mujoco_geniesim/command_controller.cpp:49.
  sub_joint_cmd_ = node_->create_subscription<sensor_msgs::msg::JointState>(
    opts_.topic_joint_command, rclcpp::SensorDataQoS(),
    [this](sensor_msgs::msg::JointState::ConstSharedPtr msg) {
      std::vector<double> pos(msg->position.begin(), msg->position.end());
      std::vector<double> eff(msg->effort.begin(), msg->effort.end());
      cmd_store_.on_joint_command(msg->name, pos, eff);
    });

  sub_cmd_4ws_ = node_->create_subscription<sensor_msgs::msg::JointState>(
    opts_.topic_cmd_4ws, rclcpp::SensorDataQoS(),
    [this](sensor_msgs::msg::JointState::ConstSharedPtr msg) {
      std::vector<double> pos(msg->position.begin(), msg->position.end());
      std::vector<double> vel(msg->velocity.begin(), msg->velocity.end());
      cmd_store_.on_cmd_4ws(msg->name, pos, vel);
    });

  // Out-of-band pose writes.  RELIABLE + depth 10 (not SensorDataQoS like
  // the command topics above): a teleport is a discrete event a client sends
  // once and expects to land, whereas a dropped /joint_command is corrected
  // by the next one 10 ms later.  Same enqueue-only discipline as the
  // lifecycle services — the callback runs on the executor thread and must
  // not touch solver state.
  sub_teleport_ = node_->create_subscription<tf2_msgs::msg::TFMessage>(
    opts_.topic_teleport, 10,
    [this](tf2_msgs::msg::TFMessage::ConstSharedPtr msg) {
      std::lock_guard<std::mutex> lock(teleport_queue_.mutex);
      for (const auto & tf : msg->transforms) {
        if (tf.child_frame_id.empty()) {
          continue;              // nothing to address; silently useless
        }
        if (teleport_queue_.requests.size() >= kMaxPendingTeleports) {
          teleport_queue_.requests.pop_front();
          ++teleport_queue_.dropped;
          continue;
        }
        TeleportRequest req;
        req.body = tf.child_frame_id;
        req.x = tf.transform.translation.x;
        req.y = tf.transform.translation.y;
        req.z = tf.transform.translation.z;
        // ROS geometry_msgs/Quaternion is x,y,z,w; this struct is WXYZ.
        // Reorder HERE so Python only ever sees one convention.
        req.qw = tf.transform.rotation.w;
        req.qx = tf.transform.rotation.x;
        req.qy = tf.transform.rotation.y;
        req.qz = tf.transform.rotation.z;
        teleport_queue_.requests.push_back(std::move(req));
      }
    });

  // Gazebo-compatible simulation controls.  The callbacks deliberately do
  // not touch solver state: they enqueue a verb for the physics thread.
  auto empty_service = [this](const std::string & command) {
      return [this, command](
        std::shared_ptr<std_srvs::srv::Empty::Request>,
        std::shared_ptr<std_srvs::srv::Empty::Response>) {
               enqueue_lifecycle(command);
             };
    };
  srv_pause_physics_ = node_->create_service<std_srvs::srv::Empty>(
    "/pause_physics", empty_service("pause"));
  srv_unpause_physics_ = node_->create_service<std_srvs::srv::Empty>(
    "/unpause_physics", empty_service("resume"));
  srv_reset_simulation_ = node_->create_service<std_srvs::srv::Empty>(
    "/reset_simulation", empty_service("reset"));
  srv_reset_world_ = node_->create_service<std_srvs::srv::Empty>(
    "/reset_world", empty_service("reset"));
  srv_pause_ = node_->create_service<std_srvs::srv::Empty>(
    "~/pause", empty_service("pause"));
  srv_resume_ = node_->create_service<std_srvs::srv::Empty>(
    "~/resume", empty_service("resume"));
  srv_reset_ = node_->create_service<std_srvs::srv::Empty>(
    "~/reset", empty_service("reset"));

  srv_change_state_ = node_->create_service<lifecycle_msgs::srv::ChangeState>(
    "~/change_state",
    [this](
      const std::shared_ptr<lifecycle_msgs::srv::ChangeState::Request> request,
      std::shared_ptr<lifecycle_msgs::srv::ChangeState::Response> response) {
      response->success = false;
      switch (request->transition.id) {
        case lifecycle_msgs::msg::Transition::TRANSITION_CONFIGURE:
          enqueue_lifecycle("configure");
          response->success = true;
          break;
        case lifecycle_msgs::msg::Transition::TRANSITION_ACTIVATE:
          enqueue_lifecycle("resume");
          response->success = true;
          break;
        case lifecycle_msgs::msg::Transition::TRANSITION_DEACTIVATE:
          enqueue_lifecycle("pause");
          response->success = true;
          break;
        case lifecycle_msgs::msg::Transition::TRANSITION_ACTIVE_SHUTDOWN:
        case lifecycle_msgs::msg::Transition::TRANSITION_INACTIVE_SHUTDOWN:
        case lifecycle_msgs::msg::Transition::TRANSITION_UNCONFIGURED_SHUTDOWN:
          enqueue_lifecycle("shutdown");
          response->success = true;
          break;
        default:
          // Cleanup/reload/destroy require rebuilding scene and solver state;
          // they are intentionally process-level operations for v0.1.
          break;
      }
    });
  srv_get_state_ = node_->create_service<lifecycle_msgs::srv::GetState>(
    "~/get_state",
    [this](
      const std::shared_ptr<lifecycle_msgs::srv::GetState::Request>,
      std::shared_ptr<lifecycle_msgs::srv::GetState::Response> response) {
      response->current_state = lifecycle_state();
    });

  rclcpp::ExecutorOptions exec_opts;
  exec_opts.context = ctx_;
  int n = std::max(opts_.executor_threads, 1);
  executor_ = std::make_unique<rclcpp::executors::MultiThreadedExecutor>(
    exec_opts, static_cast<size_t>(n));
  executor_->add_node(node_);

  spin_thread_ = std::thread(
    [this]() {
      try {
        executor_->spin();
      } catch (const std::exception & e) {
        RCLCPP_ERROR(node_->get_logger(), "executor spin terminated: %s", e.what());
      }
    });
}

RosBridge::~RosBridge()
{
  request_shutdown();
}

void RosBridge::request_shutdown()
{
  if (executor_) {
    try {
      executor_->cancel();
    } catch (...) {
    }
  }
  if (spin_thread_.joinable()) {
    try {
      spin_thread_.join();
    } catch (...) {
    }
  }
  if (ctx_ && ctx_->is_valid()) {
    try {
      ctx_->shutdown("genie_sim_engine teardown");
    } catch (...) {
    }
  }
}

bool RosBridge::ok() const
{
  return ctx_ && ctx_->is_valid();
}

void RosBridge::set_topology(
  const std::vector<std::string> & joint_names,
  const std::vector<std::string> & body_frames,
  const std::vector<bool> & body_is_robot)
{
  joint_names_ = joint_names;
  body_frames_ = body_frames;
  body_is_robot_ = body_is_robot;
}

void RosBridge::enqueue_lifecycle(const std::string & command)
{
  std::lock_guard<std::mutex> lock(lifecycle_queue_.mutex);
  lifecycle_queue_.commands.push_back(command);
}

std::vector<std::string> RosBridge::pop_lifecycle_commands()
{
  std::vector<std::string> out;
  std::lock_guard<std::mutex> lock(lifecycle_queue_.mutex);
  out.assign(lifecycle_queue_.commands.begin(), lifecycle_queue_.commands.end());
  lifecycle_queue_.commands.clear();
  return out;
}

std::vector<RosBridge::TeleportRequest> RosBridge::pop_teleports()
{
  std::vector<TeleportRequest> out;
  std::size_t dropped = 0;
  {
    std::lock_guard<std::mutex> lock(teleport_queue_.mutex);
    out.assign(teleport_queue_.requests.begin(), teleport_queue_.requests.end());
    teleport_queue_.requests.clear();
    dropped = teleport_queue_.dropped;
    teleport_queue_.dropped = 0;
  }
  if (dropped > 0) {
    RCLCPP_WARN(
      node_->get_logger(),
      "teleport queue overflowed: dropped %zu oldest request(s) — the loop was "
      "not draining fast enough, or a client is publishing faster than the "
      "physics rate", dropped);
  }
  return out;
}

void RosBridge::set_lifecycle_state(uint8_t id, const std::string & label)
{
  lifecycle_state_id_.store(id, std::memory_order_release);
  std::lock_guard<std::mutex> lock(lifecycle_state_mu_);
  lifecycle_state_label_ = label;
}

lifecycle_msgs::msg::State RosBridge::lifecycle_state() const
{
  lifecycle_msgs::msg::State state;
  state.id = lifecycle_state_id_.load(std::memory_order_acquire);
  std::lock_guard<std::mutex> lock(lifecycle_state_mu_);
  state.label = lifecycle_state_label_;
  return state;
}

void RosBridge::publish_clock(double sim_time)
{
  // Non-blocking publish: if the dedicated publishing thread is still
  // draining the previous message we drop this tick — /clock is a
  // periodic stream and the next tick recovers.
  last_sim_time_.store(sim_time, std::memory_order_relaxed);
  if (pub_clock_->trylock()) {
    pub_clock_->msg_.clock = stamp_from_seconds(sim_time);
    pub_clock_->unlockAndPublish();
  }
}

void RosBridge::publish_rtf(float rtf)
{
  if (pub_rtf_->trylock()) {
    pub_rtf_->msg_.data = rtf;
    pub_rtf_->unlockAndPublish();
  }
}

void RosBridge::publish_joint_states(
  double sim_time, const double * joint_pos, const double * joint_vel,
  const double * joint_eff, const std::string & frame_id, std::size_t n)
{
  (void)sim_time;
  if (n != joint_names_.size()) {return;}
  if (!pub_joint_->trylock()) {return;}
  auto & msg = pub_joint_->msg_;
  msg.header.stamp = node_->now();
  // reset_epoch causal credential — see the declaration comment in
  // realtime_ros_node.hpp. Set verbatim from the caller; empty string is a
  // legitimate value (backend without a reset epoch).
  msg.header.frame_id = frame_id;
  msg.name = joint_names_;
  msg.position.resize(n);
  msg.velocity.resize(n);
  // Effort is optional — a null pointer means the backend has no
  // torque feed and downstream consumers should ignore the field.
  // Publishing an empty vector matches how sensor_msgs/JointState is
  // conventionally used to signal "not measured" (as opposed to
  // "measured, zero"). When present, populate one-to-one with the
  // pos/vel arrays.
  if (joint_eff != nullptr) {
    msg.effort.resize(n);
  } else {
    msg.effort.clear();
  }
  for (std::size_t i = 0; i < n; ++i) {
    msg.position[i] = joint_pos ? joint_pos[i] : 0.0;
    msg.velocity[i] = joint_vel ? joint_vel[i] : 0.0;
    if (joint_eff != nullptr) {
      msg.effort[i] = joint_eff[i];
    }
  }
  pub_joint_->unlockAndPublish();
}

void RosBridge::publish_joint_states_scene(
  double sim_time, const std::vector<std::string> & names,
  const double * joint_pos, const double * joint_vel,
  const double * joint_eff, std::size_t n)
{
  (void)sim_time;
  // Empty scene DoF set = nothing to publish (common case: no
  // articulated scene objects). Skip so consumers can distinguish
  // "no articulated scene DoFs" from "publisher stalled".
  if (n == 0 || names.size() != n) {return;}
  if (!pub_joint_scene_->trylock()) {return;}
  auto & msg = pub_joint_scene_->msg_;
  msg.header.stamp = node_->now();
  msg.name = names;
  msg.position.resize(n);
  msg.velocity.resize(n);
  // Effort optional (same convention as ``publish_joint_states``):
  // passive scene joints (spring-loaded drawer, contact-only door)
  // still carry constraint / contact torque that benchmark checkers
  // threshold on, so backends that can measure it should pass it.
  // Backends without an effort feed pass nullptr → empty vector on
  // the wire.
  if (joint_eff != nullptr) {
    msg.effort.resize(n);
  } else {
    msg.effort.clear();
  }
  for (std::size_t i = 0; i < n; ++i) {
    msg.position[i] = joint_pos ? joint_pos[i] : 0.0;
    msg.velocity[i] = joint_vel ? joint_vel[i] : 0.0;
    if (joint_eff != nullptr) {
      msg.effort[i] = joint_eff[i];
    }
  }
  pub_joint_scene_->unlockAndPublish();
}

void RosBridge::publish_body_tf_on(
  const std::shared_ptr<RTPubTF> & pub, const double * body_xyzwxyz,
  std::size_t n_bodies, bool skip_robot)
{
  // Null when this build selected the other mode — the caller is the wrong
  // entry point for how the bridge was constructed. Silence is correct here:
  // the run loop calls one of the two every tick and must not be made to
  // care which.
  if (!pub) {return;}
  if (n_bodies != body_frames_.size()) {return;}
  if (!pub->trylock()) {return;}
  auto & tfs = pub->msg_;
  tfs.transforms.clear();
  tfs.transforms.reserve(n_bodies);
  // One stamp for the whole array: every transform in this message describes
  // the same physics tick, so a consumer differencing two of them never
  // straddles a step boundary.
  auto stamp = node_->now();
  const bool have_mask = skip_robot && body_is_robot_.size() == n_bodies;
  for (std::size_t i = 0; i < n_bodies; ++i) {
    if (have_mask && body_is_robot_[i]) {continue;}
    geometry_msgs::msg::TransformStamped t;
    t.header.stamp = stamp;
    t.header.frame_id = opts_.map_frame;
    t.child_frame_id = body_frames_[i];
    const double * p = body_xyzwxyz + i * 7;
    t.transform.translation.x = p[0];
    t.transform.translation.y = p[1];
    t.transform.translation.z = p[2];
    t.transform.rotation.w = p[3];
    t.transform.rotation.x = p[4];
    t.transform.rotation.y = p[5];
    t.transform.rotation.z = p[6];
    tfs.transforms.push_back(std::move(t));
  }
  pub->unlockAndPublish();
}

void RosBridge::publish_body_tf_scene(
  double sim_time, const double * body_xyzwxyz, std::size_t n_bodies)
{
  (void)sim_time;
  // skip_robot = true: v2 protocol keeps /tf_scene single-purpose (map-posed
  // scene bodies). Robot links reach consumers via /tf (RSP fan-out over URDF
  // FK); emitting them here too would double-write the robot xforms on the
  // OVRTX stage.
  publish_body_tf_on(pub_tf_scene_, body_xyzwxyz, n_bodies, /*skip_robot=*/ true);
}

void RosBridge::publish_tf_render(
  double sim_time, const double * body_xyzwxyz, std::size_t n_bodies)
{
  (void)sim_time;
  // skip_robot = false, unconditionally. /tf_render exists precisely because
  // there is no robot_state_publisher in this deployment, so the robot links
  // have no other route to a consumer. body_is_robot_ may well be populated
  // and correct — it is simply not this topic's business.
  publish_body_tf_on(pub_tf_render_, body_xyzwxyz, n_bodies, /*skip_robot=*/ false);
}

void RosBridge::publish_contact_states(
  double sim_time,
  const std::vector<std::string> & body1_names,
  const std::vector<std::string> & body2_names,
  const double * positions,
  const double * normals,
  const double * depths,
  const double * wrenches,
  std::size_t n_points)
{
  (void)sim_time;
  if (body1_names.size() != n_points || body2_names.size() != n_points) {return;}
  if (!pub_contact_states_->trylock()) {return;}

  auto & msg = pub_contact_states_->msg_;
  msg.header.stamp = node_->now();
  msg.header.frame_id = opts_.map_frame;
  msg.states.clear();
  // Reserve the upper bound of unique pairs so subsequent emplace_back
  // never reallocates — the pair-index map below stores pointers into
  // the msg.states entries as its hash keys and would dangle on a
  // grow. n_points is an upper bound (unique pairs ≤ n_points).
  msg.states.reserve(n_points);

  // Group per-point contacts into one ContactState per unique
  // (body1, body2) pair. The pair key is canonicalised
  // lexicographically so that (A, B) and (B, A) coalesce — MuJoCo
  // reports (geom1, geom2) in solver order which can flip between
  // frames for the same physical contact, and downstream grasp
  // checkers threshold on total force per touching pair, not per
  // ordered emission.
  struct PairKey
  {
    const std::string * a;
    const std::string * b;
    bool operator==(const PairKey & other) const noexcept
    {
      return *a == *other.a && *b == *other.b;
    }
  };
  struct PairKeyHash
  {
    std::size_t operator()(const PairKey & k) const noexcept
    {
      return std::hash<std::string>{}(*k.a) ^ (std::hash<std::string>{}(*k.b) << 1);
    }
  };
  std::unordered_map<PairKey, std::size_t, PairKeyHash> pair_index;
  pair_index.reserve(n_points);

  for (std::size_t i = 0; i < n_points; ++i) {
    const std::string & raw_a = body1_names[i];
    const std::string & raw_b = body2_names[i];
    const std::string & a = (raw_a <= raw_b) ? raw_a : raw_b;
    const std::string & b = (raw_a <= raw_b) ? raw_b : raw_a;

    PairKey key{&a, &b};
    auto it = pair_index.find(key);
    std::size_t idx;
    if (it == pair_index.end()) {
      idx = msg.states.size();
      msg.states.emplace_back();
      auto & s = msg.states.back();
      s.collision1_name = a;
      s.collision2_name = b;
      // Re-anchor the key to the freshly emplaced strings so the
      // map stays valid after subsequent emplaces reallocate.
      pair_index.emplace(PairKey{&s.collision1_name, &s.collision2_name}, idx);
    } else {
      idx = it->second;
    }

    auto & s = msg.states[idx];

    geometry_msgs::msg::Vector3 pos_v;
    if (positions != nullptr) {
      pos_v.x = positions[i * 3 + 0];
      pos_v.y = positions[i * 3 + 1];
      pos_v.z = positions[i * 3 + 2];
    }
    s.contact_positions.push_back(pos_v);

    geometry_msgs::msg::Vector3 nrm_v;
    if (normals != nullptr) {
      nrm_v.x = normals[i * 3 + 0];
      nrm_v.y = normals[i * 3 + 1];
      nrm_v.z = normals[i * 3 + 2];
    }
    s.contact_normals.push_back(nrm_v);

    if (depths != nullptr) {
      s.depths.push_back(depths[i]);
    } else {
      s.depths.push_back(0.0);
    }

    geometry_msgs::msg::Wrench w;
    if (wrenches != nullptr) {
      w.force.x = wrenches[i * 6 + 0];
      w.force.y = wrenches[i * 6 + 1];
      w.force.z = wrenches[i * 6 + 2];
      w.torque.x = wrenches[i * 6 + 3];
      w.torque.y = wrenches[i * 6 + 4];
      w.torque.z = wrenches[i * 6 + 5];
    }
    s.wrenches.push_back(w);

    // Sum into ``total_wrench``. Backends publish in their native
    // frame (MuJoCo: per-contact local frame); the sum is only
    // physically meaningful when the caller has already rotated
    // every per-point wrench into a shared frame. Left as raw sum
    // here — consumers that need a world-frame total do the rotate
    // themselves against ``contact_normals[]``.
    s.total_wrench.force.x += w.force.x;
    s.total_wrench.force.y += w.force.y;
    s.total_wrench.force.z += w.force.z;
    s.total_wrench.torque.x += w.torque.x;
    s.total_wrench.torque.y += w.torque.y;
    s.total_wrench.torque.z += w.torque.z;
  }

  pub_contact_states_->unlockAndPublish();
}

void RosBridge::publish_odom(
  double sim_time, const double * base_xyzwxyz,
  const double * base_twist)
{
  (void)sim_time;
  if (!base_xyzwxyz) {return;}
  auto stamp = node_->now();

  // ``base_xyzwxyz`` carries base_link's full world pose (px,py,pz,qw,qx,qy,qz).
  // When ``opts_.base_frame`` is "base_footprint" (mobile-base scenes), we
  // split this into two TF edges:
  //
  //   odom -> base_footprint        : ground-projected pose
  //                                   (px, py, 0, yaw_only_quat).  Anchors
  //                                   the SRDF planar virtual_joint -- pure
  //                                   3-DoF (x, y, theta), no height/tilt to
  //                                   silently drop.
  //   base_footprint -> base_link   : the residual.  Translation = (0, 0, pz),
  //                                   rotation = the roll/pitch component
  //                                   left over after factoring yaw out of
  //                                   the original quat.  Dynamic, published
  //                                   each tick so chassis bobbing /
  //                                   tilt under load shows up immediately.
  //
  // Yaw extraction from quat (qw,qx,qy,qz):
  //     yaw = atan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
  // This is the standard Z-Y-X Euler-angles yaw, ignoring roll & pitch.
  // The yaw-only quat is then (cos(yaw/2), 0, 0, sin(yaw/2)) and the
  // residual (roll/pitch) quat is q_full * inverse(q_yaw).
  const double pz = base_xyzwxyz[2];
  const double qw = base_xyzwxyz[3];
  const double qx = base_xyzwxyz[4];
  const double qy = base_xyzwxyz[5];
  const double qz = base_xyzwxyz[6];
  const bool split_footprint = (opts_.base_frame == "base_footprint");

  // Ground-projected (footprint) pose: x, y kept; z=0; yaw-only orientation.
  double fp_qw = qw, fp_qx = 0.0, fp_qy = 0.0, fp_qz = qz;
  double fp_pz = pz;
  // base_footprint -> base_link residual transform (only needed when split).
  double res_qw = 1.0, res_qx = 0.0, res_qy = 0.0, res_qz = 0.0;
  if (split_footprint) {
    const double yaw = std::atan2(
      2.0 * (qw * qz + qx * qy),
      1.0 - 2.0 * (qy * qy + qz * qz));
    const double cy = std::cos(0.5 * yaw);
    const double sy = std::sin(0.5 * yaw);
    fp_qw = cy; fp_qx = 0.0; fp_qy = 0.0; fp_qz = sy;
    fp_pz = 0.0;
    // residual = inv(q_yaw) * q_full  (rotates yaw out, leaves roll/pitch).
    // inv(q_yaw) = (cy, 0, 0, -sy).  Hamilton product:
    //   (a,b,c,d) * (w,x,y,z) =
    //     ( aw - bx - cy - dz,
    //       ax + bw + cz - dy,
    //       ay - bz + cw + dx,
    //       az + by - cx + dw )
    const double a = cy, b = 0.0, c = 0.0, d = -sy;
    res_qw = a * qw - b * qx - c * qy - d * qz;
    res_qx = a * qx + b * qw + c * qz - d * qy;
    res_qy = a * qy - b * qz + c * qw + d * qx;
    res_qz = a * qz + b * qy - c * qx + d * qw;
  }

  if (pub_odom_->trylock()) {
    auto & odom = pub_odom_->msg_;
    odom.header.stamp = stamp;
    odom.header.frame_id = opts_.odom_frame;
    odom.child_frame_id = opts_.base_frame;
    odom.pose.pose.position.x = base_xyzwxyz[0];
    odom.pose.pose.position.y = base_xyzwxyz[1];
    odom.pose.pose.position.z = fp_pz;
    odom.pose.pose.orientation.w = fp_qw;
    odom.pose.pose.orientation.x = fp_qx;
    odom.pose.pose.orientation.y = fp_qy;
    odom.pose.pose.orientation.z = fp_qz;
    if (base_twist) {
      odom.twist.twist.linear.x = base_twist[0];
      odom.twist.twist.linear.y = base_twist[1];
      odom.twist.twist.linear.z = base_twist[2];
      odom.twist.twist.angular.x = base_twist[3];
      odom.twist.twist.angular.y = base_twist[4];
      odom.twist.twist.angular.z = base_twist[5];
    } else {
      odom.twist.twist.linear.x = 0.0;
      odom.twist.twist.linear.y = 0.0;
      odom.twist.twist.linear.z = 0.0;
      odom.twist.twist.angular.x = 0.0;
      odom.twist.twist.angular.y = 0.0;
      odom.twist.twist.angular.z = 0.0;
    }
    pub_odom_->unlockAndPublish();
  }

  if (pub_tf_->trylock()) {
    auto & tfs = pub_tf_->msg_;
    tfs.transforms.clear();

    // odom -> base_frame (ground-projected when split, else base_link verbatim).
    geometry_msgs::msg::TransformStamped t;
    t.header.stamp = stamp;
    t.header.frame_id = opts_.odom_frame;
    t.child_frame_id = opts_.base_frame;
    t.transform.translation.x = base_xyzwxyz[0];
    t.transform.translation.y = base_xyzwxyz[1];
    t.transform.translation.z = fp_pz;
    t.transform.rotation.w = fp_qw;
    t.transform.rotation.x = fp_qx;
    t.transform.rotation.y = fp_qy;
    t.transform.rotation.z = fp_qz;
    tfs.transforms.push_back(std::move(t));

    // base_footprint -> base_link (split path only: carries residual height + tilt).
    if (split_footprint) {
      geometry_msgs::msg::TransformStamped r;
      r.header.stamp = stamp;
      r.header.frame_id = opts_.base_frame;       // "base_footprint"
      r.child_frame_id = "base_link";
      r.transform.translation.x = 0.0;
      r.transform.translation.y = 0.0;
      r.transform.translation.z = pz;
      r.transform.rotation.w = res_qw;
      r.transform.rotation.x = res_qx;
      r.transform.rotation.y = res_qy;
      r.transform.rotation.z = res_qz;
      tfs.transforms.push_back(std::move(r));
    }

    if (opts_.fake_slam) {
      geometry_msgs::msg::TransformStamped m;
      m.header.stamp = stamp;
      m.header.frame_id = opts_.map_frame;
      m.child_frame_id = opts_.odom_frame;
      m.transform.rotation.w = 1.0;
      tfs.transforms.push_back(std::move(m));
    }
    pub_tf_->unlockAndPublish();
  }
}

// ===========================================================================
// Inline OVRtx visualizer support
// ===========================================================================

double RosBridge::last_sim_time() const
{
  return last_sim_time_.load(std::memory_order_relaxed);
}

void RosBridge::create_camera_publisher(
  const std::string & topic,
  std::uint32_t width,
  std::uint32_t height,
  bool is_depth)
{
  (void)width;
  (void)height;
  (void)is_depth;
  if (topic.empty()) {return;}

  std::lock_guard<std::mutex> lk(cam_pubs_mu_);
  if (pub_images_it_.find(topic) != pub_images_it_.end()) {
    return;  // already created
  }

  // Match render_node.cpp's plugin convention: image on <topic>/image_raw,
  // info on <topic>/camera_info.
  //
  // Use image_transport for the image side so /<topic>/image_raw/compressed
  // (and /compressedDepth for depth) are automatically advertised by
  // whatever image_transport plugins are installed (image_transport_plugins
  // pulls in compressed_image_transport + compressed_depth_image_transport,
  // and theora_image_transport when available).  This mirrors
  // genie_sim_render/src/plugins/ros_image_publisher_plugin.cpp:25.
  //
  // image_transport::create_publisher does not accept a custom QoS at
  // create time on humble — it uses the default reliable QoS which suits
  // RViz / rqt_image_view consumers.  The CameraInfo side keeps
  // SensorDataQoS via the realtime_tools wrapper to match what the cross-
  // process renderer's plugin emits and what the rest of the bridge uses.
  const std::string image_topic = topic + "/image_raw";
  const std::string info_topic = topic + "/camera_info";

  pub_images_it_[topic] = image_transport::create_publisher(node_.get(), image_topic);

  auto info_pub = node_->create_publisher<sensor_msgs::msg::CameraInfo>(
    info_topic, rclcpp::SensorDataQoS());
  pub_caminfos_[topic] = std::make_shared<RTPubCamInfo>(info_pub);
}

bool RosBridge::has_image_subscribers(const std::string & topic) const
{
  std::lock_guard<std::mutex> lk(cam_pubs_mu_);
  auto it = pub_images_it_.find(topic);
  if (it == pub_images_it_.end()) {return false;}
  // image_transport::Publisher::getNumSubscribers sums subscribers across
  // ALL advertised transports (raw + compressed + theora + ...), so a
  // /compressed-only consumer still keeps the gate open.
  return it->second.getNumSubscribers() > 0;
}

namespace
{

// Lookup helper. Caller holds ``cam_pubs_mu_`` only for the lookup; the
// publish itself uses the RealtimePublisher's own try_lock so the OVRtx
// thread is not gated on a global mutex during the hot path.
template<typename Map>
auto find_or_null(const Map & m, const std::string & key)
-> typename Map::mapped_type
{
  auto it = m.find(key);
  if (it == m.end()) {return {};}
  return it->second;
}

}  // namespace

void RosBridge::publish_camera_image_rgba8(
  double sim_time,
  const std::string & topic,
  const std::string & frame_id,
  std::uint32_t height,
  std::uint32_t width,
  const std::uint8_t * data)
{
  (void)sim_time;
  image_transport::Publisher pub;
  {
    std::lock_guard<std::mutex> lk(cam_pubs_mu_);
    auto it = pub_images_it_.find(topic);
    if (it == pub_images_it_.end()) {return;}
    pub = it->second;
  }
  if (!data) {return;}

  // image_transport handles the publish on its own thread and dispatches
  // to all loaded transport plugins (raw + compressed + theora …) from a
  // single publish() call — same fanout the cross-process renderer relies
  // on.  We allocate a unique_ptr so the message is moved into the
  // pub-side queue rather than copied per transport.
  //
  // Stamp is wall-clock (``node_->now()``), NOT ``sim_time``: RViz keeps
  // its TF buffer in wall time (use_sim_time=false project-wide), and
  // RViz's Camera display rejects images whose stamp can't be looked up
  // against the TF buffer.  ``/clock`` is published only as a debug
  // signal and intentionally not consumed by anything else.  joint_states
  // / body_tf / odom already follow this convention (see existing
  // ``(void)sim_time`` casts at lines 169, 187, 216).
  auto msg = std::make_unique<sensor_msgs::msg::Image>();
  msg->header.stamp = node_->now();
  msg->header.frame_id = frame_id;
  msg->height = height;
  msg->width = width;
  msg->encoding = "rgba8";
  msg->is_bigendian = 0;
  msg->step = width * 4;
  const std::size_t bytes = static_cast<std::size_t>(height) * msg->step;
  msg->data.resize(bytes);
  std::memcpy(msg->data.data(), data, bytes);
  pub.publish(std::move(msg));
}

void RosBridge::publish_camera_image_depth32f(
  double sim_time,
  const std::string & topic,
  const std::string & frame_id,
  std::uint32_t height,
  std::uint32_t width,
  const float * data)
{
  (void)sim_time;
  image_transport::Publisher pub;
  {
    std::lock_guard<std::mutex> lk(cam_pubs_mu_);
    auto it = pub_images_it_.find(topic);
    if (it == pub_images_it_.end()) {return;}
    pub = it->second;
  }
  if (!data) {return;}

  // Wall-clock stamp; see publish_camera_image_rgba8 for the rationale.
  auto msg = std::make_unique<sensor_msgs::msg::Image>();
  msg->header.stamp = node_->now();
  msg->header.frame_id = frame_id;
  msg->height = height;
  msg->width = width;
  msg->encoding = "32FC1";
  msg->is_bigendian = 0;
  msg->step = width * sizeof(float);
  const std::size_t bytes = static_cast<std::size_t>(height) * msg->step;
  msg->data.resize(bytes);
  std::memcpy(msg->data.data(), data, bytes);
  pub.publish(std::move(msg));
}

void RosBridge::publish_camera_info(
  double sim_time,
  const std::string & topic,
  const std::string & frame_id,
  std::uint32_t height,
  std::uint32_t width,
  const std::array<double, 9> & K,
  const std::array<double, 12> & P,
  const std::array<double, 9> & R,
  const std::vector<double> & D,
  const std::string & distortion_model)
{
  (void)sim_time;
  std::shared_ptr<RTPubCamInfo> pub;
  {
    std::lock_guard<std::mutex> lk(cam_pubs_mu_);
    pub = find_or_null(pub_caminfos_, topic);
  }
  if (!pub) {return;}
  if (!pub->trylock()) {return;}

  auto & msg = pub->msg_;
  // Wall-clock stamp; must match the paired image so RViz's Camera
  // display can sync them and look the camera frame up in TF (also
  // wall-clock).  See publish_camera_image_rgba8 for full rationale.
  msg.header.stamp = node_->now();
  msg.header.frame_id = frame_id;
  msg.height = height;
  msg.width = width;
  msg.distortion_model = distortion_model;
  msg.d = D;
  for (std::size_t i = 0; i < 9; ++i) {
    msg.k[i] = K[i];
  }
  for (std::size_t i = 0; i < 9; ++i) {
    msg.r[i] = R[i];
  }
  for (std::size_t i = 0; i < 12; ++i) {
    msg.p[i] = P[i];
  }
  pub->unlockAndPublish();
}

void RosBridge::subscribe_camera_pose(const std::string & topic)
{
  if (sub_cam_pose_) {return;}
  if (topic.empty()) {return;}
  // Use the same SensorDataQoS as the rest of the bridge subscriptions —
  // the RViz plugin (genie_sim_rviz_plugins/view_camera_pose_publisher_display.cpp:117)
  // publishes with a transient-local QoS, but a SensorData subscription
  // can still receive its samples (best-effort vs reliable mismatches
  // would only matter if both ends required reliable; we prefer fresh
  // poses and tolerate dropped ones).
  rclcpp::QoS qos(rclcpp::KeepLast(1));
  qos.reliable();
  qos.transient_local();
  sub_cam_pose_ = node_->create_subscription<geometry_msgs::msg::PoseStamped>(
    topic, qos,
    [this](geometry_msgs::msg::PoseStamped::ConstSharedPtr msg) {
      // Build the row-major mat44d in-place — matches
      // genie_sim_render/render_node.cpp:531-540 byte-for-byte so the
      // inline path produces the exact same OVRtx omni:xform write the
      // cross-process renderer would for the same RViz viewport. This
      // intentionally skips the Python boundary.
      const auto & pose = msg->pose;
      const double qx = pose.orientation.x;
      const double qy = pose.orientation.y;
      const double qz = pose.orientation.z;
      const double qw = pose.orientation.w;
      const double tx = pose.position.x;
      const double ty = pose.position.y;
      const double tz = pose.position.z;

      RCLCPP_INFO_ONCE(
        node_->get_logger(),
        "free_cam_pose first sample: pos=(%.3f, %.3f, %.3f) "
        "quat_xyzw=(%.3f, %.3f, %.3f, %.3f)",
        tx, ty, tz, qx, qy, qz, qw);

      const double xx = qx * qx, yy = qy * qy, zz = qz * qz;
      const double xy = qx * qy, xz = qx * qz, yz = qy * qz;
      const double wx = qw * qx, wy = qw * qy, wz = qw * qz;

      std::array<double, 16> m{};
      m[0] = 1.0 - 2.0 * (yy + zz); m[1] = 2.0 * (xy + wz);       m[2] = 2.0 * (xz - wy);
      m[3] = 0.0;
      m[4] = 2.0 * (xy - wz);       m[5] = 1.0 - 2.0 * (xx + zz); m[6] = 2.0 * (yz + wx);
      m[7] = 0.0;
      m[8] = 2.0 * (xz + wy);       m[9] = 2.0 * (yz - wx);       m[10] = 1.0 - 2.0 * (xx + yy);
      m[11] = 0.0;
      m[12] = tx;                    m[13] = ty;                    m[14] = tz;
      m[15] = 1.0;

      {
        std::lock_guard<std::mutex> lk(cam_pose_mu_);
        cam_pose_mat44d_ = m;
      }
      cam_pose_dirty_.store(true, std::memory_order_release);
    });
}

bool RosBridge::take_free_cam_xform(double * out_mat44d)
{
  if (!cam_pose_dirty_.load(std::memory_order_acquire)) {return false;}
  std::lock_guard<std::mutex> lk(cam_pose_mu_);
  if (!cam_pose_dirty_.load(std::memory_order_relaxed)) {return false;}
  for (std::size_t i = 0; i < 16; ++i) {
    out_mat44d[i] = cam_pose_mat44d_[i];
  }
  cam_pose_dirty_.store(false, std::memory_order_release);
  return true;
}

void RosBridge::subscribe_camera_zoom(const std::string & topic)
{
  if (sub_cam_zoom_) {return;}
  if (topic.empty()) {return;}
  rclcpp::QoS qos(rclcpp::KeepLast(1));
  qos.reliable();
  qos.transient_local();
  sub_cam_zoom_ = node_->create_subscription<std_msgs::msg::Float64>(
    topic, qos,
    [this](std_msgs::msg::Float64::ConstSharedPtr msg) {
      if (!std::isfinite(msg->data)) {
        RCLCPP_WARN(node_->get_logger(), "ignoring non-finite free_cam zoom sample");
        return;
      }
      cam_zoom_.store(msg->data, std::memory_order_relaxed);
      cam_zoom_dirty_.store(true, std::memory_order_release);
    });
}

bool RosBridge::take_free_cam_zoom(double * out_zoom)
{
  if (!cam_zoom_dirty_.exchange(false, std::memory_order_acq_rel)) {return false;}
  *out_zoom = cam_zoom_.load(std::memory_order_relaxed);
  return true;
}

}  // namespace gsi

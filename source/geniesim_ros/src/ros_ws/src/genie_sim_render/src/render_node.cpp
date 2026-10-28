#include "genie_sim_render/render_node.hpp"
#include "genie_sim_render/image_publisher_plugin.h"
#include "genie_sim_render/manifest_schema.hpp"

#include <nlohmann/json.hpp>
#include <ovrtx/ovrtx_attributes.h>
#include <tf2/exceptions.h>
#include <tf2/time.h>

#include <cstring>
#include <sstream>
#include <thread>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <filesystem>
#include <vector>

namespace genie_sim_render
{

using json = nlohmann::json;
namespace fs = std::filesystem;

// ---- fisheye black mask -------------------------------------------------
// Pixels the lens does not trace a ray for are simply left showing the scene
// environment -- a bright uniform grey/white in the supermarket scene -- so the
// invalid region outside the image circle has to be painted black here. That is
// also what the physical camera produces: the factory calibration photos read
// exactly 0.00 luma outside their image circle.
//
// The mask is still needed after the move to the ``lut`` lens model. ``lut``
// removes the 180 deg cap, not the need for a mask: past what the renderer
// traces it keeps returning a dead-flat fill (measured: pure white 255, RGB
// equal, one unique value across the whole region).

/// Widest half-angle ovrtx actually traces through a ``lut`` lens.
///
/// Measured, not documented: with LUT tables carrying a full 220.4 / 220.9 /
/// 221.7 deg field, all three head lenses stop tracing at the same radius --
/// 909-911 px is the last ring with real image data, 911-914 px is the first
/// ring that is dead-flat white (one unique value, RGB equal, boundary a perfect
/// circle), i.e. a 200.0 deg total field on every lens regardless of its own fx
/// or configured FOV. The tables are not the limit: their sentinel-free region
/// reaches 110.45 deg, and re-baking them with no out-of-FOV sentinel at all
/// moved the boundary by 0 px. ``lut`` also has no ``maxFov`` attribute to raise
/// (``ftheta``/``kannalaBrandtK3``/``radTanThinPrism`` do), so this is a hard
/// renderer-side limit. Masking at the configured FOV instead of this leaves a
/// ~90 px annulus of flat white, ~9.5 % of every frame.
static constexpr double kOvrtxLutMaxHalfFovRad = 100.0 * M_PI / 180.0;

/// Guard band, in pixels, shaved off the traced-circle radius before masking.
///
/// The traced/untraced boundary is *not* a hard edge: ovrtx antialiases it, so
/// the last few pixels are a grey ramp blending the scene into the flat white
/// no-hit fill. Measured on an unmasked frame (head_left, 0.5 px radial bins,
/// centred on the calibrated cx/cy): black/scene up to r=906.5, then "other"
/// pixels that are overwhelmingly RGB-equal grey climbing 1.8 -> 12.8 -> 50.8
/// -> 108 -> 163 -> 216 -> 245 -> 252.6, and 100 % pure white 255 from r=911.5.
/// The unclamped radius (909.5) lands mid-ramp, so masking there keeps the
/// bright half and paints a ~3 px near-white ring right on the mask edge -- the
/// artefact this constant removes. All three lenses agree: the ramp foot sits
/// 2.5-3 px inside the computed radius (left 907.0 vs 909.47, right 907.2 vs
/// 909.66, back 911.0 vs 912.48). 3 px costs at most ~1.5 px of genuine content
/// on the back lens, the only one with bright scene right out to the rim.
///
/// An earlier revision of this file put "the first dead-flat white ring" at
/// 911-914 and concluded real data survived to 909; that came from 1 px integer
/// rings, which are too coarse to separate the grey ramp from scene content.
static constexpr double kMaskEdgeGuardPx = 3.0;

/// Radius in pixels of the circle of valid image data for a fisheye camera.
///
/// This is the OpenCV fisheye forward model
///     r(theta) = fx * theta * (1 + k1*t^2 + k2*t^4 + k3*t^6 + k4*t^8)
/// evaluated at half the configured field of view, capped at what the renderer
/// can trace. It is deliberately *not* clamped to 90 deg: assemble-time bakes
/// any lens wider than 180 deg into ``lut`` tables carrying its full field (see
/// ``author_fisheye_lens`` in ``render_layer.py``), so the traced image circle
/// really does reach past 180 deg -- 913 px rather than 817 px on the ~221 deg
/// head lenses. It does not reach the calibrated 1001 px, though: see
/// ``kOvrtxLutMaxHalfFovRad``. The calibration photos put the real lens's circle
/// at 1000-1010 px, so the rendered field is ~21 deg narrower than the physical
/// one; that gap is the renderer's, not the configuration's.
///
/// The manifest keeps ``model: opencvFisheye`` for a LUT-baked lens: the lens
/// *is* still Kannala-Brandt equidistant, which is what this radius and the
/// published CameraInfo ``distortion_model`` describe. ``lut`` is only how ovrtx
/// is told to trace it. Returns 0 for anything that is not a FOV-tagged
/// fisheye, which disables masking.
static double fisheye_mask_radius_px(const CameraConfig & cam)
{
  if (cam.max_fov_deg <= 0.0 || cam.model != "opencvFisheye") {
    return 0.0;
  }
  const double theta = std::min(cam.max_fov_deg * 0.5 * M_PI / 180.0, kOvrtxLutMaxHalfFovRad);
  const double t2 = theta * theta;
  const double poly = 1.0 + t2 * (cam.k1 + t2 * (cam.k2 + t2 * (cam.k3 + t2 * cam.k4)));
  // Shave the antialiased rim off, or it survives as a bright ring on the mask
  // edge -- see kMaskEdgeGuardPx.
  return std::max(0.0, cam.fx * theta * poly - kMaskEdgeGuardPx);
}

/// Paint every pixel further than ``radius`` from (cx, cy) opaque black, in
/// place, with a hard edge.
///
/// Row-wise: each row's valid span is one sqrt, and the two invalid runs are
/// filled by memcpy from a prebuilt black row rather than a per-pixel distance
/// test over ~3 MPix. Alpha is preserved as opaque -- a plain memset would
/// zero it too and hand downstream consumers a fully transparent border.
static void apply_circular_mask(
  uint8_t * data, int width, int height, int channels,
  double cx, double cy, double radius)
{
  if (width <= 0 || height <= 0 || channels <= 0) {
    return;
  }
  std::vector<uint8_t> black(static_cast<size_t>(width) * channels, 0);
  if (channels == 4) {
    for (int x = 0; x < width; ++x) {
      black[static_cast<size_t>(x) * 4 + 3] = 255;
    }
  }
  const double r2 = radius * radius;
  for (int y = 0; y < height; ++y) {
    uint8_t * row = data + static_cast<size_t>(y) * width * channels;
    const double dy = (static_cast<double>(y) + 0.5) - cy;
    const double half2 = r2 - dy * dy;
    if (half2 <= 0.0) {
      std::memcpy(row, black.data(), black.size());
      continue;
    }
    const double half = std::sqrt(half2);
    int x0 = static_cast<int>(std::ceil(cx - half));
    int x1 = static_cast<int>(std::floor(cx + half));
    x0 = std::max(0, std::min(width, x0));
    x1 = std::max(-1, std::min(width - 1, x1));
    if (x0 > 0) {
      std::memcpy(row, black.data(), static_cast<size_t>(x0) * channels);
    }
    if (x1 < width - 1) {
      std::memcpy(
        row + static_cast<size_t>(x1 + 1) * channels, black.data(),
        static_cast<size_t>(width - 1 - x1) * channels);
    }
  }
}

template<typename ResultT>
static bool check_error(ResultT const & result, const char * op, rclcpp::Logger logger)
{
  if (result.status == OVRTX_API_ERROR) {
    ovx_string_t err = ovrtx_get_last_error();
    if (err.ptr && err.length > 0) {
      RCLCPP_ERROR(
        logger, "%s failed: %.*s", op,
        static_cast<int>(err.length), err.ptr);
    } else {
      RCLCPP_ERROR(logger, "%s failed (unknown error)", op);
    }
    return true;
  }
  return false;
}

static double ms_between(RenderNode::TimePoint a, RenderNode::TimePoint b)
{
  return std::chrono::duration<double, std::milli>(b - a).count();
}

static bool wait_for_op(
  ovrtx_renderer_t * renderer, uint64_t op_index, rclcpp::Logger logger,
  const char * op_name, bool fatal = true)
{
  ovrtx_op_wait_result_t wait_result{};
  while (ovrtx_wait_op(
      renderer, op_index,
      ovrtx_timeout_t{0}, &wait_result).status == OVRTX_API_TIMEOUT)
  {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  if (wait_result.num_error_ops > 0) {
    std::ostringstream details;
    for (size_t i = 0; i < wait_result.num_error_ops; ++i) {
      ovrtx_op_id_t op_id = wait_result.error_op_ids[i];
      ovx_string_t err = ovrtx_get_last_op_error(op_id);
      std::string err_str = (err.ptr && err.length > 0) ?
        std::string(err.ptr, err.length) : "(no detail)";
      if (i > 0) {details << "; ";}
      details << "op_id=" << op_id << ": " << err_str;
    }
    if (fatal) {
      RCLCPP_ERROR(
        logger, "%s had %zu errors: %s",
        op_name, wait_result.num_error_ops, details.str().c_str());
    } else {
      RCLCPP_WARN(
        logger, "%s had %zu errors (non-fatal): %s",
        op_name, wait_result.num_error_ops, details.str().c_str());
    }
    return false;
  }
  return true;
}

// A composite PointCloud render var exposes named per-channel DLTensors. Match by name
// (ovrtx strings carry ptr+length, not necessarily null-terminated).
static const ovrtx_render_var_tensor_t * find_pc_tensor(
  const ovrtx_render_var_output_t & output, const char * name)
{
  const size_t len = std::strlen(name);
  for (size_t i = 0; i < output.num_tensors; ++i) {
    const ovrtx_render_var_tensor_t & t = output.tensors[i];
    if (t.name && t.name->ptr && t.name->length == len &&
      std::strncmp(t.name->ptr, name, len) == 0)
    {
      return &t;
    }
  }
  return nullptr;
}

RenderNode::RenderNode(const rclcpp::NodeOptions & options)
: Node("genie_sim_render", options)
{
  declare_parameter<std::string>("stage_manifest", "");
  declare_parameter<double>("render_fps", 30.0);
  declare_parameter<std::string>("prim_paths", "");
  declare_parameter<std::string>("ovrtx_root", "");
  declare_parameter<bool>("lidar", false);
  declare_parameter<std::string>("deformable_topic", "");
  declare_parameter<std::vector<std::string>>(
    "plugin", std::vector<std::string>{"genie_sim_render/RosImagePublisherPlugin"});

  std::string manifest_path;
  get_parameter("stage_manifest", manifest_path);
  get_parameter("render_fps", render_fps_);
  get_parameter("lidar", lidar_enabled_);
  get_parameter("deformable_topic", deformable_topic_);

  std::string prim_paths_str;
  get_parameter("prim_paths", prim_paths_str);
  if (!prim_paths_str.empty()) {
    std::istringstream ss(prim_paths_str);
    std::string token;
    while (std::getline(ss, token, ',')) {
      if (!token.empty()) {
        prim_paths_.push_back(token);
      }
    }
  }

  if (manifest_path.empty()) {
    RCLCPP_ERROR(get_logger(), "Parameter 'stage_manifest' is required");
    throw std::runtime_error("stage_manifest parameter is required");
  }

  load_manifest(manifest_path);

  timing_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>("~/render_timing", 10);

  tf_scene_sub_ = create_subscription<tf2_msgs::msg::TFMessage>(
    "/tf_scene", rclcpp::SensorDataQoS(),
    std::bind(&RenderNode::on_tf_scene, this, std::placeholders::_1));

  // v2 protocol: robot links ride /tf (RSP fan-out over URDF FK).
  // TransformListener subscribes to /tf + /tf_static internally on
  // reliable QoS matching the RSP publisher's defaults, so a URDF
  // chain that arrives after startup is composed correctly.
  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_, this);

  if (!free_cam_prim_path_.empty()) {
    // Align with the latched QoS of RViz's ViewCameraPosePublisherDisplay:
    // KeepLast(1) + Reliable + TransientLocal; new subscribers immediately receive the last pose.
    rclcpp::QoS free_cam_qos(rclcpp::KeepLast(1));
    free_cam_qos.reliable();
    free_cam_qos.transient_local();
    free_cam_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
      "~/free_cam_pose", free_cam_qos,
      std::bind(&RenderNode::on_free_cam_pose, this, std::placeholders::_1));
  }

  if (!deformable_topic_.empty()) {
    // Wire format matches deformable_publisher.py exactly: a latched JSON
    // topology (TransientLocal) received once, and a best-effort per-tick
    // flat float array.
    rclcpp::QoS topology_qos(rclcpp::KeepLast(1));
    topology_qos.reliable();
    topology_qos.transient_local();
    deformable_topology_sub_ = create_subscription<std_msgs::msg::String>(
      deformable_topic_ + "_topology", topology_qos,
      std::bind(&RenderNode::on_deformable_topology, this, std::placeholders::_1));

    rclcpp::QoS flat_qos(rclcpp::KeepLast(1));
    flat_qos.best_effort();
    deformable_flat_sub_ = create_subscription<std_msgs::msg::Float32MultiArray>(
      deformable_topic_ + "_flat", flat_qos,
      std::bind(&RenderNode::on_deformable_flat, this, std::placeholders::_1));

    RCLCPP_INFO(
      get_logger(), "deformable channel enabled: %s_topology / %s_flat",
      deformable_topic_.c_str(), deformable_topic_.c_str());
  }

  init_renderer();
  load_plugins();
  setup_cameras();
  setup_lidars();

  auto period = std::chrono::duration<double>(1.0 / render_fps_);
  render_timer_ = create_wall_timer(
    std::chrono::duration_cast<std::chrono::nanoseconds>(period),
    std::bind(&RenderNode::render_timer_callback, this));

  stats_timer_ = create_wall_timer(
    std::chrono::seconds(1),
    std::bind(&RenderNode::stats_timer_callback, this));

  RCLCPP_INFO(
    get_logger(), "Render node initialized (%.1f fps, %zu cameras)",
    render_fps_, cameras_.size());
}

RenderNode::~RenderNode()
{
  shutdown_renderer();
}

void RenderNode::load_manifest(const std::string & manifest_path)
{
  fs::path mp(manifest_path);
  if (mp.is_relative()) {
    mp = fs::current_path() / mp;
  }

  // Parse through the shared schema so a field-name drift between
  // producer and reader is caught by mismatched to_json/from_json
  // overloads at review time, not deep in a ``.at("path")`` call at
  // frame 0. Version mismatches surface as a human-readable
  // ``std::runtime_error`` from ``Manifest::load``.
  Manifest manifest;
  try {
    manifest = Manifest::load(mp);
  } catch (const std::exception & exc) {
    RCLCPP_ERROR(get_logger(), "load_manifest failed: %s", exc.what());
    throw;
  }

  // Resolve every manifest-supplied path against ``base_path`` (falling back
  // to the manifest's own directory when ``base_path`` is missing). The
  // assemble pipeline stores paths *relative to* ``base_path``, so
  // resolving here lets the render node be launched from any cwd.
  const fs::path fallback_base = mp.parent_path();
  usd_path_ = resolve_manifest_path(manifest.scene_usda, manifest.base_path, fallback_base);
  if (usd_path_.empty()) {
    throw std::runtime_error("Manifest missing both 'scene_usda' and 'usd_path'");
  }
  robot_usda_ = resolve_manifest_path(manifest.robot_usda, manifest.base_path, fallback_base);
  render_layer_usda_ = resolve_manifest_path(
    manifest.render_layer_usda, manifest.base_path,
    fallback_base);
  robot_prefix_ = manifest.robot_prefix.empty() ? std::string{} : manifest.robot_prefix;
  free_cam_prim_path_ = manifest.free_cam_prim_path;

  // v2 wire protocol: pull ``manifest.robot["body_paths"]`` so
  // ``apply_robot_tf()`` can iterate URDF-link → prim mappings on
  // every render tick.  Keys arrive keyed by bare URDF link name,
  // matching ``/tf`` ``child_frame_id`` values that RSP emits.
  robot_body_paths_.clear();
  if (manifest.robot.is_object()) {
    auto it = manifest.robot.find("body_paths");
    if (it != manifest.robot.end() && it->is_object()) {
      for (auto entry = it->begin(); entry != it->end(); ++entry) {
        if (!entry.value().is_string()) {continue;}
        std::string path = entry.value().get<std::string>();
        if (entry.key().empty() || path.empty() || path.front() != '/') {continue;}
        robot_body_paths_.emplace(entry.key(), std::move(path));
      }
    }
  }

  const fs::path base_dir =
    manifest.base_path.empty() ? fallback_base : fs::path{manifest.base_path};
  RCLCPP_INFO(
    get_logger(),
    "load_manifest: manifest=%s base_dir=%s scene=%s robot=%s render_layer=%s",
    mp.string().c_str(), base_dir.string().c_str(),
    usd_path_.c_str(),
    robot_usda_.empty() ? "(none)" : robot_usda_.c_str(),
    render_layer_usda_.empty() ? "(none)" : render_layer_usda_.c_str());

  if (!fs::exists(usd_path_)) {
    RCLCPP_ERROR(
      get_logger(),
      "Scene USD not found after base_path resolution: '%s' (manifest=%s, base_dir=%s)",
      usd_path_.c_str(), mp.string().c_str(), base_dir.string().c_str());
    throw std::runtime_error("Scene USD not found: " + usd_path_);
  }
  if (!robot_usda_.empty() && !fs::exists(robot_usda_)) {
    RCLCPP_ERROR(
      get_logger(),
      "Robot USD not found after base_path resolution: '%s' (base_dir=%s)",
      robot_usda_.c_str(), base_dir.string().c_str());
    throw std::runtime_error("Robot USD not found: " + robot_usda_);
  }
  if (!render_layer_usda_.empty() && !fs::exists(render_layer_usda_)) {
    RCLCPP_ERROR(
      get_logger(),
      "Render layer USDA not found after base_path resolution: '%s' (base_dir=%s)",
      render_layer_usda_.c_str(), base_dir.string().c_str());
    throw std::runtime_error("Render layer USDA not found: " + render_layer_usda_);
  }

  for (size_t cam_idx = 0; cam_idx < manifest.cameras.size(); ++cam_idx) {
    const auto & src = manifest.cameras[cam_idx];
    CameraConfig cam;
    cam.path = src.path;
    cam.topic = src.topic;
    cam.depth_topic = src.depth_topic;
    cam.dds_topic = src.dds_topic;
    cam.dds_depth_topic = src.dds_depth_topic;
    cam.render_product_path = src.render_product_path.empty() ?
      "/RenderOVRTX/Cam_" + std::to_string(cam_idx) :
      src.render_product_path;
    cam.width = src.width;
    cam.height = src.height;
    cam.fx = src.fx;
    cam.fy = src.fy;
    cam.cx = src.cx == 0.0 ? static_cast<double>(cam.width) / 2.0 : src.cx;
    cam.cy = src.cy == 0.0 ? static_cast<double>(cam.height) / 2.0 : src.cy;
    cam.k1 = src.k1;
    cam.k2 = src.k2;
    cam.p1 = src.p1;
    cam.p2 = src.p2;
    cam.k3 = src.k3;
    cam.k4 = src.k4;
    // Map the assemble-time lens model to a ROS CameraInfo
    // distortion_model. opencvFisheye (OpenCV fisheye /
    // Kannala-Brandt) → "equidistant"; everything else → "plumb_bob".
    cam.model = src.model;
    cam.distortion_model = (src.model == "opencvFisheye") ? "equidistant" : "plumb_bob";
    cam.max_fov_deg = src.max_fov_deg;
    cam.min_range = src.min_range;
    cam.max_range = src.max_range;
    cam.is_free_cam = src.is_free_cam;
    cameras_.push_back(std::move(cam));
  }

  if (!render_layer_usda_.empty()) {
    for (const auto & cam : cameras_) {
      if (cam.is_free_cam) {continue;}
      auto slash_pos = cam.path.find('/');
      if (slash_pos != std::string::npos) {
        std::string body_name = cam.path.substr(0, slash_pos);
        std::string render_path = "/RenderOVRTX/Cameras/" + body_name;
        cam_body_to_render_path_[body_name] = render_path;
      }
    }
  }

  if (lidar_enabled_) {
    for (const auto & src : manifest.lidars) {
      LidarConfig l;
      l.render_product_path = src.render_product_path;
      l.topic = src.topic;
      l.frame_id = src.frame_id;
      l.parent_body = src.parent_body;
      // Sensor frame = topic basename (e.g. /lidar/livox_front → livox_front),
      // matching the rmagine path. The per-lidar cloud is published in this
      // frame; the merged cloud stays in ``base_link``.
      l.sensor_frame = l.topic.substr(l.topic.find_last_of('/') + 1);
      for (int k = 0; k < 3; ++k) {
        l.bl_T_s_xyz[k] = src.base_link_T_sensor.xyz[k];
      }
      for (int k = 0; k < 4; ++k) {
        l.bl_T_s_wxyz[k] = src.base_link_T_sensor.wxyz[k];
      }
      if (l.render_product_path.empty() || l.topic.empty()) {continue;}
      lidars_.push_back(std::move(l));
    }
  }
  if (!lidars_.empty()) {
    merged_lidar_topic_ = "/lidar/merged";
    for (const auto & l : lidars_) {
      if (!l.parent_body.empty()) {
        lidar_body_to_render_path_[l.parent_body] = "/RenderOVRTX/Lidars/" + l.parent_body;
      }
    }
  }

  RCLCPP_INFO(
    get_logger(),
    "Manifest loaded: scene=%s, render_layer=%s, cameras=%zu, lidars=%zu, "
    "robot.body_paths=%zu",
    usd_path_.c_str(),
    render_layer_usda_.empty() ? "(none)" : render_layer_usda_.c_str(),
    cameras_.size(),
    lidars_.size(),
    robot_body_paths_.size());
}

void RenderNode::init_renderer()
{
  RCLCPP_INFO(
    get_logger(),
    "Creating ovrtx renderer (first run compiles shaders, may be slow)...");

  std::string ovrtx_root;
  get_parameter("ovrtx_root", ovrtx_root);

  std::vector<ovrtx_config_entry_t> entries;
  if (!ovrtx_root.empty()) {
    ovx_string_t root_str{ovrtx_root.c_str(), ovrtx_root.size()};
    entries.push_back(ovrtx_config_entry_binary_package_root_path(root_str));
  }

  ovx_string_t log_path{"/tmp/ovrtx.log", 14};
  entries.push_back(ovrtx_config_entry_log_file_path(log_path));
  ovx_string_t log_level{"verbose", 7};
  entries.push_back(ovrtx_config_entry_log_level(log_level));

  // ovrtx 0.4 turns Sensor Processing Graphs on by default (0.3 had them off).
  // With SPG on, DistanceToImagePlaneSD comes back through a depth-sensor model
  // that clamps the range to roughly [0.3, 2.5] m, so far geometry reads as inf.
  // We publish geometric depth, so keep SPG off to retain the 0.3 behaviour.
  entries.push_back(ovrtx_config_entry_enable_spg(false));

  // Required by the lidar sensor pipeline; only enable it when lidar is active so
  // camera-only scenes don't pay the motion-BVH build cost.
  if (lidar_enabled_) {
#ifdef GENIE_OVRTX_HAS_MOTION_BVH_MODE
    entries.push_back(ovrtx_config_entry_motion_bvh(OVRTX_MOTION_BVH_ENABLE));
#else
    entries.push_back(ovrtx_config_entry_enable_motion_bvh(true));
#endif
  }

  ovrtx_config_t config{};
  config.entries = entries.empty() ? nullptr : entries.data();
  config.entry_count = entries.size();

  ovrtx_result_t result = ovrtx_create_renderer(&config, &renderer_);
  if (check_error(result, "create_renderer", get_logger())) {
    throw std::runtime_error("Failed to create ovrtx renderer");
  }
  RCLCPP_INFO(get_logger(), "Renderer created");

  // ovrtx 0.4 deprecated the direct USD-loading APIs in favour of ovstage
  // (ovstage_population_open_usd_from_file + ovrtx_attach_ovstage). They still
  // work and remain the only option until the ovstage migration is taken on, so
  // scope the deprecation warning off here instead of dropping -Werror package-wide.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

  auto open_root_usd = [&](const std::string & path, const char * label) {
      RCLCPP_INFO(get_logger(), "Opening %s as root: %s", label, path.c_str());
      ovx_string_t file_str{path.c_str(), path.size()};
      ovrtx_enqueue_result_t er = ovrtx_open_usd_from_file(renderer_, file_str);
      if (check_error(er, label, get_logger())) {
        throw std::runtime_error(std::string("Failed to enqueue ") + label);
      }
      const bool ok = wait_for_op(renderer_, er.op_index, get_logger(), label, true);
      if (!ok) {
        throw std::runtime_error(std::string(label) + " loading failed");
      }
      RCLCPP_INFO(get_logger(), "%s loaded", label);
    };

  auto add_usd_reference = [&](const std::string & path, const std::string & prefix,
      ovrtx_usd_handle_t & handle, const char * label, bool fatal) {
      ovx_string_t file_str{path.c_str(), path.size()};
      ovx_string_t pfx{prefix.c_str(), prefix.size()};
      RCLCPP_INFO(
        get_logger(), "Adding %s reference: %s (prefix=%s)", label, path.c_str(),
        prefix.c_str());
      ovrtx_enqueue_result_t er =
        ovrtx_add_usd_reference_from_file(renderer_, file_str, pfx, &handle);
      if (check_error(er, label, get_logger())) {
        if (fatal) {
          throw std::runtime_error(std::string("Failed to enqueue ") + label);
        }
        RCLCPP_WARN(get_logger(), "%s enqueue failed (non-fatal)", label);
        return;
      }
      const bool ok = wait_for_op(renderer_, er.op_index, get_logger(), label, fatal);
      if (!ok) {
        if (fatal) {
          throw std::runtime_error(std::string(label) + " loading failed");
        }
        RCLCPP_WARN(get_logger(), "%s loaded with errors (non-fatal)", label);
      }
      RCLCPP_INFO(get_logger(), "%s loaded", label);
    };

#pragma GCC diagnostic pop

  open_root_usd(usd_path_, "scene");
  scene_handle_ = 0;  // Root layer; no add-reference handle.

  if (!robot_usda_.empty()) {
    std::string robot_pfx = "/" + robot_prefix_;
    add_usd_reference(robot_usda_, robot_pfx, robot_handle_, "robot", false);
  }

  if (!render_layer_usda_.empty()) {
    add_usd_reference(
      render_layer_usda_, "/RenderOVRTX", camera_layer_handle_,
      "render_layer", true);
  } else {
    RCLCPP_WARN(
      get_logger(),
      "No render_layer_usda in manifest. Run genie_sim_engine's assemble_scene first.");
  }

  scene_loaded_ = true;
}

void RenderNode::setup_cameras()
{
  if (cameras_.empty()) {
    RCLCPP_WARN(get_logger(), "No cameras configured");
    return;
  }

  for (size_t i = 0; i < cameras_.size(); ++i) {
    const auto & cam = cameras_[i];
    RCLCPP_INFO(
      get_logger(), "Camera[%zu]: product=%s -> %s (%dx%d)%s",
      i, cam.render_product_path.c_str(), cam.topic.c_str(),
      cam.width, cam.height,
      cam.depth_topic.empty() ? "" : " +depth");
    for (auto & plugin : image_plugins_) {
      // Register ROS->DDS mapping first, then create the publisher; if the mapping is empty the plugin skips this camera.
      if (!cam.dds_topic.empty()) {
        // All camera channels must match the real robot's DDS format: a
        // JPEG-payload vendor IMAGE packet (encoding=JPEG, color_format=RGB,
        // bit_depth=8) rather than the sensor_msgs CompressedImage that
        // register_dds_mapping builds. Confirmed by a hardware engineer
        // (high-confidence) that the GDK-side subscriber expects
        // CosineDataPacket for every channel, not just fisheye -- an earlier
        // retest of this exact change showed the opposite mismatch signal,
        // but that test's restart may not have cleared a stale DDS
        // participant/type registration.
        plugin->register_dds_image_packet_mapping(cam.topic, cam.dds_topic);
      }
      // Depth-side mapping is dispatched separately so the plugin can build a depth-typed
      // publisher (e.g. Z16 raw over the vendor IMAGE packet) instead of a JPEG one.
      if (!cam.depth_topic.empty() && !cam.dds_depth_topic.empty()) {
        plugin->register_dds_depth_mapping(cam.depth_topic, cam.dds_depth_topic);
      }
      plugin->create_camera_publisher(
        cam.topic, static_cast<uint32_t>(cam.width),
        static_cast<uint32_t>(cam.height));
      // Depth reuses the same plugin interface; the ROS side publishes <depth_topic>/image_raw + camera_info,
      // and the DDS side picks it up via the depth mapping registered above (when dds_depth_topic was set).
      if (!cam.depth_topic.empty()) {
        plugin->create_camera_publisher(
          cam.depth_topic, static_cast<uint32_t>(cam.width),
          static_cast<uint32_t>(cam.height));
      }
    }
  }
}

void RenderNode::setup_lidars()
{
  if (lidars_.empty()) {
    return;
  }
  for (const auto & l : lidars_) {
    RCLCPP_INFO(
      get_logger(), "Lidar: product=%s -> %s (frame=%s, body=%s)",
      l.render_product_path.c_str(), l.topic.c_str(),
      l.frame_id.c_str(), l.parent_body.c_str());
    for (auto & plugin : image_plugins_) {
      plugin->create_pointcloud_publisher(l.topic);
    }
  }
  if (!merged_lidar_topic_.empty()) {
    for (auto & plugin : image_plugins_) {
      plugin->create_pointcloud_publisher(merged_lidar_topic_);
    }
    RCLCPP_INFO(
      get_logger(), "Merged lidar topic: %s (base_link frame)",
      merged_lidar_topic_.c_str());
  }
  // Set up the /tf_static + /tf broadcasters; the actual transforms are published from
  // on_tf_scene once base_link's world pose is known (parity with the rmagine path).
  for (auto & plugin : image_plugins_) {
    plugin->create_tf_publisher();
  }
}

void RenderNode::load_plugins()
{
  plugin_loader_ =
    std::make_shared<pluginlib::ClassLoader<IImagePublisherPlugin>>(
    "genie_sim_render", "genie_sim_render::IImagePublisherPlugin");

  const auto plugin_names = get_parameter("plugin").as_string_array();
  for (const auto & plugin_name : plugin_names) {
    if (plugin_name.empty()) {continue;}
    try {
      auto plugin = plugin_loader_->createSharedInstance(plugin_name);
      plugin->initialize(this);
      image_plugins_.push_back(plugin);
      RCLCPP_INFO(get_logger(), "Loaded image publisher plugin: %s", plugin_name.c_str());
    } catch (const pluginlib::PluginlibException & e) {
      RCLCPP_ERROR(
        get_logger(), "Failed to load plugin '%s': %s",
        plugin_name.c_str(), e.what());
    }
  }
}

void RenderNode::shutdown_renderer()
{
  if (renderer_) {
    ovrtx_destroy_renderer(renderer_);
    renderer_ = nullptr;
    RCLCPP_INFO(get_logger(), "Renderer destroyed");
  }
  // Renderer destruction takes the OVRTX stage with it; the lazy
  // resetXformStack tracking is per-renderer, so drop it here so a
  // subsequent init_renderer() rebuilds the flags from scratch.
  reset_xform_done_.clear();
}

void RenderNode::on_tf_scene(const tf2_msgs::msg::TFMessage::SharedPtr msg)
{
  // Track the sim-time stamp so published clouds/images share the TF tree's time base.
  if (!msg->transforms.empty()) {
    const auto & s = msg->transforms[0].header.stamp;
    latest_tf_time_ = static_cast<double>(s.sec) + static_cast<double>(s.nanosec) * 1e-9;
  }

  if (!scene_loaded_) {
    return;
  }

  RCLCPP_INFO_ONCE(
    get_logger(), "tf_scene received (%zu transforms)",
    msg->transforms.size());

  size_t tf_count = msg->transforms.size();
  std::vector<std::string> paths;
  std::vector<ovrtx_xform_matrix44d_t> xforms;
  paths.reserve(tf_count);
  xforms.reserve(tf_count);

  for (size_t i = 0; i < tf_count; ++i) {
    const auto & tf = msg->transforms[i];
    const std::string & body_name = tf.child_frame_id;

    // v2 wire contract: /tf_scene child_frame_id is always an absolute
    // USD prim path (starts with '/'). Bare body names are a v1
    // convention we deliberately dropped; warn-once and skip so a
    // lagging producer surfaces itself instead of silently guessing
    // a target prim.
    if (body_name.empty() || body_name.front() != '/') {
      RCLCPP_WARN_ONCE(
        get_logger(),
        "tf_scene: rejecting non-absolute child_frame_id %s — v2 requires "
        "absolute USD prim paths (starts with '/'); check the physics-side "
        "publisher.",
        body_name.c_str());
      continue;
    }

    double qx = tf.transform.rotation.x;
    double qy = tf.transform.rotation.y;
    double qz = tf.transform.rotation.z;
    double qw = tf.transform.rotation.w;
    double tx = tf.transform.translation.x;
    double ty = tf.transform.translation.y;
    double tz = tf.transform.translation.z;

    double xx = qx * qx, yy = qy * qy, zz = qz * qz;
    double xy = qx * qy, xz = qx * qz, yz = qy * qz;
    double wx = qw * qx, wy = qw * qy, wz = qw * qz;

    ovrtx_xform_matrix44d_t xform{};
    auto & m = xform.v;
    m[0] = 1.0 - 2.0 * (yy + zz); m[1] = 2.0 * (xy + wz);       m[2] = 2.0 * (xz - wy);
    m[3] = 0.0;
    m[4] = 2.0 * (xy - wz);       m[5] = 1.0 - 2.0 * (xx + zz); m[6] = 2.0 * (yz + wx);
    m[7] = 0.0;
    m[8] = 2.0 * (xz + wy);       m[9] = 2.0 * (yz - wx);       m[10] = 1.0 - 2.0 * (xx + yy);
    m[11] = 0.0;
    m[12] = tx;                     m[13] = ty;                     m[14] = tz;
    m[15] = 1.0;

    paths.push_back(body_name);
    xforms.push_back(xform);
  }

  size_t count = paths.size();
  std::vector<ovx_string_t> ovx_paths(count);
  for (size_t i = 0; i < count; ++i) {
    ovx_paths[i] = {paths[i].c_str(), paths[i].size()};
  }

  // First-touch invariant: assert omni:resetXformStack=1 on every prim
  // before we write a world-space matrix to it. Without this, OVRTX
  // composes our matrix with the parent's authored xform stack — e.g.
  // /World/objects/<bottle>'s authored translate would be added to the
  // physics-derived world pose, so the rendered object drifts away from
  // its physical position. The in-process Newton visualizer does the
  // same write at bind time (see engine/newton/visualizers/ovrtx.py);
  // doing it lazily here keeps the cross-process render node
  // backend-agnostic — any /tf_scene publisher (present and future
  // physics backends) gets correct world-space placement without
  // producer-side coordination.
  std::vector<ovx_string_t> new_paths;
  std::vector<uint8_t> new_flags;  // bool: OVRTX expects uint8 for omni:resetXformStack
  for (size_t i = 0; i < count; ++i) {
    auto [it, inserted] = reset_xform_done_.insert(paths[i]);
    if (inserted) {
      new_paths.push_back(ovx_paths[i]);
      new_flags.push_back(1);
    }
  }
  if (!new_paths.empty()) {
    ovrtx_enqueue_result_t rr = ovrtx_set_reset_xform_stack(
      renderer_, new_paths.data(), new_paths.size(),
      reinterpret_cast<const bool *>(new_flags.data()));
    if (rr.status != OVRTX_API_SUCCESS) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 1000,
        "tf_scene set_reset_xform_stack failed for %zu new prim(s); "
        "world-space writes may be composed with parent xform stacks",
        new_paths.size());
    } else {
      RCLCPP_INFO(
        get_logger(),
        "tf_scene: flagged %zu new prim(s) with omni:resetXformStack=1 "
        "(total tracked: %zu)",
        new_paths.size(), reset_xform_done_.size());
    }
  }

  ovrtx_enqueue_result_t wr = ovrtx_set_xform_mat(
    renderer_, ovx_paths.data(), count, xforms.data());
  if (wr.status != OVRTX_API_SUCCESS) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 1000,
      "tf_scene set_xform_mat failed");
  }
  // base_link->world is published with the lidar cloud (~10 Hz, same stamp) in
  // publish_lidar_pointclouds, not here. Publishing it on every /tf_scene (~85 Hz)
  // overran the /tf publish queue (queue_size 5) and dropped historical TF samples, so
  // a late cloud stamp had no matching transform and the cloud jittered around Z during
  // rotation. The rmagine path likewise emits base_link->world from its 10 Hz lidar loop.
}

void RenderNode::on_deformable_topology(const std_msgs::msg::String::SharedPtr msg)
{
  if (!deformable_objects_.empty()) {
    return;  // latched topology only needs to be parsed once
  }
  try {
    json parsed = json::parse(msg->data);
    for (const auto & obj : parsed.at("objects")) {
      DeformableObject d;
      d.name = obj.at("name").get<std::string>();
      d.prim_path = obj.at("prim_path").get<std::string>();
      d.n_particles = obj.at("n_particles").get<size_t>();
      int mesh_id = obj.at("mesh_id").get<int>();
      deformable_objects_[mesh_id] = d;
    }
  } catch (const json::exception & e) {
    RCLCPP_ERROR(get_logger(), "deformable topology JSON parse failed: %s", e.what());
    deformable_objects_.clear();
    return;
  }
  RCLCPP_INFO(
    get_logger(), "deformable topology received: %zu object(s)",
    deformable_objects_.size());
  for (const auto & [mesh_id, obj] : deformable_objects_) {
    RCLCPP_INFO(
      get_logger(), "  mesh_id=%d name=%s prim_path=%s n_particles=%zu",
      mesh_id, obj.name.c_str(), obj.prim_path.c_str(), obj.n_particles);
  }
}

void RenderNode::on_deformable_flat(const std_msgs::msg::Float32MultiArray::SharedPtr msg)
{
  if (!scene_loaded_ || deformable_objects_.empty()) {
    return;  // topology hasn't arrived yet (or the topic is unused this scene)
  }
  const std::vector<float> & data = msg->data;
  if (data.size() < 2) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 1000,
      "deformable flat message too short (%zu floats)", data.size());
    return;
  }
  const int tick = static_cast<int>(std::lround(data[1]));
  size_t i = 2;
  while (i + 2 <= data.size()) {
    const int mesh_id = static_cast<int>(std::lround(data[i]));
    const size_t count = static_cast<size_t>(std::lround(data[i + 1]));
    i += 2;
    if (i + count * 3 > data.size()) {
      RCLCPP_ERROR_THROTTLE(
        get_logger(), *get_clock(), 1000,
        "deformable flat message truncated: mesh_id=%d wants %zu points, only %zu floats left",
        mesh_id, count, data.size() - i);
      break;
    }
    auto it = deformable_objects_.find(mesh_id);
    if (it == deformable_objects_.end()) {
      i += count * 3;
      continue;
    }
    const DeformableObject & obj = it->second;
    if (count != obj.n_particles) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 1000,
        "deformable flat: mesh_id=%d got %zu points, topology declared %zu",
        mesh_id, count, obj.n_particles);
    }
    write_deformable_points(obj, data.data() + i, count);
    i += count * 3;
  }
  deformable_frames_received_++;
  if (tick != deformable_last_tick_) {
    deformable_last_tick_ = tick;
    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 5000,
      "deformable channel: tick=%d (%lu message(s) received)",
      tick, static_cast<unsigned long>(deformable_frames_received_));
  }
}

void RenderNode::write_deformable_points(
  const DeformableObject & obj, const float * xyz, size_t n_points)
{
  // literal_to_ovx_string is a compile-time-literal-only macro (it deliberately
  // fails to compile on a non-literal argument via "" concatenation), so a
  // runtime path needs its ovx_string_t built directly instead.
  const ovx_string_t path{obj.prim_path.c_str(), obj.prim_path.size()};

  // First-touch: flag omni:resetXformStack=1 and set an identity xform, so
  // the world-space points we write below aren't re-composed with whatever
  // authored xform the mesh prim's ancestors carry. Reuses the same
  // reset_xform_done_ set /tf_scene's rigid-body path populates -- a prim
  // path only needs this flag asserted once, ever, regardless of which
  // channel drives it.
  auto [it, inserted] = reset_xform_done_.insert(obj.prim_path);
  if (inserted) {
    const bool reset_true = true;
    ovrtx_enqueue_result_t rr = ovrtx_set_reset_xform_stack(renderer_, &path, 1, &reset_true);
    check_error(rr, "deformable set_reset_xform_stack", get_logger());
    ovrtx_xform_matrix44d_t identity{};
    identity.v[0] = identity.v[5] = identity.v[10] = identity.v[15] = 1.0;
    ovrtx_enqueue_result_t xr = ovrtx_set_xform_mat(renderer_, &path, 1, &identity);
    check_error(xr, "deformable set_xform_mat(identity)", get_logger());
  }

  // No convenience wrapper exists for array-valued mesh attributes (unlike
  // xform/path/token in ovrtx_attributes.h), so build the binding + tensor
  // directly. One point = one dtype "element" with lanes=3 (matching how
  // ovrtx_set_xform_mat treats one 4x4 matrix as one element with
  // lanes=16) -- this mirrors ovrtx_set_path_attributes' one-tensor-per-prim
  // pattern for array attributes, just with a float3 element type instead
  // of ovx_string_t.
  DLDataType dtype;
  dtype.code = kDLFloat;
  dtype.bits = 32;
  dtype.lanes = 3;

  int64_t shape = static_cast<int64_t>(n_points);
  int64_t stride = 1;
  DLTensor tensor{};
  tensor.data = const_cast<float *>(xyz);
  tensor.device = {kDLCPU, 0};
  tensor.ndim = 1;
  tensor.dtype = dtype;
  tensor.shape = &shape;
  tensor.strides = &stride;
  tensor.byte_offset = 0;

  ovrtx_input_buffer_t buffer{};
  buffer.tensors = &tensor;
  buffer.tensor_count = 1;
  buffer.dirty_bits = nullptr;

  ovx_string_t points_attr = literal_to_ovx_string("points");
  ovrtx_binding_desc_or_handle_t binding{};
  binding.binding_desc.prim_list = {&path, 1};
  binding.binding_desc.attribute_name = {{}, points_attr};
  binding.binding_desc.attribute_type = {dtype, /*is_array=*/true, OVRTX_SEMANTIC_NONE};
  binding.binding_desc.prim_mode = OVRTX_BINDING_PRIM_MODE_EXISTING_ONLY;
  binding.binding_desc.flags = OVRTX_BINDING_FLAG_NONE;

  // ovrtx_write_attribute is soft-deprecated in favor of an ovstage-based
  // API in this SDK version, but there is no drop-in replacement for
  // arbitrary array attributes (only ovrtx_set_xform_mat/etc.'s convenience
  // wrappers exist, and this file uses those elsewhere) -- those wrappers
  // are themselves thin callers of this same function, just defined inside
  // the SDK's own header where GCC suppresses the warning. Points have no
  // wrapper, so the direct call needs the same suppression here.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  ovrtx_enqueue_result_t wr = ovrtx_write_attribute(
    renderer_, &binding, &buffer, OVRTX_DATA_ACCESS_SYNC);
#pragma GCC diagnostic pop
  if (check_error(wr, "deformable write_attribute(points)", get_logger())) {
    RCLCPP_ERROR_THROTTLE(
      get_logger(), *get_clock(), 1000,
      "deformable points write failed for %s (%zu points)",
      obj.prim_path.c_str(), n_points);
  }
}

void RenderNode::apply_robot_tf()
{
  // v2 protocol: robot links ride /tf, composed by RSP from URDF FK on
  // /joint_states plus the engine's physics-side world→base_link edge.
  // We ask the tf2 buffer for each URDF link's world-frame pose and
  // write it to the composed OVRTX prim via ``manifest.robot.body_paths``.
  //
  // Missing transforms — an entry the URDF declares but tf hasn't
  // populated yet — are skipped silently on the hot path and diagnosed
  // by a throttled warn so a stuck lookup is visible without spamming.
  if (!scene_loaded_ || robot_body_paths_.empty() || !tf_buffer_) {
    return;
  }

  std::vector<std::string> paths;
  std::vector<ovrtx_xform_matrix44d_t> xforms;
  paths.reserve(robot_body_paths_.size());
  xforms.reserve(robot_body_paths_.size());

  std::size_t missing = 0;
  for (const auto & entry : robot_body_paths_) {
    const std::string & link_name = entry.first;
    const std::string & prim_path = entry.second;

    geometry_msgs::msg::TransformStamped tf_msg;
    try {
      // v2 protocol: ``map`` is the canonical global fixed frame (REP
      // 105). The world → map static identity keeps v1 consumers
      // that still reference ``world`` working, but every v2
      // producer / consumer talks in map coordinates directly.
      tf_msg = tf_buffer_->lookupTransform("map", link_name, tf2::TimePointZero);
    } catch (const tf2::TransformException & ex) {
      ++missing;
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "apply_robot_tf: lookupTransform map->%s failed: %s "
        "(URDF link declared in manifest.robot.body_paths but /tf tree "
        "does not resolve it yet; RSP may still be initialising)",
        link_name.c_str(), ex.what());
      continue;
    }

    const double qx = tf_msg.transform.rotation.x;
    const double qy = tf_msg.transform.rotation.y;
    const double qz = tf_msg.transform.rotation.z;
    const double qw = tf_msg.transform.rotation.w;
    const double tx = tf_msg.transform.translation.x;
    const double ty = tf_msg.transform.translation.y;
    const double tz = tf_msg.transform.translation.z;

    const double xx = qx * qx, yy = qy * qy, zz = qz * qz;
    const double xy = qx * qy, xz = qx * qz, yz = qy * qz;
    const double wx = qw * qx, wy = qw * qy, wz = qw * qz;

    ovrtx_xform_matrix44d_t xform{};
    auto & m = xform.v;
    m[0] = 1.0 - 2.0 * (yy + zz); m[1] = 2.0 * (xy + wz);       m[2] = 2.0 * (xz - wy);
    m[3] = 0.0;
    m[4] = 2.0 * (xy - wz);       m[5] = 1.0 - 2.0 * (xx + zz); m[6] = 2.0 * (yz + wx);
    m[7] = 0.0;
    m[8] = 2.0 * (xz + wy);       m[9] = 2.0 * (yz - wx);       m[10] = 1.0 - 2.0 * (xx + yy);
    m[11] = 0.0;
    m[12] = tx;                    m[13] = ty;                    m[14] = tz;
    m[15] = 1.0;

    paths.push_back(prim_path);
    xforms.push_back(xform);

    // Cameras/lidars authored under ``/RenderOVRTX/Cameras/<body>``
    // follow the parent link's world pose here — the manifest side
    // authored them at identity, so mirroring the parent xform every
    // tick keeps them in the right place. Pre-v2 this lookup lived in
    // ``on_tf_scene`` but robot links no longer arrive there.
    auto cam_it = cam_body_to_render_path_.find(link_name);
    if (cam_it != cam_body_to_render_path_.end()) {
      paths.push_back(cam_it->second);
      xforms.push_back(xform);
    }
    auto lidar_it = lidar_body_to_render_path_.find(link_name);
    if (lidar_it != lidar_body_to_render_path_.end()) {
      paths.push_back(lidar_it->second);
      xforms.push_back(xform);
    }

    // Capture base_link's world pose for the lidar-tf publisher's
    // dynamic base_link→world edge (unchanged from the v1 path — it
    // previously read this off /tf_scene, now it reads it here).
    if (link_name == "base_link") {
      base_link_world_q_[0] = qx; base_link_world_q_[1] = qy;
      base_link_world_q_[2] = qz; base_link_world_q_[3] = qw;
      base_link_world_t_[0] = tx; base_link_world_t_[1] = ty; base_link_world_t_[2] = tz;
      has_base_link_world_ = true;
    }
  }

  if (paths.empty()) {
    return;
  }

  std::vector<ovx_string_t> ovx_paths(paths.size());
  for (std::size_t i = 0; i < paths.size(); ++i) {
    ovx_paths[i] = {paths[i].c_str(), paths[i].size()};
  }

  // Same first-touch invariant as on_tf_scene: reset_xform_stack once
  // per prim so world-space writes aren't composed with any authored
  // parent xform stack. See on_tf_scene for the full rationale.
  std::vector<ovx_string_t> new_paths;
  std::vector<uint8_t> new_flags;
  for (std::size_t i = 0; i < paths.size(); ++i) {
    auto [it, inserted] = reset_xform_done_.insert(paths[i]);
    (void)it;
    if (inserted) {
      new_paths.push_back(ovx_paths[i]);
      new_flags.push_back(1);
    }
  }
  if (!new_paths.empty()) {
    ovrtx_enqueue_result_t rr = ovrtx_set_reset_xform_stack(
      renderer_, new_paths.data(), new_paths.size(),
      reinterpret_cast<const bool *>(new_flags.data()));
    if (rr.status != OVRTX_API_SUCCESS) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 1000,
        "apply_robot_tf set_reset_xform_stack failed for %zu new prim(s)",
        new_paths.size());
    }
  }

  ovrtx_enqueue_result_t wr = ovrtx_set_xform_mat(
    renderer_, ovx_paths.data(), paths.size(), xforms.data());
  if (wr.status != OVRTX_API_SUCCESS) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 1000,
      "apply_robot_tf set_xform_mat failed");
  }
}

// Publish the lidar TF tree (parity with the rmagine path): a one-shot static
// base_link->sensor edge per lidar, and a dynamic base_link->world edge (= inverse of base_link's
// world pose) every /tf_scene so consumers can place the base_link/sensor-frame clouds as the
// robot moves. Only active when lidar publishing is enabled.
void RenderNode::publish_lidar_tf()
{
  if (!lidar_enabled_ || lidars_.empty() || !has_base_link_world_) {
    return;
  }

  // Static base_link -> sensor (the mount), published once.
  if (!tf_static_published_) {
    std::vector<TransformMsg> statics;
    statics.reserve(lidars_.size());
    for (const auto & l : lidars_) {
      if (l.sensor_frame.empty()) {continue;}
      TransformMsg t;
      t.parent_frame = "base_link";
      t.child_frame = l.sensor_frame;
      t.timestamp = latest_tf_time_;
      t.tx = l.bl_T_s_xyz[0]; t.ty = l.bl_T_s_xyz[1]; t.tz = l.bl_T_s_xyz[2];
      // manifest stores wxyz; TransformMsg holds xyzw.
      t.qw = l.bl_T_s_wxyz[0]; t.qx = l.bl_T_s_wxyz[1];
      t.qy = l.bl_T_s_wxyz[2]; t.qz = l.bl_T_s_wxyz[3];
      statics.push_back(std::move(t));
    }
    if (!statics.empty()) {
      for (auto & plugin : image_plugins_) {
        plugin->publish_transforms(statics, true);
      }
      tf_static_published_ = true;
    }
  }

  // Dynamic base_link -> world = inverse of base_link's world pose (world->base_link).
  const double qx = base_link_world_q_[0], qy = base_link_world_q_[1];
  const double qz = base_link_world_q_[2], qw = base_link_world_q_[3];
  const double tx = base_link_world_t_[0], ty = base_link_world_t_[1], tz = base_link_world_t_[2];
  // Inverse rotation = conjugate; inverse translation = -R(conj) * t.
  const double iqx = -qx, iqy = -qy, iqz = -qz, iqw = qw;
  const double xx = iqx * iqx, yy = iqy * iqy, zz = iqz * iqz;
  const double xy = iqx * iqy, xz = iqx * iqz, yz = iqy * iqz;
  const double wx = iqw * iqx, wy = iqw * iqy, wz = iqw * iqz;
  const double r00 = 1 - 2 * (yy + zz), r01 = 2 * (xy - wz), r02 = 2 * (xz + wy);
  const double r10 = 2 * (xy + wz), r11 = 1 - 2 * (xx + zz), r12 = 2 * (yz - wx);
  const double r20 = 2 * (xz - wy), r21 = 2 * (yz + wx), r22 = 1 - 2 * (xx + yy);
  TransformMsg dyn;
  dyn.parent_frame = "base_link";
  dyn.child_frame = "world";
  dyn.timestamp = latest_tf_time_;
  dyn.tx = -(r00 * tx + r01 * ty + r02 * tz);
  dyn.ty = -(r10 * tx + r11 * ty + r12 * tz);
  dyn.tz = -(r20 * tx + r21 * ty + r22 * tz);
  dyn.qx = iqx; dyn.qy = iqy; dyn.qz = iqz; dyn.qw = iqw;
  std::vector<TransformMsg> dynamics{dyn};
  for (auto & plugin : image_plugins_) {
    plugin->publish_transforms(dynamics, false);
  }
}

void RenderNode::on_free_cam_pose(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
  if (!scene_loaded_ || free_cam_prim_path_.empty()) {
    return;
  }

  RCLCPP_INFO_ONCE(
    get_logger(), "free_cam_pose received, writing to %s",
    free_cam_prim_path_.c_str());

  double qx = msg->pose.orientation.x;
  double qy = msg->pose.orientation.y;
  double qz = msg->pose.orientation.z;
  double qw = msg->pose.orientation.w;
  double tx = msg->pose.position.x;
  double ty = msg->pose.position.y;
  double tz = msg->pose.position.z;

  // The RViz view_camera_pose_publisher_display stamps the pose in RViz's
  // *Fixed Frame* (msg->header.frame_id), but OVRTX prim xforms are authored in
  // the ``map`` world frame (see apply_robot_tf, which looks up ``map``->link).
  // When the operator's Fixed Frame isn't ``map`` — e.g. the fallback
  // view_robot.rviz uses base_link — compose ``map``<-frame from tf so the free
  // camera lands correctly regardless of the Fixed Frame. Empty frame_id or
  // ``map`` = a publisher already speaking map coordinates, applied as-is.
  const std::string & src_frame = msg->header.frame_id;
  if (!src_frame.empty() && src_frame != "map" && tf_buffer_) {
    try {
      const auto T = tf_buffer_->lookupTransform("map", src_frame, tf2::TimePointZero);
      const double rx = T.transform.rotation.x, ry = T.transform.rotation.y;
      const double rz = T.transform.rotation.z, rw = T.transform.rotation.w;
      // q_map = q_T * q_pose  (Hamilton product)
      const double mqw = rw * qw - rx * qx - ry * qy - rz * qz;
      const double mqx = rw * qx + rx * qw + ry * qz - rz * qy;
      const double mqy = rw * qy - rx * qz + ry * qw + rz * qx;
      const double mqz = rw * qz + rx * qy - ry * qx + rz * qw;
      // t_map = R(q_T) * t_pose + t_T
      const double rxx = rx * rx, ryy = ry * ry, rzz = rz * rz;
      const double rxy = rx * ry, rxz = rx * rz, ryz = ry * rz;
      const double rwx = rw * rx, rwy = rw * ry, rwz = rw * rz;
      const double px = (1.0 - 2.0 * (ryy + rzz)) * tx + 2.0 * (rxy - rwz) * ty +
        2.0 * (rxz + rwy) * tz;
      const double py = 2.0 * (rxy + rwz) * tx + (1.0 - 2.0 * (rxx + rzz)) * ty +
        2.0 * (ryz - rwx) * tz;
      const double pz = 2.0 * (rxz - rwy) * tx + 2.0 * (ryz + rwx) * ty +
        (1.0 - 2.0 * (rxx + ryy)) * tz;
      qx = mqx; qy = mqy; qz = mqz; qw = mqw;
      tx = px + T.transform.translation.x;
      ty = py + T.transform.translation.y;
      tz = pz + T.transform.translation.z;
    } catch (const tf2::TransformException & ex) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "free_cam_pose: lookupTransform map->%s failed: %s; applying pose as-is",
        src_frame.c_str(), ex.what());
    }
  }

  double xx = qx * qx, yy = qy * qy, zz = qz * qz;
  double xy = qx * qy, xz = qx * qz, yz = qy * qz;
  double wx = qw * qx, wy = qw * qy, wz = qw * qz;

  ovrtx_xform_matrix44d_t xform{};
  auto & m = xform.v;
  m[0] = 1.0 - 2.0 * (yy + zz); m[1] = 2.0 * (xy + wz);       m[2] = 2.0 * (xz - wy);
  m[3] = 0.0;
  m[4] = 2.0 * (xy - wz);       m[5] = 1.0 - 2.0 * (xx + zz); m[6] = 2.0 * (yz + wx);
  m[7] = 0.0;
  m[8] = 2.0 * (xz + wy);       m[9] = 2.0 * (yz - wx);       m[10] = 1.0 - 2.0 * (xx + yy);
  m[11] = 0.0;
  m[12] = tx;                     m[13] = ty;                     m[14] = tz;
  m[15] = 1.0;

  ovx_string_t prim_path{free_cam_prim_path_.c_str(), free_cam_prim_path_.size()};
  ovrtx_enqueue_result_t wr = ovrtx_set_xform_mat(renderer_, &prim_path, 1, &xform);
  if (wr.status != OVRTX_API_SUCCESS) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 1000,
      "free_cam set_xform_mat failed: %d", wr.status);
  }
}

void RenderNode::render_timer_callback()
{
  if (!scene_loaded_ || cameras_.empty()) {
    return;
  }

  // v2: push RSP-composed URDF-chain world poses to OVRTX before we
  // step the renderer, so this frame captures the current /tf state.
  apply_robot_tf();

  auto frame_start = Clock::now();
  FrameTiming timing{};

  if (last_frame_time_.time_since_epoch().count() > 0) {
    timing.interval_ms = ms_between(last_frame_time_, frame_start);
  }
  last_frame_time_ = frame_start;

  // --- step (all cameras in one call) ---
  auto t0 = Clock::now();

  std::vector<ovx_string_t> rp_strings;
  rp_strings.reserve(cameras_.size() + lidars_.size());
  for (const auto & cam : cameras_) {
    rp_strings.push_back({cam.render_product_path.c_str(), cam.render_product_path.size()});
  }
  for (const auto & l : lidars_) {
    rp_strings.push_back({l.render_product_path.c_str(), l.render_product_path.size()});
  }

  ovrtx_render_product_set_t render_products{};
  render_products.render_products = rp_strings.data();
  render_products.num_render_products = rp_strings.size();

  ovrtx_step_result_handle_t step_handle = 0;
  double dt = 1.0 / render_fps_;
  ovrtx_enqueue_result_t enqueue_result =
    ovrtx_step(renderer_, render_products, dt, &step_handle);
  if (check_error(enqueue_result, "step", get_logger())) {
    ++drop_count_;
    return;
  }

  auto t1 = Clock::now();
  timing.step_ms = ms_between(t0, t1);

  // --- wait ---
  // First call to wait_op also covers shader compilation + JIT warmup,
  // which can block for 30-90s on a fresh machine. Log once before that
  // wait so the operator knows blank output is expected; the cache makes
  // subsequent runs instant.
  if (!first_frame_done_) {
    RCLCPP_INFO(
      get_logger(),
      "OVRTX first frame: compiling shaders / warming GPU "
      "(typically 30-90s on a fresh machine, instant on subsequent "
      "runs once the cache is warm). Render output stays blank until "
      "this completes.");
  }

  ovrtx_op_wait_result_t wait_result{};
  ovrtx_result_t result = ovrtx_wait_op(
    renderer_, enqueue_result.op_index, ovrtx_timeout_infinite, &wait_result);

  if (!first_frame_done_) {
    first_frame_done_ = true;
    RCLCPP_INFO(get_logger(), "OVRTX first frame complete — renderer is live.");
  }

  if (check_error(result, "wait_op(step)", get_logger())) {
    ++drop_count_;
    ovrtx_destroy_results(renderer_, step_handle);
    return;
  }

  auto t2 = Clock::now();
  timing.wait_ms = ms_between(t1, t2);

  // --- fetch ---
  ovrtx_render_product_set_outputs_t outputs{};
  result = ovrtx_fetch_results(renderer_, step_handle, ovrtx_timeout_infinite, &outputs);
  if (check_error(result, "fetch_results", get_logger())) {
    ++drop_count_;
    ovrtx_destroy_results(renderer_, step_handle);
    return;
  }

  auto t3 = Clock::now();
  timing.fetch_ms = ms_between(t2, t3);

  // --- map + publish all cameras ---
  publish_camera_images(outputs);
  publish_lidar_pointclouds(outputs);

  auto t4 = Clock::now();
  timing.map_ms = ms_between(t3, t4);

  ovrtx_destroy_results(renderer_, step_handle);

  timing.total_ms = ms_between(frame_start, Clock::now());
  ++frame_count_;

  timing_history_.push_back(timing);
  if (timing_history_.size() > kHistorySize) {
    timing_history_.pop_front();
  }

  publish_timing(timing);
}

void RenderNode::publish_camera_images(
  const ovrtx_render_product_set_outputs_t & outputs)
{
  // Use the TF tree's time base (sim time from /tf_scene) so RViz can transform the output;
  // fall back to wall clock only before the first /tf_scene arrives.
  double timestamp = latest_tf_time_ > 0.0 ? latest_tf_time_ : now().seconds();

  // PROF (temporary, remove after diagnosing FPS): per-stage timing buckets accumulated over all
  // cameras this frame, printed once/sec at the end. Splits the coarse map_ms bucket into GPU
  // readback (ovrtx_map) vs memcpy vs per-plugin publish (plugin0=ROS, plugin1=DDS/JPEG).
  double prof_map_ms = 0.0;
  double prof_copy_ms = 0.0;
  std::vector<double> prof_plugin_ms(image_plugins_.size(), 0.0);
  std::vector<double> prof_cam_map_ms(cameras_.size(), 0.0);
  std::vector<double> prof_cam_pub_ms(cameras_.size(), 0.0);

  for (size_t cam_idx = 0; cam_idx < cameras_.size(); ++cam_idx) {
    const auto & cam = cameras_[cam_idx];

    ovrtx_render_var_output_handle_t ldr_handle =
      find_output_for_product(outputs, cam_idx, "LdrColor");
    if (ldr_handle == static_cast<ovrtx_render_var_output_handle_t>(-1)) {
      continue;
    }

    ovrtx_map_output_description_t map_desc{};
    map_desc.device_type = OVRTX_MAP_DEVICE_TYPE_CPU;
    ovrtx_render_var_output_t rendered_output{};
    auto _prof_t0 = Clock::now();
    ovrtx_result_t result = ovrtx_map_render_var_output(
      renderer_, ldr_handle, &map_desc, ovrtx_timeout_infinite, &rendered_output);
    {
      double _dt = ms_between(_prof_t0, Clock::now());
      prof_map_ms += _dt;
      prof_cam_map_ms[cam_idx] += _dt;
    }
    if (result.status != OVRTX_API_SUCCESS) {
      continue;
    }

    if (rendered_output.status == OVRTX_EVENT_COMPLETED &&
      rendered_output.num_tensors >= 1 && rendered_output.tensors[0].dl)
    {
      const DLTensor & tensor = *rendered_output.tensors[0].dl;
      int height = static_cast<int>(tensor.shape[0]);
      int width = static_cast<int>(tensor.shape[1]);
      int channels = 4;

      // Shared ownership: an async plugin may hold the frame rather than
      // copy the whole image again.
      auto img_owned = std::make_shared<CameraImageMsg>();
      CameraImageMsg & img_msg = *img_owned;
      img_msg.frame_id = cam.path;
      img_msg.timestamp = timestamp;
      img_msg.width = static_cast<uint32_t>(width);
      img_msg.height = static_cast<uint32_t>(height);
      img_msg.encoding = "rgba8";
      img_msg.step = static_cast<uint32_t>(width * channels);
      auto _prof_cp0 = Clock::now();
      img_msg.data.resize(height * width * channels);
      std::memcpy(img_msg.data.data(), tensor.data, img_msg.data.size());
      // Must happen before the plugins publish, so ROS and DDS consumers see the same masked frame.
      const double mask_r = fisheye_mask_radius_px(cam);
      if (mask_r > 0.0) {
        apply_circular_mask(
          img_msg.data.data(), width, height, channels, cam.cx, cam.cy, mask_r);
      }
      prof_copy_ms += ms_between(_prof_cp0, Clock::now());

      CameraInfoMsg info_msg;
      info_msg.frame_id = cam.path;
      info_msg.timestamp = timestamp;
      info_msg.width = static_cast<uint32_t>(cam.width);
      info_msg.height = static_cast<uint32_t>(cam.height);
      info_msg.distortion_model = cam.distortion_model;
      info_msg.d = (cam.distortion_model == "equidistant") ?
        std::vector<double>{cam.k1, cam.k2, cam.k3, cam.k4} :
      std::vector<double>{cam.k1, cam.k2, cam.p1, cam.p2, cam.k3};
      info_msg.k = {cam.fx, 0.0, cam.cx, 0.0, cam.fy, cam.cy, 0.0, 0.0, 1.0};
      info_msg.r = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
      info_msg.p = {cam.fx, 0.0, cam.cx, 0.0, 0.0, cam.fy, cam.cy, 0.0, 0.0, 0.0, 1.0, 0.0};

      for (size_t _p = 0; _p < image_plugins_.size(); ++_p) {
        auto _prof_p0 = Clock::now();
        image_plugins_[_p]->publish_camera_image(cam.topic, img_owned);
        image_plugins_[_p]->publish_camera_info(cam.topic, info_msg);
        double _dt = ms_between(_prof_p0, Clock::now());
        prof_plugin_ms[_p] += _dt;
        prof_cam_pub_ms[cam_idx] += _dt;
      }
    }

    ovrtx_cuda_sync_t no_sync{};
    ovrtx_unmap_render_var_output(renderer_, rendered_output.map_handle, no_sync);

    // ----- Depth map (DistanceToImagePlaneSD, float32, unit: meters along optical axis) -----
    // Only when scene yaml has a non-empty topic.depth does the isaac_scene stage author
    // DistanceToImagePlaneSD render var on the render layer; this mirrors that condition.
    if (cam.depth_topic.empty()) {
      continue;
    }

    ovrtx_render_var_output_handle_t depth_handle =
      find_output_for_product(outputs, cam_idx, "DistanceToImagePlaneSD");
    if (depth_handle == static_cast<ovrtx_render_var_output_handle_t>(-1)) {
      continue;
    }

    ovrtx_map_output_description_t depth_map_desc{};
    depth_map_desc.device_type = OVRTX_MAP_DEVICE_TYPE_CPU;
    ovrtx_render_var_output_t depth_output{};
    auto _prof_td0 = Clock::now();
    ovrtx_result_t depth_result = ovrtx_map_render_var_output(
      renderer_, depth_handle, &depth_map_desc, ovrtx_timeout_infinite, &depth_output);
    {
      double _dt = ms_between(_prof_td0, Clock::now());
      prof_map_ms += _dt;
      prof_cam_map_ms[cam_idx] += _dt;
    }
    if (depth_result.status != OVRTX_API_SUCCESS) {
      continue;
    }

    if (depth_output.status == OVRTX_EVENT_COMPLETED &&
      depth_output.num_tensors >= 1 && depth_output.tensors[0].dl)
    {
      const DLTensor & dt = *depth_output.tensors[0].dl;
      // Shape (H, W, 1), dtype=float32 (OVRTX contract), maps directly to sensor_msgs/Image 32FC1.
      int dheight = static_cast<int>(dt.shape[0]);
      int dwidth = static_cast<int>(dt.shape[1]);

      auto depth_owned = std::make_shared<CameraImageMsg>();

      CameraImageMsg & depth_msg = *depth_owned;
      depth_msg.frame_id = cam.path;
      depth_msg.timestamp = timestamp;
      depth_msg.width = static_cast<uint32_t>(dwidth);
      depth_msg.height = static_cast<uint32_t>(dheight);
      depth_msg.encoding = "32FC1";
      depth_msg.step = static_cast<uint32_t>(dwidth * sizeof(float));
      auto _prof_dcp0 = Clock::now();
      depth_msg.data.resize(static_cast<size_t>(dheight) * dwidth * sizeof(float));
      // ovrtx returns +inf for rays that hit nothing (e.g. 3DGS background, which
      // the ray tracer does not intersect). Forwarding inf verbatim breaks any
      // consumer that auto-normalises the float image: rviz computes max=inf, so
      // every finite pixel collapses to 0 and the whole image renders black unless
      // the user pins a manual range. Normalise to 0 — the ROS convention for "no
      // depth" — which is what the DDS bridge already does for its Z16 conversion.
      {
        const float * depth_src = static_cast<const float *>(dt.data);
        float * depth_dst = reinterpret_cast<float *>(depth_msg.data.data());
        const size_t depth_count = static_cast<size_t>(dheight) * static_cast<size_t>(dwidth);
        for (size_t i = 0; i < depth_count; ++i) {
          const float d = depth_src[i];
          depth_dst[i] = std::isfinite(d) ? d : 0.0f;
        }
      }
      prof_copy_ms += ms_between(_prof_dcp0, Clock::now());

      // CameraInfo shares the same path as RGB (same camera intrinsics); constructed separately to avoid coupling with the LDR scope above.
      CameraInfoMsg depth_info;
      depth_info.frame_id = cam.path;
      depth_info.timestamp = timestamp;
      depth_info.width = static_cast<uint32_t>(cam.width);
      depth_info.height = static_cast<uint32_t>(cam.height);
      depth_info.distortion_model = cam.distortion_model;
      depth_info.d = (cam.distortion_model == "equidistant") ?
        std::vector<double>{cam.k1, cam.k2, cam.k3, cam.k4} :
      std::vector<double>{cam.k1, cam.k2, cam.p1, cam.p2, cam.k3};
      depth_info.k = {cam.fx, 0.0, cam.cx, 0.0, cam.fy, cam.cy, 0.0, 0.0, 1.0};
      depth_info.r = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
      depth_info.p = {cam.fx, 0.0, cam.cx, 0.0, 0.0, cam.fy, cam.cy, 0.0, 0.0, 0.0, 1.0, 0.0};

      for (size_t _p = 0; _p < image_plugins_.size(); ++_p) {
        auto _prof_dp0 = Clock::now();
        image_plugins_[_p]->publish_camera_image(cam.depth_topic, depth_owned);
        image_plugins_[_p]->publish_camera_info(cam.depth_topic, depth_info);
        double _dt = ms_between(_prof_dp0, Clock::now());
        prof_plugin_ms[_p] += _dt;
        prof_cam_pub_ms[cam_idx] += _dt;
      }
    }

    ovrtx_unmap_render_var_output(renderer_, depth_output.map_handle, no_sync);
  }

  // PROF (temporary): once/sec breakdown of the publish stage (= the coarse map_ms bucket).
  double _prof_pub_sum = 0.0;
  for (double v : prof_plugin_ms) {
    _prof_pub_sum += v;
  }
  std::ostringstream _prof_pl;
  for (size_t _p = 0; _p < prof_plugin_ms.size(); ++_p) {
    _prof_pl << " plugin" << _p << "=" << prof_plugin_ms[_p] << "ms";
  }
  std::ostringstream _prof_cams;
  for (size_t _c = 0; _c < cameras_.size(); ++_c) {
    _prof_cams << "\n    cam[" << _c << "] " << cameras_[_c].topic << " "
               << cameras_[_c].width << "x" << cameras_[_c].height
               << ": map=" << prof_cam_map_ms[_c] << "ms pub=" << prof_cam_pub_ms[_c] << "ms";
  }
  RCLCPP_INFO_THROTTLE(
    get_logger(), *get_clock(), 1000,
    "[prof] cams=%zu gpu_readback(ovrtx_map)=%.1fms memcpy=%.1fms publish_total=%.1fms%s%s",
    cameras_.size(), prof_map_ms, prof_copy_ms, _prof_pub_sum, _prof_pl.str().c_str(),
    _prof_cams.str().c_str());
}

void RenderNode::publish_lidar_pointclouds(
  const ovrtx_render_product_set_outputs_t & outputs)
{
  if (lidars_.empty()) {
    return;
  }

  // Use the TF tree's time base (sim time from /tf_scene) so RViz can transform the output;
  // fall back to wall clock only before the first /tf_scene arrives.
  double timestamp = latest_tf_time_ > 0.0 ? latest_tf_time_ : now().seconds();
  auto wall_now = Clock::now();
  constexpr uint32_t point_step = 22;
  constexpr uint8_t reflectivity_default = 128;

  std::vector<uint8_t> merged;  // accumulated base_link-frame points across all lidars
  bool published_any = false;   // gate the (10 Hz) base_link->world TF publish to cloud cadence

  for (auto & lidar : lidars_) {
    ovrtx_render_var_output_handle_t handle =
      find_output_by_path(outputs, lidar.render_product_path, "PointCloud");
    if (handle == static_cast<ovrtx_render_var_output_handle_t>(-1)) {
      continue;
    }

    ovrtx_map_output_description_t map_desc{};
    map_desc.device_type = OVRTX_MAP_DEVICE_TYPE_CPU;
    ovrtx_render_var_output_t pc{};
    ovrtx_result_t result = ovrtx_map_render_var_output(
      renderer_, handle, &map_desc, ovrtx_timeout_infinite, &pc);
    if (result.status != OVRTX_API_SUCCESS) {
      continue;
    }

    ovrtx_cuda_sync_t no_sync{};
    if (pc.status != OVRTX_EVENT_COMPLETED) {
      ovrtx_unmap_render_var_output(renderer_, pc.map_handle, no_sync);
      continue;
    }

    const ovrtx_render_var_tensor_t * counts_t = find_pc_tensor(pc, "Counts");
    const ovrtx_render_var_tensor_t * coords_t = find_pc_tensor(pc, "Coordinates");
    const ovrtx_render_var_tensor_t * inten_t = find_pc_tensor(pc, "Intensity");
    const ovrtx_render_var_tensor_t * flags_t = find_pc_tensor(pc, "Flags");
    if (!counts_t || !counts_t->dl || !coords_t || !coords_t->dl) {
      ovrtx_unmap_render_var_output(renderer_, pc.map_handle, no_sync);
      continue;
    }

    const int32_t * counts = static_cast<const int32_t *>(counts_t->dl->data);
    const int32_t valid = counts ? counts[0] : 0;
    // partialOutputs=false => a full scan only materializes at the lidar's scan rate (~10 Hz);
    // empty/partial frames carry Counts==0, so this also throttles publish to that rate.
    if (valid <= 0) {
      ovrtx_unmap_render_var_output(renderer_, pc.map_handle, no_sync);
      continue;
    }
    // Safety throttle: never publish a given lidar faster than ~10 Hz.
    if (lidar.last_publish.time_since_epoch().count() != 0 &&
      std::chrono::duration<double, std::milli>(wall_now - lidar.last_publish).count() < 90.0)
    {
      ovrtx_unmap_render_var_output(renderer_, pc.map_handle, no_sync);
      continue;
    }
    lidar.last_publish = wall_now;

    const float * coords = static_cast<const float *>(coords_t->dl->data);
    const int64_t alloc = coords_t->dl->shape[1];  // 3 x alloc, row-major (x row, y row, z row)
    const float * inten = (inten_t && inten_t->dl) ?
      static_cast<const float *>(inten_t->dl->data) : nullptr;
    // VALID bit (1<<6). With skipDroppingInvalidPoints=false this is set on every delivered
    // entry; kept as a safety so stray no-return rays never reach the cloud.
    const uint8_t * flags = (flags_t && flags_t->dl) ?
      static_cast<const uint8_t *>(flags_t->dl->data) : nullptr;
    constexpr uint8_t kFlagValid = 0x40;

    // base_link <- sensor rotation matrix from the static quaternion.
    const double w = lidar.bl_T_s_wxyz[0], qx = lidar.bl_T_s_wxyz[1];
    const double qy = lidar.bl_T_s_wxyz[2], qz = lidar.bl_T_s_wxyz[3];
    const double xx = qx * qx, yy = qy * qy, zz = qz * qz;
    const double xy = qx * qy, xz = qx * qz, yz = qy * qz;
    const double wx = w * qx, wy = w * qy, wz = w * qz;
    const double r00 = 1 - 2 * (yy + zz), r01 = 2 * (xy - wz), r02 = 2 * (xz + wy);
    const double r10 = 2 * (xy + wz), r11 = 1 - 2 * (xx + zz), r12 = 2 * (yz - wx);
    const double r20 = 2 * (xz - wy), r21 = 2 * (yz + wx), r22 = 1 - 2 * (xx + yy);
    const double tx = lidar.bl_T_s_xyz[0], ty = lidar.bl_T_s_xyz[1], tz = lidar.bl_T_s_xyz[2];

    PointCloudMsg msg;
    msg.frame_id = lidar.sensor_frame;  // per-lidar cloud is in the sensor frame (rmagine parity)
    msg.timestamp = timestamp;
    msg.point_step = point_step;
    msg.is_dense = true;
    msg.height = 1;
    msg.data.resize(static_cast<size_t>(valid) * point_step);
    const size_t merged_base = merged.size();
    if (!merged_lidar_topic_.empty()) {
      merged.resize(merged_base + static_cast<size_t>(valid) * point_step);
    }

    uint32_t out = 0;
    for (int32_t j = 0; j < valid; ++j) {
      if (flags && !(flags[j] & kFlagValid)) {
        continue;
      }
      // ovrtx emits points already in the original sensor (livox) frame; base_link_T_sensor
      // (the mount) maps them to base_link. Per-lidar cloud keeps the raw sensor-frame point;
      // the merged cloud accumulates the base_link-frame point.
      const double xs = coords[0 * alloc + j];
      const double ys = coords[1 * alloc + j];
      const double zs = coords[2 * alloc + j];
      const float sx = static_cast<float>(xs);
      const float sy = static_cast<float>(ys);
      const float sz = static_cast<float>(zs);
      uint8_t refl = reflectivity_default;
      if (inten) {
        const float v = inten[j] * 255.0f;
        refl = static_cast<uint8_t>(v < 0.0f ? 0.0f : (v > 255.0f ? 255.0f : v));
      }
      uint8_t * row = msg.data.data() + static_cast<size_t>(out) * point_step;
      std::memcpy(row + 0, &sx, sizeof(float));
      std::memcpy(row + 4, &sy, sizeof(float));
      std::memcpy(row + 8, &sz, sizeof(float));
      row[12] = refl;
      row[13] = 0;
      std::memcpy(row + 14, &timestamp, sizeof(double));
      if (!merged_lidar_topic_.empty()) {
        const float bx = static_cast<float>(r00 * xs + r01 * ys + r02 * zs + tx);
        const float by = static_cast<float>(r10 * xs + r11 * ys + r12 * zs + ty);
        const float bz = static_cast<float>(r20 * xs + r21 * ys + r22 * zs + tz);
        uint8_t * mrow = merged.data() + merged_base + static_cast<size_t>(out) * point_step;
        std::memcpy(mrow + 0, &bx, sizeof(float));
        std::memcpy(mrow + 4, &by, sizeof(float));
        std::memcpy(mrow + 8, &bz, sizeof(float));
        mrow[12] = refl;
        mrow[13] = 0;
        std::memcpy(mrow + 14, &timestamp, sizeof(double));
      }
      ++out;
    }
    msg.data.resize(static_cast<size_t>(out) * point_step);
    msg.width = out;
    if (!merged_lidar_topic_.empty()) {
      merged.resize(merged_base + static_cast<size_t>(out) * point_step);
    }
    if (out == 0) {
      ovrtx_unmap_render_var_output(renderer_, pc.map_handle, no_sync);
      continue;
    }

    for (auto & plugin : image_plugins_) {
      plugin->publish_pointcloud(lidar.topic, msg);
    }
    published_any = true;

    ovrtx_unmap_render_var_output(renderer_, pc.map_handle, no_sync);
  }

  if (!merged_lidar_topic_.empty() && !merged.empty()) {
    PointCloudMsg merged_msg;
    merged_msg.frame_id = "base_link";
    merged_msg.timestamp = timestamp;
    merged_msg.point_step = point_step;
    merged_msg.is_dense = true;
    merged_msg.height = 1;
    merged_msg.width = static_cast<uint32_t>(merged.size() / point_step);
    merged_msg.data = std::move(merged);
    for (auto & plugin : image_plugins_) {
      plugin->publish_pointcloud(merged_lidar_topic_, merged_msg);
    }
    published_any = true;
  }

  // Emit base_link->world together with the cloud (same `timestamp`, ~10 Hz) so every
  // published cloud has an exactly-matching transform in the consumer's TF buffer.
  if (published_any) {
    publish_lidar_tf();
  }
}

void RenderNode::stats_timer_callback()
{
  if (timing_history_.empty()) {
    return;
  }

  auto extract = [&](auto field) {
      std::vector<double> v;
      v.reserve(timing_history_.size());
      for (const auto & t : timing_history_) {
        v.push_back(t.*field);
      }
      return v;
    };

  auto stats = [](const std::vector<double> & v)
    -> std::tuple<double, double, double, double> {
      double sum = std::accumulate(v.begin(), v.end(), 0.0);
      double mean = sum / static_cast<double>(v.size());
      double mn = *std::min_element(v.begin(), v.end());
      double mx = *std::max_element(v.begin(), v.end());
      double sq_sum = 0.0;
      for (auto x : v) {
        sq_sum += (x - mean) * (x - mean);
      }
      double stddev = std::sqrt(sq_sum / static_cast<double>(v.size()));
      return {mean, mn, mx, stddev};
    };

  auto [total_mean, total_min, total_max, total_std] =
    stats(extract(&FrameTiming::total_ms));
  auto [step_mean, step_min, step_max, step_std] =
    stats(extract(&FrameTiming::step_ms));
  auto [wait_mean, wait_min, wait_max, wait_std] =
    stats(extract(&FrameTiming::wait_ms));
  auto [interval_mean, interval_min, interval_max, interval_std] =
    stats(extract(&FrameTiming::interval_ms));

  double target_interval_ms = 1000.0 / render_fps_;
  double jitter_ms = interval_std;
  double actual_fps = (interval_mean > 0.0) ? 1000.0 / interval_mean : 0.0;

  RCLCPP_INFO(
    get_logger(),
    "--- Render Stats (last %zu frames, %zu cameras) ---\n"
    "  Frames: %lu  Drops: %lu  Actual FPS: %.1f (target %.1f)\n"
    "  Total:    mean=%.2f  min=%.2f  max=%.2f  std=%.2f ms\n"
    "  Step:     mean=%.2f  min=%.2f  max=%.2f  std=%.2f ms\n"
    "  Wait:     mean=%.2f  min=%.2f  max=%.2f  std=%.2f ms\n"
    "  Interval: mean=%.2f  min=%.2f  max=%.2f  jitter=%.2f ms (target=%.2f ms)",
    timing_history_.size(), cameras_.size(),
    frame_count_, drop_count_, actual_fps, render_fps_,
    total_mean, total_min, total_max, total_std,
    step_mean, step_min, step_max, step_std,
    wait_mean, wait_min, wait_max, wait_std,
    interval_mean, interval_min, interval_max, jitter_ms, target_interval_ms);
}

void RenderNode::publish_timing(const FrameTiming & t)
{
  auto msg = std::make_unique<std_msgs::msg::Float64MultiArray>();
  msg->layout.dim.resize(1);
  msg->layout.dim[0].label = "timing";
  msg->layout.dim[0].size = 7;
  msg->layout.dim[0].stride = 7;
  msg->data = {
    t.step_ms,
    t.wait_ms,
    t.fetch_ms,
    t.map_ms,
    t.publish_ms,
    t.total_ms,
    t.interval_ms,
  };
  timing_pub_->publish(std::move(msg));
}

ovrtx_render_var_output_handle_t RenderNode::find_output_for_product(
  const ovrtx_render_product_set_outputs_t & outputs,
  size_t product_index,
  const char * render_var_name)
{
  const auto & cam = cameras_[product_index];
  const std::string & target_path = cam.render_product_path;

  static bool logged_once = false;
  if (!logged_once) {
    logged_once = true;
    for (size_t p = 0; p < outputs.output_count; ++p) {
      const auto & product = outputs.outputs[p];
      std::string out_path(product.render_product_path.ptr,
        product.render_product_path.length);
      RCLCPP_INFO(
        get_logger(), "Output[%zu] product=%s frames=%zu",
        p, out_path.c_str(), product.output_frame_count);
      for (size_t f = 0; f < product.output_frame_count; ++f) {
        const auto & frame = product.output_frames[f];
        for (size_t v = 0; v < frame.render_var_count; ++v) {
          const auto & var = frame.output_render_vars[v];
          std::string vname(var.render_var_name.ptr ? var.render_var_name.ptr : "(null)",
            var.render_var_name.length);
          RCLCPP_INFO(get_logger(), "  var[%zu]: name='%s'", v, vname.c_str());
        }
      }
    }
  }

  for (size_t p = 0; p < outputs.output_count; ++p) {
    const auto & product = outputs.outputs[p];
    std::string out_path(product.render_product_path.ptr,
      product.render_product_path.length);
    if (out_path != target_path) {
      continue;
    }
    for (size_t f = 0; f < product.output_frame_count; ++f) {
      const auto & frame = product.output_frames[f];
      for (size_t v = 0; v < frame.render_var_count; ++v) {
        const auto & var = frame.output_render_vars[v];
        if (var.render_var_name.ptr &&
          strncmp(
            var.render_var_name.ptr, render_var_name,
            var.render_var_name.length) == 0)
        {
          return var.output_handle;
        }
      }
    }
    break;
  }
  return static_cast<ovrtx_render_var_output_handle_t>(-1);
}

ovrtx_render_var_output_handle_t RenderNode::find_output_by_path(
  const ovrtx_render_product_set_outputs_t & outputs,
  const std::string & product_path,
  const char * render_var_name)
{
  const size_t var_len = std::strlen(render_var_name);
  for (size_t p = 0; p < outputs.output_count; ++p) {
    const auto & product = outputs.outputs[p];
    std::string out_path(product.render_product_path.ptr, product.render_product_path.length);
    if (out_path != product_path) {
      continue;
    }
    for (size_t f = 0; f < product.output_frame_count; ++f) {
      const auto & frame = product.output_frames[f];
      for (size_t v = 0; v < frame.render_var_count; ++v) {
        const auto & var = frame.output_render_vars[v];
        if (var.render_var_name.ptr && var.render_var_name.length == var_len &&
          strncmp(var.render_var_name.ptr, render_var_name, var_len) == 0)
        {
          return var.output_handle;
        }
      }
    }
    break;
  }
  return static_cast<ovrtx_render_var_output_handle_t>(-1);
}

}  // namespace genie_sim_render

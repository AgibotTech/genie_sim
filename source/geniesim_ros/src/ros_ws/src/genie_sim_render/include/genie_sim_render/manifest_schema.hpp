// Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
// Author: Genie Sim Team
// License: Mozilla Public License Version 2.0

#pragma once

// C++ mirror of ``genie_sim_engine_core.schema.manifest.Manifest``.
//
// Every downstream C++ reader (render_node, rmagine renderers,
// mujoco_gui_main) parses ``manifest.json`` via nlohmann::json. Before
// this header, each reader spelled out its own field-access chain
// (``manifest["cameras"][i].value("path", "")``), and a rename in the
// Python producer went unnoticed until the first runtime frame. Two
// field-drift bugs on the mujoco path (missing ``path``, empty
// ``render_layer_usda``) survived to launch because of that silent
// drift — this header exists to eliminate that class of bug.
//
// This header declares the ABI-shaped structs and hand-written
// ``to_json``/``from_json`` overloads that mirror
// :meth:`Manifest.to_dict` / :meth:`Manifest.from_dict` in the Python
// schema. Fields use ``value()`` with defaults so extra keys are
// ignored and missing keys fall back — matches the human tolerance we
// want for stale/partial stage directories, while ``Manifest::load``
// still hard-checks ``version``.
//
// (We don't use NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT — that
// macro landed in nlohmann/json 3.11 and this workspace pins 3.10.)
//
// ``kExpectedVersion`` must stay in lockstep with
// ``MANIFEST_VERSION`` in ``manifest.py``. Change one, change both.

#include <nlohmann/json.hpp>

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace genie_sim_render
{

//! One ``cameras[]`` entry in ``manifest.json``.
//!
//! Field-for-field mirror of the Python ``CameraManifest`` dataclass.
//! Every default here matches the Python default so an
//! older-schema/newer-writer mismatch stays diagnosable field by field
//! rather than exploding on the first ``at("path")``.
struct CameraManifest
{
  std::string path;
  std::string topic;
  std::string depth_topic;
  std::string dds_topic;
  std::string dds_depth_topic;
  std::string render_product_path;
  std::string frame_id;
  int32_t width{1280};
  int32_t height{800};
  double fx{610.0};
  double fy{610.0};
  double cx{640.0};
  double cy{400.0};
  double k1{0.0};
  double k2{0.0};
  double p1{0.0};
  double p2{0.0};
  double k3{0.0};
  double k4{0.0};
  std::string model;
  // Total field of view of a >180 deg fisheye, in degrees, straight from the
  // scene yaml (``mask_fov_deg`` / ``max_fov_deg``). Only used to size the
  // circular black mask -- see ``fisheye_mask_radius_px`` in render_node.cpp.
  // 0 means "not a masked fisheye".
  double max_fov_deg{0.0};
  double min_range{0.01};
  double max_range{10000.0};
  bool is_free_cam{false};
};

//! ``lidars[].base_link_T_sensor`` xform.
struct LidarBaseLinkT
{
  //! Translation, xyz in metres.
  std::array<double, 3> xyz{0.0, 0.0, 0.0};
  //! Rotation, w-first unit quaternion.
  std::array<double, 4> wxyz{1.0, 0.0, 0.0, 0.0};
};

//! One ``lidars[]`` entry.
struct LidarManifest
{
  std::string render_product_path;
  std::string topic;
  std::string frame_id{"base_link"};
  std::string parent_body;
  LidarBaseLinkT base_link_T_sensor{};
};

//! Root ``manifest.json``. Path fields are stored **relative to
//! ``base_path``** on disk (or to the manifest's own directory when
//! ``base_path`` is empty). Callers should resolve via
//! :func:`resolve_manifest_path` after loading.
struct Manifest
{
  //! Bumped in lockstep with ``MANIFEST_VERSION`` in ``manifest.py``.
  //! ``Manifest::load`` compares ``version`` against this and throws
  //! ``std::runtime_error`` on mismatch — the launcher can then prompt
  //! the operator to re-run the assemble CLI.
  static constexpr int32_t kExpectedVersion{4};

  int32_t version{kExpectedVersion};
  std::string physics_engine;
  //! Launcher YAML name / path passed via ``launcher_config:=…`` at
  //! bringup time. See Python schema for details.
  std::string launcher_config;
  std::string base_path;
  std::string stage_dir;
  std::string scene_yaml;
  std::string robot_prefix{"genie"};
  bool robot_from_urdf{false};
  std::string scene_usda;
  std::string robot_usda;
  std::string robot_visual_usda;
  std::string render_layer_usda;
  std::string free_cam_prim_path;
  std::vector<CameraManifest> cameras;
  std::vector<LidarManifest> lidars;
  //! Logical-name → path map for stage artefacts. Opaque top-level.
  nlohmann::json files{nlohmann::json::object()};
  //! Cross-backend robot metadata (URDF hash stamp, ``body_paths``, …).
  nlohmann::json robot{nlohmann::json::object()};
  //! Cross-backend scene metadata (``body_paths``, ``joint_names``, …).
  //! Opaque top-level bucket — mirrors ``robot`` on purpose.
  nlohmann::json scene{nlohmann::json::object()};
  //! Backend-specific slice keyed by engine name. Opaque to core.
  nlohmann::json physics{nlohmann::json::object()};

  //! Load and validate ``manifest.json``.
  //!
  //! Version mismatches throw ``std::runtime_error`` with a
  //! human-readable diagnostic naming the on-disk version and the
  //! version this binary expects. All other JSON errors propagate
  //! unchanged (missing/unreadable file, malformed JSON).
  static Manifest load(const std::filesystem::path & path);
};

// -- nlohmann ADL hooks --------------------------------------------
// Hand-written because the workspace's nlohmann/json (3.10) doesn't
// ship NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT. The 3.10-shipped
// NLOHMANN_JSON_FROM uses ``at()`` which throws on missing keys — we
// want ``value(key, default)`` so a partial or extra-fielded stage
// dir stays diagnosable. When (if) the workspace upgrades to 3.11+,
// these can collapse back to the macro form.

void to_json(nlohmann::json & j, const CameraManifest & c);
void from_json(const nlohmann::json & j, CameraManifest & c);

void to_json(nlohmann::json & j, const LidarBaseLinkT & t);
void from_json(const nlohmann::json & j, LidarBaseLinkT & t);

void to_json(nlohmann::json & j, const LidarManifest & l);
void from_json(const nlohmann::json & j, LidarManifest & l);

void to_json(nlohmann::json & j, const Manifest & m);
void from_json(const nlohmann::json & j, Manifest & m);

//! Resolve a manifest-stored path relative to ``base_path``.
//!
//! Empty input yields empty output. Absolute paths pass through. When
//! ``base_path`` on the manifest is empty, ``fallback_base`` is used
//! (typically the manifest file's own parent directory), matching the
//! historical ``RenderNode::load_manifest`` behaviour.
std::string resolve_manifest_path(
  const std::string & rel_or_abs,
  const std::string & base_path,
  const std::filesystem::path & fallback_base);

}  // namespace genie_sim_render

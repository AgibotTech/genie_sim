// Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
// Author: Genie Sim Team
// License: Mozilla Public License Version 2.0

#include "genie_sim_render/manifest_schema.hpp"

#include <fstream>
#include <sstream>

namespace genie_sim_render
{

namespace fs = std::filesystem;

// -- CameraManifest -------------------------------------------------

void to_json(nlohmann::json & j, const CameraManifest & c)
{
  j = nlohmann::json{
    {"path", c.path},
    {"topic", c.topic},
    {"depth_topic", c.depth_topic},
    {"dds_topic", c.dds_topic},
    {"dds_depth_topic", c.dds_depth_topic},
    {"render_product_path", c.render_product_path},
    {"frame_id", c.frame_id},
    {"width", c.width},
    {"height", c.height},
    {"fx", c.fx},
    {"fy", c.fy},
    {"cx", c.cx},
    {"cy", c.cy},
    {"k1", c.k1},
    {"k2", c.k2},
    {"p1", c.p1},
    {"p2", c.p2},
    {"k3", c.k3},
    {"k4", c.k4},
    {"model", c.model},
    {"max_fov_deg", c.max_fov_deg},
    {"min_range", c.min_range},
    {"max_range", c.max_range},
    {"is_free_cam", c.is_free_cam},
  };
}

void from_json(const nlohmann::json & j, CameraManifest & c)
{
  c = CameraManifest{};
  c.path = j.value("path", c.path);
  c.topic = j.value("topic", c.topic);
  c.depth_topic = j.value("depth_topic", c.depth_topic);
  c.dds_topic = j.value("dds_topic", c.dds_topic);
  c.dds_depth_topic = j.value("dds_depth_topic", c.dds_depth_topic);
  c.render_product_path = j.value("render_product_path", c.render_product_path);
  c.frame_id = j.value("frame_id", c.frame_id);
  c.width = j.value("width", c.width);
  c.height = j.value("height", c.height);
  c.fx = j.value("fx", c.fx);
  c.fy = j.value("fy", c.fy);
  c.cx = j.value("cx", c.cx);
  c.cy = j.value("cy", c.cy);
  c.k1 = j.value("k1", c.k1);
  c.k2 = j.value("k2", c.k2);
  c.p1 = j.value("p1", c.p1);
  c.p2 = j.value("p2", c.p2);
  c.k3 = j.value("k3", c.k3);
  c.k4 = j.value("k4", c.k4);
  c.model = j.value("model", c.model);
  c.max_fov_deg = j.value("max_fov_deg", c.max_fov_deg);
  c.min_range = j.value("min_range", c.min_range);
  c.max_range = j.value("max_range", c.max_range);
  c.is_free_cam = j.value("is_free_cam", c.is_free_cam);
}

// -- LidarBaseLinkT / LidarManifest ---------------------------------

void to_json(nlohmann::json & j, const LidarBaseLinkT & t)
{
  j = nlohmann::json{
    {"xyz", nlohmann::json::array({t.xyz[0], t.xyz[1], t.xyz[2]})},
    {"wxyz", nlohmann::json::array({t.wxyz[0], t.wxyz[1], t.wxyz[2], t.wxyz[3]})},
  };
}

void from_json(const nlohmann::json & j, LidarBaseLinkT & t)
{
  t = LidarBaseLinkT{};
  if (j.contains("xyz") && j["xyz"].is_array() && j["xyz"].size() == 3) {
    for (int k = 0; k < 3; ++k) {
      t.xyz[k] = j["xyz"][k].get<double>();
    }
  }
  if (j.contains("wxyz") && j["wxyz"].is_array() && j["wxyz"].size() == 4) {
    for (int k = 0; k < 4; ++k) {
      t.wxyz[k] = j["wxyz"][k].get<double>();
    }
  }
}

void to_json(nlohmann::json & j, const LidarManifest & l)
{
  j = nlohmann::json{
    {"render_product_path", l.render_product_path},
    {"topic", l.topic},
    {"frame_id", l.frame_id},
    {"parent_body", l.parent_body},
    {"base_link_T_sensor", l.base_link_T_sensor},
  };
}

void from_json(const nlohmann::json & j, LidarManifest & l)
{
  l = LidarManifest{};
  l.render_product_path = j.value("render_product_path", l.render_product_path);
  l.topic = j.value("topic", l.topic);
  l.frame_id = j.value("frame_id", l.frame_id);
  l.parent_body = j.value("parent_body", l.parent_body);
  if (j.contains("base_link_T_sensor")) {
    l.base_link_T_sensor = j["base_link_T_sensor"].get<LidarBaseLinkT>();
  }
}

// -- Manifest -------------------------------------------------------

void to_json(nlohmann::json & j, const Manifest & m)
{
  j = nlohmann::json{
    {"version", m.version},
    {"physics_engine", m.physics_engine},
    {"launcher_config", m.launcher_config},
    {"base_path", m.base_path},
    {"stage_dir", m.stage_dir},
    {"scene_yaml", m.scene_yaml},
    {"robot_prefix", m.robot_prefix},
    {"robot_from_urdf", m.robot_from_urdf},
    {"scene_usda", m.scene_usda},
    {"robot_usda", m.robot_usda},
    {"robot_visual_usda", m.robot_visual_usda},
    {"render_layer_usda", m.render_layer_usda},
    {"free_cam_prim_path", m.free_cam_prim_path},
    {"cameras", m.cameras},
    {"lidars", m.lidars},
    {"files", m.files},
    {"robot", m.robot},
    {"scene", m.scene},
    {"physics", m.physics},
  };
}

void from_json(const nlohmann::json & j, Manifest & m)
{
  m = Manifest{};
  m.version = j.value("version", m.version);
  m.physics_engine = j.value("physics_engine", m.physics_engine);
  m.launcher_config = j.value("launcher_config", m.launcher_config);
  m.base_path = j.value("base_path", m.base_path);
  m.stage_dir = j.value("stage_dir", m.stage_dir);
  m.scene_yaml = j.value("scene_yaml", m.scene_yaml);
  m.robot_prefix = j.value("robot_prefix", m.robot_prefix);
  m.robot_from_urdf = j.value("robot_from_urdf", m.robot_from_urdf);
  m.scene_usda = j.value("scene_usda", m.scene_usda);
  m.robot_usda = j.value("robot_usda", m.robot_usda);
  m.robot_visual_usda = j.value("robot_visual_usda", m.robot_visual_usda);
  m.render_layer_usda = j.value("render_layer_usda", m.render_layer_usda);
  m.free_cam_prim_path = j.value("free_cam_prim_path", m.free_cam_prim_path);
  if (j.contains("cameras") && j["cameras"].is_array()) {
    m.cameras = j["cameras"].get<std::vector<CameraManifest>>();
  }
  if (j.contains("lidars") && j["lidars"].is_array()) {
    m.lidars = j["lidars"].get<std::vector<LidarManifest>>();
  }
  if (j.contains("files")) {
    m.files = j["files"];
  }
  if (j.contains("robot")) {
    m.robot = j["robot"];
  }
  if (j.contains("scene")) {
    m.scene = j["scene"];
  }
  if (j.contains("physics")) {
    m.physics = j["physics"];
  }
}

// -- Public API -----------------------------------------------------

std::string resolve_manifest_path(
  const std::string & rel_or_abs,
  const std::string & base_path,
  const fs::path & fallback_base)
{
  if (rel_or_abs.empty()) {
    return std::string{};
  }
  fs::path p{rel_or_abs};
  if (p.is_absolute()) {
    return p.lexically_normal().string();
  }
  fs::path base = base_path.empty() ? fallback_base : fs::path{base_path};
  return (base / p).lexically_normal().string();
}


Manifest Manifest::load(const fs::path & path)
{
  fs::path resolved = path;
  if (resolved.is_relative()) {
    resolved = fs::current_path() / resolved;
  }
  if (!fs::exists(resolved)) {
    throw std::runtime_error("Manifest not found: " + resolved.string());
  }

  std::ifstream in{resolved};
  if (!in) {
    throw std::runtime_error("Manifest unreadable: " + resolved.string());
  }
  nlohmann::json raw = nlohmann::json::parse(in);
  if (!raw.is_object()) {
    throw std::runtime_error("Manifest is not a JSON object: " + resolved.string());
  }

  int32_t on_disk_version = raw.value("version", -1);
  if (on_disk_version != Manifest::kExpectedVersion) {
    std::ostringstream msg;
    msg << "manifest at " << resolved.string()
        << " has version=" << on_disk_version
        << "; this binary expects " << Manifest::kExpectedVersion
        << ". Re-run the assemble CLI.";
    throw std::runtime_error(msg.str());
  }

  Manifest m = raw.get<Manifest>();
  // Legacy alias: pre-schema producers wrote ``usd_path`` for what is
  // now ``scene_usda``. Fall back so a stale stage dir still resolves
  // its scene until the operator re-runs assemble.
  if (m.scene_usda.empty() && raw.contains("usd_path")) {
    m.scene_usda = raw.at("usd_path").get<std::string>();
  }
  return m;
}

}  // namespace genie_sim_render

// Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
// Author: Genie Sim Team
// License: Mozilla Public License Version 2.0

#include "genie_sim_render/manifest_schema.hpp"

#include <gtest/gtest.h>

namespace genie_sim_render
{

TEST(Manifest, RoundTripsRobotAndSceneBodyPaths)
{
  Manifest m;
  m.physics_engine = "genie_sim_engine_newton";
  m.robot = {
    {"body_paths", {
        {"base_link", "/genie/Geometry/base_link"},
        {"body_link1", "/genie/Geometry/base_link/body_link1"},
      }},
  };
  m.scene = {
    {"body_paths", {
        {"_World_props_bottle_3", "/World/props/bottle_3"},
      }},
  };

  nlohmann::json j = m;
  Manifest reloaded = j.get<Manifest>();

  EXPECT_EQ(
    reloaded.robot.at("body_paths").at("base_link").get<std::string>(),
    "/genie/Geometry/base_link");
  EXPECT_EQ(
    reloaded.robot.at("body_paths").at("body_link1").get<std::string>(),
    "/genie/Geometry/base_link/body_link1");
  EXPECT_EQ(
    reloaded.scene.at("body_paths").at("_World_props_bottle_3").get<std::string>(),
    "/World/props/bottle_3");
}

TEST(Manifest, LauncherConfigStampSurvivesRoundtrip)
{
  Manifest m;
  m.launcher_config = "launcher_ovrtx_isaac_physx";

  nlohmann::json j = m;
  Manifest reloaded = j.get<Manifest>();

  EXPECT_EQ(reloaded.launcher_config, "launcher_ovrtx_isaac_physx");
}

}  // namespace genie_sim_render

# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Register libuipc asset assembly without the Newton extras stage."""

from genie_sim_engine_core.assemble import Pipeline
from .urdf_to_usd_stage import UrdfToUsdStage
from .isaac_scene_stage import IsaacSceneStage


@Pipeline.register("libuipc")
def libuipc_pipeline():
    return Pipeline(name="libuipc", stages=[UrdfToUsdStage, IsaacSceneStage])

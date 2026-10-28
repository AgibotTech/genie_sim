# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""genie_sim_engine_isaacsim.assemble — Isaac Sim pipeline factory.

Importing this module registers the ``isaac_newton`` pipeline with
:class:`genie_sim_engine_core.assemble.Pipeline`. The core CLI's
``--pipeline-module genie_sim_engine_isaacsim.assemble`` flag drives
the import so the decorator fires.

The pipeline composes:

* :class:`UrdfToUsdStage` — URDF → ``robot.usda`` via Isaac Sim's URDF
  importer (this package).
* :class:`IsaacSceneStage` — Isaac Sim scene assembly, writes
  ``render_layer.usda`` and returns the manifest slice to core (this package).
* :class:`NewtonExtrasStage` — ``newton_solvers.json`` for cloth /
  soft / particle entries (imported from :mod:`genie_sim_engine_newton.assemble`).
"""

from __future__ import annotations

from genie_sim_engine_newton.assemble.newton_extras_stage import NewtonExtrasStage
from genie_sim_engine_core.assemble import Pipeline

from .isaac_scene_stage import IsaacSceneStage
from .urdf_to_usd_stage import UrdfToUsdStage


@Pipeline.register("isaac_newton")
def isaac_newton_pipeline() -> Pipeline:
    """Canonical 3-stage pipeline for the Isaac+Newton backend."""
    return Pipeline(
        name="isaac_newton",
        stages=[UrdfToUsdStage, IsaacSceneStage, NewtonExtrasStage],
    )


__all__ = [
    "IsaacSceneStage",
    "UrdfToUsdStage",
    "isaac_newton_pipeline",
]

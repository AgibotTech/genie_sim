# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""genie_sim_engine_newton.assemble — Newton-only pipeline factories.

Importing this module registers the ``newton_only`` pipeline with
:class:`genie_sim_engine_core.assemble.Pipeline`. The core CLI's
``--pipeline-module genie_sim_engine_newton.assemble`` flag drives the import
so the decorator fires.

The ``isaac_newton`` pipeline lives in
:mod:`genie_sim_engine_isaacsim.assemble` — Isaac Sim's URDF→USD and
scene assemblers live in that package. Launchers pass BOTH modules to
``--pipeline-module`` so the union of pipelines is registered before
dispatch.
"""

from __future__ import annotations

from genie_sim_engine_core.assemble import Pipeline

from .newton_extras_stage import NewtonExtrasStage


@Pipeline.register("newton_only")
def newton_only_pipeline() -> Pipeline:
    """Single-stage pipeline that emits ``newton_solvers.json`` only.

    Useful for stripped-down flows that already have their USD assets
    staged and only need the Newton sidecar.
    """
    return Pipeline(name="newton_only", stages=[NewtonExtrasStage])


__all__ = ["NewtonExtrasStage", "newton_only_pipeline"]

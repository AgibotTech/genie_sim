# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Concrete Newton-standalone engine class.

``NewtonHeadlessEngine`` is the only engine class produced by this
subpackage — Kit-free, opens the USD stage with pure pxr, and inherits
its entire build / step / state surface from :class:`_NewtonStandaloneBase`
(in :mod:`engine_base`) which composes the build-phase mixins from
:mod:`engine.newton.setup` plus the runtime mixins (cloth, control,
plugin, state, stats, topology).

This module is intentionally tiny: it exists so the public surface
``from engine.newton import NewtonHeadlessEngine`` lands here without
dragging in the 900-line composition machinery.  Anything that touches
the build pipeline lives in :mod:`engine.newton.setup`; anything that
touches runtime state lives next to its mixin.
"""

from __future__ import annotations

from pathlib import Path

from genie_sim_engine_newton.engine.newton.engine_base import _NewtonStandaloneBase
from genie_sim_engine_core.engine.base import PhysicsEngine


@PhysicsEngine.register("newton")
class NewtonHeadlessEngine(_NewtonStandaloneBase):
    """Newton engine — Kit-free, the only newton-standalone engine class.

    The stage is opened via pure pxr.  ``_warmup_renders`` and
    ``_configure_viewport`` are no-ops inherited from
    :class:`engine.newton.setup._RuntimeMixin`.

    For the Kit viewport with Newton physics, use
    ``physics_engine:=isaac_newton`` — that path runs Isaac Sim's Newton
    wrapper inside Kit.
    """

    def _open_stage(self, newton_scene) -> None:
        """Open the stage with pure pxr — no omni.usd context needed."""
        from pxr import Usd as _Usd

        self._stage = _Usd.Stage.Open(self._scene_usda)
        if self._robot_usda and Path(self._robot_usda).exists():
            prim = self._stage.DefinePrim(f"/{self._robot_prefix_str}", "Xform")
            prim.GetReferences().AddReference(self._robot_usda)
        if newton_scene and newton_scene.is_file():
            self._stage.GetRootLayer().subLayerPaths.append(str(newton_scene))

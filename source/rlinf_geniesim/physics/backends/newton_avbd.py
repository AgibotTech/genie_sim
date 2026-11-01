#!/usr/bin/env python3

# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Newton standalone + AVBD (augmented vertex block descent) backend.

Same upstream engine subprocess as ``newton_mujoco_warp``; only the
``physics_solver`` ROS param differs.  AVBD is a position-based solver
with a completely different contact model from mjwarp's convex GJK
narrowphase — the second cheap A/B for the grasp-contact problem
(doc §7.10).
"""

from __future__ import annotations

from . import register_engine
from .newton_mjwarp import NewtonMuJoCoWarpBackend


@register_engine("newton_avbd")
class NewtonAVBDBackend(NewtonMuJoCoWarpBackend):
    UPSTREAM_SOLVER = "avbd"
    BACKEND_NAME = "newton_avbd"


__all__ = ["NewtonAVBDBackend"]

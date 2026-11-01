#!/usr/bin/env python3

# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Newton standalone + Featherstone solver backend.

Same upstream engine subprocess as ``newton_mujoco_warp``; only the
``physics_solver`` ROS param differs.  Featherstone handles the gripper
closed-loop ball joints natively (no loop-strip + connect-equality
rebuild), which makes it the cheapest A/B against mjwarp's convex
single-point contact behaviour (doc §7.10).
"""

from __future__ import annotations

from . import register_engine
from .newton_mjwarp import NewtonMuJoCoWarpBackend


@register_engine("newton_featherstone")
class NewtonFeatherstoneBackend(NewtonMuJoCoWarpBackend):
    UPSTREAM_SOLVER = "featherstone"
    BACKEND_NAME = "newton_featherstone"


__all__ = ["NewtonFeatherstoneBackend"]

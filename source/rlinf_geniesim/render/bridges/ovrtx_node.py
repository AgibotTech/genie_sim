# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""OVRTX render backend.

Wraps ``genie_sim_render_node`` (Kit-free OVRTX OptiX renderer, runs
as a separate ROS 2 process). Subscribes to ``/tf_render`` for poses
and publishes ``<topic>/image_raw`` for each camera; the bridge
copies those images into Frame SHM.
"""

from __future__ import annotations

from typing import Any

from . import register_bridge
from ._upstream_base import _UpstreamRenderBridge


@register_bridge("ovrtx_node")
class OVRTXNodeBackend(_UpstreamRenderBridge):
    UPSTREAM_PACKAGE = "genie_sim_render"
    UPSTREAM_EXEC = "genie_sim_render_node"
    BACKEND_NAME = "ovrtx_node"
    OUTPUT_CHANNELS = ["rgb", "depth"]
    SUPPORTS_BATCHING = False  # one process per stage today

    def __init__(
        self,
        *args: Any,
        prim_paths: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._prim_paths = prim_paths

    def start(self) -> None:
        # Multi-env organization is THIS renderer's job: assemble the N-robot
        # scene_multi.usda from the single-env manifest, then launch the OVRTX
        # node against it. (The generic layer hands us the single-env manifest.)
        self._assemble_multi_env_stage()
        super().start()

    def _extra_ros_params(self) -> dict[str, str]:
        params: dict[str, str] = {}
        if self._prim_paths:
            params["prim_paths"] = self._prim_paths
        return params


__all__ = ["OVRTXNodeBackend"]

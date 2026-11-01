# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""No-op renderer selected with ``render: none`` for state-only simulation.

Lifecycle hooks require no graphics runtime. ``render()`` reports a dropped frame."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from . import register_bridge
from .base import RenderResult, VecRenderBackend


@register_bridge("none")
class NullRenderBackend(VecRenderBackend):
    """No-op renderer whose camera count follows the configured camera list."""

    BACKEND_NAME = "none"

    def start(self) -> None:
        self._started = True

    def close(self) -> None:
        self._started = False

    def attach_shm(self, frame_shm: Any, layout: Any) -> None:
        self._shm_attached = True

    def bind_scene_schema(self, schema: Any) -> None:
        self._schema_bound = True

    def on_reset(
        self, env_ids: Sequence[int], seeds: Optional[Sequence[int]] = None, material_bindings: Optional[dict] = None
    ) -> None:
        pass

    def update_poses(
        self,
        body_poses: np.ndarray,
        sim_time: np.ndarray,
        camera_poses: Optional[np.ndarray] = None,
        object_poses: Optional[np.ndarray] = None,
        cloth_vertices: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        # Absorb extra kwargs the handler forwards (e.g. body_names). Remember
        # sim_time only, so render() can stamp it consistently.
        self._last_sim_time = np.asarray(sim_time)

    def render(self) -> RenderResult:
        st = getattr(self, "_last_sim_time", np.zeros(self._num_envs))
        return RenderResult(sim_time=np.asarray(st), frame_index=0, latency_ms=0.0, dropped=True)

    @property
    def supports_batching(self) -> bool:
        return True

    @property
    def output_channels(self) -> list[str]:
        return []

    @property
    def backend_name(self) -> str:
        return self.BACKEND_NAME

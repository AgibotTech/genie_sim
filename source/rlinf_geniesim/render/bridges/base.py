# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Vec-env render backend base class.

Symmetric to ``VecPhysicsBackend``. Concrete subclasses wrap upstream
geniesim_ros renderers (``genie_sim_render_node`` for OVRTX, the
Kit-based ``isaacsim_render.py`` for IsaacSim RTX) as subprocesses
and bridge their output into the rlinf Frame SHM.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional, Sequence, TypedDict

import numpy as np

if TYPE_CHECKING:
    from rlinf_geniesim.physics.backends.base import SceneSchema


# ---------------------------------------------------------------------------
# Configuration / result data structures
# ---------------------------------------------------------------------------


class CameraSpec(TypedDict):
    """One camera mounted on a URDF link.

    Fields:
        name      Camera identifier (used in ROS topic suffix / SHM slice).
        link      URDF link the camera is attached to (e.g. "head_link").
        pose      [x, y, z, qw, qx, qy, qz] in the link's local frame.
        hfov_deg  Horizontal field of view (degrees).
        image_hw  (height, width) in pixels.
    """

    name: str
    link: str
    pose: list[float]
    hfov_deg: float
    image_hw: tuple[int, int]


class RenderResult(TypedDict):
    """Metadata returned by ``render()`` (the image bytes live in Frame SHM)."""

    sim_time: np.ndarray  # (num_envs,) float64
    frame_index: int
    latency_ms: float
    dropped: bool


class RenderHealthStatus(TypedDict):
    alive: bool
    pid: Optional[int]
    last_render_ms: float
    cuda_oom: bool
    error_msg: str


# ---------------------------------------------------------------------------
# The base class
# ---------------------------------------------------------------------------


class VecRenderBackend(ABC):
    """Vector-env-shaped render backend.

    Lifecycle::

        bridge = SomeBridge(num_envs=N, cameras=[...], render_hz=30)
        bridge.start()
        bridge.attach_shm(frame_shm, layout)
        bridge.bind_scene_schema(schema_from_engine)
        # ... main loop calls update_poses / render ...
        bridge.close()
    """

    def __init__(
        self,
        num_envs: int,
        cameras: list[CameraSpec],
        render_hz: float,
        **backend_kwargs: Any,
    ) -> None:
        self._num_envs = int(num_envs)
        self._cameras = list(cameras)
        self._render_hz = float(render_hz)
        self._backend_kwargs = dict(backend_kwargs)
        self._started = False
        self._shm_attached = False
        self._schema_bound = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    # ------------------------------------------------------------------
    # Wiring contracts
    # ------------------------------------------------------------------

    @abstractmethod
    def attach_shm(self, frame_shm: Any, layout: Any) -> None:
        """Tell the renderer where to deposit images in shared memory.

        Both ``frame_shm`` and ``layout`` are opaque from the base
        class's perspective — concrete backends interpret them per their
        SHM scheme. ``layout`` typically describes per-env / per-camera
        offsets and shapes; ``frame_shm`` is the
        ``multiprocessing.shared_memory.SharedMemory`` (or equivalent)
        object whose buffer the renderer writes into.
        """

    @abstractmethod
    def bind_scene_schema(self, schema: "SceneSchema") -> None:
        """Receive the physics engine's scene schema and build any
        index → USD-prim lookup tables needed by ``update_poses``.

        Must be called once after ``start()`` and before the first
        ``update_poses``.
        """

    def on_reset(
        self, env_ids: Sequence[int], seeds: Optional[Sequence[int]] = None, material_bindings: Optional[dict] = None
    ) -> None:
        """Default no-op. Override to clear motion-blur history,
        reset particle systems, etc., when the physics engine resets
        the listed envs.

        ``seeds`` (render-side generalization command channel):
        the RESET seed for each entry in ``env_ids``, aligned by index (same
        convention as the engine's own ``reset(env_ids=..., seeds=...)``) —
        ``>= 0`` means "this env is being generalized this reset", ``-1`` is
        the seed contract's verbatim-replay sentinel. ``None`` means the
        caller has no seed info for this reset at all (e.g. the legacy no-seed
        ``sim_server.py`` reset path — see ``_handle_step``/
        ``_handle_reset``) — a backend that needs seeds (``IsaacTiledBackend``)
        simply treats that the same as "nothing to generalize this reset".
        A backend that doesn't need seeds (this default, ``NullRenderBackend``)
        ignores the argument entirely; it exists on the base signature so
        every real call site can pass it uniformly without a per-backend
        ``hasattr`` check.
        """
        # Authored bindings are keyed by env id. Fused applies them inside
        # its physics reset response; render-only backends apply them here.
        return

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    @abstractmethod
    def update_poses(
        self,
        body_poses: np.ndarray,
        sim_time: np.ndarray,
        camera_poses: Optional[np.ndarray] = None,
        object_poses: Optional[np.ndarray] = None,
        cloth_vertices: Optional[np.ndarray] = None,
        tick: int = 0,
        will_collect: bool = True,
    ) -> None:
        """Stage the latest physics state for the next render call.

        Args:
            body_poses:    (num_envs, num_bodies, 7), aligned with
                           ``SceneSchema.body_paths``.
            sim_time:      (num_envs,) float64 — sim time for this frame.
            tick:          global monotonic int64 tick labelling these poses;
                           rides on the ``/tf_render`` stamp.
            camera_poses:  (num_envs, num_cameras, 7) or ``None``.
            object_poses:  (num_envs, num_objects, 7) or ``None``.
            cloth_vertices: (num_envs, num_clusters, max_pts, 3) or ``None``.
            will_collect:  whether the caller will call ``render()`` for this
                           tick.

        Does **not** trigger a render — call ``render()`` for that. That is
        true of this method but not of every backend's render PROCESS: the
        tiled node starts a frame whenever the tick it has seen changes, so
        there publishing poses is itself a render trigger. ``will_collect``
        lets such a backend skip a frame nobody will read. Backends whose
        renderer is passive, or that drive a viewer someone is watching,
        ignore it.
        """

    @abstractmethod
    def render(self) -> RenderResult:
        """Block until one frame is rendered and copied into Frame SHM,
        then return its metadata."""

    def render_barrier(self) -> bool:
        """Wait for the render to complete (barrier phase only).

        Default implementation delegates to ``render()`` and stashes the
        result for ``render_copy()``.  Subclasses (e.g. IsaacTiledBackend)
        override to split the barrier from the copy so the caller can
        schedule CPU work in between.

        Returns True if a valid frame was received, False if dropped.
        """
        self._stashed_render_result = self.render()
        return not self._stashed_render_result.dropped

    def render_copy(self) -> RenderResult:
        """Finalize the frame copy (copy phase after barrier).

        Default implementation returns the result stashed by
        ``render_barrier()``.  Subclasses override for a real split.
        """
        return self._stashed_render_result

    # ------------------------------------------------------------------
    # Health / metrics
    # ------------------------------------------------------------------

    def health_check(self) -> RenderHealthStatus:
        return RenderHealthStatus(
            alive=self._started,
            pid=getattr(self, "_pid", None),
            last_render_ms=float(getattr(self, "_last_render_ms", 0.0)),
            cuda_oom=False,
            error_msg="",
        )

    def get_metrics(self) -> dict[str, float]:
        return {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def num_cameras(self) -> int:
        return len(self._cameras)

    @property
    def render_hz(self) -> float:
        return self._render_hz

    @property
    def cameras(self) -> list[CameraSpec]:
        return list(self._cameras)

    @property
    def is_started(self) -> bool:
        return self._started

    @property
    @abstractmethod
    def supports_batching(self) -> bool: ...

    @property
    @abstractmethod
    def output_channels(self) -> list[str]: ...

    @property
    @abstractmethod
    def backend_name(self) -> str: ...

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "VecRenderBackend":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Implementation guards
    # ------------------------------------------------------------------

    def _ensure_started(self) -> None:
        if not self._started:
            raise RuntimeError(f"{type(self).__name__}.start() must be called first")

    def _ensure_shm_attached(self) -> None:
        if not self._shm_attached:
            raise RuntimeError(f"{type(self).__name__}.attach_shm() must be called before render()")

    def _ensure_schema_bound(self) -> None:
        if not self._schema_bound:
            raise RuntimeError(f"{type(self).__name__}.bind_scene_schema() must be called " f"before update_poses()")


__all__ = [
    "CameraSpec",
    "RenderHealthStatus",
    "RenderResult",
    "VecRenderBackend",
]

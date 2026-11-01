# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Vector physics interface used by RLinf's simulation handler.

step advances all environments, and getters expose batched state. Concrete
backends own their upstream processes and may batch worlds within one process.
This interface is separate from the engine's internal PhysicsEngine API."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal, Optional, Sequence, TypedDict

import numpy as np


# ---------------------------------------------------------------------------
# Data structures used by the physics ↔ render alignment contract.
# ---------------------------------------------------------------------------

BodyKind = Literal["robot_link", "rigid_object", "cloth_cluster", "static_prop"]


class ClothMeshSpec(TypedDict):
    """Schema for a cloth cluster.

    ``cluster_index`` must equal the index in
    ``SceneSchema.cloth_meshes`` (so the renderer can look up by index).
    """

    name: str
    num_vertices: int
    cluster_index: int


class SceneSchema(TypedDict):
    """Physics → render scene-structure contract.

    Emitted by ``VecPhysicsBackend.get_scene_schema()`` once after
    ``start()`` and consumed by ``VecRenderBackend.bind_scene_schema()``.

    Alignment guarantee (do not violate):
        ``body_paths[i]``, ``body_kinds[i]``, ``body_link_names[i]`` and
        ``get_body_poses()[:, i, :]`` all refer to the same body.
        ``joint_names[j]`` aligns with ``get_joint_states()[:, j, :]``.
        ``object_names[k]`` aligns with ``get_object_poses()[:, k, :]``.
        ``cloth_meshes[m].cluster_index == m``.
    """

    body_paths: list[str]
    body_kinds: list[BodyKind]
    body_link_names: list[str]
    joint_names: list[str]
    object_names: list[str]
    cloth_meshes: list[ClothMeshSpec]
    static_props: list[str]


class DynamicsParams(TypedDict, total=False):
    """Per-env dynamics overrides (domain randomization)."""

    mass_scale: dict[str, float]
    friction: dict[str, float]
    damping: dict[str, float]
    com_offset: dict[str, list[float]]


class HealthStatus(TypedDict):
    """Backend health snapshot consumed by the process supervisor."""

    alive: bool
    pid: Optional[int]
    last_step_ms: float
    cuda_oom: bool
    error_msg: str


# ---------------------------------------------------------------------------
# The base class.
# ---------------------------------------------------------------------------


class VecPhysicsBackend(ABC):
    """Vector-env-shaped physics backend.

    Lifecycle::

        backend = SomeBackend(scene_yaml=..., num_envs=N, ...)
        backend.start()
        schema = backend.get_scene_schema()
        # ... main loop calls step / reset / get_*  ...
        backend.close()

    All ``get_*`` methods return arrays whose first axis is ``num_envs``.
    Subclasses must keep body / joint / object orderings identical to
    the schema returned by ``get_scene_schema()``.
    """

    def __init__(
        self,
        scene_yaml: str,
        robot_urdf: str,
        num_envs: int,
        physics_hz: float,
        device: str = "cuda:0",
        **backend_kwargs: Any,
    ) -> None:
        """Stash configuration. **Do not** spawn subprocesses or touch
        USD here — defer that to ``start()`` so callers can construct a
        backend cheaply.
        """
        self.scene_yaml = scene_yaml
        self.robot_urdf = robot_urdf
        self._num_envs = int(num_envs)
        self._physics_hz = float(physics_hz)
        self._device = device
        self._backend_kwargs = dict(backend_kwargs)
        self._started = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def start(self) -> None:
        """Bring the backend online: spawn subprocesses, attach SHM,
        compile CUDA graphs, etc. After return, ``step``/``reset``/
        ``get_*`` are usable."""

    @abstractmethod
    def close(self) -> None:
        """Release all resources. **Must be idempotent.**"""

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    @abstractmethod
    def step(self, actions: np.ndarray) -> None:
        """Advance physics by ``round(physics_hz / train_hz)`` ticks.

        Args:
            actions: shape ``(num_envs, action_dim)``.
        """

    @abstractmethod
    def reset(self, env_ids: Optional[Sequence[int]] = None) -> None:
        """Reset selected envs to their initial qpos / qvel.

        Args:
            env_ids: list of env indices, or ``None`` to reset all.
        """

    # ------------------------------------------------------------------
    # State readers
    # ------------------------------------------------------------------

    @abstractmethod
    def get_state(self) -> np.ndarray:
        """Return shape ``(num_envs, state_dim)`` — the policy-facing state."""

    @abstractmethod
    def get_joint_states(self) -> np.ndarray:
        """Return shape ``(num_envs, num_joints, 2)`` — ``[pos, vel]``
        per joint, ordered by ``SceneSchema.joint_names``."""

    @abstractmethod
    def get_body_poses(self) -> np.ndarray:
        """Return shape ``(num_envs, num_bodies, 7)`` — world-frame
        ``[x, y, z, qw, qx, qy, qz]``, ordered by
        ``SceneSchema.body_paths``."""

    @abstractmethod
    def get_object_poses(self) -> np.ndarray:
        """Return shape ``(num_envs, num_objects, 7)`` for non-robot
        movable rigid bodies. Empty array if scene has no objects."""

    @abstractmethod
    def get_camera_poses(self) -> np.ndarray:
        """Return shape ``(num_envs, num_cameras, 7)`` — camera world
        poses with FK already applied, ordered as in the renderer's
        ``CameraSpec`` list."""

    def get_cloth_vertices(self) -> Optional[np.ndarray]:
        """Return ``(num_envs, num_clusters, max_vertices, 3)`` or ``None``.

        Default returns ``None``. Cloth-aware backends override.
        """
        return None

    def get_body_aabbs(self) -> np.ndarray:
        """Return ``(num_envs, num_bodies, 6)`` — world-frame axis-aligned
        bounding boxes ``[min_x, min_y, min_z, max_x, max_y, max_z]`` per body,
        ordered like ``get_body_poses`` / ``SceneSchema.body_paths``.

        Used by the eval layer (the benchmark checkers' ``get_obj_aabb``).
        Default raises — backends that can report geometry bounds override
        this. Kept engine-agnostic on purpose: eval reads only this generic
        interface, never engine internals."""
        raise NotImplementedError(f"{type(self).__name__} does not implement get_body_aabbs")

    def get_static_prop_aabbs(self) -> dict[str, np.ndarray]:
        """Return world AABBs for static props, keyed by checker-facing prim path.

        Values are [min_x, min_y, min_z, max_x, max_y, max_z]. Static props may have no
        rigid-body row, so they are reported separately from get_body_aabbs. Return an
        empty mapping when the backend has no prop geometry to report."""
        return {}

    @abstractmethod
    def get_sim_time(self) -> np.ndarray:
        """Return shape ``(num_envs,)`` — sim time (seconds) since the
        episode start. Used by the renderer to stamp frames so the
        trainer can verify image/state alignment."""

    # ------------------------------------------------------------------
    # Scene-structure contract
    # ------------------------------------------------------------------

    @abstractmethod
    def get_scene_schema(self) -> SceneSchema:
        """Emit the body / joint / object / cloth ordering the
        renderer needs to wire its USD prims correctly.

        Called once after ``start()``. If the structure ever changes at
        runtime (rare), call again and re-bind on the renderer.
        """

    # ------------------------------------------------------------------
    # Direct state writes (DR / replay / IK init)
    # ------------------------------------------------------------------

    @abstractmethod
    def set_qpos(self, env_id: int, qpos: np.ndarray) -> None:
        """Overwrite ``qpos`` for one env without integrating physics.
        Does not touch ``qvel`` — call ``reset`` for a full reset."""

    def set_dynamics_params(self, env_id: int, params: DynamicsParams) -> None:
        """Domain-randomization hook. Default raises — backends that
        support per-world parameter overrides (e.g. ``mujoco_warp``)
        override this."""
        raise NotImplementedError(f"{type(self).__name__} does not support dynamics randomization")

    # ------------------------------------------------------------------
    # Health / metrics (default implementations; override as needed)
    # ------------------------------------------------------------------

    def health_check(self) -> HealthStatus:
        """Default: report ``alive == _started`` plus best-effort PID
        and last-step latency. Override in backends that have richer
        health info (CUDA OOM detection etc.)."""
        return HealthStatus(
            alive=self._started,
            pid=getattr(self, "_pid", None),
            last_step_ms=float(getattr(self, "_last_step_ms", 0.0)),
            cuda_oom=False,
            error_msg="",
        )

    def get_metrics(self) -> dict[str, float]:
        """Default: empty. Backends may expose ``step_ms_p99``,
        ``contact_count``, etc."""
        return {}

    # ------------------------------------------------------------------
    # Read-only properties (basic getters; subclasses must implement
    # the abstract @property's below)
    # ------------------------------------------------------------------

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def physics_hz(self) -> float:
        return self._physics_hz

    @property
    def device(self) -> str:
        return self._device

    @property
    def is_started(self) -> bool:
        return self._started

    @property
    @abstractmethod
    def supports_batching(self) -> bool:
        """``True`` = single-process batched (one process holds N
        worlds); ``False`` = N independent processes."""

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """Action dimension per env. Only valid after ``start()``."""

    @property
    @abstractmethod
    def state_dim(self) -> int:
        """State dimension per env. Only valid after ``start()``."""

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """String identifier matching the registered factory name."""

    # ------------------------------------------------------------------
    # Context-manager sugar
    # ------------------------------------------------------------------

    def __enter__(self) -> "VecPhysicsBackend":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Implementation guards (subclasses call from step/reset/get_*)
    # ------------------------------------------------------------------

    def _ensure_started(self) -> None:
        if not self._started:
            raise RuntimeError(f"{type(self).__name__}.start() must be called before " f"step/reset/get_*")

    def _validate_actions(self, actions: np.ndarray) -> None:
        expected = (self._num_envs, self.action_dim)
        if actions.shape != expected:
            raise ValueError(f"actions shape {actions.shape} != expected {expected}")


__all__ = [
    "BodyKind",
    "ClothMeshSpec",
    "DynamicsParams",
    "HealthStatus",
    "SceneSchema",
    "VecPhysicsBackend",
]

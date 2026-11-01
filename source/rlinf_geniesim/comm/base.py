# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Transport-agnostic comm contract: dims, result buffer, handler/transport/client ABCs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class SimDims:
    num_envs: int
    action_dim: int
    state_dim: int
    num_joints: int
    num_bodies: int
    num_objects: int
    num_cameras: int
    image_h: int  # MAX obs height across cameras (frame buffer alloc)
    image_w: int  # common obs width (all cameras share it)
    # Per-camera obs HEIGHT (<= image_h). Cameras with different benchmark aspect
    # ratios render at a common width but their own height (head 640x400, wrist
    # 640x528); the frame buffer is allocated at the MAX height and each camera's
    # rows [0:cam_heights[c]] hold its real frame, the rest zero-padded. The obs
    # is cropped back to each camera's height. Empty => all cameras use image_h
    # (uniform, backward compatible). JSON round-trips as a list.
    cam_heights: tuple = ()
    info_blob_bytes: int = 16 * 1024
    error_msg_bytes: int = 512


# Per-env lifecycle status ``GenieSimVecEnv`` reports in ``infos["status"]``.
# Every env is always READY: a reset returns with its envs ready to act (see
# ``SimHandler.reset``). The field is kept because RLinf's GenieSim env
# reads it.
STATUS_READY = 0


@dataclass
class ResultBuffer:
    """Transport-owned arrays filled by the simulation handler.

    Per-environment fields start with the environment axis; tick fields are
    scalars. SHM provides shared-memory views. Optional fields may be absent
    on older peers."""

    states: np.ndarray  # (N, state_dim)
    joint_states: np.ndarray  # (N, J, 2)
    body_poses: np.ndarray  # (N, B, 7)
    object_poses: np.ndarray  # (N, O, 7)
    images: np.ndarray  # (N, C, H, W, 3) uint8
    rewards: np.ndarray  # (N,)
    terminations: np.ndarray  # (N,) bool
    truncations: np.ndarray  # (N,) bool
    sim_time: np.ndarray  # (N,) float64
    success: Optional[np.ndarray] = None  # (N,) bool — task-success (eval checker)
    global_tick: Optional[np.ndarray] = None  # () int64 — step's global tick
    frame_tick: Optional[np.ndarray] = None  # () int64 — rendered frame's tick


class SimHandler(ABC):
    """The transport-agnostic sim core, called back by SimTransport.serve."""

    @abstractmethod
    def handshake(self) -> tuple[SimDims, dict]:
        """Return ``(dims, scene_schema)``; called once when serve() starts."""

    @abstractmethod
    def step(self, actions: np.ndarray, out: ResultBuffer, need_images: bool = True, pre_copy_callback=None) -> None:
        """Advance physics and evaluation, then fill out.

        need_images=False permits skipping the render barrier and image copy.
        Pose updates still run, but a bridge may defer publication until an image
        is requested. SHM callers may always request images."""

    @abstractmethod
    def reset(self, mask: np.ndarray, out: ResultBuffer) -> None:
        """Reset the envs selected by ``mask``. When this returns every one
        of them is ready to act and ``out`` holds its first observation."""

    @abstractmethod
    def close(self) -> None: ...

    def on_serve_start(self, frame_region: Optional[object] = None) -> None:
        """Optional lifecycle hook — called once by the transport after it has
        allocated its output buffers and before the first step/reset.

        ``frame_region`` is the transport's image output target: the buffer a
        separate render subprocess writes pixels into (e.g. the Frame SHM
        region for the SHM transport), or ``None`` for transports without one.
        It is opaque here (transport-agnostic). Default: no-op. Handlers that
        drive a separate renderer override this to wire it up — point the
        renderer at ``frame_region`` and bind the scene schema."""
        return


class SimTransport(ABC):
    """Server side: owns the receive loop and dispatches verbs to a handler."""

    @abstractmethod
    def serve(self, handler: SimHandler) -> Optional[dict]:
        """Block until shutdown, then return ``None``.

        Backends that support the ``reload`` verb (zmq) instead return the
        task_init dict the request carried; the caller tears down, rebuilds,
        and calls ``serve()`` again with the new handler. Backends without
        reload always return ``None``.
        """

    @abstractmethod
    def close(self) -> None: ...


class SimClient(ABC):
    """Client side: blocking verb calls. This is the only interface a
    training framework depends on."""

    @abstractmethod
    def handshake(self) -> tuple[SimDims, dict]: ...

    @abstractmethod
    def step(self, actions: np.ndarray, need_images: bool = True) -> ResultBuffer:
        """Advance one step and return its result.

        For ZMQ, need_images=False skips image transfer and returns images=None.
        SHM accepts the flag but always renders into its shared frame buffer."""

    @abstractmethod
    def reset(self, mask: np.ndarray) -> ResultBuffer:
        """Reset the envs selected by ``mask`` and return their first
        observation; every one of them is ready to act."""

    @abstractmethod
    def close(self) -> None: ...

    def set_backends(self, backends: dict) -> None:
        """Declare the engine backends: physics, solver, substep, render,
        device.

        Control plane, describing how this machine runs rather than what it
        runs. Cached locally and merged into the payload by ``env.init()``.

        Backends are fixed once the server's first task takes effect; a
        different value afterwards is refused loudly.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support set_backends")

    def get_backends(self) -> dict:
        """A copy of the declared backend configuration, or this backend's
        built-in defaults when ``set_backends`` was never called."""
        raise NotImplementedError(f"{type(self).__name__} does not support get_backends")

    def backends_declared(self) -> bool:
        """Did the caller declare backends itself, i.e. was ``set_backends``
        called?

        ``get_backends()`` cannot answer this, because it falls back to
        built-in defaults. ``env.init()`` needs the distinction: with no
        declaration it omits the engine section entirely so the server keeps
        its own configuration, whereas sending a guessed value would be
        rejected by the backends lock.
        """
        return getattr(self, "_backends", None) is not None

    def init(self, task_init: dict, timeout_s: float = 300.0) -> tuple[SimDims, dict]:
        """Initialize or replace the task over the current connection.

        Wait for the rebuilt handler and return its dimensions and schema. Callers
        must replace cached values because either may change. ZMQ supports this
        operation; SHM's fixed layout does not."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support init() " "(only the zmq backend does; see comm/zmq/client.py)"
        )

    def reload(self, task_init: dict, timeout_s: float = 300.0) -> tuple[SimDims, dict]:
        """Compatibility alias for init."""
        return self.init(task_init, timeout_s=timeout_s)

    def task_metadata(self, sub_task_name: Optional[str] = None) -> dict:
        """Return the task catalog independently of the active task.

        sub_tasks maps each released task to its official G2op task_names,
        instance IDs and instruction tables. suites lists the exact pairings
        accepted by ZMQ task initialization and reload.
        Naming sub_task_name also computes switchable_instance_ids, which may be
        a subset of instance_ids for the selected backend. Only ZMQ supports this
        variable-length query."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support task_metadata() "
            "(only the zmq backend does; see comm/zmq/client.py)"
        )

    def randomization_readback(self) -> dict:
        """Return the server's requested, resolved and applied randomization state.

        requested contains the accepted init configuration. resolved contains
        physics, sensor and render specifications; unresolved physics/sensor are
        None, whereas empty mappings mean resolved and disabled. resolved_available
        indicates whether a stage build has resolved the configuration.

        applied.build_time contains per-environment joint/background plans, or None
        before a build. applied.last_reset contains render plans, or None before a
        reset or on unsupported bridges. Sensor imaging is added locally by
        GenieSimVecEnv.effective_randomization. Only ZMQ supports this query."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support randomization_readback() "
            "(only the zmq backend does; see comm/zmq/client.py)"
        )


__all__ = ["SimDims", "ResultBuffer", "SimHandler", "SimTransport", "SimClient", "STATUS_READY"]

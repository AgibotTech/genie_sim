# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""SHM wire layout (transport-specific)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from multiprocessing.shared_memory import SharedMemory
from typing import Iterable, Optional

import numpy as np

from rlinf_geniesim.comm.base import SimDims


# ---------------------------------------------------------------------------
# Phase-state-machine enums
# ---------------------------------------------------------------------------


class RequestPhase(IntEnum):
    """Trainer → sim_server request kind (written to step_shm.request_phase)."""

    IDLE = 0
    STEP = 1
    RESET = 2
    SHUTDOWN = 3
    SET_QPOS = 4
    SET_DYN = 5


class ResponsePhase(IntEnum):
    """sim_server → Trainer response kind (written to step_shm.response_phase)."""

    IDLE = 0
    STEP_DONE = 1
    RESET_DONE = 2
    ERROR = 3
    SHUTDOWN_ACK = 4


# ---------------------------------------------------------------------------
# Field specifications
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldSpec:
    """Description of one SHM field used by the layout planner."""

    name: str
    shape: tuple[int, ...]
    dtype: np.dtype
    description: str = ""

    @property
    def nbytes(self) -> int:
        n = int(np.dtype(self.dtype).itemsize)
        for s in self.shape:
            n *= int(s)
        return n


# ---------------------------------------------------------------------------
# Layout planners
# ---------------------------------------------------------------------------


def step_shm_fields(dims: SimDims) -> list[FieldSpec]:
    """Step SHM = the small request/response handshake region."""
    return [
        FieldSpec("request_phase", (), np.int32, "RequestPhase enum value"),
        FieldSpec("request_tick", (), np.int64, "monotone counter; trainer increments per request"),
        FieldSpec("response_phase", (), np.int32, "ResponsePhase enum value"),
        FieldSpec("response_tick", (), np.int64, "must equal request_tick when handshake completes"),
        FieldSpec("error_code", (), np.int32, "0 = OK, non-zero = backend-defined error code"),
        FieldSpec(
            "error_msg", (dims.error_msg_bytes,), np.uint8, "fixed-length UTF-8 byte buffer; first NUL terminates"
        ),
    ]


def ctrl_shm_fields(dims: SimDims) -> list[FieldSpec]:
    """Ctrl SHM = the bulk action / state / reward / info region.

    Kept reasonably small (~tens of KB at N=64). Frame SHM holds the
    images separately because they're orders of magnitude larger.
    """
    N = dims.num_envs
    return [
        # Trainer-write
        FieldSpec("actions_buf", (N, dims.action_dim), np.float32, "policy actions"),
        FieldSpec("reset_mask_buf", (N,), np.bool_, "True at index i = reset env i this tick"),
        FieldSpec(
            "reset_seed_buf",
            (N,),
            np.int64,
            "per-env randomization seed for THIS reset; -1 = verbatim "
            "snapshot (see engine _randomize_world_objects)",
        ),
        FieldSpec(
            "reset_instance_buf",
            (N,),
            np.int64,
            "per-env task-instance id for THIS reset; -1 = keep the "
            "current instance (in-topology-group switch only)",
        ),
        FieldSpec(
            "reset_instruction_buf",
            (N,),
            np.int64,
            "per-env instruction id (instructions.json 1-based id) for "
            "THIS reset; selects BOTH the language string (client side) "
            "and the success checker problem{id} (sim side); -1 = keep",
        ),
        # Sim-write
        FieldSpec("states_buf", (N, dims.state_dim), np.float32, "policy-facing state vector"),
        FieldSpec("joint_states_buf", (N, dims.num_joints, 2), np.float32, "[pos, vel] per joint"),
        FieldSpec("body_poses_buf", (N, dims.num_bodies, 7), np.float32, "world pose [xyz, qw, qx, qy, qz] per body"),
        FieldSpec("object_poses_buf", (N, dims.num_objects, 7), np.float32, "world pose per movable object"),
        FieldSpec("sim_time_buf", (N,), np.float64, "sim-time (seconds) per env"),
        FieldSpec("rewards_buf", (N,), np.float32, "per-env scalar reward (sim-side default; trainer may overwrite)"),
        FieldSpec("dones_buf", (N,), np.bool_, "True if episode ended this tick"),
        FieldSpec("truncated_buf", (N,), np.bool_, "True if ended due to time-limit (vs natural termination)"),
        FieldSpec("success_buf", (N,), np.bool_, "task-success flag"),
        FieldSpec("episode_step_buf", (N,), np.int32, "step counter within current episode"),
        FieldSpec("episode_return_buf", (N,), np.float32, "cumulative reward in current episode"),
        FieldSpec(
            "info_blob",
            (dims.info_blob_bytes,),
            np.uint8,
            "msgpack/pickle-serialized free-form metadata " "(per-tick diagnostics, derived reward terms, ...)",
        ),
        FieldSpec(
            "global_tick",
            (),
            np.int64,
            "monotonic global step tick (never resets); handler writes "
            "it each step so obs carries an exact, replay-stable clock",
        ),
    ]


def frame_shm_fields(dims: SimDims) -> list[FieldSpec]:
    """Frame SHM = camera images. Big — kept separate from Ctrl SHM."""
    N, C, H, W = (
        dims.num_envs,
        dims.num_cameras,
        dims.image_h,
        dims.image_w,
    )
    return [
        FieldSpec("rgb_buf", (N, C, H, W, 3), np.uint8, "RGB frames"),
        FieldSpec(
            "depth_buf",
            (N, C, H, W),
            np.float32,
            "depth frames in meters; allocated even if backend " "doesn't produce depth (left zeroed)",
        ),
        FieldSpec(
            "segmentation_buf", (N, C, H, W), np.int32, "instance segmentation; backend-optional, zero when absent"
        ),
        FieldSpec("frame_index", (), np.int64, "monotone counter incremented per render"),
        FieldSpec(
            "frame_sim_time",
            (N,),
            np.float64,
            "sim-time corresponding to this frame's poses; " "trainer compares vs ctrl_shm.sim_time_buf to detect lag",
        ),
        FieldSpec(
            "frame_ready", (), np.bool_, "sim sets True after writing the frame; " "trainer flips False after consuming"
        ),
        FieldSpec(
            "frame_tick",
            (),
            np.int64,
            "global tick the frame in rgb_buf was actually rendered at "
            "(bridge writes it after the render() barrier); trainer "
            "compares vs ctrl_shm.global_tick for exact sync",
        ),
    ]


# ---------------------------------------------------------------------------
# Layout: pack a list of fields into a flat buffer
# ---------------------------------------------------------------------------

# 8-byte alignment is enough for everything we use (largest scalar is
# float64). Keeping it modest avoids wasted padding on small fields.
_ALIGN = 8


def _aligned(offset: int, align: int = _ALIGN) -> int:
    rem = offset % align
    return offset if rem == 0 else offset + (align - rem)


@dataclass(frozen=True)
class _PackedField:
    name: str
    offset: int
    shape: tuple[int, ...]
    dtype: np.dtype
    nbytes: int


def _pack_layout(fields: Iterable[FieldSpec]) -> tuple[list[_PackedField], int]:
    """Compute byte offsets for every field and the total region size."""
    packed: list[_PackedField] = []
    cursor = 0
    for spec in fields:
        cursor = _aligned(cursor)
        packed.append(
            _PackedField(
                name=spec.name,
                offset=cursor,
                shape=spec.shape,
                dtype=np.dtype(spec.dtype),
                nbytes=spec.nbytes,
            )
        )
        cursor += spec.nbytes
    total = _aligned(cursor)
    return packed, total


# ---------------------------------------------------------------------------
# ShmRegion — owns one named SharedMemory and exposes named numpy views
# ---------------------------------------------------------------------------


class ShmRegion:
    """One named POSIX shared memory region holding a packed FieldSpec list.

    Two construction modes:

      * ``ShmRegion.allocate(name, fields)`` — creates a fresh region
        sized to fit ``fields``. Caller is the *owner*; the region is
        unlinked on ``close(unlink=True)``.

      * ``ShmRegion.attach(name, fields)`` — opens an existing region by
        name. The trainer (or any non-owner consumer) uses this.

    Both modes give you the same API: ``self.field("rgb_buf")`` returns
    a zero-copy numpy view writable in place.
    """

    def __init__(
        self,
        shm: SharedMemory,
        layout: list[_PackedField],
        owns: bool,
    ) -> None:
        self._shm = shm
        self._layout = {p.name: p for p in layout}
        self._views: dict[str, np.ndarray] = {}
        self._owns = owns

        # Build numpy views once; subsequent ``field()`` calls reuse them.
        for p in layout:
            view = np.ndarray(
                shape=p.shape if p.shape else (1,),
                dtype=p.dtype,
                buffer=shm.buf,
                offset=p.offset,
            )
            if not p.shape:
                # Scalar fields use a 0-d view; expose as a 1-element
                # writable buffer.
                view = view.reshape(())
            self._views[p.name] = view

    # ------------------------------------------------------------------

    @classmethod
    def allocate(cls, name: str, fields: Iterable[FieldSpec]) -> "ShmRegion":
        """Create a brand-new SHM region. Fails if ``name`` already exists."""
        layout, total = _pack_layout(fields)
        shm = SharedMemory(name=name, create=True, size=total)
        # Zero the region so consumers don't see stale bytes.
        shm.buf[:total] = bytes(total)
        return cls(shm, layout, owns=True)

    @classmethod
    def attach(cls, name: str, fields: Iterable[FieldSpec]) -> "ShmRegion":
        """Open an existing SHM region created by another process."""
        layout, _ = _pack_layout(fields)
        shm = SharedMemory(name=name, create=False)
        # Only the owning server may unlink this region. Unregister attached
        # clients so their resource tracker cannot remove it when they exit.
        try:
            from multiprocessing import resource_tracker  # noqa: PLC0415

            resource_tracker.unregister(shm._name, "shared_memory")  # noqa: SLF001
        except Exception:  # noqa: BLE001
            pass
        return cls(shm, layout, owns=False)

    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._shm.name

    @property
    def size(self) -> int:
        return self._shm.size

    def field(self, name: str) -> np.ndarray:
        """Return the zero-copy numpy view for ``name``."""
        if name not in self._views:
            raise KeyError(f"unknown SHM field '{name}'; " f"valid: {sorted(self._views)}")
        return self._views[name]

    def fields(self) -> list[str]:
        return sorted(self._views)

    def close(self, unlink: bool = False) -> None:
        """Release this handle. Pass ``unlink=True`` exactly once
        (from the owner) to remove the OS-level region."""
        # Drop numpy views so they release their references to shm.buf
        # before we close it (some platforms get fussy otherwise).
        self._views.clear()
        try:
            self._shm.close()
        except Exception:  # noqa: BLE001
            pass
        if unlink and self._owns:
            try:
                self._shm.unlink()
            except FileNotFoundError:
                pass

    # ------------------------------------------------------------------

    def __enter__(self) -> "ShmRegion":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close(unlink=self._owns)


# ---------------------------------------------------------------------------
# Convenience helpers for the three regions
# ---------------------------------------------------------------------------


def allocate_step_shm(name: str, dims: SimDims) -> ShmRegion:
    return ShmRegion.allocate(name, step_shm_fields(dims))


def allocate_ctrl_shm(name: str, dims: SimDims) -> ShmRegion:
    return ShmRegion.allocate(name, ctrl_shm_fields(dims))


def allocate_frame_shm(name: str, dims: SimDims) -> ShmRegion:
    return ShmRegion.allocate(name, frame_shm_fields(dims))


def attach_step_shm(name: str, dims: SimDims) -> ShmRegion:
    return ShmRegion.attach(name, step_shm_fields(dims))


def attach_ctrl_shm(name: str, dims: SimDims) -> ShmRegion:
    return ShmRegion.attach(name, ctrl_shm_fields(dims))


def attach_frame_shm(name: str, dims: SimDims) -> ShmRegion:
    return ShmRegion.attach(name, frame_shm_fields(dims))


# ---------------------------------------------------------------------------
# Bootstrap message
# ---------------------------------------------------------------------------


@dataclass
class BootstrapInfo:
    """Tiny one-shot dict the sim_server sends to the trainer at startup
    so it knows how to interpret the SHM regions.

    Sent over a unix domain socket / file on disk / env var — NOT over
    SHM itself (chicken-and-egg).
    """

    sim_dims: SimDims
    step_shm_name: str
    ctrl_shm_name: str
    frame_shm_name: str
    schema_dump_path: str = ""  # path to JSON containing SceneSchema
    output_channels: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "sim_dims": self.sim_dims.__dict__,
            "step_shm_name": self.step_shm_name,
            "ctrl_shm_name": self.ctrl_shm_name,
            "frame_shm_name": self.frame_shm_name,
            "schema_dump_path": self.schema_dump_path,
            "output_channels": list(self.output_channels),
        }


# ---------------------------------------------------------------------------
# Frequency contract reminder
# ---------------------------------------------------------------------------
#
# physics_hz : ticks the upstream physics loop runs at (set in backends.yaml)
# render_hz  : target FPS for the renderer (set in backends.yaml)
# train_hz   : implicit — trainer's STEP request rate
#
# Each STEP request advances the physics by
#     physics_substeps = round(physics_hz / train_hz)
# physical ticks. Render runs at most once per STEP, latest-state-wins;
# its returned RenderResult.sim_time is what the trainer should compare
# against ctrl_shm.sim_time_buf to detect lag.


__all__ = [
    "BootstrapInfo",
    "FieldSpec",
    "RequestPhase",
    "ResponsePhase",
    "ShmRegion",
    "allocate_ctrl_shm",
    "allocate_frame_shm",
    "allocate_step_shm",
    "attach_ctrl_shm",
    "attach_frame_shm",
    "attach_step_shm",
    "ctrl_shm_fields",
    "frame_shm_fields",
    "step_shm_fields",
]

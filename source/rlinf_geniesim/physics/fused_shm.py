# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Shared-memory actions and state for the fused Isaac subprocess.

Images use TiledFrameShm separately. This engine channel has its own layout,
independent of the trainer-facing comm/shm protocol. Each region has one writer;
the SemPair request/reply handshake orders writes and reads without atomics."""

from __future__ import annotations

from multiprocessing.shared_memory import SharedMemory
from typing import Optional

import numpy as np

# Header: int64[16]. Index constants are module-level so both sides name the
# same slot instead of hardcoding an integer twice (that drifts).
_H_MAGIC = 0
_H_NUM_ENVS = 1
_H_ACTION_DIM = 2
_H_NUM_JOINTS = 3
_H_NUM_BODIES = 4
_H_NUM_OBJECTS = 5
_H_SEQ = 6
_H_CMD = 7
_H_RESET_MASK_VALID = 8
_H_STATUS = 9
#: 0 = this STEP does not need a rendered frame, so the server may skip the
#: render and publish no new tiled frame. Stored INVERTED (0 means "render")
#: so a zeroed header -- an older peer, or a segment before this field existed
#: -- still means render, the pre-existing behaviour. See ``want_image``.
_H_SKIP_RENDER = 10
_H_NUM_SCENE_JOINTS = 11
_HEADER_LEN = 16
_HEADER_BYTES = _HEADER_LEN * 8

_ERR_BYTES = 512

_MAGIC = 0x4655534544535433  # "FUSEDST3": includes scene joint positions

# ``cmd`` values. The server blocks on the request semaphore and reads this to
# decide what the client is asking for.
CMD_IDLE = 0
CMD_STEP = 1
CMD_RESET = 2
CMD_SHUTDOWN = 3

# ``status`` values written by the server alongside each response.
STATUS_OK = 0
STATUS_ERROR = 1


class FusedStateShm:
    """Action/state channel owned by the fused Isaac subprocess.

    The client creates SemPair and launches the subprocess. The subprocess builds
    the stage, creates this segment with the resulting dimensions, writes the schema,
    and signals readiness. The client then attaches using schema-derived dimensions;
    header validation rejects stale or mismatched layouts. The subprocess owns unlinking."""

    def __init__(
        self,
        name: str,
        *,
        num_envs: int,
        action_dim: int,
        num_joints: int,
        num_bodies: int,
        num_objects: int,
        num_scene_joints: int = 0,
        create: bool = False,
    ) -> None:
        self.num_envs = int(num_envs)
        self.action_dim = int(action_dim)
        self.num_joints = int(num_joints)
        self.num_bodies = int(num_bodies)
        self.num_objects = int(num_objects)
        self.num_scene_joints = int(num_scene_joints)

        n, a, j, b, o, s = (
            self.num_envs,
            self.action_dim,
            self.num_joints,
            self.num_bodies,
            self.num_objects,
            self.num_scene_joints,
        )
        # Region byte counts, in the order they are laid out below.
        self._sz_actions = n * a * 4  # float32
        self._sz_reset_mask = n * 4  # int32
        self._sz_reset_seeds = n * 8  # int64
        self._sz_reset_instances = n * 8  # int64
        self._sz_joint_states = n * j * 2 * 4  # float32
        self._sz_body_poses = n * b * 7 * 4  # float32
        self._sz_object_poses = n * o * 7 * 4  # float32
        self._sz_sim_time = n * 8  # float64
        self._sz_scene_joint_positions = n * s * 4  # float32
        size = (
            _HEADER_BYTES
            + _ERR_BYTES
            + self._sz_actions
            + self._sz_reset_mask
            + self._sz_reset_seeds
            + self._sz_reset_instances
            + self._sz_joint_states
            + self._sz_body_poses
            + self._sz_object_poses
            + self._sz_sim_time
            + self._sz_scene_joint_positions
        )

        if create:
            # Best-effort cleanup of a segment left behind by a crashed
            # predecessor: SharedMemory(create=True) raises FileExistsError on a
            # stale name, and a fused process that was SIGKILLed never gets to
            # unlink. Without this, one hard kill makes every subsequent run on
            # this server tag fail to start with an error that points at the
            # name, not at the cause.
            try:
                _stale = SharedMemory(name=name, create=False)
                _stale.close()
                _stale.unlink()
            except FileNotFoundError:
                pass
            except Exception:  # noqa: BLE001 -- never let cleanup mask the real open
                pass
            self._shm = SharedMemory(name=name, create=True, size=size)
        else:
            self._shm = SharedMemory(name=name, create=False)
            if self._shm.size < size:
                self._shm.close()
                raise ValueError(
                    f"fused state shm '{name}' too small: {self._shm.size} < "
                    f"{size} -- the two sides disagree on (num_envs, "
                    f"action_dim, num_joints, num_bodies, num_objects, "
                    f"num_scene_joints)"
                )

        buf = self._shm.buf
        off = 0
        self._hdr = np.ndarray((_HEADER_LEN,), dtype=np.int64, buffer=buf[off : off + _HEADER_BYTES])
        off += _HEADER_BYTES
        self._err = np.ndarray((_ERR_BYTES,), dtype=np.uint8, buffer=buf[off : off + _ERR_BYTES])
        off += _ERR_BYTES

        def _region(shape, dtype, nbytes):
            nonlocal off
            arr = np.ndarray(shape, dtype=dtype, buffer=buf[off : off + nbytes])
            off += nbytes
            return arr

        #: client writes, server reads -- joint position targets, ordered by
        #: ``SceneSchema.joint_names``.
        self.actions = _region((n, a), np.float32, self._sz_actions)
        #: client writes, server reads -- 1 = reset this env.
        self.reset_mask = _region((n,), np.int32, self._sz_reset_mask)
        #: client writes, server reads -- per-env seed, -1 = verbatim replay
        #: (no randomization), matching the seed contract used everywhere else
        #: in this repo.
        self.reset_seeds = _region((n,), np.int64, self._sz_reset_seeds)
        #: client writes, server reads -- per-env benchmark instance to switch
        #: INTO on this reset, -1 = leave this env on whatever it is already
        #: running. Present because the whole point of a superset stage is
        #: that every instance's objects are already loaded; switching is then
        #: just writing that instance's authored object poses, which needs no
        #: stage rebuild -- only a way to say WHICH instance, which is what
        #: this region is.
        self.reset_instances = _region((n,), np.int64, self._sz_reset_instances)
        #: server writes, client reads -- ``[pos, vel]`` per joint.
        self.joint_states = _region((n, j, 2), np.float32, self._sz_joint_states)
        #: server writes, client reads -- ``[x,y,z,qw,qx,qy,qz]`` per body.
        self.body_poses = _region((n, b, 7), np.float32, self._sz_body_poses)
        #: server writes, client reads -- same layout, non-robot rigid bodies.
        self.object_poses = _region((n, o, 7), np.float32, self._sz_object_poses)
        #: server writes, client reads -- seconds since episode start.
        self.sim_time = _region((n,), np.float64, self._sz_sim_time)
        #: server writes, client reads -- positions of non-robot scene joints,
        #: ordered by the fused schema's ``scene_joint_paths``.
        self.scene_joint_positions = _region((n, s), np.float32, self._sz_scene_joint_positions)

        if create:
            self._hdr[:] = 0
            self._hdr[_H_MAGIC] = _MAGIC
            self._hdr[_H_NUM_ENVS] = n
            self._hdr[_H_ACTION_DIM] = a
            self._hdr[_H_NUM_JOINTS] = j
            self._hdr[_H_NUM_BODIES] = b
            self._hdr[_H_NUM_OBJECTS] = o
            self._hdr[_H_NUM_SCENE_JOINTS] = s
        else:
            self._validate_header()

    def _validate_header(self) -> None:
        """Attach-side check: the creator's dimensions must match the ones this
        side was constructed with. Catches the case the size check above cannot
        -- a segment that is big enough but shaped differently (e.g. more envs
        and fewer joints), where every read would silently return garbage."""
        if int(self._hdr[_H_MAGIC]) != _MAGIC:
            raise ValueError(
                f"fused state shm '{self.name}': bad magic "
                f"0x{int(self._hdr[_H_MAGIC]):x} (expected 0x{_MAGIC:x}) -- "
                f"this segment was not created by FusedStateShm"
            )
        found = (
            int(self._hdr[_H_NUM_ENVS]),
            int(self._hdr[_H_ACTION_DIM]),
            int(self._hdr[_H_NUM_JOINTS]),
            int(self._hdr[_H_NUM_BODIES]),
            int(self._hdr[_H_NUM_OBJECTS]),
            int(self._hdr[_H_NUM_SCENE_JOINTS]),
        )
        want = (
            self.num_envs,
            self.action_dim,
            self.num_joints,
            self.num_bodies,
            self.num_objects,
            self.num_scene_joints,
        )
        if found != want:
            raise ValueError(
                f"fused state shm '{self.name}' dimension mismatch: segment "
                f"says (num_envs, action_dim, num_joints, num_bodies, "
                f"num_objects, num_scene_joints)={found}, this side expects {want}"
            )

    # ------------------------------------------------------------------
    # Header accessors
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._shm.name

    @property
    def seq(self) -> int:
        """Server-side response counter. Bumped once per completed command, so
        the client can assert it is reading the answer to the request it just
        sent rather than the previous one."""
        return int(self._hdr[_H_SEQ])

    @property
    def cmd(self) -> int:
        return int(self._hdr[_H_CMD])

    @property
    def status(self) -> int:
        return int(self._hdr[_H_STATUS])

    # ------------------------------------------------------------------
    # Client side
    # ------------------------------------------------------------------

    def set_command(self, cmd: int, *, reset_mask_valid: bool = False, want_image: bool = True) -> None:
        """Publish a command after filling its payload and before posting the semaphore.

        want_image=False allows a step to skip rendering. Set it on every command so
        a previous request cannot suppress the next requested image."""
        self._hdr[_H_CMD] = int(cmd)
        self._hdr[_H_RESET_MASK_VALID] = 1 if reset_mask_valid else 0
        self._hdr[_H_SKIP_RENDER] = 0 if want_image else 1

    # ------------------------------------------------------------------
    # Server side
    # ------------------------------------------------------------------

    def read_command(self) -> tuple[int, bool]:
        """Server: what the client asked for, as ``(cmd, reset_mask_valid)``."""
        return int(self._hdr[_H_CMD]), bool(self._hdr[_H_RESET_MASK_VALID])

    def want_image(self) -> bool:
        """Server: must this command publish a rendered frame?

        Kept separate from ``read_command`` rather than widening its tuple:
        every existing call site unpacks exactly two values, and RESET always
        renders regardless of the flag (its caller is handed an observation),
        so only the STEP branch has any use for this."""
        return not bool(self._hdr[_H_SKIP_RENDER])

    def publish_ok(self) -> None:
        """Server: mark the command complete and successful. Call AFTER the
        state regions are fully written -- the client reads them the moment the
        response semaphore is posted."""
        self._hdr[_H_STATUS] = STATUS_OK
        self._err[0] = 0
        self._hdr[_H_SEQ] += 1

    def publish_error(self, message: str) -> None:
        """Server: mark the command failed, with a message the client re-raises.

        Truncated to fit; the truncation is deliberate and one-way. An engine
        traceback can be arbitrarily long, and the alternative (a second
        variable-length channel just for error text) is machinery nobody would
        exercise until the day it is needed. The full text is printed on the
        server's own stderr, which is where a real diagnosis happens anyway --
        this field only needs to say enough for the client to name the failure.
        """
        raw = (message or "").encode("utf-8", errors="replace")[: _ERR_BYTES - 1]
        self._err[: len(raw)] = np.frombuffer(raw, dtype=np.uint8)
        self._err[len(raw)] = 0
        self._hdr[_H_STATUS] = STATUS_ERROR
        self._hdr[_H_SEQ] += 1

    def read_error(self) -> str:
        """Client: the server's message for the last failed command."""
        nul = int(np.argmax(self._err == 0)) if (self._err == 0).any() else _ERR_BYTES
        return bytes(self._err[:nul]).decode("utf-8", errors="replace")

    # ------------------------------------------------------------------

    def close(self, *, unlink: bool = False) -> None:
        """Detach. ``unlink=True`` (creator only) also removes the name.

        Idempotent: the numpy views must be dropped before ``SharedMemory.close``
        or CPython raises ``BufferError: cannot close exported pointers exist``,
        so every view is released here first.
        """
        for attr in (
            "_hdr",
            "_err",
            "actions",
            "reset_mask",
            "reset_seeds",
            "reset_instances",
            "joint_states",
            "body_poses",
            "object_poses",
            "sim_time",
            "scene_joint_positions",
        ):
            if getattr(self, attr, None) is not None:
                setattr(self, attr, None)
        shm: Optional[SharedMemory] = getattr(self, "_shm", None)
        if shm is None:
            return
        try:
            shm.close()
        except Exception:  # noqa: BLE001
            pass
        if unlink:
            try:
                shm.unlink()
            except FileNotFoundError:
                pass
            except Exception:  # noqa: BLE001
                pass
        self._shm = None


__all__ = [
    "FusedStateShm",
    "CMD_IDLE",
    "CMD_STEP",
    "CMD_RESET",
    "CMD_SHUTDOWN",
    "STATUS_OK",
    "STATUS_ERROR",
]

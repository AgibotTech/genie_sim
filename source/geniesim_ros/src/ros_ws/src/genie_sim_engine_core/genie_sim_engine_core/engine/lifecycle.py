#!/usr/bin/env python3

# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Lifecycle/control records: the data the control verbs pass around.

The verbs themselves are declared on ``PhysicsEngine`` (``engine/base.py``),
and that ABC — not a Protocol here — is this tree's lifecycle contract.

The ROS surface for the verbs already existed in C++ (``/pause_physics``,
``/unpause_physics``, ``/reset_simulation``, ``/reset_world``, ``~/pause``,
``~/resume``, ``~/reset``, ``~/change_state``, ``~/get_state``) and the run
loop already drained a verb queue. What was missing was the Python side: the
loop discovered each verb with ``getattr(sim, "pause", None)`` and silently
did nothing when the probe came back empty — which is what happened for
``pause`` and ``resume``, because no backend in this tree implemented either.
A declared surface turns that class of silence into either a working call or
a documented no-op.

Two contracts, one process, deliberately
----------------------------------------
An integration tree consumes ``src/ros_ws/src`` by symlink rather than by
copy, so its layers and this one's run in one process:

* that tree's own ``SimulationControl`` Protocol, ``Snapshot``, hook surface,
  engine factory, and a shipped conformance suite. The contract for the
  backends in THAT tree.
* ``PhysicsEngine`` here — the contract for THIS tree.
* the backends, which satisfy one or both. At least one satisfies both: it
  implements that contract while subclassing this ABC.

Neither contract chases the other, and that is a decision rather than a gap.
``episode_reset`` is ``(env_ids, seeds, instances)`` here against
``(env_ids, seed)`` there, because this tree is multi-env and a GRPO group
needs each member seeded independently, which one scalar cannot express.
Three live consequences:

* The run loop's ``~/reset`` calls ``episode_reset()`` with NO arguments —
  the one form that binds under both arities and means the same thing under
  each. Its ``try`` has only a ``finally``, so a TypeError there would not
  degrade the reset, it would end the process on an operator's keystroke.
* ``RosResetCoordinator.drain_resets`` cannot make that call — the per-env
  mapping is its whole job — so it always passes ``seeds=``/``instances=``,
  and a backend on the two-argument arity raises on EVERY reset through it.
  The raise is caught and warned, and the epoch deliberately does not
  advance, which means a client waiting on the credential waits forever.
  ``PhysicsEngine``'s ``audit_lifecycle`` names such a backend once at
  bring-up rather than once per reset, and names the fix with it.
* That tree's conformance suite does not pass against ``PhysicsEngine``,
  and is not meant to: it is Tier 1's suite for Tier 2 backends of THAT
  package, and this ABC is not one of them. Do not "fix" this tree to
  satisfy it.

There is deliberately no ``SimulationControl`` mirror in this module any more.
One used to live here, member for member, so that
``isinstance(engine, SimulationControl)`` would pass against either repo. It
did — and it taught nothing: ``@runtime_checkable`` compares member NAMES and
nothing else, so it could not see the one divergence that changes a call. A
consuming tree's engine factory uses exactly that isinstance as its admission
gate, which is how a backend passes the gate and then rejects the call the
caller makes. A mirror with no importers, hand-synced against the ABC
beside it, whose only check was blind to the thing worth checking, was a
liability. The ABC's documented defaults plus ``audit_lifecycle`` — which
checks the divergence that actually changes a call, and says so by name — are
what replaced it.

What stays here
---------------
The records the verbs exchange — ``EpisodeHandle``, ``EpisodeSummary``,
``Snapshot`` with its two mismatch errors, ``TeleportRequest`` — and
``ActivitySignal``, which duplicates nothing: it is the read side the pacer
needs to tell "idle client" from "client between commands" (previously reached
for as ``sim._idle_activity_ok`` / ``sim._last_cmd_wall`` /
``sim._pending_reset_envs``), and its one implementation is
``RosResetCoordinator``, not an engine.

``TeleportRequest`` is local as well. A tree that applies teleports from raw
``tf2_msgs/TFMessage`` inside its own C++ loop needs no Python shape; this
tree routes them to the backend, so the request needs a typed shape and an
``env_ids`` selector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol, Tuple, runtime_checkable

# ----------------------------------------------------------------------
# Episode records
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class EpisodeHandle:
    """Handle returned by ``PhysicsEngine.episode_begin``.

    ``started_at_tick`` is the backend's own step counter, not sim time — a
    consumer comparing it against :attr:`EpisodeSummary.ended_at_tick` gets
    the exact step count regardless of ``dt`` changes or time scaling.
    """

    episode_id: int
    seed: int
    started_at_tick: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EpisodeSummary:
    """Summary returned by ``PhysicsEngine.episode_end``.

    ``reason`` is free-form and consumer-defined (``"success"``,
    ``"timeout"``, ``"shutdown"``, ...); nothing in the engine branches on it.
    """

    episode_id: int
    reason: str
    ended_at_tick: int
    tick_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)


# ----------------------------------------------------------------------
# Snapshots
# ----------------------------------------------------------------------


class SnapshotBackendMismatch(RuntimeError):
    """Restore attempted on a backend other than the one that took the snapshot."""


class SnapshotVersionMismatch(RuntimeError):
    """Snapshot's backend version is incompatible with the running build."""


@dataclass(frozen=True)
class Snapshot:
    """Opaque-payload simulation state with structured metadata alongside.

    ``payload`` is backend-native serialisation — consumers treat it as
    bytes. The typed fields sit OUTSIDE it so cross-backend tooling can
    inspect what determinism actually depends on (RNG streams, event
    schedules, curriculum rows, command targets) without a backend-specific
    decoder.

    The contract, unchanged from the reference:

    * restore is legal from any lifecycle state — idle, running, or paused;
    * ``restore(snap)`` then ``step()`` reproduces the original run
      bit-for-bit on the same backend and build;
    * a cross-backend or cross-version restore raises
      :class:`SnapshotBackendMismatch` / :class:`SnapshotVersionMismatch`
      rather than producing a plausible-looking divergent run.
    """

    backend: str
    backend_version: str
    tick: int
    payload: bytes

    rng: Tuple[bytes, ...] = field(default_factory=tuple)
    """One RNG state per env — the pickled ``numpy.random.Generator`` state."""

    manager_rng: Dict[str, Tuple[bytes, ...]] = field(default_factory=dict)
    """Per-manager, per-env RNG streams. Keys: ``reward``, ``observation``, ...."""

    event_schedule: Dict[str, Any] = field(default_factory=dict)
    """Next-fire tick for every ``interval`` event term."""

    curriculum: Dict[str, Any] = field(default_factory=dict)
    """Per-env curriculum row / difficulty / promotion counters."""

    commands: Dict[str, Any] = field(default_factory=dict)
    """Current target for each command term."""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Free-form backend / task metadata. Not required for replay determinism."""


# ----------------------------------------------------------------------
# Teleport
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class TeleportRequest:
    """Move one body to a pose, out of band with the physics.

    Free-joint bodies only. A body welded to the world, or attached by a
    revolute/prismatic joint, has no 6-DoF state to write: the reference
    rejects those (``ApplyTeleportTransform``) rather than approximating, and
    so does this tree.

    Attributes
    ----------
    body
        Body label. Backends accept their own naming forms; the newton
        backend takes the full body path, the short leaf name, or
        ``env_<w>/<name>`` to address one world out of many.
    position
        ``(x, y, z)`` in world frame.
    orientation_wxyz
        Quaternion as ``(w, x, y, z)``. Canonical for this codebase's Python
        layer — the ROS wire is ``x, y, z, w`` and Newton's free-joint ``q``
        slots are ``xyzw``, so exactly one reorder happens at each boundary
        and neither is left to the caller.
    zero_velocity
        Zero the body's 6 velocity DoFs as part of the write. Default True,
        matching the reference: a body that keeps its pre-teleport velocity
        immediately drifts off the pose that was just requested.
    env_ids
        Worlds to apply to, or ``None`` for all of them. Ignored by
        single-env backends.
    """

    body: str
    position: Tuple[float, float, float]
    orientation_wxyz: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    zero_velocity: bool = True
    env_ids: Optional[Tuple[int, ...]] = None


# ----------------------------------------------------------------------
# Activity signal
# ----------------------------------------------------------------------


@runtime_checkable
class ActivitySignal(Protocol):
    """Read side of "is a client actually driving this simulation?".

    The pacer uses it to decide whether idling is safe. Backends that expose
    one return it from ``PhysicsEngine.activity``; ``None`` there means "no
    signal", and the pacer then free-runs rather than guessing.

    Every member is a property so the pacer samples live state each tick —
    a snapshot taken at bring-up would freeze the answer for the whole run.
    """

    @property
    def idle_activity_ok(self) -> bool:
        """True once idling is safe — i.e. a client has been seen at all.

        False before the first command keeps the loop stepping at full rate
        during bring-up, when a client may be waiting on ``/clock`` to
        advance before it sends anything.
        """
        ...

    @property
    def last_cmd_wall(self) -> float:
        """``time.monotonic()`` of the most recent client command."""
        ...

    @property
    def reset_pending(self) -> bool:
        """True while a reset is queued and not yet carried out.

        Suppresses idling: the reset needs a tick to be applied on, and a
        client that just asked for one is by definition not idle.
        """
        ...

    @property
    def reset_epoch(self) -> int:
        """Monotonic counter, bumped once per completed reset.

        Published as the credential a wrapper waits on to know the state it
        is reading belongs to the episode it asked for.
        """
        ...


__all__ = [
    "ActivitySignal",
    "EpisodeHandle",
    "EpisodeSummary",
    "Snapshot",
    "SnapshotBackendMismatch",
    "SnapshotVersionMismatch",
    "TeleportRequest",
]

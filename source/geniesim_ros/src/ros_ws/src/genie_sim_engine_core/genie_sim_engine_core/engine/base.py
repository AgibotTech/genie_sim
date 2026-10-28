#!/usr/bin/env python3

# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Abstract physics engine interface + registry-based factory.

All concrete engines subclass :class:`PhysicsEngine` and self-register
via :meth:`PhysicsEngine.register` (typically as a class decorator).
The run loop depends only on this interface — no engine-specific
conditionals here.

Concrete backends own their own state-readback (joint states, body
transforms, odom). The base class does not import any backend module;
:meth:`get_joint_states` / :meth:`get_body_transforms` / :meth:`get_odom`
are abstract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import inspect
from typing import Any, Callable, ClassVar, Dict, List, Optional, Sequence, Tuple, Type

import numpy as np

from genie_sim_engine_core.engine.lifecycle import (
    ActivitySignal,
    EpisodeHandle,
    EpisodeSummary,
    Snapshot,
    TeleportRequest,
)


@dataclass(frozen=True)
class EngineParameterABI:
    """Constructor parameter accepted by a registered physics backend."""

    name: str
    required: bool
    kind: str
    annotation: str = ""
    default: str = ""


@dataclass(frozen=True)
class EngineBackendABI:
    """Constructor compatibility summary for a registered physics backend."""

    name: str
    class_name: str
    parameters: Tuple[EngineParameterABI, ...]
    accepts_extra_kwargs: bool = False

    @property
    def accepted_names(self) -> Tuple[str, ...]:
        return tuple(param.name for param in self.parameters)

    @property
    def required_names(self) -> Tuple[str, ...]:
        return tuple(param.name for param in self.parameters if param.required)


class PhysicsEngine(ABC):
    """Minimal interface the physics step loop depends on.

    Lifecycle::

        engine = PhysicsEngine.create(physics_engine, ...)
        engine.startup(headless)
        while running:
            engine.step(dt, step_start)          # simulate + fetch
            engine.tick_extras(usd_update_fn)    # e.g. cloth writeback
        engine.shutdown()
    """

    # Backend self-registration.  Subclasses call
    # ``PhysicsEngine.register("isaac_physx")(IsaacPhysXEngine)`` (or
    # use it as a decorator) to add themselves; ``create`` looks them
    # up here.  Keeping the map on the base class avoids a hard
    # dependency on any specific backend module.
    _registry: ClassVar[Dict[str, Type["PhysicsEngine"]]] = {}
    _abi_registry: ClassVar[Dict[str, EngineBackendABI]] = {}

    # ------------------------------------------------------------------
    # Properties the run loop reads every tick
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def stage(self) -> Any:
        """Live USD stage."""

    @property
    @abstractmethod
    def robot_prefix(self) -> str:
        """USD prim prefix for the robot root (e.g. ``"ur5"``)."""

    @property
    @abstractmethod
    def joint_names(self) -> List[str]:
        """Ordered list of joint DOF names used for state publish."""

    @property
    def scene_joint_names(self) -> List[str]:
        """Articulated-scene DoF names for ``/joint_states_scene``.

        OPTIONAL capability, default empty. Two backend shapes exist and
        ``EngineSession`` handles both:

        * Backends whose :attr:`joint_names` spans EVERY articulated DoF
          (the Isaac Sim paths — ``kit/stage.py``'s joint collection
          traverses the whole USD stage) leave this empty. The session then
          derives the scene set by subtracting the manifest's
          ``robot.joint_names`` out of :attr:`joint_names`.

        * Backends that keep scene articulations off the controllable
          surface entirely (newton-standalone — a door hinge must not get a
          PD drive or an RL action slot) report them HERE instead, since the
          subtraction above would find nothing. Such backends must also
          implement :meth:`get_scene_joint_states`, and the names must be
          disjoint from :attr:`joint_names`.

        Names are bare joint names, ``env_<w>/``-prefixed at ``num_envs > 1``
        like every other multi-env label.
        """
        return []

    def get_scene_joint_states(self) -> Optional[Tuple[Any, Any, Any]]:
        """``(positions, velocities, efforts)`` aligned with
        :attr:`scene_joint_names`.

        OPTIONAL, default ``None`` meaning "this backend has no separate
        scene DoF readback" — the publish loop then falls back to slicing
        :meth:`get_joint_states` by the session's computed scene indices.

        ``None`` is also the correct return for a backend that DOES
        implement the channel but whose current scene has no articulated
        DoFs; the loop leaves ``/joint_states_scene`` quiet either way.
        """
        return None

    @property
    @abstractmethod
    def joint_prim_map(self) -> Dict[str, str]:
        """Joint name → USD prim path mapping."""

    @property
    @abstractmethod
    def body_paths(self) -> List[str]:
        """USD prim paths for all rigid bodies (for tf_scene)."""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def startup(self, headless: bool) -> None:
        """One-time setup just before entering the physics loop."""

    @abstractmethod
    def step(self, dt: float, step_start: float) -> float:
        """Run one physics tick.

        Returns the wall-clock seconds spent in the physics solve.
        Implementations should NOT call ``simulation_app.update()``
        here — the caller handles the render budget.
        """

    @abstractmethod
    def tick_extras(self) -> None:
        """Per-tick work that ALWAYS fires (every physics step).

        For engines that publish state to a parallel renderer (Newton-direct
        with C++ GIL-released render thread), this is where Fabric writeback
        happens — every tick, so the renderer always sees fresh data when
        it polls Fabric on its own cadence.
        """

    def sync_visual_state(self) -> None:
        """Optional explicit force-sync hook.

        Default no-op. Engines that defer Fabric writeback to render-time
        (single-threaded inline-render use cases) can override this and the
        run loop will call it just before ``simulation_app.update()``.
        """
        return

    def note_render(self, render_ms: float, did_render: bool) -> None:
        """Called by the run loop after each potential render tick.

        Default no-op. Override to gate expensive per-render work to render
        cadence rather than physics cadence.
        """
        return

    def note_render_target(self, render_hz: float) -> None:
        """Tell the engine the run loop's target render Hz. Default no-op."""
        return

    def note_phase_timing(
        self,
        step_ms: float = 0.0,
        extras_ms: float = 0.0,
        render_ms: float = 0.0,
        render_sync_ms: float = 0.0,
        did_render: bool = False,
    ) -> None:
        """Per-tick phase timings from the run loop. Default no-op."""
        return

    def note_publish_phase(
        self,
        clock_ms: float = 0.0,
        joints_ms: float = 0.0,
        bodies_ms: float = 0.0,
        odom_ms: float = 0.0,
    ) -> None:
        """Publish-phase timing breakdown from the run loop. Default no-op."""
        return

    # ------------------------------------------------------------------
    # State readback — abstract, no default backend dependency.
    # ------------------------------------------------------------------

    @abstractmethod
    def get_joint_states(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(positions, velocities)`` aligned with ``joint_names``."""

    @abstractmethod
    def get_body_transforms(self) -> Tuple[np.ndarray, List[str]]:
        """Return ``(Nx7 pose array, frame_paths)`` for ``body_paths``."""

    @abstractmethod
    def get_odom(self, sim_time: float) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Return ``(pose7, twist6)`` for the base_link, or ``None``."""

    def get_contact_states(
        self,
    ) -> Optional[Tuple[List[str], List[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Return per-point contact state, or ``None`` if unsupported.

        When implemented, returns a 6-tuple of parallel per-point arrays
        matching the ``publish_contact_states`` C++ pybind:

        * ``body1_names`` / ``body2_names`` — length-N string lists.
        * ``positions`` — ``(N, 3)`` float64, world frame.
        * ``normals`` — ``(N, 3)`` float64, world frame.
        * ``depths`` — ``(N,)`` float64, penetration depths.
        * ``wrenches`` — ``(N, 6)`` float64, engine-native frame
          (``(fx, fy, fz, tx, ty, tz)``). MuJoCo emits contact-local
          frame per ``mj_contactForce``; other backends document their
          own convention.

        An empty tuple (all ``N == 0``) is a meaningful reading — "no
        contacts this frame" — and gets published as an empty
        ``ContactsState``. Returning ``None`` means the backend does
        not produce contact state; the run loop skips the publish
        entirely and the topic stays quiet.
        """
        return None

    @abstractmethod
    def apply_commands(
        self,
        cmd_positions: Dict[str, float],
        cmd_4ws_steer_pos: Any,
        cmd_4ws_drive_vel: Any,
        cmd_4ws_stamp: Any,
    ) -> None:
        """Forward joint/drive commands to the active articulation."""

    @abstractmethod
    def shutdown(self) -> None:
        """Clean up resources."""

    # ------------------------------------------------------------------
    # Lifecycle / control — THIS is the contract, not a Protocol elsewhere.
    #
    # Every member has a default so no existing backend breaks, following
    # the ``sync_visual_state`` / ``note_render`` no-op pattern above. The
    # defaults are declared rather than discovered: before this section the
    # run loop probed for each verb with ``getattr(sim, "pause", None)`` and
    # skipped it when the probe came back empty, which is exactly what
    # happened to ``~/pause`` and ``~/resume`` for as long as no backend
    # implemented them. A no-op that is written down is reviewable; a
    # no-op that is a failed attribute lookup is not.
    #
    # ``engine/lifecycle.py`` holds the record types these exchange, and
    # the note on why a consuming tree's lifecycle contract is a second
    # contract rather than one this ABC should converge onto.
    # ------------------------------------------------------------------

    @property
    def autostart(self) -> bool:
        """Whether the backend starts spinning on construction.

        Default True — the ROS realtime path, which steps as soon as it is
        built. Override to False for consumers that drive the clock
        themselves (RL, benchmarks, tests, SysID): the backend then stays
        idle until :meth:`episode_begin`.
        """
        return True

    @property
    def num_envs(self) -> int:
        """Parallel worlds this backend simulates. Default 1.

        Read by the session to decide whether names get an ``env_<w>/``
        prefix, so a multi-env backend MUST override it — a wrong 1 here
        publishes world 0's labels for every world's data.
        """
        return 1

    @property
    def reset_epoch(self) -> int:
        """Causal credential: how many resets this backend has COMPLETED.

        Default 0 — a backend that never resets never bumps it, and the run
        loop publishes a stuck 0 rather than pretending it has no epoch. Own
        it on the backend, not on whatever queued the reset: the state write
        is what the credential certifies, so it must advance in the same place
        the write lands and only after it has landed. ``episode_reset`` called
        directly by an RL consumer counts exactly as much as one arriving over
        ROS.

        Published in ``/joint_states`` ``header.frame_id`` as
        ``reset_epoch:{N}``; a consumer waits for it to pass a pre-reset
        baseline before trusting the state alongside it.
        """
        return 0

    @property
    def activity(self) -> Optional[ActivitySignal]:
        """Client-activity signal for the pacer, or ``None`` if unavailable.

        ``None`` means "this backend cannot tell whether a client is
        driving it", and the pacer then free-runs rather than guessing —
        idling a simulation whose client is merely between commands would
        stall that client for as long as the idle lasts.
        """
        return None

    def pause(self) -> None:
        """Stop advancing physics. Default no-op; idempotent when implemented.

        The run loop keeps its own ``paused`` flag either way, so the loop
        stops stepping regardless. Implementing this matters for backends
        with machinery that runs BESIDE the step call — a parallel render
        thread, an async state mirror, a solver graph replaying on its own
        cadence — which a loop-level flag cannot reach.
        """
        return

    def resume(self) -> None:
        """Resume advancing physics. Default no-op; idempotent when implemented."""
        return

    def set_time_scale(self, scale: float) -> None:
        """Multiply the target sim-time rate; ``1.0`` is realtime.

        Validates and then does nothing: pacing normally lives in the run
        loop's pacer, so most backends have no clock of their own to scale.
        Override when the backend owns one the pacer does not drive.

        The validation is NOT optional even though the default is a no-op.
        A non-positive scale is not a slow simulation, it is a stopped or
        time-reversed one, and a backend that accepts ``0.0`` silently hands
        back a sim that never advances with no error to explain why. Raising
        here means every backend inherits the rejection rather than each one
        having to remember it.

        Raises:
            ValueError: ``scale`` is zero or negative.
        """
        if scale <= 0:
            raise ValueError(f"time scale must be positive, got {scale!r}")
        return

    def set_seed(self, seed: int) -> None:
        """Seed the backend's RNG streams. Default no-op."""
        return

    def episode_begin(self, seed: int) -> EpisodeHandle:
        """Start an episode and return its handle. Default: no state change.

        The default reports ``backend_tracks_episodes: False`` in the
        handle's metadata rather than inventing an id and a tick count, so a
        consumer can see that the numbers are placeholders instead of
        reading zeros as a real episode boundary.
        """
        return EpisodeHandle(
            episode_id=0,
            seed=seed,
            started_at_tick=0,
            metadata={"backend_tracks_episodes": False},
        )

    def episode_reset(
        self,
        env_ids: Optional[Sequence[int]] = None,
        seeds: Optional[Dict[int, int]] = None,
        instances: Optional[Dict[int, int]] = None,
    ) -> None:
        """Reset the listed worlds, or all of them when ``env_ids`` is ``None``.

        Default no-op. Both mappings are keyed by **env id**, matching what
        ``RosResetCoordinator._per_env_map`` produces and what a backend
        looks up as ``if seeds and w in seeds``:

        * ``seeds`` — ``{env_id: seed}``. Per-env rather than one scalar
          because a GRPO group needs each member seeded independently.
        * ``instances`` — ``{env_id: instance_id}``, selecting which of a
          scene's pre-authored object placements that world resets into.

        A mapping keyed by anything else is not a type error, it is a silent
        no-op: every ``w in instances`` misses and the reset proceeds without
        the randomisation the caller asked for.
        """
        return

    def episode_end(self, reason: str) -> EpisodeSummary:
        """End the current episode and return its summary. Default: no state change."""
        return EpisodeSummary(
            episode_id=0,
            reason=reason,
            ended_at_tick=0,
            tick_count=0,
            metadata={"backend_tracks_episodes": False},
        )

    def teleport(self, requests: Sequence[TeleportRequest]) -> int:
        """Apply out-of-band pose writes; returns the number APPLIED.

        Default 0 — "this backend does not teleport" — which lets the caller
        log the shortfall honestly instead of reporting a batch as landed
        because nothing raised. Legal while paused, and that is the main
        use: reposition the world with the sim held still.
        """
        return 0

    def snapshot(self) -> Snapshot:
        """Capture restorable state. Opt-in capability.

        Raises rather than returning an empty :class:`Snapshot`: a snapshot
        that restores to nothing is worse than no snapshot, because the
        divergence only shows up later as an unreproducible run.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement snapshot()")

    def restore(self, snap: Snapshot) -> None:
        """Restore state captured by :meth:`snapshot`. Opt-in capability.

        Implementations must reject a snapshot from another backend or an
        incompatible build with ``SnapshotBackendMismatch`` /
        ``SnapshotVersionMismatch``.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement restore()")

    def raw_model(self) -> Any:
        """Escape hatch — the backend-native model handle, or ``None``.

        No cross-backend guarantees. For SysID and contract debugging;
        everything else stays above this interface.
        """
        return None

    def raw_data(self) -> Any:
        """Escape hatch — the backend-native runtime data handle, or ``None``."""
        return None

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def register(cls, name: str) -> Callable[[Type["PhysicsEngine"]], Type["PhysicsEngine"]]:
        """Class decorator: register ``cls`` under ``name`` for :meth:`create`.

        Usage::

            @PhysicsEngine.register("isaac_physx")
            class IsaacPhysXEngine(PhysicsEngine):
                ...

        Re-registering the same name is allowed (last-write-wins) so
        backends can shadow defaults during testing.
        """

        def _decorator(target: Type["PhysicsEngine"]) -> Type["PhysicsEngine"]:
            cls._registry[name] = target
            cls._abi_registry[name] = cls._inspect_backend_abi(name, target)
            return target

        return _decorator

    @classmethod
    def _inspect_backend_abi(cls, name: str, target: Type["PhysicsEngine"]) -> EngineBackendABI:
        parameters: List[EngineParameterABI] = []
        accepts_extra_kwargs = False
        signature = inspect.signature(target.__init__)

        for param in signature.parameters.values():
            if param.name in ("self", "cls"):
                continue
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                accepts_extra_kwargs = True
                continue
            if param.kind == inspect.Parameter.VAR_POSITIONAL:
                continue

            required = param.default is inspect.Parameter.empty
            annotation = "" if param.annotation is inspect.Parameter.empty else str(param.annotation)
            default = "" if required else repr(param.default)
            parameters.append(
                EngineParameterABI(
                    name=param.name,
                    required=required,
                    kind=param.kind.name,
                    annotation=annotation,
                    default=default,
                )
            )

        return EngineBackendABI(
            name=name,
            class_name=f"{target.__module__}.{target.__qualname__}",
            parameters=tuple(parameters),
            accepts_extra_kwargs=accepts_extra_kwargs,
        )

    @classmethod
    def backend_abi(cls, physics_engine: str) -> EngineBackendABI:
        try:
            return cls._abi_registry[physics_engine]
        except KeyError as exc:
            available = ", ".join(sorted(cls._abi_registry)) or "<none>"
            raise ValueError(
                f"unknown physics_engine {physics_engine!r}; registered ABI backends: {available}"
            ) from exc

    @classmethod
    def registered_backend_abis(cls) -> Dict[str, EngineBackendABI]:
        return dict(cls._abi_registry)

    @classmethod
    def _validate_create_kwargs(cls, physics_engine: str, kwargs: Dict[str, Any]) -> None:
        abi = cls.backend_abi(physics_engine)
        accepted = set(abi.accepted_names)
        missing = [name for name in abi.required_names if name not in kwargs]
        unexpected = sorted(name for name in kwargs if name not in accepted)

        if abi.accepts_extra_kwargs:
            unexpected = []

        if not missing and not unexpected:
            return

        accepted_msg = ", ".join(abi.accepted_names) or "<none>"
        details: List[str] = []
        if missing:
            details.append(f"missing required parameter(s): {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected parameter(s): {', '.join(unexpected)}")
        if abi.accepts_extra_kwargs:
            details.append("backend accepts extra **kwargs")

        raise TypeError(
            f"physics_engine {physics_engine!r} constructor ABI mismatch for {abi.class_name}: "
            f"{'; '.join(details)}. Accepted parameters: {accepted_msg}."
        )

    @classmethod
    def create(cls, physics_engine: str, **kwargs: Any) -> "PhysicsEngine":
        """Instantiate the engine registered for ``physics_engine``.

        ``physics_engine`` must already be normalized — callers should
        validate via ``runtime.bootstrap._validate_engine_id`` before
        reaching this factory.

        Backend modules must have been imported at least once before
        calling this (importing them populates the registry via the
        ``@PhysicsEngine.register`` decorator).
        ``genie_sim_engine_newton.common.session._register_backends``
        imports ``genie_sim_engine_newton.engine.newton`` /
        ``genie_sim_engine_isaacsim.kit.isaac_physx`` /
        ``genie_sim_engine_isaacsim.kit.isaac_newton`` once at startup.
        """
        try:
            target = cls._registry[physics_engine]
        except KeyError as exc:
            available = ", ".join(sorted(cls._registry)) or "<none>"
            raise ValueError(
                f"unknown physics_engine {physics_engine!r}; "
                f"registered backends: {available}. Ensure the backend "
                f"module has been imported so it can self-register."
            ) from exc
        cls._validate_create_kwargs(physics_engine, kwargs)
        return target(**kwargs)


# ----------------------------------------------------------------------
# Bring-up conformance
# ----------------------------------------------------------------------


def audit_lifecycle(sim: PhysicsEngine, logger: Any) -> None:
    """Warn once, at bring-up, when a backend narrows a lifecycle contract.

    Called from ``EngineSession.__init__``, so it covers every backend that
    session builds — including the ones in an integration tree that consumes
    ``src/ros_ws/src`` by symlink and subclasses ``EngineSession`` to get
    here.

    Why a diagnostic and not a shim
    -------------------------------
    ``PhysicsEngine.episode_reset(env_ids, seeds, instances)`` is this tree's
    contract; a consuming tree's ``episode_reset(env_ids, seed)`` is that
    tree's. Neither chases the other (see ``engine/lifecycle.py``). What is
    left is a backend that subclasses THIS ABC while implementing THAT arity,
    and for such a backend the reset path degrades in ways nothing reports:

    * A backend that does not override ``episode_reset`` inherits the ABC
      no-op, so ``~/reset`` returns success having done nothing at all.
    * ``RosResetCoordinator.drain_resets`` always passes ``seeds=`` and
      ``instances=`` — the per-env mapping is its entire job — so a
      two-argument backend raises ``TypeError`` on EVERY reset through it, not
      only on seeded ones. That raise is caught and warned per reset, and the
      epoch deliberately does not advance, so an rlinf client waiting on the
      ``reset_epoch`` credential waits forever with nothing in its own logs.
    * A backend that resets but never bumps ``reset_epoch`` publishes
      ``reset_epoch:0`` in ``/joint_states`` frame_id for the whole run, which
      is the same stall by a different route: the write landed and the
      credential says it did not.

    Each of those is silent, and each is at bring-up already decidable from the
    class object. So decide it there, name the class, and name the fix — one
    line per finding, rather than a per-reset warning in ``drain_resets`` or a
    signature-driven dispatch that would guess which seed a scalar-seed backend
    wanted.

    The fix this points at is the reset design the rlinf path here already
    runs on, and it is three things, not one:

    1. ``episode_reset(env_ids, seeds, instances)`` — ``seeds`` as
       ``{env_id: seed}`` because a GRPO group needs each member seeded
       independently, ``instances`` as ``{env_id: instance_id}`` for per-reset
       placements.
    2. bump ``reset_epoch`` at the tail, after every state write has landed.
    3. return ``RosResetCoordinator`` from ``activity``, which then supplies
       ``/rlinf/reset_envs`` and its seeded variant, the thread-safe queue, the
       credential, and the ``/joint_command`` activity stamp the idle gate and
       lockstep read. The backend writes state; it does not write ROS.

    ``activity`` returning ``None`` is deliberately NOT reported here —
    ``RealtimePacer.build`` already warns about it, and only when
    ``step_idle_timeout > 0`` makes it matter. Two warnings for one omission
    train an operator to read neither.

    The pause axis
    --------------
    ``pause``/``resume`` have the same shape of failure and are checked the same
    way. Two cases: overriding one and not the other is unambiguously broken,
    and overriding neither is broken only for a backend with machinery running
    beside ``step()`` — which is not decidable from the class object, so that
    one is reported as a question rather than a defect. It is worth reporting
    anyway because the live case is real: ``MujocoBackendAdapter`` runs physics
    on a C++ thread spawned by ``start_physics()`` (its ``step`` is a no-op
    afterwards), so ``/pause_physics`` parks the run loop, stops the publishing,
    and leaves the world integrating.

    Never raises. A conformance report that can abort bring-up is worse than
    the silence it replaces. Each axis is guarded separately so a backend whose
    reset members misbehave still gets its pause findings, and vice versa.
    """
    for check in (_audit_reset_contract, _audit_pause_contract):
        try:
            check(sim, logger)
        except Exception as exc:  # noqa: BLE001
            logger.warn(f"[lifecycle] {check.__name__} skipped: {exc}")


def _audit_reset_contract(sim: PhysicsEngine, logger: Any) -> None:
    """The three reset omissions, worst first.

    Returns after the first hit rather than reporting all three: they compound
    on one root cause, and a bring-up log carrying three lines for one class
    gets skimmed as noise.
    """
    cls = type(sim)
    name = cls.__name__

    if cls.episode_reset is PhysicsEngine.episode_reset:
        logger.warn(
            f"[lifecycle] {name} does not override episode_reset, so it inherits "
            f"the PhysicsEngine no-op: ~/reset and /rlinf/reset_envs will "
            f"report success and change nothing. Implement "
            f"episode_reset(env_ids, seeds, instances), bump reset_epoch at its "
            f"tail, and return a RosResetCoordinator "
            f"(genie_sim_engine_core.common.lifecycle_ros) from activity"
        )
        return

    # Bind the call ``drain_resets`` actually makes, rather than reading
    # parameter names: a backend is free to spell them differently or take
    # ``**kwargs``, and what matters is only whether that one call lands.
    try:
        inspect.signature(sim.episode_reset).bind(None, seeds=None, instances=None)
    except TypeError:
        try:
            # Parameters only. The return annotation is noise in a log line
            # and, unannotated, reads as though the method returned nothing
            # meaningful when the point is the arguments it refuses.
            shape = str(inspect.signature(sim.episode_reset)).split(") ->")[0] + ")"
        except (TypeError, ValueError):  # pragma: no cover - exotic callables
            shape = "(?)"
        logger.warn(
            f"[lifecycle] {name}.episode_reset{shape} narrows "
            f"PhysicsEngine.episode_reset(env_ids, seeds, instances). Every reset "
            f"arriving on /rlinf/reset_envs raises TypeError inside "
            f"RosResetCoordinator.drain_resets, is swallowed there, and leaves "
            f"reset_epoch un-advanced — so a client waiting on the credential "
            f"stalls with nothing in its own logs. ~/reset still works (the run "
            f"loop calls episode_reset() with no arguments on purpose). Widen to "
            f"the three-argument form: seeds is {{env_id: seed}}, per-env because "
            f"a GRPO group needs each member seeded independently and one scalar "
            f"cannot express that; instances is {{env_id: instance_id}}"
        )
        return

    if cls.reset_epoch is PhysicsEngine.reset_epoch:
        logger.warn(
            f"[lifecycle] {name} implements episode_reset but not reset_epoch, so "
            f"/joint_states frame_id carries reset_epoch:0 for the whole run. An "
            f"rlinf client waits on that credential to know the state it is "
            f"reading is post-reset, so the reset lands and the client never "
            f"learns: bump a counter at the tail of episode_reset, after every "
            f"state write"
        )


def _audit_pause_contract(sim: PhysicsEngine, logger: Any) -> None:
    """``pause``/``resume``: an asymmetry is a defect, absence is a question.

    The loop's ``pause`` verb parks its own flag AND calls ``sim.pause()``,
    because a flag the loop keeps to itself cannot stop a backend whose sim
    advances somewhere other than inside ``step()``.  Whether a given backend
    is such a backend is not decidable from the class object, so the two cases
    are reported differently — see ``audit_lifecycle``'s docstring.
    """
    cls = type(sim)
    name = cls.__name__
    overrides = {v for v in ("pause", "resume") if getattr(cls, v) is not getattr(PhysicsEngine, v)}

    if len(overrides) == 1:
        have = overrides.pop()
        missing = "resume" if have == "pause" else "pause"
        logger.warn(
            f"[lifecycle] {name} overrides {have} but not {missing}, so the two are "
            f"not a matched pair: {missing} lands on the PhysicsEngine no-op while "
            f"{have} does real work. ~/pause and ~/resume (and the lifecycle "
            f"DEACTIVATE/ACTIVATE transitions) drive both, so this is a backend that "
            f"can be "
            + ("stopped and never restarted" if have == "pause" else "restarted and never stopped")
            + f" — implement {missing} too, or neither"
        )
        return

    if not overrides:
        logger.warn(
            f"[lifecycle] {name} overrides neither pause nor resume, so ~/pause parks "
            f"the run loop and the backend is not told. Correct IF step() is the only "
            f"thing advancing this sim — expected for a loop-driven backend, and "
            f"nothing to do. Wrong if anything advances state beside step(): a "
            f"physics thread, an async state mirror, a render loop that integrates. "
            f"In that case /pause_physics stops the publishing and not the world, and "
            f"an operator reads a frozen /joint_states off a sim that kept going. "
            f"Gate that driver in pause()/resume() if so"
        )

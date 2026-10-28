# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Shared physics-loop primitives for both engine entry points.

``_merge_params_file``, ``_parse_args`` — ROS-arg helpers.
``_publish_tick`` — clock/joints/bodies/odom publish; identical in both.
``EngineRunLoop`` — common physics-step scaffold; render is injected via
  a callable so Kit-specific and GL-specific code stays in the entry point.
"""

from __future__ import annotations

import time

from ..engine.lifecycle import TeleportRequest
from ..engine.publish_plan import PublishPlan
from .pacing import DEFAULT_STEP_MODE, make_pacer


def _merge_params_file(out: dict, path: str, prefix: str = "genie_sim_engine") -> None:
    try:
        import yaml
    except Exception as exc:
        print(f"[{prefix}] WARN: cannot import yaml to read {path}: {exc}", flush=True)
        return
    try:
        with open(path, "r") as f:
            doc = yaml.safe_load(f) or {}
    except Exception as exc:
        print(f"[{prefix}] WARN: cannot read params file {path}: {exc}", flush=True)
        return
    if not isinstance(doc, dict):
        return
    for _node_key, node_block in doc.items():
        if not isinstance(node_block, dict):
            continue
        ros_params = node_block.get("ros__parameters")
        if not isinstance(ros_params, dict):
            continue
        for k, v in ros_params.items():
            if isinstance(v, bool):
                out[str(k)] = "true" if v else "false"
            elif isinstance(v, dict):
                # A nested MAPPING is the one shape ``str()`` destroys beyond
                # recovery for the consumer: every reader downstream wants the
                # tree, not ``"{'newton': {...}}"``.  ``scene_yaml_defaults``
                # is the live one (``EngineSession`` merges it beneath the
                # scene yaml); keep any other block usable for the same reason.
                # Scalars stay strings — every ``EngineNodeParams`` field
                # coerces from ``str`` and changing that would ripple.
                out[str(k)] = v
            else:
                out[str(k)] = str(v)


def _parse_args(argv, prefix: str = "genie_sim_engine") -> dict:
    """Parse ``--ros-args`` style argv into a flat ``{name: str}`` dict.

    Supports ``--params-file <path>`` and ``-p key:=value``. CLI ``-p``
    overrides take precedence over ``--params-file`` values (same as rclcpp).
    """
    out: dict = {}
    cli_overrides: dict = {}
    it = iter(argv)
    for a in it:
        if a == "--ros-args":
            continue
        if a == "-p":
            kv = next(it, "")
            if ":=" in kv:
                k, v = kv.split(":=", 1)
                cli_overrides[k.strip()] = v.strip()
        elif a.startswith("-p="):
            kv = a[3:]
            if ":=" in kv:
                k, v = kv.split(":=", 1)
                cli_overrides[k.strip()] = v.strip()
        elif a == "--params-file":
            path = next(it, "")
            if path:
                _merge_params_file(out, path, prefix)
        elif a.startswith("--params-file="):
            path = a[len("--params-file=") :]
            if path:
                _merge_params_file(out, path, prefix)
    out.update(cli_overrides)
    return out


def _publish_tick(sim, plan: PublishPlan, sim_time: float, _core) -> float:
    """Publish clock / joints / bodies / odom.  Returns total ms.

    ``plan`` carries the v2 routing decision ``EngineSession`` made at
    bootstrap — which joint index lands on which topic, and which body-pose
    channel is live.  See ``engine/publish_plan.py``; a legacy caller that
    built the loop without a session passes ``PublishPlan.legacy()``.
    """
    t0 = time.monotonic()
    _core.publish_clock(sim_time)
    t1 = time.monotonic()

    # Backends may return either (pos, vel) — legacy — or
    # (pos, vel, effort). Effort is optional; when a backend supplies
    # it, ``sensor_msgs/JointState.effort`` gets populated (mujoco
    # backend reads it from the actuator's ``sensordata`` / torque
    # sensor). Legacy backends keep the two-tuple shape and effort
    # stays default-empty on the wire.
    js = sim.get_joint_states()
    if len(js) == 3:
        jpos, jvel, jeff = js
    else:
        jpos, jvel = js
        jeff = None
    # reset_epoch causal credential (see engine/newton/reset.py — the reset
    # owner object bumps this ONLY after a drained reset batch's full
    # application, including sync_state_into_solver). Stamped into THIS SAME
    # /joint_states message's header.frame_id so a wrapper watching for the
    # epoch to advance can never observe it alongside still-stale state (the
    # two are written together, under one lock, in the wrapper bridge's
    # callback). ``PhysicsEngine.reset_epoch`` defaults to 0, so a backend
    # that never resets publishes a stuck ``reset_epoch:0`` rather than an
    # empty frame_id — which is the truth (no reset has completed) and keeps
    # the wrapper-side parser on its epoch path instead of the seq-advance
    # fallback it uses when the prefix is missing entirely.
    #
    # NOTE — WORKAROUND: header.frame_id is a TF frame name field; encoding
    # a monotonically-bumped integer here as ``reset_epoch:<n>`` is a
    # stop-gap because /joint_states is the only signal we already share
    # with the wrapper on the hot path, and the credential must be observed
    # ATOMICALLY with the post-reset state (a sidecar topic would race it).
    # Replace once a dedicated channel exists — e.g. a /reset_envs service
    # ack or a /reset_epoch topic published under the same trylock. Full
    # rationale in src/realtime_ros_node.hpp above the C++ declaration.
    _js_frame_id = f"reset_epoch:{sim.reset_epoch}"

    # v2 protocol: slice full engine joint state into robot-only (goes on
    # ``/joint_states``, RSP fans out onto ``/tf``) and scene-only (goes on
    # ``/joint_states_scene`` for benchmark / logger consumers).
    # ``session.py`` computed the split at bootstrap from
    # ``manifest.robot["joint_names"]`` and handed it over as ``plan``; if
    # that manifest field was absent (legacy bake), robot_joint_indices spans
    # the whole set and scene_joint_indices is empty — same wire behaviour as
    # v1. A backend that never exposes its scene DoFs through
    # ``sim.joint_names`` gets ``scene_joint_direct`` instead and is read
    # through ``get_scene_joint_states`` below.
    if plan.planned:
        import numpy as _np

        robot_idx = plan.robot_joint_indices
        scene_idx = plan.scene_joint_indices
        rpos = _np.take(jpos, robot_idx) if robot_idx else _np.empty(0)
        rvel = _np.take(jvel, robot_idx) if robot_idx else _np.empty(0)
        reff = _np.take(jeff, robot_idx) if (jeff is not None and robot_idx) else None
        _core.publish_joint_states(sim_time, rpos, rvel, reff, frame_id=_js_frame_id)
        # Two ways to reach the scene DoFs, chosen at bootstrap by
        # ``session.py`` (see the comment there for why a backend would keep
        # them out of ``sim.joint_names`` entirely):
        #   direct → the backend reports them on its own channel, because
        #            they are NOT in ``jpos`` and no index could reach them.
        #   sliced → they ride ``sim.joint_names``; take the scene indices.
        if plan.scene_joint_direct:
            _scene = sim.get_scene_joint_states()
            if _scene is not None:
                spos, svel, seff = _scene
                _core.publish_joint_states_scene(sim_time, plan.scene_joint_names, spos, svel, seff)
        elif scene_idx:
            spos = _np.take(jpos, scene_idx)
            svel = _np.take(jvel, scene_idx)
            seff = _np.take(jeff, scene_idx) if jeff is not None else None
            _core.publish_joint_states_scene(sim_time, plan.scene_joint_names, spos, svel, seff)
    else:
        # Legacy path — a caller wired EngineRunLoop without going through
        # EngineSession; publish everything on /joint_states.
        _core.publish_joint_states(sim_time, jpos, jvel, jeff, frame_id=_js_frame_id)
    t2 = time.monotonic()

    body_xyzwxyz, _frames = sim.get_body_transforms()
    if body_xyzwxyz.shape[0] > 0:
        # Two body-pose channels, mutually exclusive, chosen at bootstrap by
        # ``EngineNodeParams.use_tf_render`` (session.py puts the flag on the
        # plan; init_ros built the matching publisher). Calling the wrong one
        # is a silent no-op on the C++ side — the unselected publisher is
        # null — so this branch is what actually keeps poses on the wire.
        #   off → /tf_scene,  scene bodies only (robot rides /tf via RSP)
        #   on  → /tf_render, every body incl. robot links (no RSP deployed)
        if plan.use_tf_render:
            _core.publish_tf_render(sim_time, body_xyzwxyz)
        else:
            _core.publish_body_tf_scene(sim_time, body_xyzwxyz)
    t3 = time.monotonic()

    base = sim.get_odom(sim_time)
    if base is not None:
        base_pose, base_twist = base
        _core.publish_odom(sim_time, base_pose, base_twist)

    # /contact_states — optional; only backends that implement it
    # produce output. Empty per-point arrays are still a meaningful
    # signal ("no contacts this frame") so they get published; a
    # ``None`` return means the backend does not support contact
    # readback and we leave the topic quiet.
    cs = sim.get_contact_states()
    if cs is not None:
        c_names1, c_names2, c_pos, c_nrm, c_dep, c_wr = cs
        _core.publish_contact_states(sim_time, c_names1, c_names2, c_pos, c_nrm, c_dep, c_wr)

    publish_ms = (time.monotonic() - t0) * 1000.0
    sim.note_publish_phase(
        clock_ms=(t1 - t0) * 1000.0,
        joints_ms=(t2 - t1) * 1000.0,
        bodies_ms=(t3 - t2) * 1000.0,
        odom_ms=(time.monotonic() - t3) * 1000.0,
    )
    return publish_ms


class _Acc:
    """Lightweight mean+max accumulator for Python-side perf supplements."""

    __slots__ = ("_sum", "_count", "mx")

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._sum = 0.0
        self._count = 0
        self.mx = 0.0

    def push(self, v: float) -> None:
        self._sum += v
        self._count += 1
        if v > self.mx:
            self.mx = v

    @property
    def mean(self) -> float:
        return self._sum / self._count if self._count else 0.0


class EngineRunLoop:
    """Shared physics-step scaffold.

    ``render_hook(now, next_tick, sim_time) -> (render_ms, render_sync_ms, did_render)``
        Called each tick after ``tick_extras()`` and any ``post_step_hooks``.
        Return ``(0.0, 0.0, False)`` when no render fires this tick.
        ``render_sync_ms`` is the total including any post-render GPU sync
        (equals ``render_ms`` when no explicit sync is performed).

    ``exit_check() -> bool``
        If provided, called once per tick BEFORE the timing wait.  Return
        ``False`` to break the loop (e.g. GL window closed).

    ``post_step_hooks``
        Optional list of ``Callable[[float], None]`` called after
        ``tick_extras()`` each tick, receiving ``sim_time``.  Timed together
        with ``tick_extras`` under ``t_phase_extras``.  Append to extend
        step behaviour without subclassing (IsaacLab EventManager pattern).

    Performance stats
    -----------------
    The C++ ring (``note_step_timing`` → ``log_stats_if_due``) reports
    mean/min/max/std/p95/p99 for: interval, step, physx, render (sync-
    inclusive), publish, spin, plus overrun count and scheduler counters.

    The Python supplement appended to each stats message adds dimensions
    the C++ ring doesn't capture:

    * ``Cmd``      — ``pop_commands + apply_commands`` latency.
    * ``Extras``   — ``tick_extras + post_step_hooks`` latency.
    * ``GPU-sync`` — ``render_sync_ms − render_ms`` on render ticks: time
                     the CPU waited for the GPU to finish the render frame
                     after dispatch (zero when GENIESIM_RENDER_SYNC is off).
    * ``Budget``   — avg and worst-case ``(step+publish) / dt`` utilisation.
    """

    def __init__(
        self,
        core,
        sim,
        dt: float,
        logger,
        post_step_hooks=None,
        rtf: float = 1.0,
        dds_hz: float = 0.0,
        step_mode: str = DEFAULT_STEP_MODE,
        step_idle_timeout: float = 0.0,
        step_lockstep_ticks: int = 0,
        publish_plan: PublishPlan | None = None,
    ):
        self._core = core
        self._sim = sim
        self._dt = dt
        # The v2 routing decision, computed once by ``EngineSession``. ``None``
        # means nobody planned this run (a caller wired the loop directly):
        # fall back to the legacy plan, which publishes the backend's whole
        # joint vector on ``/joint_states`` and keeps the scene channel quiet.
        self._publish_plan = publish_plan if publish_plan is not None else PublishPlan.legacy()
        self._wall_tick = dt / max(rtf, 1e-6)
        self._logger = logger
        self._post_step_hooks = list(post_step_hooks) if post_step_hooks else []
        # Call log_stats_if_due at most once per simulated second to avoid
        # paying a pybind round-trip every tick for a check that returns "" ~99%
        # of the time.  The C++ side still owns the actual log interval.
        self._stats_interval_ticks = max(1, round(1.0 / dt))
        # DDS/ROS publish cadence throttling.  ``dds_hz == 0`` is the legacy
        # sentinel — publish every tick, matching pre-existing behaviour for
        # every backend that hasn't opted in.  When set, gate ``_publish_tick``
        # to fire every ``physics_hz / dds_hz`` ticks so /joint_states etc.
        # land at ``dds_hz`` regardless of physics_hz.  Keeps the physics
        # loop inside its budget at physics_hz=1000 without dropping the
        # simulator's tick rate.
        self._pub_period_ticks = 1
        if dds_hz > 0.0 and dt > 0.0:
            self._pub_period_ticks = max(1, int(round((1.0 / dt) / dds_hz)))

        # Pacing policy — WHEN a tick may run and WHETHER it publishes.
        # ``spin()`` owns the tick body; ``common/pacing.py`` owns the
        # policy, so a new mode is a class plus a registry line and never
        # a new branch down there.  Built in ``spin()`` rather than here
        # because a pacer captures the clock at construction, and a loop
        # may be built well before it is spun.
        self._step_mode = str(step_mode or DEFAULT_STEP_MODE).strip().lower()
        self._step_idle_timeout = max(0.0, float(step_idle_timeout))
        self._step_lockstep_ticks = max(0, int(step_lockstep_ticks))

    def spin(self, render_hook=None, exit_check=None) -> None:
        core = self._core
        sim = self._sim
        plan = self._publish_plan
        dt = self._dt
        wall_tick = self._wall_tick
        logger = self._logger
        post_step_hooks = self._post_step_hooks
        stats_interval = self._stats_interval_ticks
        pub_period = self._pub_period_ticks
        target_ms = wall_tick * 1000.0

        # The pacing policy for this run.  Built here, not in __init__,
        # because a pacer captures the wall clock at construction and a
        # loop may be built long before it is spun.  An unknown mode
        # raises out of ``make_pacer`` — see the rationale there.
        pacer = make_pacer(
            self._step_mode,
            sim,
            wall_tick,
            pub_period,
            logger,
            # Pass the clock explicitly rather than letting the pacer bind
            # its own default: the loop's ``now`` and the pacer's internal
            # deadline must come from the SAME source, or the two drift
            # under any harness that scripts the clock.
            clock=time.monotonic,
            sleep=time.sleep,
            idle_timeout=self._step_idle_timeout,
            lockstep_ticks=self._step_lockstep_ticks,
        )
        logger.info(f"[loop] step_mode={self._step_mode}: {pacer.describe()}")

        sim_time = 0.0
        behind_count = 0
        stats_tick = 0

        # spin_ms = idle time from end of previous tick to start of this tick.
        # Seeded to now so the first tick reports 0 spin (no previous tick).
        t_prev_tick_end = time.monotonic()

        # /rtf is measured and published every physics step. The value is
        # the instantaneous ratio dt_sim / wall_delta — the wall time
        # actually elapsed for this step's wall_tick. No smoothing: the
        # topic reflects each step's true factor.
        _rtf_value = 1.0
        _rtf_prev_wall = time.monotonic()

        # Python-side rolling accumulators — reset when C++ stats fire.
        _acc_cmd = _Acc()  # pop_commands + apply_commands
        _acc_extras = _Acc()  # tick_extras + post_step_hooks
        _acc_gpu_sync = _Acc()  # render_sync_ms − render_ms (render ticks only)
        _acc_budget = _Acc()  # step_ms + publish_ms (budget consumption per tick)

        # Lifecycle services are owned by the shared C++ ROS bridge.  The
        # bridge only queues verbs; this loop is the serialization point that
        # may safely call backend lifecycle methods around a solver step.
        paused = False
        if hasattr(core, "set_lifecycle_state"):
            core.set_lifecycle_state(3, "active")

        def process_lifecycle_commands() -> None:
            nonlocal paused, sim_time
            poll = getattr(core, "poll_lifecycle_commands", None)
            if poll is None:
                return
            for command in poll():
                if command == "configure":
                    paused = True
                    if hasattr(core, "set_lifecycle_state"):
                        core.set_lifecycle_state(2, "inactive")
                elif command == "pause":
                    # The backend gets told too, not just the loop flag: a
                    # backend with machinery running BESIDE step() (parallel
                    # render thread, async state mirror) cannot be stopped by
                    # a flag the loop keeps to itself. Declared on
                    # PhysicsEngine, default no-op — see engine/lifecycle.py.
                    sim.pause()
                    paused = True
                    if hasattr(core, "set_lifecycle_state"):
                        core.set_lifecycle_state(2, "inactive")
                elif command == "resume":
                    sim.resume()
                    paused = False
                    if hasattr(core, "set_lifecycle_state"):
                        core.set_lifecycle_state(3, "active")
                    # The pause happened in the ``if paused: continue`` branch
                    # below, which never reaches ``before_tick``, so the pacer
                    # could not observe that time passing.  Tell it, or it owes
                    # one catch-up tick per millisecond spent paused.
                    pacer.notify_resumed(time.monotonic())
                elif command == "reset":
                    # Called with NO arguments, deliberately.  Every default
                    # is the lifecycle meaning already — all worlds, no seeds,
                    # no instance placements, because a lifecycle reset is not
                    # a task reset — so spelling them out buys nothing and
                    # costs compatibility: a backend carrying the upstream
                    # ``episode_reset(env_ids, seed)`` arity (a consuming
                    # tree's contract, which at least one backend implements
                    # while subclassing THIS PhysicsEngine) takes two
                    # positional args, and a third is a TypeError.  This
                    # try block has no ``except`` — only a ``finally`` — so
                    # that would not degrade, it would take the server down on
                    # an operator's ~/reset.  An empty call binds on both
                    # arities and means the same thing under each.
                    sim.episode_reset()
                    sim_time = 0.0
                    pacer.notify_resumed(time.monotonic())
                elif command == "shutdown":
                    if hasattr(core, "set_lifecycle_state"):
                        core.set_lifecycle_state(4, "finalized")
                    # ``shutdown`` is @abstractmethod on PhysicsEngine
                    # (engine/base.py) — every backend has it, so there is
                    # nothing to probe for.
                    sim.shutdown()
                    core.shutdown()
                    # Stop draining the batch.  ``pop_lifecycle_commands``
                    # returns the WHOLE queue in one call, so a single poll can
                    # hand back ``["shutdown", "pause"]`` -- a lifecycle manager
                    # sending SHUTDOWN and DEACTIVATE back to back, or a scripted
                    # pair of service calls.  Without this return the loop
                    # dispatches those later verbs at a backend it has just torn
                    # down: ``NewtonROSNode.shutdown`` drops ``self._engine``
                    # (nodes/ros_node.py) and its ``pause``/``resume``/
                    # ``episode_reset`` all dereference it, so the next verb is an
                    # AttributeError on None -- and this try block has only a
                    # ``finally``, so it escapes ``spin()`` instead of degrading.
                    # A verb arriving after shutdown has nothing left to act on
                    # anyway; dropping the rest of the batch IS the correct
                    # semantics, not just crash avoidance.
                    return

        def drain_teleports() -> None:
            """Hand queued ``/genie_sim/teleport`` requests to the backend.

            Called at the top of every tick BEFORE ``pop_commands``, and again
            from the paused branch below.  Both placements are contracts, not
            conveniences:

            * BEFORE commands, because ``pop_commands`` LATCHES -- it returns a
              snapshot and never clears, so a hold-position command sent before
              the teleport is re-applied on every later tick.  Draining after it
              would let that stale target drag the body straight back off the
              pose that was just requested.  The reference states the invariant
              directly: a teleport followed by a hold-position command should
              land on the teleported pose.
            * WHILE PAUSED, because repositioning the world with the sim held
              still is the main reason to teleport at all.  The reference drains
              its queue under the sim mutex ahead of the ``run`` check for
              exactly this.

            The bridge queue deliberately does NOT latch: each request is
            delivered once.  Re-delivering the last one every tick would pin the
            body in place instead of releasing it to physics.

            Probed rather than called outright because ``core`` is the C++
            bridge, not a ``PhysicsEngine`` -- a bridge built before this topic
            existed degrades to "no teleports" instead of raising.
            """
            pop = getattr(core, "pop_teleports", None)
            if pop is None:
                return
            raw = pop()
            if not raw:
                return
            reqs = [
                TeleportRequest(
                    body=str(body),
                    position=(float(x), float(y), float(z)),
                    orientation_wxyz=(float(qw), float(qx), float(qy), float(qz)),
                )
                for body, x, y, z, qw, qx, qy, qz in raw
            ]
            applied = int(sim.teleport(reqs))
            # Report the shortfall, not just the success: ``teleport`` returns a
            # COUNT precisely so a backend that ignored a body (no free joint,
            # unknown name) cannot look like a batch that landed.
            if applied == len(reqs):
                logger.info(f"[teleport] applied {applied} pose(s): {[r.body for r in reqs]}")
            else:
                logger.warn(
                    f"[teleport] applied {applied} of {len(reqs)} pose(s); "
                    f"requested: {[r.body for r in reqs]} -- the rest have no "
                    f"FREE joint under that name in this scene"
                )

        try:
            while core.ok():
                process_lifecycle_commands()
                if not core.ok():
                    break
                if paused:
                    # A teleport is legal -- and most useful -- with the sim
                    # held.  Applying it here means the backend's own state
                    # write (and its mirror flush) runs while nothing steps, so
                    # the next publish carries the teleported pose at a standstill.
                    drain_teleports()
                    time.sleep(0.001)
                    # A lifecycle pause is unbounded operator time, not idle
                    # time between ticks, so it must not be charged to
                    # ``spin_ms``.  Without this, a 5 min pause injects a
                    # 300000 ms sample into the C++ stats ring and destroys
                    # max/p95/p99 for the rest of the run -- the same rule
                    # ``Tick.PAUSED.resets_spin_baseline`` encodes for the
                    # idle gate.  Re-stamped every poll, so the baseline is
                    # <=1 ms old when the resume verb lands.
                    t_prev_tick_end = time.monotonic()
                    continue
                if exit_check is not None and not exit_check():
                    break

                now = time.monotonic()
                # The one place the pacing policy is consulted.  Everything
                # below is the tick BODY and is mode-independent.
                decision = pacer.before_tick(now)
                if not decision.runs_step:
                    # ``resets_spin_baseline`` is a property of the decision,
                    # not of the pacer, so a new mode inherits the correct
                    # accounting by picking the right Tick member.  A bounded
                    # sub-tick wall wait keeps accumulating into spin_ms (it
                    # IS tick-budget idle time); an unbounded wait on a peer
                    # must not, or one 300 s sample destroys max/p95/p99 for
                    # the rest of the run.
                    if decision.publishes_without_stepping:
                        # State changed underneath us with no step to carry it
                        # out (an out-of-band reset).  Give the backend its
                        # apply hook, then publish immediately: the RL wrapper
                        # is blocked reading post-reset joints, and the command
                        # that would otherwise flush them is what it cannot
                        # send until it has read them.
                        sim.apply_commands(
                            cmd_positions={},
                            cmd_4ws_steer_pos={},
                            cmd_4ws_drive_vel={},
                            cmd_4ws_stamp=0.0,
                        )
                        _publish_tick(sim, plan, sim_time, core)
                    if decision.resets_spin_baseline:
                        t_prev_tick_end = time.monotonic()
                    continue
                next_tick = pacer.next_deadline(now)
                # Idle time since last tick's work finished.
                spin_ms = (now - t_prev_tick_end) * 1000.0

                # --- teleports, then commands ---
                # Order is load-bearing; see ``drain_teleports``.
                drain_teleports()

                _t_cmd = time.monotonic()
                pos_dict, _eff_dict, steer_dict, drive_dict, cmd_4ws_stamp = core.pop_commands()
                # Activity = command ARRIVAL time (rclpy stamp set in the reset-
                # listener node), NOT dict non-emptiness: pop_commands LATCHES
                # (snapshot, never clears), so the dicts stay non-empty forever
                # after the first message. Gate only when the activity sub is
                # actually up (else fall back to free-run — never risk pausing
                # a live client we can't hear).
                # Lockstep engage: switch from startup free-run to command-driven
                # stepping at the FIRST command arrival. This burst consumes it.
                sim.apply_commands(
                    cmd_positions=pos_dict,
                    cmd_4ws_steer_pos=steer_dict,
                    cmd_4ws_drive_vel=drive_dict,
                    cmd_4ws_stamp=cmd_4ws_stamp,
                )
                cmd_ms = (time.monotonic() - _t_cmd) * 1000.0

                step_start = time.monotonic()
                # Time spent inside ``sim.step()`` — the physics solver call,
                # whichever backend is active (PhysX, Newton-in-Isaac, mjwarp,
                # Featherstone, ...).  Logged as the "Solver" line in the
                # 1 Hz physics-stats block.
                solver_ms = 0.0
                render_ms = 0.0
                render_sync_ms = 0.0
                did_render = False
                t_phase_step = 0.0
                t_phase_extras = 0.0
                ok = False

                try:
                    _t = time.monotonic()
                    solver_ms = sim.step(dt, step_start)
                    t_phase_step = (time.monotonic() - _t) * 1000.0

                    _t = time.monotonic()
                    sim.tick_extras()
                    for hook in post_step_hooks:
                        hook(sim_time)
                    t_phase_extras = (time.monotonic() - _t) * 1000.0

                    if render_hook is not None:
                        render_ms, render_sync_ms, did_render = render_hook(now, next_tick, sim_time)

                    sim.note_render(render_ms, did_render)
                    sim.note_phase_timing(
                        step_ms=t_phase_step,
                        extras_ms=t_phase_extras,
                        render_ms=render_ms,
                        render_sync_ms=render_sync_ms,
                        did_render=did_render,
                    )
                    ok = True
                except Exception as exc:
                    logger.warn(f"step failed: {exc}")

                step_ms = (time.monotonic() - step_start) * 1000.0

                if not ok:
                    # The tick's clock still advances — the wall moved — but
                    # the tick did not happen, so it must consume neither a
                    # publish slot nor a lockstep burst credit.  ``after_step``
                    # owns both of those, hence calling it with ok=False rather
                    # than skipping it.
                    pacer.after_step(False)
                    t_prev_tick_end = time.monotonic()
                    continue

                sim_time += dt
                if pacer.after_step(True):
                    publish_ms = _publish_tick(sim, plan, sim_time, core)
                else:
                    publish_ms = 0.0

                # /rtf — instantaneous dt_sim / wall_delta for this physics
                # step, published every step.
                _wall_delta = now - _rtf_prev_wall
                if _wall_delta > 1e-6:
                    _rtf_value = dt / _wall_delta
                _rtf_prev_wall = now
                core.publish_rtf(float(_rtf_value))

                # Mark end of all active work; next spin_ms starts here.
                t_prev_tick_end = time.monotonic()

                core.note_step_timing(step_ms, solver_ms, render_sync_ms, publish_ms, spin_ms, did_render, now)

                # Python-side accumulation.
                _acc_cmd.push(cmd_ms)
                _acc_extras.push(t_phase_extras)
                _acc_budget.push(step_ms + publish_ms)
                if did_render:
                    _acc_gpu_sync.push(render_sync_ms - render_ms)

                stats_tick += 1
                if stats_tick >= stats_interval:
                    stats_tick = 0
                    msg = core.log_stats_if_due(now)
                    if msg:
                        n_render = _acc_gpu_sync._count
                        py_lines = (
                            f"  Cmd:      mean={_acc_cmd.mean:.2f}  max={_acc_cmd.mx:.2f} ms"
                            f"  (pop_commands + apply_commands)\n"
                            f"  Extras:   mean={_acc_extras.mean:.2f}  max={_acc_extras.mx:.2f} ms"
                            f"  (tick_extras + hooks)\n"
                            f"  GPU-sync: mean={_acc_gpu_sync.mean:.2f}  max={_acc_gpu_sync.mx:.2f} ms"
                            f"  ({n_render} render ticks, sync_ms − dispatch_ms)\n"
                            f"  Budget:   mean={_acc_budget.mean:.2f}ms"
                            f" ({_acc_budget.mean / target_ms * 100:.0f}%)"
                            f"  worst={_acc_budget.mx:.2f}ms"
                            f" ({_acc_budget.mx / target_ms * 100:.0f}%)"
                            f"  of {target_ms:.1f}ms target"
                        )
                        logger.info(msg + "\n" + py_lines)
                        _acc_cmd.reset()
                        _acc_extras.reset()
                        _acc_gpu_sync.reset()
                        _acc_budget.reset()

                # ``after_step`` already advanced the pacer's clock; this
                # only asks whether it had to be clamped forward.  Unpaced
                # modes have no clock to fall behind and answer False.
                if pacer.note_overrun(now):
                    behind_count += 1
                    if behind_count % 20 == 1:
                        logger.warn(
                            f"loop fell behind by >1 tick: clamped {behind_count} times "
                            f"(step={step_ms:.1f} ms, target={target_ms:.1f} ms)"
                        )

        finally:
            try:
                sim.shutdown()
            except Exception:
                pass
            try:
                core.shutdown()
            except Exception:
                pass

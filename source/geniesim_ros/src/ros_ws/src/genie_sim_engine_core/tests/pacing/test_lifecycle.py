# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Lifecycle resume must not leave the pacer owing catch-up ticks.

``EngineRunLoop.spin.process_lifecycle_commands`` handles ``pause`` /
``resume`` / ``reset``.  It parks the loop in the
``if paused: sleep(0.001); continue`` branch, which is reached BEFORE
``before_tick`` — so a pacer never sees that time pass.  The pre-refactor
loop compensated by writing ``next_tick = time.monotonic()`` under
``nonlocal``; now that the clock is pacer-private, the lifecycle path calls
``notify_resumed(now)`` instead.

These tests pin what breaks without it, which is not one bug but two:

  * the wall clock owes ~pause/dt ticks and bursts to catch up;
  * the idle-gate baseline is stale by the pause duration, so the gate
    trips on the first tick after resume — the operator presses resume and
    the GPU stays quiet.
"""

from __future__ import annotations

from conftest import DT, PAUSE_S, FakeClock, FakeLog, FakeSim, LoopCore, LoopSim, run_loop

from genie_sim_engine_core.common.pacing import (
    STEP_MODES,
    LockstepPacer,
    RealtimePacer,
    Tick,
    make_pacer,
)


def count_catchup_ticks(pacer, clock, limit=200):
    """How many ticks fire back-to-back without the clock advancing?"""
    n = 0
    while n < limit and pacer.before_tick(clock()).runs_step:
        pacer.after_step(True)
        n += 1
    return n


def test_without_notify_a_pause_owes_a_runaway_burst(clock, sim):
    """Negative control: this is the bug ``notify_resumed`` exists to fix.

    A 30 s pause at 100 Hz owes ~3000 ticks.  Bounded at 200 by the helper;
    hitting the cap is the runaway.
    """
    p = RealtimePacer(sim, DT, 1, 0.0, None, clock=clock, sleep=clock.sleep)
    clock.advance(PAUSE_S)  # pacer never told
    assert count_catchup_ticks(p, clock) >= 200


def test_notify_resumed_clears_the_debt(clock, sim):
    p = RealtimePacer(sim, DT, 1, 0.0, None, clock=clock, sleep=clock.sleep)
    clock.advance(PAUSE_S)
    p.notify_resumed(clock())
    assert count_catchup_ticks(p, clock) == 1  # exactly one tick is due


def test_stale_idle_baseline_would_pause_instantly(clock, sim, log):
    """The second-order bug: without the re-baseline, resume looks like a hang."""
    sim.activity.idle_activity_ok = True
    sim.activity.last_cmd_wall = clock()  # a command did arrive
    p = RealtimePacer(sim, DT, 1, 1.0, log, clock=clock, sleep=clock.sleep)
    clock.advance(PAUSE_S)  # 30 s pause vs a 1 s timeout
    assert p.before_tick(clock()) is Tick.PAUSED


def test_notify_resumed_rebaselines_the_idle_gate(clock, sim, log):
    sim.activity.idle_activity_ok = True
    sim.activity.last_cmd_wall = clock()
    p = RealtimePacer(sim, DT, 1, 1.0, log, clock=clock, sleep=clock.sleep)
    clock.advance(PAUSE_S)
    p.notify_resumed(clock())
    assert p.before_tick(clock()) is Tick.RUN


def test_the_gate_still_arms_after_a_resume(clock, sim, log):
    """Re-baselining must not disable the gate — only move it forward."""
    sim.activity.idle_activity_ok = True
    sim.activity.last_cmd_wall = clock()
    p = RealtimePacer(sim, DT, 1, 1.0, log, clock=clock, sleep=clock.sleep)
    clock.advance(PAUSE_S)
    p.notify_resumed(clock())
    assert p.before_tick(clock()) is Tick.RUN
    clock.advance(5.0)  # quiet again, past the timeout
    assert p.before_tick(clock()) is Tick.PAUSED


def test_resume_is_logged(clock, sim, log):
    sim.activity.idle_activity_ok = True
    sim.activity.last_cmd_wall = clock()
    p = RealtimePacer(sim, DT, 1, 1.0, log, clock=clock, sleep=clock.sleep)
    clock.advance(PAUSE_S)
    p.before_tick(clock())  # trips the gate, logs "paused"
    p.notify_resumed(clock())
    assert any("resumed" in line for line in log.lines)


def test_lockstep_warmup_debt_is_cleared(clock, sim):
    """Lockstep delegates its startup grace, so it must delegate the resume."""
    warm = RealtimePacer(sim, DT, 1, 0.0, None, clock=clock, sleep=clock.sleep)
    lp = LockstepPacer(sim, 4, warm, None, sleep=clock.sleep)
    clock.advance(PAUSE_S)
    lp.notify_resumed(clock())
    assert count_catchup_ticks(lp, clock) == 1


def test_mid_burst_pause_preserves_the_k_tick_step(clock, sim):
    """A command bought K ticks of sim time and still owes them on resume.

    Clearing ``_burst_left`` here would silently shorten one episode step,
    which is invisible in the logs and shows up only as a training curve
    that will not reproduce.
    """
    sim.activity.idle_activity_ok = True
    sim.activity.last_cmd_wall = clock()
    warm = RealtimePacer(sim, DT, 1, 0.0, None, clock=clock, sleep=clock.sleep)
    lp = LockstepPacer(sim, 4, warm, None, sleep=clock.sleep)
    lp.before_tick(clock())  # engages, burst = 4
    lp.after_step(True)  # consume 1 -> 3 left
    clock.advance(PAUSE_S)
    lp.notify_resumed(clock())
    remaining = 0
    while lp.before_tick(clock()).runs_step and remaining < 10:
        remaining += 1
        if lp.after_step(True):
            break  # published => burst ended
    assert remaining == 3


def test_resume_does_not_replay_the_consumed_command(clock, sim):
    """``_last_consumed`` survives the pause, so no phantom extra burst."""
    sim.activity.idle_activity_ok = True
    sim.activity.last_cmd_wall = clock()
    warm = RealtimePacer(sim, DT, 1, 0.0, None, clock=clock, sleep=clock.sleep)
    lp = LockstepPacer(sim, 1, warm, None, sleep=clock.sleep)
    lp.before_tick(clock())
    lp.after_step(True)  # burst done
    clock.advance(PAUSE_S)
    lp.notify_resumed(clock())
    assert lp.before_tick(clock()) is Tick.PAUSED  # same stamp — still waiting


def test_every_mode_accepts_notify_resumed(sim):
    """freerun's inherited no-op is correct (no clock, no baseline) but the
    loop calls this unconditionally, so it must still be callable."""
    for mode in STEP_MODES:
        make_pacer(mode, sim, DT, 1, None).notify_resumed(1000.0)


# ----------------------------------------------------------------------
# The verbs must reach the BACKEND, not just the loop's paused flag
# ----------------------------------------------------------------------
#
# ``process_lifecycle_commands`` used to look each verb up with
# ``getattr(sim, "pause", None)`` and skip it when the probe came back
# empty — which it always did, because nothing in this tree implemented
# ``pause`` or ``resume``.  The loop parked itself and the backend was
# never told, so a backend with machinery running beside ``step()`` kept
# running through the pause.  ``PhysicsEngine`` now declares all three, so
# the loop calls them straight and these pin that it does.


def _run_verbs(*verbs):
    """Feed one verb per poll to the real ``spin()``, then return the stub."""
    sim = LoopSim()
    schedule = {i + 1: [v] for i, v in enumerate(verbs)}
    core = LoopCore(max_iters=len(verbs) + 40, lifecycle=schedule)
    run_loop(core, sim, mode="lockstep")
    return sim


def test_pause_and_resume_reach_the_backend():
    sim = _run_verbs("pause", "resume")
    assert sim.pauses == 1, "the backend was never told to pause"
    assert sim.resumes == 1, "the backend was never told to resume"


def test_reset_calls_episode_reset_for_every_world():
    """A lifecycle reset is not a task reset: no env selection, no seeds.

    ``None`` env_ids means every world; passing seeds or instance ids here
    would silently apply domain randomisation to an operator's
    ``~/reset``.  The stub's own defaults are what record as ``None`` — the
    loop passes nothing, which is the point of the next test.
    """
    sim = _run_verbs("reset")
    assert sim.resets == [(None, None, None)]


def test_reset_survives_a_backend_on_the_upstream_two_arg_arity():
    """``~/reset`` must not take the server down on a two-arg backend.

    A downstream backend subclasses this tree's ``PhysicsEngine`` while
    implementing a consuming tree's ``episode_reset(env_ids, seed)``.
    ``spin()``'s main ``try`` has only a ``finally``, no ``except``, so a
    TypeError raised here does not degrade the reset — it unwinds out of
    ``spin``, runs the shutdown teardown, and ends the process on an
    operator's keystroke.  The loop therefore calls ``episode_reset()`` with
    no arguments, which binds under both arities and means the same thing
    under each.

    This stub deliberately does NOT accept the third parameter: widening the
    loop's call back to ``episode_reset(None, None, None)`` fails here rather
    than in a downstream integration.
    """

    class UpstreamAritySim(LoopSim):
        def __init__(self):
            super().__init__()
            self.upstream_resets = 0

        def episode_reset(self, env_ids=None, seed=None):  # two, not three
            self.upstream_resets += 1

    sim = UpstreamAritySim()
    core = LoopCore(max_iters=8, lifecycle={1: ["reset"]})
    run_loop(core, sim, mode="lockstep")  # a TypeError would escape this call
    assert sim.upstream_resets == 1, "the backend was never told to reset"
    assert sim.steps > 0, "the loop died instead of carrying on past the reset"


def test_shutdown_reaches_the_backend():
    """Twice, and that is correct: once for the verb, once in ``spin``'s
    ``finally``.  Backend ``shutdown`` has always had to be idempotent —
    the loop's teardown runs on every exit path, verb or not."""
    sim = _run_verbs("shutdown")
    assert sim.shutdowns >= 1


def test_the_verb_batch_stops_at_shutdown():
    """``pop_lifecycle_commands`` hands back the WHOLE queue in one call.

    So one poll can return ``["shutdown", "pause"]`` — a lifecycle manager
    sending SHUTDOWN then DEACTIVATE, or a scripted pair of service calls — and
    the drain used to keep going, dispatching verbs at a backend it had just
    torn down.  A verb arriving after shutdown has nothing left to act on, so
    dropping the rest of the batch is the semantics, not only crash avoidance.
    """
    sim = LoopSim()
    core = LoopCore(max_iters=8, lifecycle={1: ["shutdown", "pause"]})
    run_loop(core, sim, mode="lockstep")
    assert sim.shutdowns >= 1
    assert sim.pauses == 0, "a verb was dispatched at an already-shut-down backend"


def test_a_post_shutdown_verb_does_not_escape_spin():
    """The concrete cost on a real backend, pinned.

    A downstream ROS node's ``shutdown`` drops its engine handle, and its
    ``pause`` / ``resume`` / ``episode_reset`` all dereference it.  A verb dispatched after shutdown is therefore an
    ``AttributeError`` on ``None`` inside a ``try`` that has only a
    ``finally`` — it unwinds out of ``spin()`` and turns a clean teardown into
    a traceback and a nonzero exit, skipping whatever the caller runs after
    ``spin``.  Same failure mode as the ``~/reset`` arity crash, one verb over.
    """

    class NewtonNextShape(LoopSim):
        def __init__(self):
            super().__init__()
            self._engine = object()

        def shutdown(self):
            super().shutdown()
            self._engine = None

        def pause(self):
            self._engine.pause()  # AttributeError once shutdown nulled it
            super().pause()

    sim = NewtonNextShape()
    core = LoopCore(max_iters=8, lifecycle={1: ["shutdown", "pause"]})
    run_loop(core, sim, mode="lockstep")  # an AttributeError would escape this
    assert sim.pauses == 0


# ----------------------------------------------------------------------
# /genie_sim/teleport — WHEN the drain happens is the whole contract
# ----------------------------------------------------------------------
#
# The C++ bridge only queues; this loop is the serialization point that may
# safely touch backend state around a solver step.  Two orderings are
# contracts inherited from the reference implementation, and both are
# invisible in the code unless something pins them:
#
#   * teleports drain BEFORE ``pop_commands``, because pop_commands LATCHES.
#     A hold-position command sent before the teleport is re-applied on every
#     later tick, so draining after it would let that stale target drag the
#     body straight back off the pose just requested.
#   * teleports drain WHILE PAUSED too — repositioning the world with the sim
#     held still is the main reason to teleport at all.

_TP = ("/World/Objects/cup_000/body", 0.3, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0)


def test_teleport_reaches_the_backend_ahead_of_commands():
    sim = LoopSim()
    core = LoopCore(max_iters=6, teleports=[_TP])
    run_loop(core, sim, mode="lockstep")
    assert len(sim.teleports) == 1
    # Not "both happened" — teleport happened FIRST, on the same tick.
    assert sim.order[:2] == ["teleport", "commands"], sim.order


def test_teleport_applies_while_paused():
    """Queued before the pause verb, applied inside the paused branch.

    ``sim.applied`` staying empty is the load-bearing half: the loop never
    reached ``pop_commands``, so the write demonstrably landed with the sim
    held rather than on the first tick after resume.
    """
    sim = LoopSim()
    core = LoopCore(max_iters=8, lifecycle={1: ["pause"]}, teleports=[_TP])
    run_loop(core, sim, mode="lockstep")
    assert sim.pauses == 1
    assert len(sim.teleports) == 1, "a paused loop swallowed the teleport"
    assert sim.applied == [], "a tick ran; this no longer tests the paused path"


def test_the_teleport_queue_does_not_latch():
    """One delivery per request, however many ticks follow.

    ``pop_commands`` latching is correct — a position target is a standing
    order.  A teleport is an event: re-applying it every tick would pin the
    body at the teleported pose and physics would never get it back.
    """
    sim = LoopSim()
    core = LoopCore(max_iters=12, teleports=[_TP])
    run_loop(core, sim, mode="lockstep")
    assert core.teleport_pops > 1, "the loop only drained once; not a real test"
    assert len(sim.teleports) == 1
    assert sim.order.count("teleport") == 1


def test_the_request_the_backend_gets_is_canonical_wxyz():
    """The wire is xyzw and Newton's free-joint q slots are xyzw; the Python
    layer in between is wxyz.  The C++ callback does the first reorder, so by
    the time a ``TeleportRequest`` exists the fields are already named."""
    sim = LoopSim()
    body, x, y, z, qw, qx, qy, qz = _TP
    core = LoopCore(max_iters=6, teleports=[(body, x, y, z, qw, qx, qy, qz)])
    run_loop(core, sim, mode="lockstep")
    req = sim.teleports[0]
    assert req.body == body
    assert req.position == (x, y, z)
    assert req.orientation_wxyz == (qw, qx, qy, qz)
    # Reference defaults, carried through untouched.
    assert req.zero_velocity is True
    assert req.env_ids is None


def test_a_bridge_without_the_topic_is_not_an_error():
    """An older C++ bridge has no ``pop_teleports``.  The loop probes for it
    because ``core`` is the bridge, not a ``PhysicsEngine`` — degrading to
    "no teleports" beats refusing to spin."""

    class OldCore(LoopCore):
        pop_teleports = None

    sim = LoopSim()
    run_loop(OldCore(max_iters=6), sim, mode="lockstep")
    assert sim.teleports == []
    assert sim.steps > 0

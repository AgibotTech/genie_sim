# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""``Tick.FLUSH`` — a reset landing between lockstep bursts must reach the client.

This is a DEADLOCK guard, not a latency optimisation.  The RL wrapper blocks
reading post-reset joints, and the next command — which would otherwise carry
the new state out on the following burst — is precisely what the wrapper
cannot send until it has read them.  A FLUSH that no loop path consumes hangs
the run.

Unlike the other files here, these tests drive the REAL
``EngineRunLoop.spin()`` through stub core/sim objects, so they pin the
wiring and not just the pacer's return value.
"""

from __future__ import annotations

from conftest import DT, LoopCore, LoopSim, run_loop

from genie_sim_engine_core.common.pacing import LockstepPacer, RealtimePacer, Tick

# ----------------------------------------------------------------------
# The flush path, through the real loop
# ----------------------------------------------------------------------


def _engaged(reset_after, ticks):
    """One command, one burst of ``ticks``, a reset landing on its last tick."""
    sim = LoopSim(reset_after=reset_after)
    sim.activity.idle_activity_ok = True
    sim.activity.last_cmd_wall = 1.0  # the one command
    return sim, LoopCore(max_iters=300)


def test_reset_between_bursts_is_flushed_by_the_real_loop():
    """The headline: a reset with no command behind it still reaches the wire."""
    sim, core = _engaged(reset_after=4, ticks=4)
    run_loop(core, sim, ticks=4)
    assert sim.steps == 4  # exactly the one burst
    assert sim.drained == 1  # the reset was drained...
    assert sim.published == 2  # ...by a flush, which published


def test_a_reset_already_pending_needs_no_flush():
    """Control for the above: a reset present before the burst is drained by
    the first tick's ordinary apply_commands, and no flush is involved."""
    sim, core = LoopSim(), LoopCore(max_iters=300)
    sim.activity.idle_activity_ok = True
    sim.activity.last_cmd_wall = 1.0
    sim.activity.reset_pending = True
    run_loop(core, sim, ticks=4)
    assert sim.drained == 1
    assert sim.published == 1  # the burst's publish, only


def test_flush_applies_empty_commands():
    """``pop_commands`` LATCHES, so re-applying its snapshot would drive the
    robot straight back off the pose the reset just established."""
    sim, core = _engaged(reset_after=2, ticks=2)
    run_loop(core, sim, ticks=2)
    # The burst's applies carry the latched dict; the flush's must be empty.
    assert sim.applied[-1][0] == {}
    assert sim.applied[-1][1] == 0.0  # stamp 0.0 -> chassis brake
    assert any(a[0] for a in sim.applied[:-1])  # and the burst DID carry one


def test_flush_does_not_pop_or_step():
    """FLUSH publishes state; it does not consume a command or advance time."""
    sim, core = _engaged(reset_after=3, ticks=3)
    run_loop(core, sim, ticks=3)
    assert sim.steps == 3  # the flush added no step
    assert core.pops == 3  # one pop per stepped tick only


def test_flush_does_not_charge_the_spin_baseline():
    """The wait before a flush is unbounded, so it must not enter the ring."""
    sim, core = _engaged(reset_after=2, ticks=2)
    run_loop(core, sim, ticks=2)
    # Every sample the C++ ring saw must be tick-scale, not wait-scale.
    assert core.step_timings
    assert max(core.step_timings) < 100.0


def test_ignoring_flush_would_deadlock():
    """Negative control — the shape without a FLUSH consumer never recovers.

    Reproduces the pre-fix behaviour: the pacer returns FLUSH, the loop
    treats it as an ordinary skip, so the reset is never drained and never
    published, and the pacer waits forever for a command the wrapper cannot
    send until it has read the reset state.
    """
    sim = LoopSim()
    sim.activity.idle_activity_ok = True
    sim.activity.last_cmd_wall = 1.0
    warm = RealtimePacer(sim, DT, 1, 0.0, None)
    pacer = LockstepPacer(sim, 1, warm, None)
    pacer.before_tick(2.0)  # engage
    pacer.after_step(True)  # burst done
    sim.activity.reset_pending = True
    for _ in range(500):
        decision = pacer.before_tick(3.0)
        if decision.runs_step:
            break  # would have escaped — it does not
    assert sim.drained == 0 and sim.published == 0


# ----------------------------------------------------------------------
# Flush is lockstep-only
# ----------------------------------------------------------------------


def test_realtime_never_flushes(clock, sim):
    """A pending reset on the realtime path is drained by the ordinary tick's
    ``apply_commands``; emitting FLUSH there would double-drain it."""
    p = RealtimePacer(sim, DT, 1, 0.0, None, clock=clock, sleep=clock.sleep)
    sim.activity.reset_pending = True
    seen = set()
    for _ in range(30):
        seen.add(p.before_tick(clock()))
        clock.advance(DT / 3)
    assert Tick.FLUSH not in seen


def test_realtime_drains_resets_through_the_normal_tick():
    """And the drain still happens — via the tick body, not a flush."""
    sim, core = LoopSim(), LoopCore(max_iters=6)
    sim.activity.reset_pending = True
    run_loop(core, sim, mode="realtime")
    assert sim.drained == 1
    assert sim.steps >= 1


def test_flush_semantics_live_on_the_enum():
    assert Tick.FLUSH.resets_spin_baseline
    assert not Tick.FLUSH.runs_step
    assert Tick.FLUSH.publishes_without_stepping
    assert not any(t.publishes_without_stepping for t in (Tick.RUN, Tick.IDLE, Tick.PAUSED))


# ----------------------------------------------------------------------
# The loop body stayed mode-independent
# ----------------------------------------------------------------------


def test_realtime_loop_still_steps_and_publishes():
    """The default path is unchanged behaviour, driven end to end.

    Budget generously: realtime still gates on the REAL wall clock, and at
    ``rtf=1000`` a tick is due every 10 us, so a good fraction of iterations
    are wall-waits that step nothing.  The point of the check is the ratios
    (one pop and one publish per step), not the absolute count.
    """
    sim, core = LoopSim(), LoopCore(max_iters=400)
    run_loop(core, sim, mode="realtime")
    assert sim.steps >= 5
    assert sim.published == sim.steps  # pub_period 1: every tick
    assert core.pops == sim.steps  # one pop per tick, as before


def test_freerun_loop_runs_unpaced():
    sim, core = LoopSim(), LoopCore(max_iters=50)
    run_loop(core, sim, mode="freerun")
    assert sim.steps >= 40  # no wall pacing at all


def test_failed_step_does_not_publish():
    """A raised step still advances the clock but must not put state out."""
    sim, core = LoopSim(fail_after=3), LoopCore(max_iters=20)
    run_loop(core, sim, mode="freerun")
    assert sim.published == 3  # only the 3 good ticks


def test_unknown_step_mode_fails_the_run():
    """Strict: a typo must stop the run, not silently redefine a step."""
    sim, core = LoopSim(), LoopCore(max_iters=5)
    try:
        run_loop(core, sim, mode="lockstepp")
    except ValueError as exc:
        assert "expected one of" in str(exc)
    else:
        raise AssertionError("an unknown step_mode must raise out of spin()")

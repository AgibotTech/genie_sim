# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Behavioural contract of each pacer, mode by mode.

These are the semantics the loop body relies on and can no longer see:
publish cadence, when the idle gate arms, and the lockstep burst contract
(one command == exactly K ticks, exactly one publish).
"""

from __future__ import annotations

from conftest import DT, FakeClock, FakeLog, FakeSim

from genie_sim_engine_core.common.pacing import (
    STEP_MODES,
    FreerunPacer,
    LockstepPacer,
    RealtimePacer,
    Tick,
    make_pacer,
)

# ----------------------------------------------------------------------
# realtime
# ----------------------------------------------------------------------


def test_publish_cadence(clock, sim, log):
    """dds_hz throttling: publish every pub_period-th tick, and no other."""
    p = RealtimePacer(sim, wall_tick=0.0, pub_period=3, logger=log, clock=clock, sleep=clock.sleep)
    assert [p.after_step() for _ in range(9)] == [False, False, True] * 3


def test_failed_step_consumes_no_publish_slot(clock, sim):
    """A step that raised must not advance the publish counter."""
    p = RealtimePacer(sim, wall_tick=0.0, pub_period=2, logger=None, clock=clock, sleep=clock.sleep)
    assert p.after_step(True) is False
    assert p.after_step(False) is False  # failed — no credit consumed
    assert p.after_step(True) is True  # so the 2nd good tick publishes


def test_idle_gate_off_by_default(clock, sim, log):
    """The gate is opt-in: unset means the historical free-run behaviour."""
    p = RealtimePacer(sim, wall_tick=0.0, idle_timeout=0.0, logger=log, clock=clock, sleep=clock.sleep)
    sim.activity.idle_activity_ok = True
    sim.activity.last_cmd_wall = clock() - 10_000.0
    # Still RUN after ~3 hours of quiet.
    assert p.before_tick(clock()) is Tick.RUN


def test_idle_gate_needs_a_first_command(clock, sim, log):
    """Arms only after the first command — pausing during bring-up would
    starve the client's readiness polling."""
    p = RealtimePacer(sim, wall_tick=0.0, idle_timeout=0.5, logger=log, clock=clock, sleep=clock.sleep)
    sim.activity.idle_activity_ok = True
    sim.activity.last_cmd_wall = 0.0  # no command has ever arrived
    clock.advance(10_000.0)
    assert p.before_tick(clock()) is Tick.RUN


def test_idle_gate_needs_the_activity_subscription(clock, sim, log):
    """Never pause a live client we cannot hear."""
    p = RealtimePacer(sim, wall_tick=0.0, idle_timeout=0.5, logger=log, clock=clock, sleep=clock.sleep)
    sim.activity.idle_activity_ok = False  # subscription not up
    sim.activity.last_cmd_wall = clock()
    clock.advance(10_000.0)
    assert p.before_tick(clock()) is Tick.RUN


def test_idle_gate_pauses_then_a_command_resumes(clock, sim, log):
    p = RealtimePacer(sim, wall_tick=0.0, idle_timeout=0.5, logger=log, clock=clock, sleep=clock.sleep)
    sim.activity.idle_activity_ok = True
    sim.activity.last_cmd_wall = clock() - 5.0  # a real command, long ago
    assert p.before_tick(clock() + 10.0) is Tick.PAUSED
    sim.activity.last_cmd_wall = clock() + 10.0  # fresh command
    assert p.before_tick(clock() + 10.0) is Tick.RUN
    assert any("paused" in m for m in log.lines)
    assert any("resumed" in m for m in log.lines)


def test_stale_stamp_never_pulls_the_baseline_back(clock, sim, log):
    """``_last_active`` moves forward only — a late-delivered old stamp must
    not un-pause a gate that legitimately tripped."""
    p = RealtimePacer(sim, wall_tick=0.0, idle_timeout=0.5, logger=log, clock=clock, sleep=clock.sleep)
    sim.activity.idle_activity_ok = True
    sim.activity.last_cmd_wall = clock()
    p.before_tick(clock() + 10.0)  # trips
    sim.activity.last_cmd_wall = clock() - 100.0  # a stale stamp arrives
    assert p.before_tick(clock() + 10.0) is Tick.PAUSED


def test_pending_reset_counts_as_activity(clock, sim, log):
    """A reset in flight is work, even with no command behind it."""
    p = RealtimePacer(sim, wall_tick=0.0, idle_timeout=0.5, logger=log, clock=clock, sleep=clock.sleep)
    sim.activity.idle_activity_ok = True
    sim.activity.last_cmd_wall = clock() - 5.0
    sim.activity.reset_pending = True
    assert p.before_tick(clock() + 10.0) is Tick.RUN


# ----------------------------------------------------------------------
# ``activity is None`` — the honest degrade, at RUN time
# ----------------------------------------------------------------------
#
# Both pacers used to ask the ENGINE for ``_idle_activity_ok`` /
# ``_last_cmd_wall``, names that live on the reset coordinator.  Every probe
# missed, so both features were off on every backend in the tree and the only
# trace was a bring-up warning.  ``activity`` is now a declared member with a
# documented ``None``, which turns "we cannot hear the client" from an
# accident into a state — and a state can be tested.
#
# ``test_lockstep_warns_without_an_activity_signal`` covers the bring-up
# warning; these cover what the pacer DOES for the rest of the run.


class NoSignal(FakeSim):
    """A backend that reports no client activity, the ABC's way."""

    def __init__(self) -> None:
        super().__init__()
        self.activity = None


def test_the_idle_gate_free_runs_without_a_signal(clock, log):
    """Never idle a client we cannot hear.

    The gate is requested (1 s) and the run then goes quiet for an hour. A
    backend WITH a signal would pause here; this one must keep stepping,
    because "no command seen" and "no way to see commands" look identical
    from inside the pacer and only one of them means the client left.
    """
    p = RealtimePacer(NoSignal(), DT, 1, 1.0, log, clock=clock, sleep=clock.sleep)
    clock.advance(3600.0)
    assert p.before_tick(clock()) is Tick.RUN


def test_lockstep_never_engages_without_a_signal(clock, log):
    """Engagement keys off a command arrival, which is invisible here.

    It must fall through to the warmup pacer's wall pacing rather than
    engaging on a zero stamp — an engaged lockstep with no arrivals to
    consume would park the loop forever after its first burst.
    """
    sim = NoSignal()
    warm = RealtimePacer(sim, DT, 1, 0.0, None, clock=clock, sleep=clock.sleep)
    lp = LockstepPacer(sim, 4, warm, log, sleep=clock.sleep)
    for _ in range(20):
        assert lp.before_tick(clock()) in (Tick.RUN, Tick.IDLE)
        clock.advance(DT)
    assert not lp._engaged


def test_lockstep_flushes_on_a_pending_reset_only_with_a_signal(clock, log):
    """The FLUSH path reads ``reset_pending`` off the same object.

    Reached only once engaged, so the negative control is the one that
    matters: without a signal there is nothing to read and nothing to flush.
    """
    sim = NoSignal()
    warm = RealtimePacer(sim, DT, 1, 0.0, None, clock=clock, sleep=clock.sleep)
    lp = LockstepPacer(sim, 1, warm, log, sleep=clock.sleep)
    seen = set()
    for _ in range(20):
        seen.add(lp.before_tick(clock()))
        clock.advance(DT)
    assert Tick.FLUSH not in seen


def test_a_signal_that_appears_late_still_engages(clock, log):
    """``idle_activity_ok`` flips true when the backend's ``/joint_command``
    subscriber comes up, which can be after the loop starts — so the pacer
    must sample it every tick, not latch the answer at construction."""
    sim = FakeSim()
    warm = RealtimePacer(sim, DT, 1, 0.0, None, clock=clock, sleep=clock.sleep)
    lp = LockstepPacer(sim, 2, warm, log, sleep=clock.sleep)
    lp.before_tick(clock())
    assert not lp._engaged
    sim.activity.idle_activity_ok = True
    sim.activity.last_cmd_wall = clock()
    assert lp.before_tick(clock()) is Tick.RUN
    assert lp._engaged


def test_overrun_clamp(clock, sim):
    """A tick that blew its budget clamps the clock forward once."""
    p = RealtimePacer(sim, wall_tick=DT, pub_period=1, logger=None, clock=clock, sleep=clock.sleep)
    p.before_tick(clock())
    p.after_step(True)
    assert p.note_overrun(clock()) is False  # on schedule
    clock.advance(10 * DT)  # a 100 ms stall
    assert p.note_overrun(clock()) is True
    assert p.note_overrun(clock()) is False  # already clamped


# ----------------------------------------------------------------------
# lockstep
# ----------------------------------------------------------------------


def _lockstep(clock, sim, ticks=4, log=None):
    warm = RealtimePacer(sim, DT, 1, 0.0, log, clock=clock, sleep=clock.sleep)
    return LockstepPacer(sim, ticks, warm, log, sleep=clock.sleep)


def test_lockstep_startup_grace(clock, sim, log):
    """Free-run until the first command: the handshake needs state to flow."""
    p = _lockstep(clock, sim, log=log)
    assert p.before_tick(clock()) is Tick.RUN
    assert p.after_step() is True  # and warmup publishes


def test_lockstep_burst_is_exactly_k_ticks(clock, sim, log):
    """The RL contract: one command == K ticks, exactly one publish, on the
    LAST tick — the wrapper waits for the state after all K."""
    p = _lockstep(clock, sim, ticks=4, log=log)
    p.before_tick(clock())  # warmup
    p.after_step()
    sim.activity.idle_activity_ok = True
    sim.activity.last_cmd_wall = clock()
    assert p.before_tick(clock()) is Tick.RUN  # engages
    assert [p.after_step() for _ in range(4)] == [False, False, False, True]


def test_lockstep_waits_between_commands(clock, sim, log):
    p = _lockstep(clock, sim, ticks=2, log=log)
    sim.activity.idle_activity_ok = True
    sim.activity.last_cmd_wall = clock()
    p.before_tick(clock())
    p.after_step()
    p.after_step()
    assert p.before_tick(clock()) is Tick.PAUSED


def test_lockstep_failed_step_keeps_its_burst_credit(clock, sim):
    """A step that raised did not happen, so it must not shorten the burst."""
    p = _lockstep(clock, sim, ticks=2)
    sim.activity.idle_activity_ok = True
    sim.activity.last_cmd_wall = clock()
    p.before_tick(clock())
    assert p.after_step(False) is False
    assert p.after_step(True) is False  # 1st real tick of 2
    assert p.after_step(True) is True  # 2nd real tick publishes


def test_lockstep_one_command_starts_one_burst(clock, sim):
    """The same command must not be replayed into a second burst."""
    p = _lockstep(clock, sim, ticks=1)
    sim.activity.idle_activity_ok = True
    sim.activity.last_cmd_wall = clock()
    assert p.before_tick(clock()) is Tick.RUN
    p.after_step(True)
    assert p.before_tick(clock()) is Tick.PAUSED  # same stamp — no new burst
    clock.advance(0.5)
    sim.activity.last_cmd_wall = clock()  # a genuinely new command
    assert p.before_tick(clock()) is Tick.RUN


def test_lockstep_warns_without_an_activity_signal(clock, log):
    """A backend with no command-arrival signal can never engage lockstep.

    That must be loud: a run that silently free-runs produces plausible,
    wrong episode timing, which is the exact hazard the strict step_mode
    parsing exists to prevent.
    """

    class Bare:
        # ``PhysicsEngine.activity`` defaults to None: "this backend cannot
        # tell whether a client is driving it".
        activity = None

    make_pacer("lockstep", Bare(), DT, 1, log, clock=clock, sleep=clock.sleep, lockstep_ticks=4)
    assert any("NEVER engage" in m for m in log.lines), log.lines


# ----------------------------------------------------------------------
# freerun
# ----------------------------------------------------------------------


def test_freerun_never_waits(sim):
    p = FreerunPacer(sim, pub_period=1)
    assert all(p.before_tick(t) is Tick.RUN for t in (0.0, 1.0, 1e9))


def test_freerun_honours_publish_cadence(sim):
    p = FreerunPacer(sim, pub_period=3)
    assert [p.after_step() for _ in range(6)] == [False, False, True] * 2


# ----------------------------------------------------------------------
# registry
# ----------------------------------------------------------------------


def test_registry_keys_match_class_attributes():
    """The registry is keyed off ``cls.mode``; a mismatch would make a mode
    unreachable by the name its own docstring advertises."""
    assert all(key == cls.mode for key, cls in STEP_MODES.items())
    assert set(STEP_MODES) == {"realtime", "lockstep", "freerun"}


def test_every_mode_builds_from_the_same_param_bag(clock, sim, log):
    """The maintenance contract: a new mode must need no new call site.

    Every pacer is constructed through ``from_params`` with the identical
    argument bag, so adding one is a class plus a registry line.
    """
    for mode in STEP_MODES:
        p = make_pacer(mode, sim, DT, 1, log, clock=clock, sleep=clock.sleep, idle_timeout=1.0, lockstep_ticks=4)
        assert p.mode == mode
        assert p.describe()


def test_irrelevant_knobs_are_warned_not_silently_dropped(clock, sim, log):
    """Setting step_lockstep_ticks under realtime is a mistake worth hearing."""
    make_pacer("realtime", sim, DT, 1, log, clock=clock, sleep=clock.sleep, lockstep_ticks=8)
    assert any("ignored under step_mode=realtime" in m for m in log.lines)

    log.lines.clear()
    make_pacer("lockstep", sim, DT, 1, log, clock=clock, sleep=clock.sleep, idle_timeout=2.0, lockstep_ticks=4)
    assert any("ignored under step_mode=lockstep" in m for m in log.lines)

# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Differential check: ``spin_ms`` must survive the pacer refactor unchanged.

``spin_ms`` is fed to the C++ stats ring (``note_step_timing``), which
reports mean/min/max/std/p95/p99 and is the primary triage signal in
``docs/perf.md``.  The refactor moved the wall-pacing ``continue`` behind a
pacer, and the tempting simplification is a single uniform
``if not decision.runs_step: t_prev_tick_end = now; continue``.

That is wrong, and silently so:

  * the wall-wait ``continue`` deliberately does NOT touch
    ``t_prev_tick_end`` — the sub-tick wait is charged to the next tick's
    spin_ms, which is the whole point of the metric;
  * an idle-gate pause MUST reset it — an unbounded pause charged to
    spin_ms injects one 300000 ms sample and destroys max/p95/p99 for the
    rest of the run.

Both are correct, for opposite reasons, so no uniform choice works.  These
tests drive a scripted clock through a transcription of the pre-refactor
loop and through the pacer, and assert the emitted streams match exactly.
"""

from __future__ import annotations

from conftest import (
    DT,
    POLL_S,
    WALL_TICK,
    WORK_S,
    FakeClock,
    FakeSim,
    LoopCore,
    LoopSim,
    run_loop,
)

from genie_sim_engine_core.common.pacing import RealtimePacer, Tick, make_pacer


def run_baseline(clock, sim, n_ticks, idle_timeout=0.0):
    """The pre-refactor loop, verbatim in structure, instrumented for spin_ms.

    Mirrors the old wall wait (no baseline reset) and the idle-gate block
    (baseline reset + next_tick advance).  This is the reference the pacer
    must reproduce; it is deliberately NOT written in terms of the pacer.
    """
    next_tick = clock()
    t_prev_tick_end = clock()
    last_active = clock()
    out = []
    ran = 0
    guard = 0
    while ran < n_ticks:
        guard += 1
        assert guard < 100000, "baseline failed to converge"
        clock.advance(POLL_S)
        now = clock()
        if now < next_tick:
            remaining = next_tick - now
            if remaining > 0.0005:
                clock.sleep(remaining * 0.9)
            continue  # <-- does NOT touch t_prev_tick_end
        # --- the idle gate ---
        if idle_timeout > 0 and sim.activity.idle_activity_ok and float(sim.activity.last_cmd_wall or 0.0) > 0.0:
            lc = float(sim.activity.last_cmd_wall or 0.0)
            if lc > last_active:
                last_active = lc
            if sim.activity.reset_pending:
                last_active = now
            if (now - last_active) > idle_timeout:
                t_prev_tick_end = clock()  # <-- DOES reset the baseline
                next_tick += WALL_TICK
                continue
        spin_ms = (now - t_prev_tick_end) * 1000.0
        out.append(round(spin_ms, 6))
        clock.advance(WORK_S)  # the tick body
        t_prev_tick_end = clock()
        next_tick += WALL_TICK
        ran += 1
    return out


def run_pacer(clock, sim, n_ticks, idle_timeout=0.0):
    """The shipped loop shape: one pacer call, baseline driven by the enum."""
    pacer = RealtimePacer(sim, WALL_TICK, 1, idle_timeout, None, clock=clock, sleep=clock.sleep)
    t_prev_tick_end = clock()
    out = []
    ran = 0
    guard = 0
    while ran < n_ticks:
        guard += 1
        assert guard < 100000, "pacer failed to converge"
        clock.advance(POLL_S)
        now = clock()
        decision = pacer.before_tick(now)
        if not decision.runs_step:
            if decision.resets_spin_baseline:
                t_prev_tick_end = clock()
            continue
        spin_ms = (now - t_prev_tick_end) * 1000.0
        out.append(round(spin_ms, 6))
        clock.advance(WORK_S)
        pacer.after_step(True)
        t_prev_tick_end = clock()
        ran += 1
    return out


def run_uniform(clock, sim, n_ticks):
    """The tempting, WRONG simplification — reset the baseline on every skip.

    Kept as a negative control: a suite that only agrees with the
    implementation has no teeth.  This must be detectably different from
    ``run_baseline``, or ``test_spin_ms_matches_baseline`` proves nothing.
    """
    pacer = RealtimePacer(sim, WALL_TICK, 1, 0.0, None, clock=clock, sleep=clock.sleep)
    t_prev = clock()
    out, ran, guard = [], 0, 0
    while ran < n_ticks:
        guard += 1
        assert guard < 100000
        clock.advance(POLL_S)
        now = clock()
        if not pacer.before_tick(now).runs_step:
            t_prev = clock()  # <-- the wrong simplification
            continue
        out.append(round((now - t_prev) * 1000.0, 6))
        clock.advance(WORK_S)
        pacer.after_step(True)
        t_prev = clock()
        ran += 1
    return out


def test_spin_ms_matches_baseline():
    """The headline: an identical spin_ms stream on the realtime path."""
    assert run_baseline(FakeClock(), FakeSim(), 40) == run_pacer(FakeClock(), FakeSim(), 40)


def test_spin_ms_is_non_trivial():
    """And non-trivial: spin_ms reflects the real wall wait (~dt - work)."""
    a = run_baseline(FakeClock(), FakeSim(), 40)
    # dt=10 ms, work=4 ms, so the wait is ~6 ms.  A uniform baseline reset
    # yields ~0 here, which is what makes this assertion the one that
    # actually catches the regression.
    assert all(s > 1.0 for s in a[1:])
    assert sum(a[1:]) / len(a[1:]) > 5.0


def test_uniform_reset_is_detectably_wrong():
    """Negative control — the wrong version must fail this comparison."""
    correct = run_baseline(FakeClock(), FakeSim(), 40)
    uniform = run_uniform(FakeClock(), FakeSim(), 40)
    assert uniform != correct
    # Specifically: it reports ~0 spin where the truth is ~6 ms.
    assert max(uniform[1:]) < 1.0


def _with_pause(runner):
    clock, sim = FakeClock(), FakeSim()
    sim.activity.idle_activity_ok = True
    sim.activity.last_cmd_wall = clock() - 5.0  # quiet for 5 s already
    return runner(clock, sim, 12, idle_timeout=1.0)


def test_pause_not_charged_to_spin_ms():
    """An unbounded pause must not land in the stats ring."""
    baseline = _with_pause(run_baseline)
    paced = _with_pause(run_pacer)
    assert baseline == paced
    # A charged 5 s pause would read ~5000 ms.
    assert max(paced) < 100.0


def test_enum_owns_the_baseline_rule():
    """Every mode inherits correct accounting by picking the right member."""
    assert not Tick.IDLE.resets_spin_baseline
    assert Tick.PAUSED.resets_spin_baseline
    assert Tick.FLUSH.resets_spin_baseline
    assert Tick.RUN.runs_step
    assert not any(t.runs_step for t in (Tick.IDLE, Tick.PAUSED, Tick.FLUSH))


def test_unknown_step_mode_raises():
    """Strict by design — a typo must not silently change what a step means."""
    try:
        make_pacer("lockstepp", FakeSim(), WALL_TICK, 1)
    except ValueError as exc:
        assert "expected one of" in str(exc)
        assert "lockstep" in str(exc)
    else:
        raise AssertionError("a mis-typed step_mode must raise, not fall back")


def test_default_mode_is_realtime():
    """An unset step_mode must keep the historical behaviour."""
    assert make_pacer("", FakeSim(), WALL_TICK, 1).mode == "realtime"
    assert make_pacer(None, FakeSim(), WALL_TICK, 1).mode == "realtime"


def test_dt_constant_matches_wall_tick():
    """Guard the fixture's own assumption (realtime_factor = 1.0)."""
    assert DT == WALL_TICK


# ----------------------------------------------------------------------
# A lifecycle pause must not be charged to spin_ms either
# ----------------------------------------------------------------------
#
# This is the same rule as ``Tick.PAUSED.resets_spin_baseline``, but the
# lifecycle pause never reaches ``before_tick`` — it parks in the loop's own
# ``if paused: sleep(0.001); continue`` branch.  So the enum cannot carry
# the rule here and the branch has to re-stamp the baseline itself.  These
# drive the REAL ``spin()``, because a pacer-level test cannot see a code
# path the pacer is not on.


def _paused_for(polls, extra=60):
    """Pause on poll 1, resume ``polls`` iterations later, then run on.

    The loop polls lifecycle verbs once per iteration including while
    parked, so ``polls`` is also the number of 1 ms sleeps spent paused.
    The iteration budget must therefore clear ``polls`` with room to spare,
    or the run ends still parked and the test measures nothing.

    Unlike everything else in this suite these two run on the REAL clock:
    the parked branch calls ``time.sleep`` directly and takes no injected
    clock, so a scripted one cannot reach it.  Cost is ~0.2 s each.
    """
    sim = LoopSim()
    core = LoopCore(max_iters=polls + extra, lifecycle={1: ["pause"], polls: ["resume"]})
    run_loop(core, sim, mode="realtime")
    return sim, core


def test_a_lifecycle_pause_is_not_charged_to_spin_ms():
    """The headline.  ~200 ms of pause must not land in the stats ring.

    Sampled rather than exact: the pause length here is real wall time
    (the parked branch calls the real ``time.sleep``), so the bound is
    generous — but the broken version reports the FULL pause, which is
    an order of magnitude clear of it.
    """
    sim, core = _paused_for(polls=200)
    assert sim.steps >= 1, "the loop never resumed"
    assert core.step_timings, "no spin_ms samples recorded"
    assert max(core.step_timings) < 50.0, (
        f"max spin_ms = {max(core.step_timings):.1f} ms; a ~200 ms pause " f"was charged to the metric"
    )


def test_the_pause_really_did_happen():
    """Control for the above: prove the pause is long enough to be detected.

    Without this, a bug that skipped the pause entirely would make the
    previous test pass for the wrong reason.
    """
    import time as _time

    t0 = _time.monotonic()
    _paused_for(polls=200)
    elapsed_ms = (_time.monotonic() - t0) * 1000.0
    assert elapsed_ms > 100.0, (
        f"run took {elapsed_ms:.1f} ms — the pause did not hold, so the " f"spin_ms bound above proves nothing"
    )

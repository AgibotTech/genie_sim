# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Loop pacing strategies — WHEN a tick runs, and WHETHER it publishes.

``EngineRunLoop.spin()`` owns the tick BODY (commands, step, publish,
stats).  It does not own the POLICY of when a tick is allowed to run.
That policy is what varies between a wall-clock simulator, an RL
lockstep harness, and a flat-out batch run, so it lives here behind two
questions asked once each per loop iteration:

    decision = pacer.before_tick(now)   # may this tick run?
    published = pacer.after_step(ok)    # did this tick's state go out?

Adding a mode means adding a class and a registry entry.  It must never
mean adding a branch to the loop body — that is the failure mode this
module exists to prevent.

Selected by the ``step_mode`` launcher parameter; see
``docs/step_modes.md`` for the full design rationale.
"""

from __future__ import annotations

import time
from enum import Enum


class Tick(Enum):
    """What the loop should do this iteration.

    The IDLE / PAUSED split is not cosmetic — it decides whether the
    waiting time is charged to ``spin_ms`` in the stats ring.

    ``spin_ms`` answers "how much of the tick budget went to waiting
    rather than working", so it is only meaningful when bounded by the
    tick period.  A sub-tick wall wait IS that: the loop is waiting on
    its own clock for work it is about to do, and the wait belongs in
    the sample.  An unbounded pause is NOT: the loop is waiting on an
    external party (no command for 300 s), and charging that to
    ``spin_ms`` puts a 300000 ms sample into the C++ ring, destroying
    max/p95/p99 for the rest of the run.

    Hence the rule, which is a property of the enum rather than of any
    pacer, so every mode gets it right by construction:

        IDLE   — waiting on the clock, bounded    -> keep accumulating
        PAUSED — waiting on a peer, unbounded     -> reset the baseline
        FLUSH  — no step, but publish; unbounded  -> reset the baseline
    """

    RUN = "run"  # normal tick: commands, step, maybe publish
    IDLE = "idle"  # sub-tick wall wait; spin_ms keeps accumulating
    PAUSED = "paused"  # unbounded wait; reset the spin baseline
    FLUSH = "flush"  # no step, but apply pending state and publish

    @property
    def resets_spin_baseline(self) -> bool:
        return self in (Tick.PAUSED, Tick.FLUSH)

    @property
    def runs_step(self) -> bool:
        return self is Tick.RUN

    @property
    def publishes_without_stepping(self) -> bool:
        """FLUSH is the only decision that makes the loop DO work.

        The other three are kinds of skip; FLUSH means "state changed
        underneath us without a step, get it on the wire now".  It exists
        because a reset that lands between lockstep bursts must reach the
        client immediately: the RL wrapper is blocked reading post-reset
        joints, and the next command — which would carry the state out on
        the next burst — is exactly what it is waiting to be able to send.
        Without this the run deadlocks.
        """
        return self is Tick.FLUSH


class LoopPacer:
    """Decides when a tick runs and whether its state is published.

    Subclasses implement :meth:`before_tick` and :meth:`after_step`, and
    are constructed through :meth:`from_params` so the registry can build
    any mode from the same launcher-parameter bag.
    """

    #: ``step_mode`` value that selects this pacer.  Set by subclasses.
    mode: str = ""

    def before_tick(self, now: float) -> Tick:
        """May this tick run?  May sleep.  Called once per iteration."""
        raise NotImplementedError

    def after_step(self, ok: bool = True) -> bool:
        """Publish the state this tick produced?

        ``ok`` is False when the step raised — the tick's clock still
        advances, but the tick did not happen, so it must not consume a
        publish slot or a lockstep burst credit.
        """
        raise NotImplementedError

    def next_deadline(self, now: float) -> float:
        """Wall time this tick was scheduled for — the render budget anchor.

        ``render_hook(now, next_tick, sim_time)`` receives this as its second
        argument, and the Isaac decoupled-render path turns it into
        ``budget_s = (next_tick + wall_tick) - t_after_phys``: how much wall
        time is left before the following tick must start.

        Unpaced modes enforce no deadline, so the base answers ``now`` — the
        same slack a perfectly on-schedule realtime tick has.  Answering
        infinity would let a decoupled render fire every tick and defeat the
        point of an unpaced mode; answering zero would starve it entirely.
        """
        return now

    def note_overrun(self, now: float) -> bool:
        """Called at the end of a RUN body.  True if the clock was clamped."""
        return False

    def notify_resumed(self, now: float) -> None:
        """The loop was stopped by something OUTSIDE the pacer, and resumed.

        Lifecycle ``resume`` and ``reset`` (see
        ``EngineRunLoop.spin.process_lifecycle_commands``) park the loop in a
        branch that ``continue``s BEFORE ``before_tick`` is reached, so a
        pacer cannot observe that time passing.  The pre-pacer loop handled
        this by writing ``next_tick = time.monotonic()`` directly under
        ``nonlocal``; now that the clock is pacer-private, the lifecycle path
        must say so explicitly instead.

        Skipping this call is not a cosmetic bug.  After a 30 s pause a
        wall-clock pacer owes ~3000 ticks and will burst to catch up, and an
        idle gate whose baseline is 30 s stale trips on the very first tick
        after resume — the operator presses resume and the GPU stays quiet.
        """
        return None

    def describe(self) -> str:
        """One line, logged once at spin start, naming the live policy."""
        return self.mode


class RealtimePacer(LoopPacer):
    """Wall-clock paced — ticks fire on a fixed wall cadence.

    This is the historical ``EngineRunLoop`` behaviour, unchanged: the
    same ``next_tick`` accumulation, the same ``sleep(remaining * 0.9)``
    convergence, the same ``pub_tick`` counter, the same overrun clamp.

    The idle gate is a PARAMETER of this mode, not a mode of its own.
    It is wall pacing with a longer period once the client goes quiet,
    and making it a mode would give ``realtime`` / ``realtime_idle`` /
    ``lockstep`` / ``lockstep_idle`` — a product where a sum was wanted.
    """

    mode = "realtime"

    def __init__(
        self,
        sim,
        wall_tick: float,
        pub_period: int = 1,
        idle_timeout: float = 0.0,
        logger=None,
        clock=time.monotonic,
        sleep=time.sleep,
    ) -> None:
        self._sim = sim
        self._wall_tick = wall_tick
        self._pub_period = max(1, int(pub_period))
        self._idle_timeout = max(0.0, float(idle_timeout))
        self._logger = logger
        self._clock = clock
        self._sleep = sleep
        self._next_tick = clock()
        self._pub_tick = 0
        self._last_active = clock()
        self._idle_paused = False

    @classmethod
    def from_params(
        cls,
        sim,
        wall_tick,
        pub_period,
        logger,
        clock=time.monotonic,
        sleep=time.sleep,
        *,
        idle_timeout=0.0,
        lockstep_ticks=0,
        **_ignored,
    ):
        if lockstep_ticks:
            _warn(logger, f"step_lockstep_ticks={lockstep_ticks} ignored under " f"step_mode=realtime")
        pacer = cls(sim, wall_tick, pub_period, idle_timeout, logger, clock, sleep)
        if pacer._idle_timeout > 0.0 and not _has_activity_signal(sim):
            _warn(
                logger,
                f"step_idle_timeout={pacer._idle_timeout:g}s requested, but this "
                f"backend returns no ActivitySignal from ``activity`` — the idle "
                f"gate will never fire and physics free-runs, which is the "
                f"pre-existing behaviour",
            )
        return pacer

    def describe(self) -> str:
        hz = 1.0 / self._wall_tick if self._wall_tick > 0 else float("inf")
        gate = f"idle gate {self._idle_timeout:.1f}s" if self._idle_timeout > 0 else "idle gate off"
        return f"realtime — {hz:.1f} Hz wall, publish every " f"{self._pub_period} tick(s), {gate}"

    def notify_resumed(self, now: float) -> None:
        # Drop the catch-up debt: the paused wall time was never ours to
        # simulate.  Mirrors the pre-pacer ``next_tick = time.monotonic()``.
        self._next_tick = now
        # Re-baseline the idle gate too.  ``_last_active`` only ever moves
        # forward (a stale command stamp must not pull it back), so without
        # this a lifecycle pause longer than the timeout would leave the gate
        # already tripped at the moment of resume.
        if now > self._last_active:
            self._last_active = now
        if self._idle_paused:
            self._idle_paused = False
            _info(self._logger, "[pacer] resumed — physics active")

    def _quiet(self, now: float) -> bool:
        """Has the client gone quiet long enough to stop stepping?"""
        if self._idle_timeout <= 0.0:
            return False
        act = self._sim.activity
        # Never pause a live client we cannot hear: no signal at all, or the
        # activity subscription not yet up, and we fall back to free-running.
        if act is None or not act.idle_activity_ok:
            return False
        last_cmd = float(act.last_cmd_wall or 0.0)
        # Arms only AFTER the first command ever — pausing during bring-up
        # starves the RL wrapper's readiness polling.
        if last_cmd <= 0.0:
            return False
        # Monotonic only: a stale stamp must never pull the baseline back.
        if last_cmd > self._last_active:
            self._last_active = last_cmd
        if act.reset_pending:
            self._last_active = now
        return (now - self._last_active) > self._idle_timeout

    def before_tick(self, now: float) -> Tick:
        if now < self._next_tick:
            remaining = self._next_tick - now
            if remaining > 0.0005:
                self._sleep(remaining * 0.9)
            return Tick.IDLE
        if self._quiet(now):
            if not self._idle_paused:
                self._idle_paused = True
                _info(
                    self._logger,
                    f"[pacer] no commands for {self._idle_timeout:.1f}s — physics "
                    f"paused (GPU quiet; wakes within one tick)",
                )
            self._next_tick += self._wall_tick
            return Tick.PAUSED
        if self._idle_paused:
            self._idle_paused = False
            _info(self._logger, "[pacer] command received — physics resumed")
        return Tick.RUN

    def after_step(self, ok: bool = True) -> bool:
        self._next_tick += self._wall_tick
        if not ok:
            return False
        self._pub_tick += 1
        if self._pub_tick >= self._pub_period:
            self._pub_tick = 0
            return True
        return False

    def next_deadline(self, now: float) -> float:
        return self._next_tick

    def note_overrun(self, now: float) -> bool:
        if self._next_tick < now - self._wall_tick:
            self._next_tick = now + self._wall_tick
            return True
        return False


class LockstepPacer(LoopPacer):
    """Command-driven — one ``/joint_command`` advances physics K ticks.

    One control step becomes a fixed ``K*dt`` of sim time regardless of
    the wall clock, which is the property free-running cannot give: sim
    duration per action would otherwise track client latency.

    Startup grace: free-run until the first command arrives, because the
    handshake and warm-up need state to flow.  That grace is delegated to
    a real :class:`RealtimePacer` rather than reimplemented behind a flag.
    """

    mode = "lockstep"

    #: The C++ command latch is a SEPARATE subscription from the arrival
    #: stamp; give it a moment so the loop's pop_commands() sees the new
    #: values rather than the previous action.
    _LATCH_SETTLE_S = 0.002
    _POLL_S = 0.0005

    def __init__(self, sim, ticks: int, warmup: RealtimePacer, logger=None, sleep=time.sleep) -> None:
        self._sim = sim
        self._ticks = max(1, int(ticks))
        self._warmup = warmup
        self._logger = logger
        self._sleep = sleep
        self._engaged = False
        self._burst_left = 0
        self._last_consumed = 0.0

    @classmethod
    def from_params(
        cls,
        sim,
        wall_tick,
        pub_period,
        logger,
        clock=time.monotonic,
        sleep=time.sleep,
        *,
        idle_timeout=0.0,
        lockstep_ticks=1,
        **_ignored,
    ):
        if idle_timeout:
            _warn(
                logger,
                f"step_idle_timeout={idle_timeout} ignored under "
                f"step_mode=lockstep (bursts already idle between commands)",
            )
        warmup = RealtimePacer(sim, wall_tick, pub_period, 0.0, logger, clock, sleep)
        if not _has_activity_signal(sim):
            # Do not raise: ``idle_activity_ok`` flips true when the
            # backend's command subscription comes up, which can be after the
            # loop starts.  But never let this be silent — a lockstep run that
            # never engages produces plausible-looking, wrong training curves,
            # which is the exact hazard the strict step_mode parsing exists
            # to prevent.
            _warn(
                logger,
                "step_mode=lockstep selected, but this backend returns no "
                "ActivitySignal from ``activity``, so command arrivals are "
                "invisible here. The loop will free-run on the wall clock and "
                "lockstep will NEVER engage — one command will not equal "
                f"{max(1, int(lockstep_ticks))} tick(s). Check the backend "
                "supports command-activity reporting before trusting any "
                "episode timing from this run.",
            )
        return cls(sim, lockstep_ticks, warmup, logger, sleep)

    def describe(self) -> str:
        return (
            f"lockstep — {self._ticks} tick(s) per command, publish on the "
            f"last tick, free-run until the first command"
        )

    def _cmd_stamp(self) -> float:
        act = self._sim.activity
        return 0.0 if act is None else float(act.last_cmd_wall or 0.0)

    def before_tick(self, now: float) -> Tick:
        if not self._engaged:
            act = self._sim.activity
            if act is not None and act.idle_activity_ok and self._cmd_stamp() > 0.0:
                self._engaged = True
                self._last_consumed = self._cmd_stamp()
                self._burst_left = self._ticks
                _info(self._logger, f"[pacer] lockstep engaged: 1 command == " f"{self._ticks} physics tick(s)")
                return Tick.RUN
            return self._warmup.before_tick(now)
        if self._burst_left > 0:
            return Tick.RUN  # inside a burst: no wall pacing, run flat out
        act = self._sim.activity
        if act is not None and act.reset_pending:
            # Apply + publish immediately so the RL reset path reads
            # post-reset joints, then keep waiting for the next command.
            return Tick.FLUSH
        stamp = self._cmd_stamp()
        if stamp <= self._last_consumed:
            self._sleep(self._POLL_S)
            return Tick.PAUSED
        self._last_consumed = stamp
        self._sleep(self._LATCH_SETTLE_S)
        self._burst_left = self._ticks
        return Tick.RUN

    def notify_resumed(self, now: float) -> None:
        # The warmup pacer owns the wall clock whether or not we are engaged.
        self._warmup.notify_resumed(now)
        # ``_burst_left`` is deliberately NOT cleared: a command already
        # bought K ticks of sim time, and a lifecycle pause partway through
        # must not silently shorten that episode step.  ``_last_consumed`` is
        # likewise kept, so the command consumed before the pause is not
        # replayed on resume.

    def next_deadline(self, now: float) -> float:
        # While the warmup pacer is driving, its clock is the real deadline.
        # Inside a burst there is none — bursts run flat out — so the base
        # answer (one tick of slack) applies.
        if not self._engaged:
            return self._warmup.next_deadline(now)
        return now

    def after_step(self, ok: bool = True) -> bool:
        if not self._engaged:
            return self._warmup.after_step(ok)
        if not ok:
            return False
        self._burst_left -= 1
        # Publish on the burst's LAST tick only: the wrapper waits for the
        # state after all K ticks.  This publish is the RL contract ("exactly
        # one publish per command"), not throughput shaping, so it bypasses
        # the dds_hz cadence gate that the realtime path goes through.
        return self._burst_left == 0


class FreerunPacer(LoopPacer):
    """Unpaced — step as fast as the hardware allows.

    What ``realtime_factor: 1000`` is reaching for, without the clock
    arithmetic and the rounding sleeps it still pays today.
    """

    mode = "freerun"

    def __init__(self, sim, pub_period: int = 1, logger=None) -> None:
        self._sim = sim
        self._pub_period = max(1, int(pub_period))
        self._logger = logger
        self._pub_tick = 0

    @classmethod
    def from_params(
        cls,
        sim,
        wall_tick,
        pub_period,
        logger,
        clock=time.monotonic,
        sleep=time.sleep,
        *,
        idle_timeout=0.0,
        lockstep_ticks=0,
        **_ignored,
    ):
        for name, val in (("step_idle_timeout", idle_timeout), ("step_lockstep_ticks", lockstep_ticks)):
            if val:
                _warn(logger, f"{name}={val} ignored under step_mode=freerun")
        return cls(sim, pub_period, logger)

    def describe(self) -> str:
        return f"freerun — unpaced, publish every {self._pub_period} tick(s)"

    def before_tick(self, now: float) -> Tick:
        return Tick.RUN

    def after_step(self, ok: bool = True) -> bool:
        if not ok:
            return False
        self._pub_tick += 1
        if self._pub_tick >= self._pub_period:
            self._pub_tick = 0
            return True
        return False


# ----------------------------------------------------------------------
# Registry.  Adding a mode = a class above + one line here.
# ----------------------------------------------------------------------
STEP_MODES = {cls.mode: cls for cls in (RealtimePacer, LockstepPacer, FreerunPacer)}

DEFAULT_STEP_MODE = RealtimePacer.mode


def make_pacer(
    mode, sim, wall_tick, pub_period, logger=None, clock=time.monotonic, sleep=time.sleep, **params
) -> LoopPacer:
    """Build the pacer named by ``step_mode``.

    Raises rather than falling back.  This is deliberately stricter than
    the tri-state launcher strings (``contact_publish_scope`` and friends)
    whose invalid values are documented as "treated as unset and logged":
    a mis-typed contact scope publishes slightly wrong topics, while a
    mis-typed step mode silently changes what "one training step" means
    and yields plausible-looking, wrong training curves.
    """
    key = str(mode or DEFAULT_STEP_MODE).strip().lower()
    try:
        cls = STEP_MODES[key]
    except KeyError:
        raise ValueError(f"unknown step_mode {mode!r}; expected one of " f"{', '.join(sorted(STEP_MODES))}") from None
    return cls.from_params(sim, wall_tick, pub_period, logger, clock, sleep, **params)


def _has_activity_signal(sim) -> bool:
    """Does this backend report command arrivals?

    Both the idle gate and lockstep engagement need ``last_cmd_wall`` and
    ``idle_activity_ok``; a backend whose ``activity`` is ``None`` can only
    free-run. Checked once at construction so the warning is emitted at
    bring-up rather than never (the run-time checks are silent by design —
    they are polled every tick).

    One attribute, declared on ``PhysicsEngine``, replaces two ``hasattr``
    probes for private names that lived on a different object entirely: the
    old form asked the engine for the coordinator's attributes, always got
    "no", and turned both features off without anything being wrong.
    """
    return sim.activity is not None


def _info(logger, msg: str) -> None:
    if logger is not None:
        logger.info(msg)


def _warn(logger, msg: str) -> None:
    if logger is not None:
        logger.warn(msg)

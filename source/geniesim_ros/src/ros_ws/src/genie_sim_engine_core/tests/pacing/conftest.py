# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Shared fixtures and scripted-clock fakes for the pacing tests.

Every test here runs on a FAKE clock that ``sleep()`` advances instead of
blocking.  That is what makes a 30 s lifecycle pause cost no wall time and
makes the results deterministic — these checks assert on exact tick counts
and exact ``spin_ms`` streams, which a real clock could never give twice.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make `genie_sim_engine_core` importable without going through
# ament_python_install (colcon test runs before install by default).
_HERE = Path(__file__).resolve().parent
_PKG_ROOT = _HERE.parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from genie_sim_engine_core.common.loop import EngineRunLoop  # noqa: E402

DT = 0.01  # 100 Hz
WALL_TICK = DT  # realtime_factor = 1.0
WORK_S = 0.004  # each tick body costs 4 ms
POLL_S = 0.00002  # one loop iteration costs 20 us.  Models the busy-spin:
# the loop does not sleep when remaining <= 0.5 ms, so
# without this the scripted clock would never advance and
# the sub-tick wait would spin forever.
PAUSE_S = 30.0


class FakeClock:
    """Monotonic clock the test advances explicitly.  ``sleep()`` jumps it."""

    def __init__(self, t0: float = 1000.0) -> None:
        self.t = t0

    def __call__(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.t += max(0.0, s)

    def advance(self, s: float) -> None:
        self.t += s


class FakeActivity:
    """``ActivitySignal``, as a plain mutable record.

    The real one is ``RosResetCoordinator``, where every member is a property
    over live ROS state — a command-arrival stamp written by an rclpy
    callback and a pending-reset queue. Tests need to script that state at
    exact clock values, so this is the same four names with nothing behind
    them.

    Public names on purpose: the pacers reach the engine through
    ``sim.activity`` now, and a stub that still spelled these
    ``_last_cmd_wall`` would let the old private-attribute reach-in come back
    without a single test noticing.
    """

    def __init__(self, sim) -> None:
        self._sim = sim
        self.idle_activity_ok = False
        self.last_cmd_wall = 0.0
        self.reset_pending = False

    @property
    def reset_epoch(self) -> int:
        """Read through to the backend, as the real coordinator does.

        The credential belongs to whatever performed the state write, so the
        signal never keeps its own count — mirrored here so a test cannot
        pass against a stub that answers the question the coordinator no
        longer answers.
        """
        return self._sim.reset_epoch


class FakeSim:
    """Backend stub exposing only the members the pacers and loop read."""

    def __init__(self) -> None:
        self.activity = FakeActivity(self)
        self.num_envs = 1
        self.reset_epoch = 0
        self.applied = []  # list of (positions, stamp)
        self.drained = 0
        # The lifecycle verbs the loop now calls straight (declared on
        # PhysicsEngine, engine/base.py — no getattr probe any more).
        # Counted rather than ignored: "the backend was actually told" is the
        # thing the old probe-and-skip version could not state.
        self.pauses = 0
        self.resumes = 0
        self.resets = []  # list of (env_ids, seeds, instances)
        self.shutdowns = 0
        self.teleports = []  # every TeleportRequest handed over
        # Interleaving log: "teleport" / "commands", in call order.  Which of
        # the two ran first is a CONTRACT (pop_commands latches, so a stale
        # hold-position target would drag a teleported body back), and only an
        # ordered record can state it.
        self.order = []

    def pause(self):
        self.pauses += 1

    def resume(self):
        self.resumes += 1

    def episode_reset(self, env_ids=None, seeds=None, instances=None):
        self.resets.append((env_ids, seeds, instances))

    def shutdown(self):
        self.shutdowns += 1

    def teleport(self, requests):
        reqs = list(requests)
        self.order.append("teleport")
        self.teleports.extend(reqs)
        return len(reqs)  # a real backend returns the count it APPLIED

    def apply_commands(self, cmd_positions, cmd_4ws_steer_pos, cmd_4ws_drive_vel, cmd_4ws_stamp):
        # Mirrors the backend contract: drain_resets() runs first,
        # unconditionally, before any target buffer is touched.  Clearing the
        # flag here is the model of that drain — the real coordinator empties
        # its queue, so ``reset_pending`` goes false on the same call.
        if self.activity.reset_pending:
            self.drained += 1
            self.activity.reset_pending = False
            self.reset_epoch += 1
        self.order.append("commands")
        self.applied.append((dict(cmd_positions), cmd_4ws_stamp))


class FakeCore:
    """``pop_commands`` LATCHES — it returns the same snapshot forever.

    This is the real C++ behaviour and it is why the flush path must pass
    EMPTY dicts: re-applying a latched snapshot would drive the robot back
    off a freshly reset pose.
    """

    def __init__(self, latched=None, teleports=None) -> None:
        self._latched = latched if latched is not None else {"arm_j1": 1.234}
        self.pops = 0
        # ...and ``pop_teleports`` is the opposite: a queue, delivered ONCE.
        # Modelled here as a one-shot so a test can prove the loop does not
        # re-apply it every tick, which would pin the body in place.
        self._teleports = list(teleports or [])
        self.teleport_pops = 0

    def pop_teleports(self):
        self.teleport_pops += 1
        out, self._teleports = self._teleports, []
        return out

    def pop_commands(self):
        self.pops += 1
        # (positions, efforts, steer, drive, stamp) — the 5-tuple the real
        # loop unpacks.
        return self._latched, {}, {}, {}, 99.0


class FakeLog:
    def __init__(self) -> None:
        self.lines = []

    def info(self, m):
        self.lines.append(m)

    def warn(self, m):
        self.lines.append(m)


# ----------------------------------------------------------------------
# Stubs rich enough to drive the REAL ``EngineRunLoop.spin()``.
#
# The pacer-only fakes above pin what a pacer RETURNS; these pin what the
# loop DOES with it.  Both matter: every bug this suite guards against so
# far has been a wiring bug, not a policy bug.
# ----------------------------------------------------------------------


class Bodies:
    """Stands in for the (N, 7) body-transform array; N=0 skips the tf publish."""

    shape = (0, 7)


class LoopSim(FakeSim):
    """FakeSim plus the readback surface ``_publish_tick`` calls.

    ``reset_after`` injects a pending reset once that many steps have run.
    That timing is the whole point: a reset present BEFORE the burst is
    drained by the first tick's ordinary ``apply_commands`` and never needs
    a flush.  Only a reset arriving once the burst is spent — with no
    further tick to carry it out — exercises the FLUSH path.
    """

    def __init__(self, fail_after=None, reset_after=None):
        super().__init__()
        self.steps = 0
        self.published = 0
        self._fail_after = fail_after
        self._reset_after = reset_after
        # No PublishPlan is passed to the loop in these tests, so the publish
        # path runs PublishPlan.legacy() and never reaches numpy.

    def step(self, dt, t0):
        self.steps += 1
        if self._fail_after is not None and self.steps > self._fail_after:
            raise RuntimeError("solver blew up")
        if self._reset_after is not None and self.steps == self._reset_after:
            self.activity.reset_pending = True
        return 0.0

    def tick_extras(self):
        pass

    def get_joint_states(self):
        return [], []

    def get_body_transforms(self):
        return Bodies(), []

    def get_odom(self, sim_time):
        return None

    def get_contact_states(self):
        return None

    def note_publish_phase(self, **kw):
        self.published += 1

    def note_render(self, *a):
        pass

    def note_phase_timing(self, **kw):
        pass

    # ``shutdown`` is NOT overridden here — FakeSim's counting version is the
    # one that must run, or a test asserting the verb reached the backend
    # passes against a stub that swallowed it.


class LoopCore(FakeCore):
    """FakeCore plus the publish surface, and an ``ok()`` that ends the run.

    ``max_iters`` counts LOOP ITERATIONS, not ``ok()`` calls: ``spin()`` polls
    ``core.ok()`` twice per pass (once as the ``while`` condition, once after
    ``process_lifecycle_commands`` so a shutdown verb takes effect
    immediately).  Counting raw calls would silently halve every budget here.
    """

    def __init__(self, max_iters=400, latched=None, lifecycle=None, teleports=None):
        super().__init__(latched, teleports=teleports)
        self._calls_left = 2 * max_iters
        self.iters = 0
        self.step_timings = []
        # {poll_number: [verb, ...]}.  Polls are counted from 1 and happen
        # once per loop iteration, INCLUDING iterations spent parked in the
        # ``if paused`` branch — which is what lets a test hold a pause open
        # for a known number of 1 ms sleeps.
        self._lifecycle = dict(lifecycle or {})
        self.polls = 0

    def poll_lifecycle_commands(self):
        self.polls += 1
        return self._lifecycle.pop(self.polls, [])

    def set_lifecycle_state(self, code, label):
        pass

    def ok(self):
        self.iters += 1
        self._calls_left -= 1
        return self._calls_left > 0

    def publish_clock(self, t):
        pass

    def publish_joint_states(self, *a, **kw):
        pass

    def publish_joint_states_scene(self, *a, **kw):
        pass

    def publish_body_tf_scene(self, *a):
        pass

    def publish_tf_render(self, *a):
        pass

    def publish_odom(self, *a):
        pass

    def publish_contact_states(self, *a):
        pass

    def publish_rtf(self, v):
        pass

    def note_step_timing(self, step_ms, solver_ms, render_ms, publish_ms, spin_ms, did_render, now):
        self.step_timings.append(spin_ms)

    def log_stats_if_due(self, now):
        return ""

    def shutdown(self):
        pass


class NullLog:
    def info(self, m):
        pass

    def warn(self, m):
        pass

    def error(self, m):
        pass


def run_loop(core, sim, mode="lockstep", ticks=4, idle_timeout=0.0):
    EngineRunLoop(
        core,
        sim,
        DT,
        NullLog(),
        rtf=1000.0,  # wall pacing effectively off; the mode drives
        step_mode=mode,
        step_idle_timeout=idle_timeout,
        step_lockstep_ticks=ticks,
    ).spin()


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def sim():
    return FakeSim()


@pytest.fixture
def core():
    return FakeCore()


@pytest.fixture
def log():
    return FakeLog()

# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""``audit_lifecycle`` — the bring-up report on a narrowed reset contract.

Three omissions that are decidable from the class object and otherwise
silent at runtime: no ``episode_reset`` at all, an ``episode_reset`` on the
upstream two-argument arity, and an ``episode_reset`` that never bumps
``reset_epoch``.  The shapes tested here are the real ones — the two-argument
arity is a live contract in an integration tree that consumes ``src/ros_ws/src``
by symlink, and two of its backends subclass this ABC while carrying that
arity.
"""

from __future__ import annotations

import numpy as np
import pytest

from genie_sim_engine_core.engine.base import (
    PhysicsEngine,
    _audit_pause_contract,
    _audit_reset_contract,
    audit_lifecycle,
)


class Recorder:
    """``SimpleLogger``'s surface, kept as a list."""

    def __init__(self) -> None:
        self.warnings: list = []

    def info(self, m: str) -> None:
        pass

    def warn(self, m: str) -> None:
        self.warnings.append(m)

    def error(self, m: str) -> None:
        self.warnings.append(m)


class Bare(PhysicsEngine):
    """The abstract members and nothing else — ``MujocoBackendAdapter``'s shape."""

    @property
    def stage(self):
        return None

    @property
    def robot_prefix(self) -> str:
        return "bot"

    @property
    def joint_names(self):
        return ["j0"]

    @property
    def joint_prim_map(self):
        return {}

    @property
    def body_paths(self):
        return []

    def startup(self, headless: bool) -> None:
        pass

    def step(self, dt: float, step_start: float) -> float:
        return 0.0

    def tick_extras(self) -> None:
        pass

    def get_joint_states(self):
        return np.zeros(1), np.zeros(1)

    def get_body_transforms(self):
        return np.zeros((0, 7)), []

    def get_odom(self, sim_time):
        return None

    def apply_commands(self, cmd_positions, cmd_4ws_steer_pos, cmd_4ws_drive_vel, cmd_4ws_stamp):
        pass

    def shutdown(self) -> None:
        pass


class UpstreamArity(Bare):
    """``episode_reset(env_ids, seed)`` — a consuming tree's contract.

    A downstream ROS node is exactly this: it subclasses this ABC for
    ``EngineSession`` compatibility while implementing the upstream signature.
    """

    _epoch = 0

    def episode_reset(self, env_ids=None, seed=None) -> None:
        type(self)._epoch += 1

    @property
    def reset_epoch(self) -> int:
        return type(self)._epoch


class NoCredential(Bare):
    """Correct arity, no epoch — the reset lands and no client can tell."""

    def episode_reset(self, env_ids=None, seeds=None, instances=None) -> None:
        pass


class Conformant(NoCredential):
    """Arity, credential, a matched pause pair — nothing to report."""

    def __init__(self) -> None:
        self._epoch = 0

    def pause(self) -> None:
        pass

    def resume(self) -> None:
        pass

    def episode_reset(self, env_ids=None, seeds=None, instances=None) -> None:
        self._epoch += 1

    @property
    def reset_epoch(self) -> int:
        return self._epoch


def audit(sim) -> list:
    """Everything ``EngineSession`` bring-up would print for this backend."""
    log = Recorder()
    audit_lifecycle(sim, log)
    return log.warnings


def reset_findings(sim) -> list:
    """Just the reset axis.

    Most fixtures here are built on ``Bare``, which also fails the pause axis
    (correctly — it is ``MujocoBackendAdapter``'s shape, and that backend's
    pause really is broken).  Asserting on one axis at a time keeps each test
    about the thing it names; ``test_both_axes_report_independently`` is where
    the composition is pinned.
    """
    log = Recorder()
    _audit_reset_contract(sim, log)
    return log.warnings


def pause_findings(sim) -> list:
    log = Recorder()
    _audit_pause_contract(sim, log)
    return log.warnings


def test_a_backend_with_no_episode_reset_is_named():
    """``~/reset`` on this backend reports success and does nothing.

    The inherited ABC no-op is the right default — it is what let the verb be
    declared without breaking every backend — but a backend that lands in
    production still on it has a dead reset path and no way to find out.
    """
    (msg,) = reset_findings(Bare())
    assert "Bare" in msg, "the report must name the class, not the ABC"
    assert "episode_reset" in msg
    assert "RosResetCoordinator" in msg, "name the fix, not just the fault"


def test_the_upstream_two_arg_arity_is_named_with_what_it_costs():
    """The narrowed form is reported, and the loss is stated precisely.

    ``drain_resets`` passes ``seeds=``/``instances=`` on every call, seeded or
    not, so this is not a degraded seeding path — it is every reset through
    ``/rlinf/reset_envs`` raising, being swallowed, and leaving the epoch
    where it was.  A message that said only "per-env seeding unavailable"
    would understate it by a lot.
    """
    (msg,) = reset_findings(UpstreamArity())
    assert "UpstreamArity" in msg
    assert "episode_reset(env_ids=None, seed=None)" in msg, "echo the signature that was found"
    assert "->" not in msg, "the return annotation is noise; the arguments are the finding"
    assert "reset_epoch" in msg and "stalls" in msg
    assert "~/reset still works" in msg, "do not overstate it either"


def test_the_arity_report_is_the_drain_resets_call_not_a_name_check():
    """Bound against the real call, so ``**kwargs`` passes and a rename does not.

    A backend is free to spell the parameters differently; what decides the
    question is whether ``episode_reset(ids, seeds=..., instances=...)``
    lands.  Reading parameter names instead would flag a working backend and
    clear a broken one.
    """

    class Kwargs(Bare):
        _epoch = 0

        def episode_reset(self, env_ids=None, **kw) -> None:
            type(self)._epoch += 1

        @property
        def reset_epoch(self) -> int:
            return type(self)._epoch

    assert reset_findings(Kwargs()) == []

    class Renamed(Bare):
        def episode_reset(self, worlds=None, seed=None) -> None:
            pass

    (msg,) = reset_findings(Renamed())
    assert "Renamed" in msg


def test_a_reset_without_a_credential_is_named():
    """The write lands; ``reset_epoch:0`` says it did not.

    Reported only once the arity is right — a two-arg backend has a louder
    problem and two findings for one class is how a bring-up log gets skimmed.
    """
    (msg,) = reset_findings(NoCredential())
    assert "NoCredential" in msg
    assert "reset_epoch:0" in msg


def test_a_conformant_backend_produces_no_output():
    """Silence is the pass condition; a report that always fires is ignored."""
    assert audit(Conformant()) == []


def test_a_missing_activity_signal_is_not_a_finding_of_its_own():
    """``RealtimePacer.build`` owns that warning, and only when it matters.

    ``Conformant`` returns ``None`` from ``activity`` and is still reported
    clean.  Warning about it here as well would double every bring-up line for
    one omission, and would fire even at ``step_idle_timeout=0``, where a
    free-running pacer is exactly the intended behaviour.

    The word still appears in the no-``episode_reset`` message above — as part
    of the fix being named, which is the opposite of a duplicate finding.
    """
    sim = Conformant()
    assert sim.activity is None
    assert audit(sim) == []


# ----------------------------------------------------------------------
# The pause axis
# ----------------------------------------------------------------------


def test_a_backend_that_can_stop_and_not_restart_is_named():
    """An asymmetric pair is a defect on any architecture.

    Unlike the both-missing case below, nothing about the backend's internals
    can make this correct: the same two services drive both verbs, so a
    backend implementing one has told us its sim needs telling, and then does
    not get told half the time.
    """

    class StopOnly(Bare):
        def pause(self) -> None:
            pass

    (msg,) = pause_findings(StopOnly())
    assert "StopOnly" in msg
    assert "overrides pause but not resume" in msg
    assert "stopped and never restarted" in msg


def test_a_backend_that_can_restart_and_not_stop_is_named_the_other_way():
    """The consequence clause has to follow which half is missing.

    Reporting "stopped and never restarted" for this backend would send the
    reader looking for the opposite bug.
    """

    class StartOnly(Bare):
        def resume(self) -> None:
            pass

    (msg,) = pause_findings(StartOnly())
    assert "overrides resume but not pause" in msg
    assert "restarted and never stopped" in msg


def test_neither_verb_is_reported_as_a_question_not_a_verdict():
    """Whether this is a bug is not decidable from the class object.

    A backend whose sim only advances inside ``step()`` needs no ``pause`` at
    all — the loop declining to call ``step`` IS the pause — and that is the
    common case, so the line must not read as a defect.  It is still worth
    printing: ``MujocoBackendAdapter`` is this shape and its physics runs on a
    C++ thread, so ``/pause_physics`` stops the publishing and not the world.
    """
    (msg,) = pause_findings(Bare())
    assert "Bare" in msg
    assert "neither pause nor resume" in msg
    # Both halves of the judgement, so the reader can decide which they are.
    assert "Correct IF step() is the only" in msg
    assert "Wrong if anything advances state beside step()" in msg
    assert "frozen /joint_states" in msg, "name the symptom an operator would see"


def test_a_matched_pair_is_silent():
    class Both(Bare):
        def pause(self) -> None:
            pass

        def resume(self) -> None:
            pass

    assert pause_findings(Both()) == []


def test_both_axes_report_independently():
    """``Bare`` is ``MujocoBackendAdapter``'s real shape: it fails both.

    One line per axis, and the reset one first — a dead reset path is worse
    than a pause that may or may not be a problem.  Pinned because the two
    axes were one function with early ``return``s, where the first finding
    suppressed everything after it.
    """
    reset, pause = audit(Bare())
    assert "episode_reset" in reset
    assert "neither pause nor resume" in pause


def test_one_hostile_axis_does_not_hide_the_other():
    """Each axis is guarded separately, and that is the reason why.

    A single ``try`` around both meant a backend with an exploding
    ``episode_reset`` attribute lost its pause findings as collateral.
    """

    class HostileReset(Bare):
        @property
        def episode_reset(self):
            raise RuntimeError("no")

    msgs = audit(HostileReset())
    assert any("skipped" in m for m in msgs), "the failure itself must stay visible"
    assert any("neither pause nor resume" in m for m in msgs), "the other axis still ran"


def test_the_audit_cannot_take_bring_up_down():
    """A diagnostic that aborts the run is worse than the silence it replaces."""

    class Hostile(Bare):
        @property
        def episode_reset(self):  # not callable as a method at all
            raise RuntimeError("no")

    log = Recorder()
    audit_lifecycle(Hostile(), log)  # must not raise
    assert any("skipped" in m for m in log.warnings), "a swallowed failure must still be visible"


def test_the_audit_runs_once_per_session_bring_up():
    """Pinned against ``EngineSession``, not just called directly in tests.

    The call sits next to ``self.sim = sim``, which is the one point every
    backend this session builds passes through exactly once — including the
    downstream backends that subclass ``EngineSession`` to reach it.
    """
    import inspect

    from genie_sim_engine_core.common import session as session_mod

    src = inspect.getsource(session_mod.EngineSession.__init__)
    assert src.count("audit_lifecycle(") == 1
    assert "self.sim = sim" in src

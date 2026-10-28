# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""``PhysicsEngine``'s lifecycle surface: defaults, and honest failures.

That ABC is this tree's lifecycle contract — there is no Protocol mirror of
it (see ``engine/lifecycle.py``'s module docstring for why the mirror was
retired). So the guarantee to pin is the ABC's own: the run loop calls these
verbs straight (``sim.pause()``), not through a ``getattr`` probe, which is
only safe because every one of them has a default here. A backend
implementing nothing but the abstract members must still survive every
lifecycle call the loop can make.

The probe version failed exactly here and silently — ``~/pause`` reached
``getattr(sim, "pause", None)``, came back ``None``, and the loop moved on
without pausing the backend for as long as no backend implemented it.
"""

from __future__ import annotations

import numpy as np
import pytest

from genie_sim_engine_core.engine.base import PhysicsEngine
from genie_sim_engine_core.engine.lifecycle import (
    ActivitySignal,
    EpisodeHandle,
    EpisodeSummary,
    Snapshot,
    TeleportRequest,
)


class MinimalEngine(PhysicsEngine):
    """Implements the ABSTRACT members and nothing else.

    Every lifecycle call below therefore lands on a ``PhysicsEngine``
    default. A backend author writing this much and no more is the case the
    defaults exist for.
    """

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


@pytest.fixture
def engine():
    return MinimalEngine()


# ------------------------------------------------------------------ shape


def test_every_verb_the_loop_dispatches_is_declared_on_the_abc(engine):
    """The loop's verb vocabulary, read off the loop, not off a hand list.

    This replaced an ``isinstance(engine, SimulationControl)`` check against a
    Protocol that mirrored the ABC member for member. That assertion could
    only ever pass — ``@runtime_checkable`` compares member NAMES, and the
    names were copied from the very class being checked — so it proved the
    mirror was still a mirror and nothing about the contract.

    The invariant actually worth holding is the one the loop depends on: it
    calls each verb straight, so a verb it can dispatch that the ABC does not
    declare is an ``AttributeError`` on an operator's keystroke. Parsed out of
    ``process_lifecycle_commands`` so adding a branch there without a default
    here fails, rather than waiting for the branch to be exercised.
    """
    import ast
    import inspect
    import re

    from genie_sim_engine_core.common import loop as loop_mod

    src = inspect.getsource(loop_mod)
    verbs = set(re.findall(r'command == "(\w+)"', src))
    assert verbs, "the dispatch moved; this test is no longer reading it"

    # ``configure`` is the lifecycle_msgs transition, handled by the loop's own
    # paused flag, and the ABC has nothing to do with it. ``reset`` is the only
    # verb whose ROS name is not its method name: the wire says "reset" and the
    # ABC says ``episode_reset``, because a lifecycle reset and a task reset
    # are the same state write with different arguments.
    on_the_abc = {"reset": "episode_reset"}
    expected = {on_the_abc.get(v, v) for v in verbs - {"configure"}}
    for name in sorted(expected):
        assert callable(
            getattr(type(engine), name, None)
        ), f"the loop dispatches a verb bound to {name!r} and the ABC does not declare it"

    # Declared is not enough — it must be CONCRETE. An @abstractmethod here
    # would move the failure to construction for every existing backend at
    # once, which is the reason these are documented defaults in the first
    # place.
    #
    # ``shutdown`` is the one exception and stays abstract on purpose: there is
    # no honest default for "release the resources you acquired", every backend
    # acquires some, and the loop calls it on every exit path including the
    # ``finally``. A no-op default there would leak a solver per run silently.
    tree = ast.parse(inspect.getsource(PhysicsEngine))
    abstract = {
        n.name
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef)
        and any(getattr(d, "id", getattr(d, "attr", "")) == "abstractmethod" for d in n.decorator_list)
    }
    must_default = expected - {"shutdown"}
    assert not (
        must_default & abstract
    ), f"lifecycle verbs need defaults, not @abstractmethod: {must_default & abstract}"
    assert "shutdown" in abstract, "shutdown stopped being abstract; a silent no-op teardown leaks per run"


def test_every_lifecycle_verb_the_loop_can_call_is_callable(engine):
    """The loop's whole verb vocabulary, applied to a do-nothing backend."""
    engine.pause()
    engine.resume()
    engine.set_time_scale(0.5)
    engine.set_seed(7)
    engine.episode_reset()
    engine.episode_reset(None, None, None)
    # Both mappings keyed by ENV ID.  This assertion used to pass
    # ``{"table": 1}`` for instances, which no backend can ever match: the
    # lookup is ``if instances and w in instances`` against an int world
    # index, so a name-keyed dict is a silent no-op and the check proved
    # nothing about the argument it named.
    engine.episode_reset([0], {0: 3}, {0: 1})
    engine.shutdown()


def test_set_time_scale_rejects_a_stopped_or_reversed_clock(engine):
    """Validated on the ABC even though the default does nothing.

    ``0.0`` is not a slow sim, it is a stopped one, and a backend that takes
    it hands back a sim that never advances with nothing to explain why.
    Rejecting on the base class means no backend has to remember to.
    """
    engine.set_time_scale(1.0)
    engine.set_time_scale(0.25)
    engine.set_time_scale(1e6)
    for bad in (0.0, -0.5, -1.0):
        with pytest.raises(ValueError):
            engine.set_time_scale(bad)


def test_the_lifecycle_reset_binds_under_the_upstream_arity_too(engine):
    """``episode_reset()`` is what the loop calls, and why.

    A downstream backend subclasses this ABC while implementing the upstream
    ``episode_reset(env_ids, seed)`` — two positional parameters.
    The loop's ``~/reset`` handler runs inside a ``try`` that has only a
    ``finally``, so a TypeError there is not a degraded reset, it is the
    server exiting under an operator. Pinned by binding the loop's call
    against that arity directly rather than by trusting the comment.
    """
    import inspect

    def upstream(self, env_ids=None, seed=None) -> None:
        """The arity in the consuming tree's SimulationControl."""

    sig = inspect.signature(upstream)
    sig.bind(None)  # episode_reset() — what the loop calls; binds
    with pytest.raises(TypeError):
        sig.bind(None, None, None, None)  # the three-positional form; does not
    # ...and the empty call is a real reset on this tree's arity: every
    # default is the lifecycle meaning already.
    engine.episode_reset()


def test_capability_readbacks_have_documented_defaults(engine):
    assert engine.autostart is True, "the ROS realtime path spins on construction"
    assert engine.num_envs == 1
    assert engine.activity is None, "no signal — the pacer must free-run, not guess"
    # 0, not None: the credential is published unconditionally, and "no reset
    # has completed" is a real answer where "this backend has no epoch" would
    # send the wrapper back to its seq-advance fallback.
    assert engine.reset_epoch == 0
    assert engine.raw_model() is None
    assert engine.raw_data() is None


# --------------------------------------------------------------- episodes


def test_episode_records_admit_they_are_placeholders(engine):
    """Zeros from a backend that does not track episodes must be labelled.

    Otherwise a consumer reads ``ended_at_tick=0`` as a real boundary and
    concludes the episode ran for no ticks, which is indistinguishable from
    a backend that never implemented the call.
    """
    handle = engine.episode_begin(seed=11)
    assert isinstance(handle, EpisodeHandle)
    assert handle.seed == 11
    assert handle.metadata.get("backend_tracks_episodes") is False

    summary = engine.episode_end("timeout")
    assert isinstance(summary, EpisodeSummary)
    assert summary.reason == "timeout"
    assert summary.metadata.get("backend_tracks_episodes") is False


def test_episode_records_are_frozen(engine):
    """A handle is a credential; a consumer must not be able to edit it."""
    handle = engine.episode_begin(seed=1)
    with pytest.raises(Exception):
        handle.episode_id = 99  # type: ignore[misc]


# -------------------------------------------------------------- teleport


def test_teleport_reports_zero_applied_rather_than_pretending(engine):
    """The count is what makes "not supported" visible to the caller.

    A ``None``/no-op return would let the run loop log a batch as landed on
    a backend that cannot move anything.
    """
    req = TeleportRequest(body="cup", position=(0.1, 0.2, 0.3))
    assert engine.teleport([req]) == 0


def test_teleport_request_defaults_match_the_reference_contract():
    req = TeleportRequest(body="cup", position=(0.0, 0.0, 1.0))
    assert req.orientation_wxyz == (1.0, 0.0, 0.0, 0.0), "identity, wxyz order"
    assert req.zero_velocity is True, "a body keeping its velocity drifts off the pose"
    assert req.env_ids is None, "None = every world"


# -------------------------------------------------------------- snapshots


def test_snapshot_raises_instead_of_returning_an_empty_one(engine):
    """An empty Snapshot restores to nothing and diverges silently later.

    Failing at capture time names the backend; failing at replay time names
    nothing.
    """
    with pytest.raises(NotImplementedError, match="MinimalEngine"):
        engine.snapshot()


def test_restore_raises_for_the_same_reason(engine):
    snap = Snapshot(backend="none", backend_version="0", tick=0, payload=b"")
    with pytest.raises(NotImplementedError, match="MinimalEngine"):
        engine.restore(snap)


# ------------------------------------------------------- activity signal


def test_the_ros_reset_coordinator_is_an_activity_signal():
    """``RosResetCoordinator`` is the one implementation core ships.

    The pacer used to read ``sim._idle_activity_ok`` / ``sim._last_cmd_wall``
    off the backend — names that live on THIS object, not on the engine, so
    the lookup silently missed and the idle gate could never fire. Pinning
    the Protocol against the real class is what keeps the two from drifting
    apart again.

    An ``isinstance`` check, not four ``hasattr`` probes: the pacer reaches
    this object through ``PhysicsEngine.activity``, typed as
    ``Optional[ActivitySignal]``, so "does the shipped implementation satisfy
    that annotation" is the actual question. A missing member fails here
    instead of once per tick, silently, in a run whose training curve nobody
    can explain later.
    """
    from genie_sim_engine_core.common.lifecycle_ros import RosResetCoordinator

    class Backend:
        reset_epoch = 0

        def episode_reset(self, env_ids=None, seeds=None, instances=None):
            pass

    assert isinstance(RosResetCoordinator(Backend(), None), ActivitySignal)


def test_activity_signal_is_runtime_checkable():
    """A consumer must be able to ask, not assume.

    ``runtime_checkable`` is what lets the pacer answer "does this backend
    expose a signal at all" with a check instead of four attribute probes
    that each degrade to a default.
    """

    class Signal:
        idle_activity_ok = True
        last_cmd_wall = 12.5
        reset_pending = False
        reset_epoch = 3

    assert isinstance(Signal(), ActivitySignal)
    assert not isinstance(object(), ActivitySignal)

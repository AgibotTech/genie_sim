# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""``/rlinf/reset_envs`` has to actually reach ``episode_reset``.

``RosResetCoordinator`` is the only path from the reset topics to a backend
state write, and every failure it can have is silent by construction:
``drain_resets`` wraps the call in ``except Exception`` and logs a warning,
because one malformed request must not kill the loop thread.  That is the
right shape and it is also why the whole path could sit broken — the
coordinator called ``episode_reset(env_ids, seed=..., instances=[...])``
while every backend in the tree declares ``episode_reset(env_ids, seeds,
instances)`` with mapping arguments, so every reset raised ``TypeError`` on
the keyword name, got swallowed, and the published ``reset_epoch`` never
moved off 0.

So these tests assert on three things a passing call implies and a swallowed
one does not: the recorded arguments, an EMPTY warning log, and an epoch that
advanced.  Checking only the first would pass against a backend stub lax
enough to accept the broken shape.
"""

from __future__ import annotations

import pytest

from genie_sim_engine_core.common.lifecycle_ros import RosResetCoordinator


class RecordingBackend:
    """A backend with the REAL ``episode_reset`` signature and nothing else.

    Deliberately not ``**kwargs``: the point under test is that the
    coordinator uses the parameter names and argument types the backends
    declare, and a permissive stub cannot state that.
    """

    def __init__(self, fail: bool = False) -> None:
        self.calls = []
        self._fail = fail
        self.reset_epoch = 0

    def episode_reset(self, env_ids=None, seeds=None, instances=None) -> None:
        self.calls.append((env_ids, seeds, instances))
        if self._fail:
            raise RuntimeError("solver refused the write")
        # Mirrors the backend contract: bumped only after the write landed.
        self.reset_epoch += 1


class Log:
    def __init__(self) -> None:
        self.warns = []
        self.infos = []

    def warn(self, m):
        self.warns.append(m)

    def info(self, m):
        self.infos.append(m)


@pytest.fixture
def log():
    return Log()


def _coord(log, backend):
    return RosResetCoordinator(backend, log)


# --------------------------------------------------------- the plain topic


def test_a_plain_reset_reaches_the_backend(log):
    """``/rlinf/reset_envs`` with world indices, no seeds."""
    be = RecordingBackend()
    c = _coord(log, be)
    c.push_reset([0, 2])
    c.drain_resets()
    assert be.calls == [([0, 2], None, None)]
    assert log.warns == [], log.warns


def test_an_empty_env_list_means_every_world(log):
    """The topic is an Int32MultiArray; an empty array is its only way to say
    "all worlds", and ``episode_reset``'s spelling of that is ``None``.

    Forwarding ``[]`` verbatim would reset NOTHING — the backend filters
    ``env_ids`` against its world count and returns early on an empty result.
    """
    be = RecordingBackend()
    c = _coord(log, be)
    c.push_reset([])
    c.drain_resets()
    assert be.calls == [(None, None, None)]


def test_nothing_pending_is_a_no_op(log):
    be = RecordingBackend()
    _coord(log, be).drain_resets()
    assert be.calls == []
    assert be.reset_epoch == 0


# -------------------------------------------------------- the seeded topic


def test_the_seeded_triples_arrive_as_per_env_mappings(log):
    """The wire carries flat triples ``[env, seed, inst, ...]``.

    The subscriber slices them into three parallel arrays; the backend wants
    ``{env: value}``, because a batch may name the same world twice and a
    positional array cannot say which entry wins.
    """
    be = RecordingBackend()
    c = _coord(log, be)
    wire = [0, 111, 5, 3, 222, 7]  # env 0 seed 111 inst 5, env 3 seed 222 inst 7
    c.push_reset(wire[0::3], seeds=wire[1::3], instances=wire[2::3])
    c.drain_resets()
    assert be.calls == [([0, 3], {0: 111, 3: 222}, {0: 5, 3: 7})]
    assert log.warns == []


def test_minus_one_is_the_wires_no_op_and_never_reaches_the_backend(log):
    """A seed of -1 would read as "randomize with seed -1" and an instance of
    -1 would index nothing.  Both slots exist only because the triple layout
    forces every field to be present."""
    be = RecordingBackend()
    c = _coord(log, be)
    wire = [0, -1, 4, 1, 900, -1]
    c.push_reset(wire[0::3], seeds=wire[1::3], instances=wire[2::3])
    c.drain_resets()
    assert be.calls == [([0, 1], {1: 900}, {0: 4})]


def test_an_all_noop_column_collapses_to_none(log):
    """``None``, not ``{}``: the backend guards with ``if seeds and w in
    seeds``, and an empty dict is the same answer at more cost — but a caller
    reading the recorded arguments should see "unseeded", not "seeded with
    nothing"."""
    be = RecordingBackend()
    c = _coord(log, be)
    c.push_reset([0, 1], seeds=[-1, -1], instances=[-1, -1])
    c.drain_resets()
    assert be.calls == [([0, 1], None, None)]


# ------------------------------------------------------------ batch shape


def test_each_pushed_batch_gets_its_own_call(log):
    """Two requests, two writes, two epoch bumps.

    Coalescing them into one call would be cheaper and would break the
    credential: a client that asked twice waits for the epoch to advance
    twice, and would hang forever on a single bump.
    """
    be = RecordingBackend()
    c = _coord(log, be)
    c.push_reset([0])
    c.push_reset([1], seeds=[7])
    c.drain_resets()
    assert be.calls == [([0], None, None), ([1], {1: 7}, None)]
    assert c.reset_epoch == 2


def test_the_queue_is_emptied_by_the_drain(log):
    be = RecordingBackend()
    c = _coord(log, be)
    c.push_reset([0])
    c.drain_resets()
    c.drain_resets()
    assert len(be.calls) == 1


# -------------------------------------------------------------- the epoch


def test_the_epoch_is_the_backends_own(log):
    """Read through, not counted here.

    The coordinator is not the only door to ``episode_reset`` — an RL
    consumer holding the engine calls it directly — and a counter kept here
    would answer the credential question differently depending on which door
    the reset came through.
    """
    be = RecordingBackend()
    c = _coord(log, be)
    assert c.reset_epoch == 0
    be.reset_epoch = 9  # a direct episode_reset, bypassing ROS entirely
    assert c.reset_epoch == 9


def test_a_backend_with_no_epoch_reads_as_zero(log):
    class NoEpoch:
        def episode_reset(self, env_ids=None, seeds=None, instances=None):
            pass

    assert _coord(log, NoEpoch()).reset_epoch == 0


def test_a_failed_reset_does_not_advance_the_epoch(log):
    """The credential certifies state.  A reset that raised left state as it
    was, so issuing the credential would tell a waiting wrapper that stale
    state is post-reset."""
    be = RecordingBackend(fail=True)
    c = _coord(log, be)
    c.push_reset([0])
    c.drain_resets()
    assert be.calls == [([0], None, None)]
    assert c.reset_epoch == 0
    assert any("episode_reset failed" in m for m in log.warns)


def test_one_bad_request_does_not_stop_the_next(log):
    """``drain_resets`` runs on the loop thread; a raise that escaped would
    take the simulation down with it."""
    be = RecordingBackend(fail=True)
    c = _coord(log, be)
    c.push_reset([0])
    c.push_reset([1])
    c.drain_resets()
    assert len(be.calls) == 2

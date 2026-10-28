# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""``/joint_states_scene`` sourcing: index slice vs backend direct channel.

Two backend shapes reach this topic and ``_publish_tick`` must route each
one correctly:

* **sliced** — ``sim.joint_names`` spans EVERY articulated DoF (the Isaac
  Sim paths traverse the whole USD stage), so the scene DoFs are already
  in the vector ``get_joint_states()`` returns and a plain index take
  reaches them.  ``EngineSession`` computes those indices by subtracting
  the manifest's ``robot.joint_names`` out of ``sim.joint_names``.

* **direct** — the backend deliberately keeps scene articulations OFF the
  controllable surface (newton-standalone: an entry in
  ``_joint_name_to_dof`` gets a stiff POSITION-mode PD drive from
  ``_configure_pd_drives``, which would motorize every door and drawer),
  so they are absent from ``get_joint_states()`` and NO index can reach
  them.  Such a backend reports them on ``scene_joint_names`` /
  ``get_scene_joint_states`` instead, and the session sets
  ``PublishPlan.scene_joint_direct``.

The regression these guard against is the second shape silently landing on
the first path: the subtraction finds nothing, ``scene_joint_indices`` comes
out empty, and the topic sits quiet forever with no error anywhere.

Which shape is live is a ``PublishPlan`` the caller passes in, not state on
the backend — so each test below builds the plan next to the joint vector it
describes, and a mismatch between the two is visible in one screenful.
"""

from __future__ import annotations

import numpy as np

ROBOT = ["j0", "j1", "j2"]
SCENE = ["door_hinge", "drawer_slide"]


def _sliced(PublishSim, PublishPlan):
    """Isaac-shaped backend: one joint vector spanning robot + scene."""
    sim = PublishSim(
        jpos=[0.1, 0.2, 0.3, 1.5, 0.05],
        jvel=[1.1, 1.2, 1.3, 2.5, 0.06],
        jeff=[10.0, 20.0, 30.0, 40.0, 50.0],
    )
    plan = PublishPlan(
        robot_joint_indices=(0, 1, 2),
        scene_joint_indices=(3, 4),
        scene_joint_names=tuple(SCENE),
    )
    return sim, plan


def _direct(PublishSim, PublishPlan, scene_values=(1.5, 0.05)):
    """Newton-shaped backend: joint vector holds the robot DoFs ONLY."""
    sim = PublishSim(
        jpos=[0.1, 0.2, 0.3],
        jvel=[1.1, 1.2, 1.3],
        jeff=[10.0, 20.0, 30.0],
    )
    plan = PublishPlan(
        robot_joint_indices=(0, 1, 2),
        scene_joint_indices=(),  # nothing to index into — see module docstring
        scene_joint_names=tuple(SCENE),
        scene_joint_direct=True,
    )

    def _get_scene_joint_states(_sim=sim, _vals=scene_values):
        _sim.scene_reads += 1
        if _vals is None:
            return None
        pos = np.asarray(_vals, dtype=np.float64)
        return pos, pos * 10.0, pos * 100.0

    sim.get_scene_joint_states = _get_scene_joint_states
    return sim, plan


# ---------------------------------------------------------------- sliced


def test_sliced_backend_takes_scene_indices(publish_tick, core, PublishSim, PublishPlan):
    sim, plan = _sliced(PublishSim, PublishPlan)
    publish_tick(sim, plan, 4.0, core)

    assert len(core.joint_states) == 1
    _t, rpos, rvel, reff, _fid = core.joint_states[0]
    np.testing.assert_allclose(rpos, [0.1, 0.2, 0.3])
    np.testing.assert_allclose(rvel, [1.1, 1.2, 1.3])
    np.testing.assert_allclose(reff, [10.0, 20.0, 30.0])

    assert len(core.joint_states_scene) == 1
    t, names, spos, svel, seff = core.joint_states_scene[0]
    assert t == 4.0
    assert names == SCENE
    np.testing.assert_allclose(spos, [1.5, 0.05])
    np.testing.assert_allclose(svel, [2.5, 0.06])
    np.testing.assert_allclose(seff, [40.0, 50.0])


def test_sliced_backend_without_scene_indices_stays_quiet(publish_tick, core, PublishSim, PublishPlan):
    """Robot-only scene (or a legacy manifest bake): topic quiet, no error."""
    sim, plan = _sliced(PublishSim, PublishPlan)
    plan = PublishPlan(robot_joint_indices=plan.robot_joint_indices)

    publish_tick(sim, plan, 1.0, core)

    assert len(core.joint_states) == 1
    assert core.joint_states_scene == []


# ---------------------------------------------------------------- direct


def test_direct_backend_publishes_from_its_own_channel(publish_tick, core, PublishSim, PublishPlan):
    sim, plan = _direct(PublishSim, PublishPlan)
    publish_tick(sim, plan, 7.0, core)

    # /joint_states still carries the robot DoFs and nothing else.
    _t, rpos, _rvel, _reff, _fid = core.joint_states[0]
    np.testing.assert_allclose(rpos, [0.1, 0.2, 0.3])

    assert sim.scene_reads == 1, "backend channel must be read exactly once per tick"
    assert len(core.joint_states_scene) == 1
    t, names, spos, svel, seff = core.joint_states_scene[0]
    assert t == 7.0
    assert names == SCENE
    # Values come from the backend channel, NOT from any slice of jpos —
    # jpos is only 3 long here, so a slice could not have produced these.
    np.testing.assert_allclose(spos, [1.5, 0.05])
    np.testing.assert_allclose(svel, [15.0, 0.5])
    np.testing.assert_allclose(seff, [150.0, 5.0])


def test_direct_backend_returning_none_leaves_topic_quiet(publish_tick, core, PublishSim, PublishPlan):
    """No articulated DoFs in this scene — quiet, and distinguishable from a stall."""
    sim, plan = _direct(PublishSim, PublishPlan, scene_values=None)
    publish_tick(sim, plan, 1.0, core)

    assert sim.scene_reads == 1
    assert len(core.joint_states) == 1, "robot channel must keep publishing regardless"
    assert core.joint_states_scene == []


def test_direct_flag_wins_over_stale_scene_indices(publish_tick, core, PublishSim, PublishPlan):
    """The two paths are mutually exclusive, direct first.

    Indices into ``get_joint_states()`` are meaningless for joints that are
    not in it, so if both are somehow set the direct channel must win — and
    the index take must not also fire and double-publish.
    """
    sim, plan = _direct(PublishSim, PublishPlan)
    plan = PublishPlan(
        robot_joint_indices=plan.robot_joint_indices,
        scene_joint_indices=(1, 2),  # would silently publish robot DoFs
        scene_joint_names=plan.scene_joint_names,
        scene_joint_direct=True,
    )

    publish_tick(sim, plan, 1.0, core)

    assert len(core.joint_states_scene) == 1
    _t, _names, spos, _svel, _seff = core.joint_states_scene[0]
    np.testing.assert_allclose(spos, [1.5, 0.05])


def test_pre_fix_shape_is_silent(publish_tick, core, PublishSim, PublishPlan):
    """Regression witness: the newton shape WITHOUT the direct channel.

    This is exactly the state that made ``/joint_states_scene`` permanently
    quiet on the newton backend — the joint vector holds robot DoFs only, so
    the manifest subtraction yields no indices and there is nothing to slice.
    Kept as a test so the failure mode stays visible rather than becoming a
    mystery the next time someone adds a backend.
    """
    sim = PublishSim(jpos=[0.1, 0.2, 0.3], jvel=[1.1, 1.2, 1.3])
    plan = PublishPlan(robot_joint_indices=(0, 1, 2))

    publish_tick(sim, plan, 1.0, core)

    assert len(core.joint_states) == 1
    assert core.joint_states_scene == []


# ---------------------------------------------------------------- legacy


def test_legacy_path_publishes_everything_on_joint_states(publish_tick, core, PublishSim, PublishPlan):
    """No plan at all (EngineRunLoop wired without EngineSession)."""
    sim, _plan = _sliced(PublishSim, PublishPlan)

    publish_tick(sim, PublishPlan.legacy(), 1.0, core)

    _t, pos, _vel, _eff, _fid = core.joint_states[0]
    assert len(pos) == 5
    assert core.joint_states_scene == []


def test_legacy_sentinel_is_not_an_empty_plan(publish_tick, core, PublishSim, PublishPlan):
    """``None`` means unplanned; ``()`` means planned-and-empty.

    Both are falsy, so a truthiness test on ``robot_joint_indices`` would
    conflate them and silently dump the whole joint vector on a run whose
    session decided no robot DoF belongs on the wire.
    """
    sim, _plan = _sliced(PublishSim, PublishPlan)
    plan = PublishPlan(robot_joint_indices=(), scene_joint_indices=(3, 4), scene_joint_names=tuple(SCENE))
    assert plan.planned and not PublishPlan.legacy().planned

    publish_tick(sim, plan, 1.0, core)

    _t, pos, _vel, _eff, _fid = core.joint_states[0]
    assert len(pos) == 0, "planned-and-empty must publish an empty JointState, not everything"
    assert len(core.joint_states_scene) == 1, "the scene half of the plan still runs"


def test_two_tuple_backend_publishes_scene_without_effort(publish_tick, core, PublishSim, PublishPlan):
    """Legacy 2-tuple ``get_joint_states``: effort stays absent on both topics."""
    sim = PublishSim(jpos=[0.1, 0.2, 0.3, 1.5, 0.05], jvel=[1.1, 1.2, 1.3, 2.5, 0.06])
    plan = PublishPlan(
        robot_joint_indices=(0, 1, 2),
        scene_joint_indices=(3, 4),
        scene_joint_names=tuple(SCENE),
    )

    publish_tick(sim, plan, 1.0, core)

    assert core.joint_states[0][3] is None
    assert core.joint_states_scene[0][4] is None

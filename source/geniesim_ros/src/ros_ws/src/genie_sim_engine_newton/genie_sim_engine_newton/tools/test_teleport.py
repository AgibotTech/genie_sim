#!/usr/bin/env python3

# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""``teleport`` on newton-standalone — the out-of-band pose write.

``SimulationControl.teleport`` moves a FREE-joint body to an explicit pose
between frames, legally while the sim is paused.  Four things can go wrong
silently, and each has a section below:

**Name resolution.**  Three dialects already exist in this tree for the same
body: the full Newton body label (what ``/tf_scene`` publishes), the
leaf-PARENT segment (what the authored instance-pose table keys on, see
``_apply_instance_poses``), and a trailing path segment.  All three must
resolve, and none of them may be a substring match — ``cup`` matching
``cupboard`` would teleport the furniture.

**Quaternion convention.**  Three conventions meet here.  The ROS wire is
xyzw, the Python layer is wxyz (canonical), and Newton's free-joint ``q``
slots are xyzw, Warp-native.  Exactly one reorder happens at each boundary;
this file pins the second one.  A wrong reorder is a valid unit quaternion,
so nothing raises — the object simply lands rotated.

**q-index vs qd-index.**  A FREE joint owns 7 ``joint_q`` slots and 6
``joint_qd`` slots, so the two vectors desynchronise behind every free body.
Zeroing velocity at ``q_start`` instead of ``qd_start`` corrupts a
neighbouring DoF and is invisible on a single-body scene.

**World fan-out.**  ``replicate`` does NOT rename bodies, so all worlds carry
identical labels and world membership is recoverable only from the body
INDEX.  A bare name therefore addresses every world; ``env_<w>/<name>``
addresses one.

Everything runs on the duck-typed fake arrays from
``test_scene_joint_channel.py`` — ``JointIndex`` only calls ``.numpy()`` /
``.size``, so no solver is built and no GPU is touched.  The commit path
(``_commit_state_write``, which needs a real solver and Warp streams) is
replaced by a recorder: what this file asserts on is the host arrays handed
to it, which is exactly the boundary between "computed the write" and
"landed the write".
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

_HERE = pathlib.Path(__file__).resolve()
_NEWTON_SRC_ROOT = _HERE.parents[2]  # dir containing the newton package
_SRC = _HERE.parents[3]  # ros_ws/.../src (holds both packages)
_CORE_SRC_ROOT = _SRC / "genie_sim_engine_core"

for _p in (str(_NEWTON_SRC_ROOT), str(_CORE_SRC_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, str(_p))

pytest.importorskip("warp", reason="the sibling fakes' module imports warp at module scope")

from genie_sim_engine_core.engine.lifecycle import TeleportRequest  # noqa: E402
from genie_sim_engine_newton.engine.newton.joint_index import (  # noqa: E402
    JT_FREE,
    JT_REVOLUTE,
    JointIndex,
)
from genie_sim_engine_newton.engine.newton.lifecycle import (  # noqa: E402
    _body_name_matches,
    _LifecycleMixin,
)

# The duck-typed warp-array stand-in, shared with the scene-channel tests
# rather than re-declared: one fake model shape for the whole newton tools
# suite means a change to it cannot leave half the tests describing a model
# Newton no longer produces.
from genie_sim_engine_newton.tools.test_scene_joint_channel import _Arr  # noqa: E402

_CUP = "/World/Objects/cup_000/body"
_CUPBOARD = "/World/Objects/cupboard_000/body"
_DOOR_FRAME = "/World/Objects/benchmark_door_000/body"
_DOOR_PANEL = "/World/Objects/benchmark_door_000/part_00"


# ======================================================================
# Fakes
# ======================================================================
class _J:
    """One authored joint.  ``child``/``parent`` are PER-ENV body indices.

    Indices rather than labels, unlike the scene-channel fakes: worlds share
    body labels here (that is the real ``replicate`` shape), so a label could
    not identify which world's body a joint belongs to — which is the very
    ambiguity ``_world_free_joints`` has to resolve.
    """

    def __init__(self, name, jtype, child, parent=None) -> None:
        self.name = name
        self.jtype = jtype
        self.child = child
        self.parent = parent

    @property
    def q_count(self):
        return 7 if self.jtype == JT_FREE else 1

    @property
    def dof_count(self):
        return 6 if self.jtype == JT_FREE else 1


class _ReplicatedModel:
    """World-major replica of one env's joint/body layout.

    q_start and qd_start accumulate independently, as Newton does, so a FREE
    joint genuinely desynchronises the two vectors for everything behind it.
    """

    def __init__(self, joints, body_labels, num_envs=1) -> None:
        nb = len(body_labels)
        self.body_label = list(body_labels) * num_envs
        self.joint_label = []
        types, q_starts, qd_starts, dofs, children, parents = [], [], [], [], [], []
        q = qd = 0
        for w in range(num_envs):
            for j in joints:
                self.joint_label.append(f"/World/Physics/env_{w}/{j.name}")
                types.append(j.jtype)
                q_starts.append(q)
                qd_starts.append(qd)
                q += j.q_count
                qd += j.dof_count
                dofs.append(j.dof_count)
                children.append(w * nb + j.child)
                parents.append(-1 if j.parent is None else w * nb + j.parent)
        self.joint_type = _Arr(types)
        self.joint_q_start = _Arr(q_starts)
        self.joint_qd_start = _Arr(qd_starts)
        self.joint_dof_dim = _Arr(dofs)
        self.joint_child = _Arr(children)
        self.joint_parent = _Arr(parents)
        self.joint_q = _Arr(np.zeros(q), dtype=np.float64)
        self.joint_qd = _Arr(np.zeros(qd), dtype=np.float64)
        self.n_q, self.n_qd = q, qd


class _FakeState:
    def __init__(self, n_q, n_qd) -> None:
        self.joint_q = _Arr(np.zeros(n_q), dtype=np.float64)
        self.joint_qd = _Arr(np.zeros(n_qd), dtype=np.float64)


class _RecLog:
    def __init__(self) -> None:
        self.infos = []
        self.warns = []

    def info(self, m):
        self.infos.append(str(m))

    def warn(self, m):
        self.warns.append(str(m))

    def error(self, m):
        self.warns.append(str(m))


class _Host(_LifecycleMixin):
    """The mixin under test with only the attributes ``teleport`` reads.

    ``_commit_state_write`` is overridden to RECORD instead of committing: the
    real one calls ``newton.eval_fk``, pushes into the solver's own data and
    flushes Warp mirrors, none of which exists without a built model.  The
    recording boundary is deliberate — everything above it is arithmetic this
    file can check, everything below it needs a GPU.
    """

    def __init__(self, joints, body_labels, num_envs=1) -> None:
        self._model = _ReplicatedModel(joints, body_labels, num_envs)
        self._num_envs = num_envs
        self._jindex = JointIndex(self._model)
        self._logger = _RecLog()
        self._state_0 = _FakeState(self._model.n_q, self._model.n_qd)
        self._state_1 = None
        self._control = None
        self._adapter = None
        self._reset_epoch = 0
        self.commits = []

    def _commit_state_write(self, env_ids, jq, qd, tgt=None, base_mocap_off=None, label="reset"):
        self.commits.append(
            {
                "env_ids": list(env_ids),
                "jq": None if jq is None else np.array(jq, copy=True),
                "qd": None if qd is None else np.array(qd, copy=True),
                "tgt": tgt,
                "label": label,
            }
        )


def _cup_scene(num_envs=1):
    """One free cup, one hinged door panel, per world."""
    return _Host(
        [
            _J("cup_free", JT_FREE, child=1),
            _J("door_hinge", JT_REVOLUTE, child=3, parent=2),
        ],
        [_CUPBOARD, _CUP, _DOOR_FRAME, _DOOR_PANEL],
        num_envs=num_envs,
    )


def _req(body, pos=(0.3, -0.2, 0.9), quat=(1.0, 0.0, 0.0, 0.0), **kw):
    return TeleportRequest(body=body, position=pos, orientation_wxyz=quat, **kw)


# ======================================================================
# Name resolution
# ======================================================================
@pytest.mark.parametrize(
    "name",
    [
        _CUP,  # full body label, as /tf_scene publishes it
        "cup_000",  # leaf-PARENT segment, as the instance-pose table keys it
        "body",  # trailing segment
        "cup_000/body",  # multi-segment suffix
    ],
)
def test_every_accepted_name_form_resolves(name):
    host = _cup_scene()
    assert host.teleport([_req(name)]) == 1
    assert len(host.commits) == 1


def test_matching_is_not_a_substring_test():
    """``cup`` must not reach ``cupboard``.

    Both are in this scene, and only the cup has a free joint — a substring
    match would still resolve, just to the wrong body's index in a scene where
    the cupboard floats too.
    """
    assert _body_name_matches(_CUP, "cup_000")
    assert not _body_name_matches(_CUPBOARD, "cup_000")
    assert not _body_name_matches(_CUP, "cup")
    host = _cup_scene()
    assert host.teleport([_req("cup")]) == 0


def test_an_unknown_body_warns_and_lists_the_candidates():
    """A miss is warned, never raised, and the warning names what IS
    teleportable: the overwhelmingly common cause is a body that exists but is
    welded or hinged, and a bare "not found" sends the reader hunting a typo.
    """
    host = _cup_scene()
    assert host.teleport([_req("no_such_body")]) == 0
    assert host.commits == [], "nothing matched, so nothing may be committed"
    assert len(host._logger.warns) == 1
    assert "no_such_body" in host._logger.warns[0]
    assert _CUP in host._logger.warns[0]


def test_a_hinged_body_is_rejected_not_approximated():
    """The door PANEL has a revolute joint, not a free one.

    Writing a 6-DoF pose for it could only mean solving IK for whatever chain
    holds it; the reference (``ApplyTeleportTransform``) refuses, and so does
    this.  Rejection has to be visible, hence the warning.
    """
    host = _cup_scene()
    assert host.teleport([_req("benchmark_door_000")]) == 0
    assert host.commits == []
    assert host._logger.warns


def test_one_bad_name_does_not_lose_the_rest_of_the_batch():
    host = _cup_scene()
    assert host.teleport([_req("no_such_body"), _req("cup_000")]) == 1
    assert len(host.commits) == 1, "the good half of the batch must still land"
    assert host._logger.warns


# ======================================================================
# The write itself
# ======================================================================
def test_position_and_quaternion_land_in_newtons_xyzw_slots():
    host = _cup_scene()
    # A quaternion with four DISTINCT components: (1,0,0,0) or any pair of
    # equal components would pass under a wrong reorder.
    quat_wxyz = (0.1, 0.2, 0.3, 0.4)
    assert host.teleport([_req("cup_000", pos=(1.5, -2.5, 3.5), quat=quat_wxyz)]) == 1
    jq = host.commits[0]["jq"]
    slc = next(s for s in host._jindex.slices() if s.joint_type == JT_FREE)
    q0 = slc.q_start
    assert list(jq[q0 : q0 + 3]) == [1.5, -2.5, 3.5]
    # wxyz in, xyzw out — Warp-native free-joint layout.
    assert list(jq[q0 + 3 : q0 + 7]) == [0.2, 0.3, 0.4, 0.1]


def test_velocity_is_zeroed_over_the_qd_slice_only():
    """Zeroing at ``q_start`` would corrupt a neighbouring DoF.

    The hinge sits behind the cup's FREE joint, so its ``qd_start`` (6) is one
    below its ``q_start`` (7).  A 6-wide zero written at q_start would erase
    the hinge's velocity and leave the cup's last angular DoF spinning.
    """
    host = _cup_scene()
    host._state_0.joint_qd.numpy()[:] = 7.0
    assert host.teleport([_req("cup_000")]) == 1
    qd = host.commits[0]["qd"]
    slc = next(s for s in host._jindex.slices() if s.joint_type == JT_FREE)
    hinge = next(s for s in host._jindex.slices() if s.joint_type == JT_REVOLUTE)
    # The desync this test needs is real, not assumed: without it a q_start
    # write would land on the right slots by accident.
    assert (hinge.q_start, hinge.qd_start) == (7, 6)
    assert list(qd[slc.qd_start : slc.qd_start + slc.dof_count]) == [0.0] * 6
    assert qd[hinge.qd_start] == 7.0, "the hinge's velocity was collateral damage"


def test_zero_velocity_false_leaves_velocity_alone():
    """Opt-out exists, but the default is on: a body that keeps its
    pre-teleport velocity leaves the requested pose on the very next step."""
    host = _cup_scene()
    host._state_0.joint_qd.numpy()[:] = 7.0
    assert host.teleport([_req("cup_000", zero_velocity=False)]) == 1
    qd = host.commits[0]["qd"]
    slc = next(s for s in host._jindex.slices() if s.joint_type == JT_FREE)
    assert list(qd[slc.qd_start : slc.qd_start + 6]) == [7.0] * 6


def test_the_pd_targets_are_not_touched():
    """A free body has no PD target, and rewriting the robot's targets here
    would silently undo whatever command is in flight."""
    host = _cup_scene()
    host.teleport([_req("cup_000")])
    assert host.commits[0]["tgt"] is None


def test_the_shared_commit_path_is_what_lands_the_write():
    """Teleport goes through ``_commit_state_write``, same as ``episode_reset``.

    That helper's last step flushes the async host mirrors the publishers read
    — and only ``step()`` refreshes those otherwise.  Since a teleport applies
    while PAUSED by contract, skipping the helper would publish the pose the
    teleport just replaced, every single time.
    """
    host = _cup_scene()
    host.teleport([_req("cup_000")])
    assert len(host.commits) == 1, "one commit per batch, not one per body"
    assert host.commits[0]["label"] == "teleport"


def test_teleport_does_not_bump_reset_epoch():
    """The epoch is the credential an RL wrapper waits on to know its RESET
    landed.  A teleport is not an episode boundary; issuing one here would
    tell the wrapper an episode restarted when none did."""
    host = _cup_scene()
    host.teleport([_req("cup_000")])
    assert host._reset_epoch == 0


def test_nothing_is_committed_when_the_batch_is_empty():
    host = _cup_scene()
    assert host.teleport([]) == 0
    assert host.teleport(None) == 0
    assert host.commits == []


def test_no_live_state_yet_is_a_warning_not_a_crash():
    """Teleports can arrive before the first build completes."""
    host = _cup_scene()
    host._state_0 = None
    assert host.teleport([_req("cup_000")]) == 0
    assert host.commits == []
    assert host._logger.warns


# ======================================================================
# World fan-out
# ======================================================================
def test_a_bare_name_fans_out_to_every_world():
    """What a batched RL caller means by "teleport the mug"."""
    host = _cup_scene(num_envs=3)
    assert host.teleport([_req("cup_000", pos=(0.5, 0.5, 0.5))]) == 1, "one REQUEST, applied once"
    assert host.commits[0]["env_ids"] == [0, 1, 2]
    jq = host.commits[0]["jq"]
    frees = [s for s in host._jindex.slices() if s.joint_type == JT_FREE]
    assert len(frees) == 3
    for slc in frees:
        assert list(jq[slc.q_start : slc.q_start + 3]) == [0.5, 0.5, 0.5]


def test_env_prefix_addresses_exactly_one_world():
    host = _cup_scene(num_envs=3)
    assert host.teleport([_req("env_1/cup_000", pos=(0.5, 0.5, 0.5))]) == 1
    assert host.commits[0]["env_ids"] == [1]
    jq = host.commits[0]["jq"]
    frees = sorted((s for s in host._jindex.slices() if s.joint_type == JT_FREE), key=lambda s: s.q_start)
    assert list(jq[frees[1].q_start : frees[1].q_start + 3]) == [0.5, 0.5, 0.5]
    for other in (frees[0], frees[2]):
        assert list(jq[other.q_start : other.q_start + 3]) == [0.0, 0.0, 0.0]


def test_env_ids_narrows_the_request():
    host = _cup_scene(num_envs=4)
    assert host.teleport([_req("cup_000", env_ids=(1, 3))]) == 1
    assert host.commits[0]["env_ids"] == [1, 3]


def test_an_out_of_range_world_is_dropped_not_indexed():
    """A stale env id from a client that outlived a rebuild must not reach an
    array index."""
    host = _cup_scene(num_envs=2)
    assert host.teleport([_req("cup_000", env_ids=(7,))]) == 0
    assert host.commits == []


def test_a_body_genuinely_named_env_something_is_not_mistaken_for_a_prefix():
    """``env_`` is only a world prefix when a slash and a valid index follow."""
    host = _Host(
        [_J("env_marker_free", JT_FREE, child=0)],
        ["/World/Objects/env_marker/body"],
        num_envs=1,
    )
    assert host.teleport([_req("env_marker")]) == 1
    assert host.commits[0]["env_ids"] == [0]

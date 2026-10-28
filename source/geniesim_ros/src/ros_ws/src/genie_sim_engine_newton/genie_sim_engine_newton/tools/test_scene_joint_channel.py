#!/usr/bin/env python3

# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""The ``/joint_states_scene`` backend channel on newton-standalone.

``test_scene_articulation.py`` locks down the OWNERSHIP predicate — which
joints ``_build_joint_map`` keeps off the controllable surface.  This file
locks down what happens to them afterwards: they land on a parallel
channel (``_scene_joint_names`` + aligned q/qd index arrays) that
``get_scene_joint_states`` reads and ``EngineSession`` routes onto
``/joint_states_scene``.

Why the parallel channel and not just leaving them in the joint map: an
entry in ``_joint_name_to_dof`` is an entry on the CONTROLLABLE surface,
and ``_configure_pd_drives`` forces every entry there into POSITION mode
with ke=1000 / kd=100.  Merging a door hinge in would come up motorized.

The load-bearing subtlety here is q-index vs qd-index.  ``joint_q`` gives a
FREE joint 7 slots (xyz + quat) while ``joint_qd`` gives it 6, so a single
free-floating scene body ahead of a hinge shifts every later ``q_start``
one past its ``qd_start``.  Positions must be read with the q index and
velocities/efforts with the qd index; mixing them is silent on a
robot-only scene and reads a neighbouring DoF as soon as anything floats.

Everything here runs on a duck-typed fake model — ``JointIndex`` only ever
calls ``.numpy()`` / ``.size`` on the joint arrays, so no solver is built
and no GPU is touched.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

_HERE = pathlib.Path(__file__).resolve()
# .../genie_sim_engine_newton/genie_sim_engine_newton/tools/<this>
_NEWTON_SRC_ROOT = _HERE.parents[2]  # dir containing the newton package
_SRC = _HERE.parents[3]  # ros_ws/.../src (holds both packages)
_CORE_SRC_ROOT = _SRC / "genie_sim_engine_core"

for _p in (str(_NEWTON_SRC_ROOT), str(_CORE_SRC_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

pytest.importorskip("warp", reason="topology.py / state.py import warp at module scope")

from genie_sim_engine_newton.engine.newton.joint_index import (  # noqa: E402
    JT_FIXED,
    JT_FREE,
    JT_PRISMATIC,
    JT_REVOLUTE,
)
from genie_sim_engine_newton.engine.newton.state import _StateMixin  # noqa: E402
from genie_sim_engine_newton.engine.newton.topology import _TopologyMixin  # noqa: E402

_ROBOT_PREFIX = "genie"
_ROBOT_BASE = "/genie/base_link"
_ROBOT_L1 = "/genie/arm_l_link1"
_ROBOT_L2 = "/genie/arm_l_link2"
_DOOR_FRAME = "/World/Objects/benchmark_door_000/body"
_DOOR_PANEL = "/World/Objects/benchmark_door_000/part_00"
_CUP = "/World/Objects/cup_000/body"


# ======================================================================
# Duck-typed fakes.  JointIndex reads .numpy()/.size; _build_joint_map
# additionally reads model.joint_parent / joint_child / body_label.
# ======================================================================
class _Arr:
    """Minimal stand-in for a ``warp.array``."""

    def __init__(self, data, dtype=np.int32) -> None:
        self._a = np.asarray(data, dtype=dtype)

    def numpy(self):
        return self._a

    @property
    def size(self):
        return int(self._a.size)

    def __len__(self):
        return int(self._a.shape[0])


class _Joint:
    """One authored joint in the fake model."""

    def __init__(self, name, jtype, parent, child) -> None:
        self.name = name
        self.jtype = jtype
        self.parent = parent  # body label, or None for world
        self.child = child

    @property
    def q_count(self):
        return {JT_FREE: 7, JT_FIXED: 0}.get(self.jtype, 1)

    @property
    def dof_count(self):
        return {JT_FREE: 6, JT_FIXED: 0}.get(self.jtype, 1)


class _FakeModel:
    """Assembles the ``joint_*`` / ``body_*`` arrays from a joint list.

    q_start and qd_start are accumulated independently, exactly as Newton
    does, so a FREE joint in the list genuinely desynchronises the two for
    every joint after it — which is the whole point of testing this.
    """

    def __init__(self, joints, body_labels) -> None:
        self.body_label = list(body_labels)
        _bidx = {lbl: i for i, lbl in enumerate(self.body_label)}
        self.joint_label = [f"/World/Physics/{j.name}" for j in joints]
        self.joint_type = _Arr([j.jtype for j in joints])
        q_starts, qd_starts, q, qd = [], [], 0, 0
        for j in joints:
            q_starts.append(q)
            qd_starts.append(qd)
            q += j.q_count
            qd += j.dof_count
        self.joint_q_start = _Arr(q_starts)
        self.joint_qd_start = _Arr(qd_starts)
        self.joint_dof_dim = _Arr([j.dof_count for j in joints])
        self.joint_parent = _Arr([_bidx.get(j.parent, -1) for j in joints])
        self.joint_child = _Arr([_bidx.get(j.child, -1) for j in joints])
        # Values are never read by JointIndex — only .size, for the last
        # joint's slice extent.
        self.joint_q = _Arr(np.zeros(q), dtype=np.float64)
        self.joint_qd = _Arr(np.zeros(qd), dtype=np.float64)
        self.n_q, self.n_qd = q, qd


class _NullLog:
    def info(self, *a, **kw):
        pass

    def warn(self, *a, **kw):
        pass

    def error(self, *a, **kw):
        pass


class _Host(_TopologyMixin, _StateMixin):
    """The two mixins under test, with only the attributes they read.

    Mirrors the real ``_NewtonStandaloneBase`` attribute surface without
    the other ~13 mixins or any solver.
    """

    def __init__(self, model, robot_prefix=_ROBOT_PREFIX, num_envs=1) -> None:
        self._model = model
        self._robot_prefix_str = robot_prefix
        self._num_envs = num_envs
        self._logger = _NullLog()
        self._static_joint_q = {}
        self._state_0 = None


def _build(joints, body_labels, **kw):
    host = _Host(_FakeModel(joints, body_labels), **kw)
    host._build_joint_map()
    return host


# ======================================================================
# Robot-only scene — the channel must stay empty and quiet.
# ======================================================================
def test_robot_only_scene_has_no_scene_channel():
    host = _build(
        [
            _Joint("arm_l_joint1", JT_REVOLUTE, _ROBOT_BASE, _ROBOT_L1),
            _Joint("arm_l_joint2", JT_REVOLUTE, _ROBOT_L1, _ROBOT_L2),
        ],
        [_ROBOT_BASE, _ROBOT_L1, _ROBOT_L2],
    )
    assert host._joint_names == ["arm_l_joint1", "arm_l_joint2"]
    assert host._scene_joint_names == []
    assert host._scene_joint_q_idx == []
    assert host._scene_joint_dof_idx == []
    # ``None``, not empty arrays: the publish loop reads that as "nothing to
    # publish" and leaves the topic quiet.
    assert host.get_scene_joint_states() is None


# ======================================================================
# A door hinge: on the scene channel, off the controllable surface.
# ======================================================================
def test_scene_hinge_lands_on_scene_channel_only():
    host = _build(
        [
            _Joint("arm_l_joint1", JT_REVOLUTE, _ROBOT_BASE, _ROBOT_L1),
            _Joint("door_hinge_left", JT_REVOLUTE, _DOOR_FRAME, _DOOR_PANEL),
            _Joint("drawer_slide", JT_PRISMATIC, _DOOR_FRAME, _DOOR_PANEL),
        ],
        [_ROBOT_BASE, _ROBOT_L1, _DOOR_FRAME, _DOOR_PANEL],
    )
    assert host._scene_joint_names == ["door_hinge_left", "drawer_slide"]
    # Off /joint_states and off /joint_command in one assertion each — the
    # PD-drive forcing in ``_configure_pd_drives`` walks
    # ``_joint_name_to_dof``, so absence there is what keeps the door limp.
    assert host._joint_names == ["arm_l_joint1"]
    for n in host._scene_joint_names:
        assert n not in host._joint_name_to_dof
        assert n not in host._joint_name_to_q_idx
    # Disjointness is the invariant ``EngineSession`` relies on to use this
    # channel directly instead of subtracting the manifest's robot set.
    assert not set(host._scene_joint_names) & set(host._joint_names)


def test_free_floating_scene_body_stays_off_the_channel():
    """A loose cup's 6-DoF FREE joint is scene-owned but is NOT reported here.

    The FREE skip runs ahead of the ownership test, deliberately: a free
    body has no articulated DoF to report.  Its q-slice is 7 wide (xyz +
    quaternion) against a 6-wide qd-slice, so there is no scalar
    ``JointState.position`` that means anything for it — and its pose is
    already on ``/tf_scene`` as a body transform, which is the channel a
    6-DoF pose belongs on.  Only hinges and slides reach this one.
    """
    host = _build(
        [
            _Joint("root_joint", JT_FREE, None, _ROBOT_BASE),
            _Joint("cup_free", JT_FREE, None, _CUP),
        ],
        [_ROBOT_BASE, _CUP],
    )
    assert host._scene_joint_names == []
    assert host._joint_names == []


def test_name_mangled_scene_joint_is_not_dropped_as_noise():
    """``tn__`` scene joints reach the channel.

    Newton's USD parser mangles a joint name that starts with a digit into
    ``tn__…``, which the auto-name filter drops as importer noise.  The
    ownership test runs FIRST precisely so a mangled SCENE joint still gets
    reported — an articulated prop is exactly what the eval checkers watch.
    A mangled ROBOT joint is still dropped.
    """
    host = _build(
        [
            _Joint("tn__00_door_hinge", JT_REVOLUTE, _DOOR_FRAME, _DOOR_PANEL),
            _Joint("tn__weld_0", JT_REVOLUTE, _ROBOT_BASE, _ROBOT_L1),
        ],
        [_ROBOT_BASE, _ROBOT_L1, _DOOR_FRAME, _DOOR_PANEL],
    )
    assert host._scene_joint_names == ["tn__00_door_hinge"]
    assert host._joint_names == []


def test_empty_robot_prefix_disables_the_filter():
    """No prefix → no ownership information → nothing is called scene-owned,
    and the joints stay on the robot surface exactly as before Task 2."""
    host = _build(
        [_Joint("door_hinge_left", JT_REVOLUTE, _DOOR_FRAME, _DOOR_PANEL)],
        [_DOOR_FRAME, _DOOR_PANEL],
        robot_prefix="",
    )
    assert host._scene_joint_names == []
    assert host._joint_names == ["door_hinge_left"]


# ======================================================================
# q-index vs qd-index — the silent-until-something-floats bug.
# ======================================================================
def test_scene_indices_track_q_and_qd_separately():
    host = _build(
        [
            _Joint("cup_free", JT_FREE, None, _CUP),  # q:0-6  qd:0-5
            _Joint("door_hinge_left", JT_REVOLUTE, _DOOR_FRAME, _DOOR_PANEL),  # q:7 qd:6
            _Joint("weld", JT_FIXED, _DOOR_FRAME, _DOOR_PANEL),  # 0 wide
            _Joint("drawer_slide", JT_PRISMATIC, _DOOR_FRAME, _CUP),  # q:8 qd:7
        ],
        [_ROBOT_BASE, _DOOR_FRAME, _DOOR_PANEL, _CUP],
    )
    # ``cup_free`` is not on the channel (see above) but it is still IN the
    # model, so it still spends 7 q slots against 6 qd slots — every scene
    # joint behind it now has ``q_start == qd_start + 1``.
    assert host._scene_joint_names == ["door_hinge_left", "drawer_slide"]
    assert host._scene_joint_q_idx == [7, 8]
    assert host._scene_joint_dof_idx == [6, 7]
    # If the two were ever collapsed to one array this assertion is the
    # tripwire — everything after the FREE joint is off by one.
    assert host._scene_joint_q_idx != host._scene_joint_dof_idx


# ======================================================================
# get_scene_joint_states — the index walk.
# ======================================================================
class _NoMirror:
    """Ping-pong mirror in its warmup state: ``read()`` returns None and
    the caller falls back to a synchronous ``.numpy()``."""

    ok = False

    def read(self):
        return None


class _Vec:
    def __init__(self, data) -> None:
        self._a = np.asarray(data, dtype=np.float64)

    def numpy(self):
        return self._a


class _FakeState:
    def __init__(self, q, qd, f=None) -> None:
        self.joint_q = _Vec(q)
        self.joint_qd = _Vec(qd)
        self.joint_f = None if f is None else _Vec(f)


def _hinge_host(**kw):
    host = _build(
        [
            _Joint("cup_free", JT_FREE, None, _CUP),
            _Joint("door_hinge_left", JT_REVOLUTE, _DOOR_FRAME, _DOOR_PANEL),
            _Joint("drawer_slide", JT_PRISMATIC, _DOOR_FRAME, _CUP),
        ],
        [_ROBOT_BASE, _DOOR_FRAME, _DOOR_PANEL, _CUP],
        **kw,
    )
    host._jq_mirror = _NoMirror()
    host._jqd_mirror = _NoMirror()
    return host


def test_reads_positions_by_q_and_velocities_by_qd():
    host = _hinge_host()
    # q is 9 wide (7 free + 1 + 1), qd is 8 (6 + 1 + 1).  Distinct values
    # everywhere so a q/qd mix-up cannot coincidentally pass.
    host._state_0 = _FakeState(
        q=[0, 0, 0, 1, 0, 0, 0, 1.25, 0.06],
        qd=[0, 0, 0, 0, 0, 0, 3.5, 0.7],
        f=[0, 0, 0, 0, 0, 0, -12.0, 4.0],
    )
    pos, vel, eff = host.get_scene_joint_states()

    assert len(pos) == len(vel) == len(eff) == 2
    np.testing.assert_allclose(pos, [1.25, 0.06])  # q[7], q[8]
    np.testing.assert_allclose(vel, [3.5, 0.7])  # qd[6], qd[7]
    np.testing.assert_allclose(eff, [-12.0, 4.0])  # joint_f is qd-indexed too


def test_missing_joint_f_yields_zero_efforts_not_a_crash():
    """Solvers that do not populate ``joint_f`` still get a valid message —
    zeros are the honest reading for "no constraint torque reported"."""
    host = _hinge_host()
    host._state_0 = _FakeState(q=np.arange(9, dtype=np.float64), qd=np.arange(8, dtype=np.float64), f=None)
    pos, vel, eff = host.get_scene_joint_states()
    np.testing.assert_allclose(pos, [7.0, 8.0])
    np.testing.assert_allclose(vel, [6.0, 7.0])
    np.testing.assert_allclose(eff, [0.0, 0.0])


def test_no_state_yet_returns_zeros_of_the_right_length():
    """Called before the solver's first state exists: length must already
    match ``scene_joint_names`` so the C++ length check never trips."""
    host = _hinge_host()
    host._state_0 = None
    pos, vel, eff = host.get_scene_joint_states()
    assert len(pos) == len(vel) == len(eff) == 2
    assert not pos.any() and not vel.any() and not eff.any()


def test_mirror_read_is_preferred_over_a_sync_readback():
    """Steady state: the mirrors ``get_joint_states`` already refreshed this
    tick are reused, so the extra channel costs no extra GPU sync."""

    class _Mirror:
        ok = True

        def __init__(self, data) -> None:
            self._a = np.asarray(data, dtype=np.float64)
            self.reads = 0

        def read(self):
            self.reads += 1
            return self._a

    host = _hinge_host()
    host._jq_mirror = _Mirror([0, 0, 0, 1, 0, 0, 0, 9.9, 8.8])
    host._jqd_mirror = _Mirror([0, 0, 0, 0, 0, 0, 7.7, 6.6])
    # Values here differ from the mirrors' so a fallback read is detectable.
    host._state_0 = _FakeState(q=np.zeros(9), qd=np.zeros(8))

    pos, vel, _eff = host.get_scene_joint_states()
    np.testing.assert_allclose(pos, [9.9, 8.8])
    np.testing.assert_allclose(vel, [7.7, 6.6])
    assert host._jq_mirror.reads == 1 and host._jqd_mirror.reads == 1


def test_bad_index_is_clamped_rather_than_raising():
    """A shorter-than-expected state vector must degrade to zeros, not throw
    out of the publish path and take /joint_states down with it."""
    host = _hinge_host()
    host._state_0 = _FakeState(q=[0.5], qd=[0.25])
    pos, vel, _eff = host.get_scene_joint_states()
    np.testing.assert_allclose(pos, [0.0, 0.0])
    np.testing.assert_allclose(vel, [0.0, 0.0])


# ======================================================================
# Multi-env — same world-major replicate grouping as the robot joints.
# ======================================================================
def test_multi_env_prefixes_scene_names_per_world():
    joints = []
    labels = [_ROBOT_BASE]
    for w in range(2):
        frame, panel = f"/World/env_{w}/door/body", f"/World/env_{w}/door/part_00"
        labels += [frame, panel]
    for w in range(2):
        frame, panel = f"/World/env_{w}/door/body", f"/World/env_{w}/door/part_00"
        joints.append(_Joint("door_hinge_left", JT_REVOLUTE, frame, panel))
    host = _build(joints, labels, num_envs=2)
    assert host._scene_joint_names == ["env_0/door_hinge_left", "env_1/door_hinge_left"]
    assert host._scene_joint_q_idx == [0, 1]


def test_multi_env_leaves_names_bare_when_the_count_does_not_divide():
    """An indivisible count means the replicate assumption does not hold —
    prefixing would invent a world grouping that isn't there."""
    host = _build(
        [
            _Joint("door_hinge_left", JT_REVOLUTE, _DOOR_FRAME, _DOOR_PANEL),
            _Joint("door_hinge_right", JT_REVOLUTE, _DOOR_FRAME, _DOOR_PANEL),
            _Joint("drawer_slide", JT_PRISMATIC, _DOOR_FRAME, _CUP),
        ],
        [_ROBOT_BASE, _DOOR_FRAME, _DOOR_PANEL, _CUP],
        num_envs=2,
    )
    assert host._scene_joint_names == ["door_hinge_left", "door_hinge_right", "drawer_slide"]

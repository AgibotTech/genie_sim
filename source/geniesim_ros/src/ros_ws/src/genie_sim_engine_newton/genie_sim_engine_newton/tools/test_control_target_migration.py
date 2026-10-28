# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Canonical Newton control targets using the engine's DOF target layout.

These CPU tests exercise the production topology, command and reset methods.
A floating joint before an arm makes q and qd indices differ, so accidentally
using a q index for DOF targets cannot pass as it does on a welded robot.
Deprecated aliases deliberately raise and record every access.
"""

from __future__ import annotations

import pathlib
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

_HERE = pathlib.Path(__file__).resolve()
for _root in (_HERE.parents[2], _HERE.parents[3] / "genie_sim_engine_core"):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from genie_sim_engine_newton.engine.newton.control import _ControlMixin  # noqa: E402
from genie_sim_engine_newton.engine.newton.lifecycle import _LifecycleMixin  # noqa: E402
from genie_sim_engine_newton.engine.newton.topology import _TopologyMixin  # noqa: E402


class _Array:
    def __init__(self, values):
        self._values = np.asarray(values).copy()

    @property
    def size(self):
        return self._values.size

    def numpy(self):
        return self._values.copy()

    def assign(self, values):
        values = np.asarray(values)
        assert values.shape == self._values.shape
        self._values[...] = values


class _Control:
    def __init__(self, target_q, target_qd):
        self.joint_target_q = _Array(target_q)
        self.joint_target_qd = _Array(target_qd)
        self.removed_alias_reads = []

    @property
    def joint_target_pos(self):
        self.removed_alias_reads.append("joint_target_pos")
        raise AttributeError("joint_target_pos was removed")

    @property
    def joint_target_vel(self):
        self.removed_alias_reads.append("joint_target_vel")
        raise AttributeError("joint_target_vel was removed")


class _Logger:
    def __init__(self):
        self.warnings = []

    def info(self, message):
        pass

    def warn(self, message):
        self.warnings.append(message)


class _Model:
    def __init__(self):
        self.joint_label = ["/genie/root_joint", "/genie/arm_joint1", "/genie/wheel_joint2"]
        self.joint_type = _Array([4, 1, 1])
        self.joint_q_start = _Array([0, 7, 8])
        self.joint_qd_start = _Array([0, 6, 7])
        self.joint_dof_dim = _Array([6, 1, 1])
        self.joint_q = _Array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.1, 0.2])
        self.joint_qd = _Array(np.zeros(8))


class _Engine(_TopologyMixin, _ControlMixin, _LifecycleMixin):
    pass


def _engine():
    engine = _Engine()
    engine._model = _Model()
    q = engine._model.joint_q.numpy()
    target_q = np.r_[np.zeros(6), q[7:]]
    engine._control = _Control(target_q, np.zeros(8))
    buffer = engine._control.joint_target_q
    engine._adapter = SimpleNamespace(target_buffer=lambda: buffer)
    engine._logger = _Logger()
    engine._num_envs = 1
    engine._init_joint_pos = {"arm_joint1": 30.0}
    engine._mimic_followers = {}
    engine._params = SimpleNamespace(cmd_4ws_timeout_s=1.0)
    engine._chassis_drive_dofs = {"wheel_joint2": 7}
    engine._scene_cfg = {}
    engine._body_paths = []
    engine._state_0 = SimpleNamespace(joint_q=_Array(q), joint_qd=_Array(np.zeros(8)), body_q=None)
    engine._state_1 = None
    engine._reset_init_joint_q = q.copy()
    engine._reset_init_target = target_q.copy()
    engine._reset_resnapshotted = True
    engine._reset_epoch = 0
    engine._build_joint_map()
    return engine, buffer


def test_init_pose_writes_state_and_dof_target_after_free_joint():
    engine, target = _engine()
    before_target = target.numpy()
    before_q = engine._model.joint_q.numpy()

    engine._apply_init_joint_pos()

    target_idx = 6
    assert engine._model.joint_q.numpy()[7] == pytest.approx(np.pi / 6)
    assert target.numpy()[target_idx] == pytest.approx(np.pi / 6)
    np.testing.assert_array_equal(engine._model.joint_q.numpy()[:7], before_q[:7])
    expected = before_target.copy()
    expected[target_idx] = np.pi / 6
    np.testing.assert_allclose(target.numpy(), expected)
    assert engine._control.removed_alias_reads == []
    assert engine._logger.warnings == []


def test_partial_position_and_velocity_use_canonical_buffers():
    engine, target = _engine()
    before = target.numpy()
    target_identity = id(target)

    engine.apply_commands({"arm_joint1": 0.75}, {}, {"wheel_joint2": 1.25}, time.monotonic())

    expected = before.copy()
    expected[6] = 0.75
    np.testing.assert_array_equal(target.numpy(), expected)
    np.testing.assert_array_equal(engine._control.joint_target_qd.numpy(), [0, 0, 0, 0, 0, 0, 0, 1.25])
    assert id(engine._adapter.target_buffer()) == target_identity
    assert engine._control.removed_alias_reads == []
    assert engine._logger.warnings == []


def test_reset_restores_canonical_position_targets(monkeypatch):
    # Only FK and stream synchronization are replaced: the actual reset,
    # commit and command implementations execute on the CPU arrays above.
    monkeypatch.setitem(sys.modules, "newton", SimpleNamespace(eval_fk=lambda *args: None))
    monkeypatch.setitem(sys.modules, "warp", SimpleNamespace(synchronize_device=lambda: None))
    engine, target = _engine()
    initial = target.numpy()
    target.assign(np.full(8, 0.9))

    engine.episode_reset()

    assert engine._reset_epoch == 1
    np.testing.assert_array_equal(target.numpy(), initial)
    assert engine._control.removed_alias_reads == []
    assert engine._logger.warnings == []

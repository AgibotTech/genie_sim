# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Reset writes both the physics hold and published FK without accumulating offsets."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

p = Path(__file__).resolve().parents[1] / "genie_sim_engine_newton/engine/newton/adapters/base_mocap.py"
spec = importlib.util.spec_from_file_location("base_mocap_reset_under_test", p)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Array:
    def __init__(self, value):
        self.value = np.array(value, float)
        self.size = self.value.size

    def numpy(self):
        return self.value.copy()

    def assign(self, value):
        self.value[:] = value

    def zero_(self):
        self.value.fill(0)


def adapter():
    a = module.BaseMocapResetMixin()
    a._base_mocap_idx = 0
    a._base_mocap_jnt = np.array([0])
    a._base_mocap_init_pos = np.array([[[-0.0482, 0.0181, 0.04]]])
    a._base_mocap_init_xp = np.array([[-0.0482, 0.0181, 0.04, 0, 0, 0, 1]])
    a._solver = SimpleNamespace(
        model=SimpleNamespace(joint_X_p=Array(a._base_mocap_init_xp)),
        mjw_data=SimpleNamespace(
            mocap_pos=Array(a._base_mocap_init_pos), qacc_warmstart=Array([[1, 3]]), act=Array([[5]])
        ),
    )
    return a


def test_two_layouts_then_original_restore_both_physics_and_fk_in_place():
    a = adapter()
    physics = a._solver.mjw_data.mocap_pos
    fk = a._solver.model.joint_X_p
    for xy in [(0.0732, -0.0331), (0.01, 0.02), (0, 0)]:
        assert a.apply_base_mocap_offsets({0: xy}) == 1
        np.testing.assert_allclose(physics.numpy()[0, 0, :2], a._base_mocap_init_pos[0, 0, :2] + xy)
        np.testing.assert_allclose(fk.numpy()[0, :2], a._base_mocap_init_xp[0, :2] + xy)
        assert a._solver.mjw_data.mocap_pos is physics
        assert a._solver.model.joint_X_p is fk


def test_single_world_clears_warmstart_and_activation():
    a = adapter()
    a.clear_solver_carryover([0])
    assert not np.any(a._solver.mjw_data.qacc_warmstart.numpy())
    assert not np.any(a._solver.mjw_data.act.numpy())
    with pytest.raises(ValueError):
        a.clear_solver_carryover([1])


def test_compact_view_and_global_fk_root_are_both_moved():
    a = adapter()
    parent = SimpleNamespace(joint_X_p=Array([[0, 0, 0, 0, 0, 0, 1], [-0.0482, 0.0181, 0.04, 0, 0, 0, 1]]))
    a._base_mocap_parent_model = parent
    a._base_mocap_parent_jnt = np.array([1])
    a._base_mocap_parent_init_xp = parent.joint_X_p.numpy()
    for offset in [(0.0569, -0.0619), (0, 0)]:
        a.apply_base_mocap_offsets({0: offset})
        np.testing.assert_allclose(parent.joint_X_p.numpy()[1, :2], [-0.0482 + offset[0], 0.0181 + offset[1]])
        np.testing.assert_allclose(a._solver.model.joint_X_p.numpy()[0, :2], parent.joint_X_p.numpy()[1, :2])
        np.testing.assert_array_equal(parent.joint_X_p.numpy()[0], [0, 0, 0, 0, 0, 0, 1])

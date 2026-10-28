# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Exercise the official MuJoCo target API with the retained DOF layout."""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

_ENGINE = Path(__file__).resolve().parents[1] / "engine/newton"


class _Array:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32).copy()
        self.size = self.values.size

    @property
    def ptr(self):
        return self.values.ctypes.data

    def numpy(self):
        return self.values.copy()

    def assign(self, values):
        np.copyto(self.values, values.values if isinstance(values, _Array) else values)


class _Control(SimpleNamespace):
    def __getattribute__(self, name):
        if name in ("joint_target_pos", "joint_target_vel"):
            raise AssertionError(f"Adapter accessed removed alias {name}")
        return super().__getattribute__(name)


def _adapter(filename):
    path = _ENGINE / "adapters" / filename
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    names = {"init_target_buffer", "target_buffer", "sync_control_into_solver"}
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), namespace)
    return type("TargetAdapter", (), {node.name: namespace[node.name] for node in methods})()


@pytest.fixture
def joint_index(monkeypatch):
    name = "genie_sim_engine_newton.engine.newton.joint_index"
    spec = importlib.util.spec_from_file_location(name, _ENGINE / "joint_index.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def _model():
    # A FREE joint precedes the arm; existing q -> qd seeding must not shift it.
    q = [1, 2, 3, 0, 0, 0, 1, 0.42]
    return SimpleNamespace(
        joint_q=_Array(q),
        joint_qd=_Array(np.zeros(7)),
        joint_label=["/robot/free", "/robot/arm"],
        joint_type=_Array([4, 1]),
        joint_q_start=_Array([0, 7, 8]),
        joint_qd_start=_Array([0, 6, 7]),
        joint_dof_dim=_Array([[3, 3], [0, 1]]),
        use_coord_layout_targets=False,
    )


@pytest.mark.parametrize("filename", ["mujoco_warp.py", "mujoco_warp_multi_env.py"])
def test_dof_seed_uses_official_api_without_replacing_buffer(filename, joint_index):
    model = _model()
    control = _Control(joint_target_q=_Array(np.full(7, -7.0)))
    target, ptr = control.joint_target_q, control.joint_target_q.ptr
    adapter = _adapter(filename)
    logger = SimpleNamespace(info=Mock(), warn=Mock())

    adapter.init_target_buffer(model, control, logger)

    assert adapter.target_buffer() is target
    assert control.joint_target_q is target
    assert target.ptr == ptr
    expected = np.asarray([0] * 6 + [0.42], dtype=np.float32)
    np.testing.assert_array_equal(target.numpy(), expected)
    arm = joint_index.JointIndex(model).get("arm")
    assert target.numpy()[arm.qd_start] == pytest.approx(0.42)
    logger.warn.assert_not_called()


def test_multi_env_reset_delegates_to_native_control_mapping():
    adapter = _adapter("mujoco_warp_multi_env.py")
    model, state, control, data = (object() for _ in range(4))
    solver = SimpleNamespace(model=model, mjw_data=data, _apply_mjc_control=Mock())
    adapter._solver = solver
    adapter.sync_control_into_solver(state, control)
    solver._apply_mjc_control.assert_called_once_with(model, state, control, data)

# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Gripper loops stripped for mjwarp are re-closed as CONNECT equalities on every Newton version."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

newton = pytest.importorskip("newton")
wp = pytest.importorskip("warp")
_PATH = Path(__file__).resolve().parents[1] / "genie_sim_engine_newton/engine/newton/setup/model.py"
_SPEC = importlib.util.spec_from_file_location("_loop_connect_model_under_test", _PATH)
model_module = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(model_module)

_ANCHOR = (0.0, 0.0, 0.2)


def _two_link_loop():
    """Two hinged links off a fixed root; their tips are meant to meet at _ANCHOR on link a."""
    builder = newton.ModelBuilder()
    root = builder.add_link(label="/robot/root")
    fixed = builder.add_joint_fixed(parent=-1, child=root)
    joints = [fixed]
    for name, x in (("a", -0.1), ("b", 0.1)):
        link = builder.add_link(label=f"/robot/{name}")
        builder.add_shape_box(link, hx=0.02, hy=0.02, hz=0.1)
        joints.append(
            builder.add_joint_revolute(
                parent=root, child=link, axis=(1.0, 0.0, 0.0),
                parent_xform=wp.transform((x, 0.0, 0.0), wp.quat_identity()),
            )
        )
    builder.add_articulation(joints)
    return builder


def test_loops_close_on_the_installed_newton():
    builder = _two_link_loop()
    coords_before = builder.joint_coord_count

    added = model_module._add_loop_connect_equalities(builder, [("/robot/a", "/robot/b", _ANCHOR)])

    assert added == 1
    model = builder.finalize(device="cpu")
    assert model.joint_coord_count == coords_before, "a CONNECT equality must not add joint coordinates"
    solver = newton.solvers.SolverMuJoCo(model, use_mujoco_cpu=True)
    assert solver.mj_model.neq == 1
    assert solver.mj_model.nq == coords_before


def test_unresolved_bodies_are_skipped_without_adding_rows():
    builder = _two_link_loop()

    added = model_module._add_loop_connect_equalities(builder, [("/robot/a", "/robot/missing", _ANCHOR)])

    assert added == 0
    solver = newton.solvers.SolverMuJoCo(builder.finalize(device="cpu"), use_mujoco_cpu=True)
    assert solver.mj_model.neq == 0


def test_legacy_builder_method_is_used_when_present():
    calls = []
    builder = SimpleNamespace(
        body_label=["/robot/a", "/robot/b"],
        add_equality_constraint_connect=lambda **kw: calls.append(kw),
    )

    assert model_module._add_loop_connect_equalities(builder, [("/robot/a", "/robot/b", _ANCHOR)]) == 1
    assert calls == [{"body1": 0, "body2": 1, "anchor": _ANCHOR, "label": "loop:b"}]

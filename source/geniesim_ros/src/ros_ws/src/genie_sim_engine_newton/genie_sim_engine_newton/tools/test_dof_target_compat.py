# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Keep the standalone engine's DOF target ABI on Newton 1.4 and 1.6.1."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

newton = pytest.importorskip("newton")
_PATH = Path(__file__).resolve().parents[1] / "engine/newton/setup/runtime.py"
_SPEC = importlib.util.spec_from_file_location("_dof_runtime_under_test", _PATH)
runtime = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(runtime)


def _engine(build_model):
    return SimpleNamespace(
        _phase_open_stage_and_overrides=lambda: None,
        _phase_resolve_scene=lambda: ([], "xpbd", False, False),
        _phase_build_model=build_model,
        _phase_normalize_model=lambda: None,
        _phase_states_and_robot_solver=lambda: None,
        _phase_cloth_solver_and_readback=lambda *args: None,
        _phase_debug_publishers=lambda: None,
        _phase_finalize_init_state=lambda: None,
        _phase_publish_static_facts=lambda: None,
    )


@pytest.mark.parametrize("original_layout", [False, True])
def test_real_model_retains_dof_targets_after_global_layout_is_restored(monkeypatch, original_layout):
    monkeypatch.setattr(newton, "use_coord_layout_targets", original_layout)
    observed = {}

    def build(*args):
        observed["layout_during_build"] = newton.use_coord_layout_targets
        builder = newton.ModelBuilder()
        base = builder.add_link()
        root = builder.add_joint_free(child=base)
        arm = builder.add_link()
        hinge = builder.add_joint_revolute(parent=base, child=arm)
        builder.add_articulation([root, hinge])
        observed["model"] = builder.finalize(device="cpu")

    runtime._RuntimeMixin._build(_engine(build))

    assert observed["layout_during_build"] is False
    assert newton.use_coord_layout_targets is original_layout
    model = observed["model"]
    control = model.control()
    assert model.use_coord_layout_targets is False
    assert model.joint_coord_count == model.joint_dof_count + 1
    assert control.joint_target_q.size == model.joint_dof_count
    assert control.joint_target_qd.size == model.joint_dof_count
    np.testing.assert_array_equal(model.joint_target_q_start.numpy(), model.joint_qd_start.numpy())


def test_failed_build_restores_the_upstream_layout(monkeypatch):
    monkeypatch.setattr(newton, "use_coord_layout_targets", True)

    def build(*args):
        assert newton.use_coord_layout_targets is False
        raise RuntimeError("model build failed")

    with pytest.raises(RuntimeError, match="model build failed"):
        runtime._RuntimeMixin._build(_engine(build))

    assert newton.use_coord_layout_targets is True

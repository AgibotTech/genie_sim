# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Public configuration contract; no Kit/ROS/GPU required."""

from dataclasses import dataclass, field
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock
import ast
from pathlib import Path
import sys

import pytest

from genie_sim_engine_isaacsim.kit.newton_config import configure_runtime, verify_runtime_config


@dataclass
class SolverConfig:
    solver_type: str = "mujoco"
    cone: str = "elliptic"
    njmax: int = 1200
    nconmax: int = 200
    use_mujoco_contacts: bool = False


@dataclass
class RuntimeConfig:
    solver_cfg: SolverConfig = field(default_factory=SolverConfig)
    joint_limit_ke: float = 100
    joint_limit_kd: float = 1
    physics_frequency: float = 600
    num_substeps: int = 1


@pytest.fixture
def public_api(monkeypatch):
    api = ModuleType("isaacsim.physics.newton")
    api.MuJoCoSolverConfig = SolverConfig
    api.NewtonConfig = RuntimeConfig
    api.current = None
    api.configure_newton = Mock(side_effect=lambda cfg: setattr(api, "current", cfg))
    api.get_newton_config = lambda: api.current
    monkeypatch.setitem(sys.modules, api.__name__, api)
    return api


def _configure():
    return configure_runtime(
        physics_solver="mujoco-warp", physics_hz=100, substeps=4, logger=SimpleNamespace(info=Mock())
    )


def test_public_instances_receive_required_values_without_mutating_defaults(public_api):
    expected = _configure()
    cfg = public_api.current
    assert cfg.num_substeps == 4 and cfg.physics_frequency == 100
    assert cfg.joint_limit_ke == 10000 and cfg.joint_limit_kd == 10
    assert cfg.solver_cfg.cone == "pyramidal" and cfg.solver_cfg.use_mujoco_contacts
    assert cfg.solver_cfg.njmax == 1024 and cfg.solver_cfg.nconmax == 512
    verify_runtime_config(cfg, expected)
    assert RuntimeConfig().joint_limit_ke == 100
    assert SolverConfig().cone == "elliptic"
    assert SolverConfig.__dataclass_fields__["cone"].default == "elliptic"


def test_each_engine_gets_a_fresh_configuration(public_api):
    _configure()
    first = public_api.current
    _configure()
    assert public_api.current is not first
    assert public_api.current.solver_cfg is not first.solver_cfg
    assert public_api.configure_newton.call_count == 2


def test_default_substeps_follow_the_public_config(public_api):
    configure_runtime(physics_solver="mujoco-warp", physics_hz=100, substeps=0, logger=SimpleNamespace(info=Mock()))
    assert public_api.current.num_substeps == 1


def test_ignored_configuration_is_fatal(public_api):
    public_api.configure_newton.side_effect = lambda cfg: setattr(public_api, "current", RuntimeConfig())
    with pytest.raises(RuntimeError, match="effective.*required"):
        _configure()


def test_removed_constructor_field_is_fatal(public_api):
    public_api.MuJoCoSolverConfig = lambda: None
    with pytest.raises(RuntimeError, match="fields changed"):
        _configure()
    public_api.configure_newton.assert_not_called()


def test_other_solver_is_rejected_before_configuration(public_api):
    with pytest.raises(ValueError, match="supports MuJoCo-Warp"):
        configure_runtime(physics_solver="xpbd", physics_hz=100, substeps=1, logger=Mock())
    public_api.configure_newton.assert_not_called()


def test_post_reset_drift_is_fatal(public_api):
    expected = _configure()
    public_api.current.solver_cfg.cone = "elliptic"
    with pytest.raises(RuntimeError, match="MuJoCoSolverConfig.cone"):
        verify_runtime_config(public_api.current, expected)


def test_configuration_precedes_stage_reset_and_does_not_use_private_imports():
    package = Path(__file__).resolve().parents[1] / "genie_sim_engine_isaacsim/kit"
    tree = ast.parse((package / "isaac_newton.py").read_text())
    engine = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "IsaacNewtonEngine")
    init = next(node for node in engine.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    calls = list(ast.walk(init))
    configure = next(
        node.lineno
        for node in calls
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "configure_runtime"
    )
    stage = next(
        node.lineno
        for node in calls
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "IsaacSimStage"
    )
    assert configure < stage
    assert "_patch_isaac_newton_class_defaults" not in (package / "isaac_newton.py").read_text()
    for path in (package / "isaac_newton.py", package / "newton_config.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("isaacsim.physics.newton.impl")


def test_live_solver_readback_accepts_requested_values():
    from genie_sim_engine_isaacsim.kit.newton_config import verify_live_solver

    solver = SimpleNamespace(
        mjw_model=SimpleNamespace(opt=SimpleNamespace(cone=0, run_collision_detection=True, iterations=23))
    )
    verify_live_solver(solver, 23)
    verify_live_solver(solver, 0)


@pytest.mark.parametrize("field,value", [("cone", 1), ("run_collision_detection", False), ("iterations", 100)])
def test_live_solver_drift_is_fatal(field, value):
    from genie_sim_engine_isaacsim.kit.newton_config import verify_live_solver

    opt = SimpleNamespace(cone=0, run_collision_detection=True, iterations=23)
    setattr(opt, field, value)
    with pytest.raises(RuntimeError, match=f"live solver {field}"):
        verify_live_solver(SimpleNamespace(mjw_model=SimpleNamespace(opt=opt)), 23)


def test_missing_live_solver_is_fatal():
    from genie_sim_engine_isaacsim.kit.newton_config import verify_live_solver

    with pytest.raises(RuntimeError, match="no live MuJoCo"):
        verify_live_solver(None, 23)

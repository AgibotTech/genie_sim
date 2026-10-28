#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import importlib.util
from pathlib import Path

import pytest

UTILS_PATH = Path(__file__).resolve().parents[1] / "launch" / "utils.py"
SPEC = importlib.util.spec_from_file_location("genie_sim_bringup_launch_utils", UTILS_PATH)
UTILS = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(UTILS)


def _command(action) -> list[str]:
    return [part[0].text for part in action.cmd]


def _registered_pipeline(tmp_path: Path, *, robot_stage=None):
    artifacts = [tmp_path / "robot.urdf", tmp_path / "robot.usda"]
    for artifact in artifacts:
        artifact.touch()
    pipeline = {
        "name": "backend",
        "modules": ["backend.assemble"],
    }
    if robot_stage is not None:
        pipeline["robot_stage"] = robot_stage
    process, _, _, _ = UTILS.make_assemble_pipeline(
        scene_resolved="/tmp/scene.yaml",
        stage_dir=str(tmp_path),
        assets_folder="/tmp/assets",
        always_regenerate_robot_usd=False,
        runtime_nodes=[],
        physics_engine="backend_package",
        assemble_style="registered_pipeline",
        assemble_pipeline=pipeline,
        assemble_robot_artefacts=[str(path) for path in artifacts],
    )
    return process


def test_registered_pipeline_uses_configured_robot_stage_for_cache_hit(tmp_path):
    process = _registered_pipeline(tmp_path, robot_stage="robot_import")

    assert _command(process)[-2:] == ["--skip-stage", "robot_import"]


@pytest.mark.parametrize("scene", ["", "scene_missing_robot"])
def test_required_scene_distinguishes_missing_name_from_omitted_argument(monkeypatch, capsys, scene):
    monkeypatch.setattr(UTILS, "perform", lambda context, name: scene)
    monkeypatch.setattr(UTILS, "resolve_bringup_config_file", lambda name: "")
    with pytest.raises(SystemExit) as error:
        UTILS.resolve_scene_yaml_robot_params(None, required=True, robot_model="", body="", arm="", gripper="")
    assert error.value.code == 1
    output = capsys.readouterr().out
    assert (f"scene YAML not found: {scene}" if scene else "scene is required") in output


def test_registered_pipeline_without_robot_stage_does_not_invent_one(tmp_path):
    process = _registered_pipeline(tmp_path)

    assert "--skip-stage" not in _command(process)


@pytest.mark.parametrize("robot_stage", [[], 42])
def test_registered_pipeline_rejects_non_string_robot_stage(tmp_path, robot_stage):
    with pytest.raises(ValueError, match="robot_stage must be a string"):
        _registered_pipeline(tmp_path, robot_stage=robot_stage)


def test_pipeline_mapping_requires_registered_style(tmp_path):
    with pytest.raises(ValueError, match="only valid"):
        UTILS.make_assemble_pipeline(
            scene_resolved="/tmp/scene.yaml",
            stage_dir=str(tmp_path),
            assets_folder="/tmp/assets",
            always_regenerate_robot_usd=False,
            runtime_nodes=[],
            physics_engine="isaac_physx",
            assemble_pipeline={"name": "unexpected"},
        )


def test_system_isaac_renderer_keeps_ros_underlay(monkeypatch):
    import importlib.metadata
    import sys

    monkeypatch.delenv("GENIESIM_PY_CMD", raising=False)
    monkeypatch.setattr(importlib.metadata, "distribution", lambda name: object())
    monkeypatch.setattr(UTILS.shutil, "which", lambda name: sys.executable)
    monkeypatch.setenv("PYTHONPATH", "/opt/ros/jazzy/lib/python3.12/site-packages")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/opt/ros/jazzy/lib")
    interpreter, env = UTILS._isaacsim_render_env()
    assert interpreter == sys.executable
    assert env == {}


def test_explicit_system_renderer_keeps_ros_underlay(monkeypatch):
    import sys

    monkeypatch.setenv("GENIESIM_PY_CMD", "python3")
    monkeypatch.setattr(UTILS.shutil, "which", lambda name: sys.executable)
    assert UTILS._isaacsim_render_env() == ("python3", {})


def test_explicit_bundled_renderer_filters_native_underlay(monkeypatch, tmp_path):
    monkeypatch.setenv("GENIESIM_PY_CMD", "omni_python")
    monkeypatch.setattr(UTILS.shutil, "which", lambda name: str(tmp_path / name))
    overlay = str(tmp_path / "devel")
    monkeypatch.setenv("COLCON_PREFIX_PATH", overlay)
    monkeypatch.setenv(
        "PYTHONPATH", f"/opt/ros/jazzy/lib/python3.12/site-packages:{overlay}/lib/python3.12/site-packages"
    )
    monkeypatch.setenv("LD_LIBRARY_PATH", "/opt/ros/jazzy/lib")
    interpreter, env = UTILS._isaacsim_render_env()
    assert interpreter == "omni_python"
    assert env["PYTHONPATH"] == f"{overlay}/lib/python3.12/site-packages"
    assert "/opt/ros/" not in env["LD_LIBRARY_PATH"]

# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import copy
import json
from pathlib import Path

import pytest

from agent2act.adapters.embodiments import HARDWARE_FIELDS, bind_reference
from agent2act.adapters.planners.embodied import EmbodiedPlanner
from agent2act.configuration import load_config, parse_settings
from agent2act.domain.errors import ConfigurationError

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def referenced_config():
    data = load_config(ROOT / "examples/g2_llm_recipe.json").model_dump()
    for key in HARDWARE_FIELDS | {"embodiment"}:
        data["planner"].pop(key, None)
    data["embodiment"] = {"preset": "g2_omnipicker"}
    data["planner"]["use_gripper_center"] = True
    return data


def test_reference_round_trip_and_overrides_preserve_kinematics(referenced_config):
    original = copy.deepcopy(referenced_config)
    settings = parse_settings(referenced_config)
    assert referenced_config == original
    assert parse_settings(settings.model_dump()) == settings
    assert settings.override(port=9999).planner == settings.planner
    assert settings.motion_backend.model_dump() == original["motion_backend"]
    assert settings.execution == parse_settings({**original, "embodiment": None}).execution
    assert settings.embodiment["sha256"]
    data = settings.model_dump()
    data["planner"]["use_grasp_geometry"] = False
    assert parse_settings(data).override(gripper_center=False).planner.options.get("pose_reference") is None


@pytest.mark.parametrize(
    "change,match",
    [
        ("inline", "conflicts"),
        ("implicit", "explicit use_gripper_center"),
        ("robot", "use that robot's configuration"),
        ("bounds", "gripper_bounds"),
        ("geometry_without_tcp", "requires TCP"),
        ("frozen_mismatch", "differs from frozen"),
    ],
)
def test_conflicts_fail_during_loading(referenced_config, change, match):
    data = referenced_config
    if change == "inline":
        data["planner"]["gripper_center_offset_m"] = {"left": [0, 0, 0.185], "right": [0, 0, 0.185]}
    elif change == "implicit":
        data["planner"].pop("use_gripper_center")
    elif change == "robot":
        data["robot"]["robot_type"] = "another_robot"
    elif change == "bounds":
        data["robot"]["gripper_bounds"] = [[-0.1, 0], [-0.1, 0]]
    elif change == "geometry_without_tcp":
        data["planner"].update(use_gripper_center=False, use_grasp_geometry=True)
    else:
        data = parse_settings(data).model_dump()
        data["planner"]["pose_reference"] = "stale text"
    with pytest.raises(ConfigurationError, match=match):
        parse_settings(data)


def test_relative_reference_is_relative_to_config_and_freeze_is_portable(
    referenced_config, tmp_path, monkeypatch
):
    path = tmp_path / "preset.json"
    path.write_text(json.dumps(bind_reference({"preset": "g2_omnipicker"})["definition"]))
    referenced_config["embodiment"] = {"path": "preset.json"}
    config = tmp_path / "config.json"
    config.write_text(json.dumps(referenced_config))
    monkeypatch.chdir(ROOT)
    settings = load_config(config)
    path.unlink()
    assert settings.override(depth=False).embodiment == settings.embodiment
    assert parse_settings(settings.model_dump()) == settings


@pytest.mark.parametrize("tcp,geometry", [(True, True), (True, False), (False, False)])
@pytest.mark.parametrize("fixed", [True, False])
@pytest.mark.parametrize("planner_type", [EmbodiedPlanner])
def test_actual_request_respects_feature_gates(
    referenced_config, observation, tcp, geometry, fixed, planner_type
):
    referenced_config["planner"].update(use_gripper_center=tcp, use_grasp_geometry=geometry)
    settings = parse_settings(referenced_config)
    calls = []

    def post(body):
        calls.append(body)
        output = {
            "mode": "motion",
            "instruction": "hold",
            "reason": "offline",
            "motion": None,
            "success_condition": "unchanged",
            "status": "complete",
        }
        if "messages" in body:
            return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(output)}}]}
        return {
            "model": body["model"],
            "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(output)}]}],
        }

    options = settings.planner.options
    options.pop("reasoning_effort", None)
    options.pop("reasoning_summary", None)
    planner = planner_type(**options, post=post)
    compiler = (
        {"fixed_steps": 10, "timing": "linear interpolation", "speed_limits_enforced": False} if fixed else {}
    )
    try:
        planner.decide(
            observation,
            {
                "execution": {
                    "mode": "motion",
                    "motion_frame": "base_link",
                    "motion_frame_axes": ["forward", "left", "up"],
                    "executors": {"motion": {"compiler": compiler}},
                }
            },
        )
        text = (
            calls[0]["messages"][1]["content"][0]["text"]
            if "messages" in calls[0]
            else calls[0]["input"][0]["content"][0]["text"]
        )
        state = json.loads(text)
        assert ("pose_reference" in state) is tcp
        assert ("grasp_geometry" in state) is geometry
        if not geometry:
            assert "60.4" not in text and "26 mm" not in text and "wide jaw" not in text
        if tcp:
            assert state["pose_reference"] == settings.planner.options["pose_reference"]
    finally:
        planner.close()


def test_trace_freezes_hardware_without_sending_metadata_to_model(referenced_config, tmp_path):
    from agent2act.bootstrap import ApplicationFactory

    referenced_config["logging"]["trace_dir"] = str(tmp_path)
    settings = parse_settings(referenced_config)
    with ApplicationFactory(settings).create_session() as session:
        data = json.loads((tmp_path / session.session_id / "hardware.json").read_text())
    assert data == json.loads(json.dumps(settings.hardware_snapshot()))
    assert data["running_asset_identity_verified"] is False
    assert "endpoint" not in json.dumps(data)
    assert "api_key" not in json.dumps(data)
    assert data["embodiment"]["definition"]["revision"] == "1"
    assert data["planner_hardware"]["pose_reference"] == settings.planner.options["pose_reference"]


def test_cli_feature_override_regenerates_a_portable_snapshot(referenced_config, tmp_path):
    settings = parse_settings(referenced_config).override(grasp_geometry=False)
    file = tmp_path / "resolved.json"
    file.write_text(json.dumps(settings.model_dump()))
    disabled = load_config(file, gripper_center=False)
    assert disabled.planner.options["use_gripper_center"] is False
    assert "pose_reference" not in disabled.planner.options
    assert disabled.embodiment == settings.embodiment

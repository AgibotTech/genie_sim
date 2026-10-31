# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""The bundled G2 recipe sends the omnipicker gripper description and loads its URDFs."""

import copy
import json
from pathlib import Path

import pytest

from agent2act.adapters.planners.responses import ResponsesPlanner
from agent2act.bootstrap import default_registry
from agent2act.configuration import load_config
from agent2act.domain.errors import ContractError

ROOT = Path(__file__).resolve().parents[2]
RECIPE = ROOT / "examples/g2_llm_recipe.json"
CONTEXT = {
    "execution": {
        "mode": "motion",
        "motion_frame": "base_link",
        "motion_frame_axes": ["forward", "left", "up"],
    }
}


def _post(calls):
    def post(body):
        calls.append(copy.deepcopy(body))
        decision = {
            "mode": "motion",
            "instruction": "hold",
            "reason": "offline probe",
            "motion": None,
            "success_condition": "unchanged",
            "status": "complete",
        }
        return {
            "model": body["model"],
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": json.dumps(decision)}]}
            ],
        }

    return post


def _request(settings, observation, monkeypatch):
    artifacts = {}
    monkeypatch.setattr(
        "agent2act.adapters.planners.responses.record_artifact",
        lambda kind, value: artifacts.update({kind: copy.deepcopy(value)}),
    )
    calls = []
    planner = default_registry().planners.resolve(settings.planner.type)(
        {**settings.planner.options, "post": _post(calls)}
    )
    try:
        planner.decide(observation, copy.deepcopy(CONTEXT))
    finally:
        planner.close()
    body = calls[0]
    return (
        json.loads(body["input"][0]["content"][0]["text"]),
        body["input"][1]["content"][0]["text"],
        artifacts,
    )


def test_default_recipe_keeps_omnipicker_description(observation, monkeypatch):
    state, developer, artifacts = _request(load_config(RECIPE), observation, monkeypatch)
    assert "185 mm" not in state["pose_reference"]
    assert state["gripper_command_reference"]["left"]["fully_open_command"] == -0.785
    assert state["grasp_geometry"]["tcp_axes"]["open_close"] == "x"
    assert '"grippers":[-0.785,-0.785]' in developer
    assert artifacts["observation-options"]["embodiment"] == "G2_omnipicker"


def test_default_recipe_loads_bundled_urdfs():
    from agent2act.bootstrap import ApplicationFactory

    settings = load_config(RECIPE)
    motion = settings.motion_backend.options
    codec = settings.simulator_codec.options
    assert motion["path"] == "examples/urdf/G2_omnipicker_fixed_dual.urdf"
    assert motion["subtree_path"] == codec["path"] == "examples/urdf/G2_NO_GRIPPER.urdf"
    with ApplicationFactory(settings).create_session() as session:
        assert session.runtime.router.describe()["executors"]["motion"]["compiler"]["backend"] == "urdf_joint"


def test_pose_reference_requires_gripper_center():
    with pytest.raises(ContractError, match="pose_reference"):
        ResponsesPlanner(model="test", pose_reference="TCP text")

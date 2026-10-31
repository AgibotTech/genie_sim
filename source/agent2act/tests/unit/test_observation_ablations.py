# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Verify information boundaries and coordinate semantics without model credentials."""

import copy
import json

import pytest

from agent2act.adapters.planners.embodied import EmbodiedPlanner
from agent2act.adapters.planners.height_prior import HeightPrior
from agent2act.adapters.planners.responses import ResponsesPlanner
from agent2act.configuration import load_config
from agent2act.domain.errors import ContractError

CALIBRATION = {
    "frame": "base_link",
    "table_surface_world_z_m": 0.73,
    "base_link_world_z_m": 0.04,
    "source": "offline synthetic calibration",
}
OFFSETS = {"left": [0, 0, 0.14845], "right": [0, 0, 0.14845]}


@pytest.mark.parametrize("depth", [False, True])
@pytest.mark.parametrize("center", [False, True])
@pytest.mark.parametrize("height", [False, True])
def test_independent_information_switches(depth, center, height, observation, monkeypatch):
    calls = []
    artifacts = {}
    monkeypatch.setattr(
        "agent2act.adapters.planners.responses.record_artifact",
        lambda kind, value: artifacts.update({kind: copy.deepcopy(value)}),
    )

    def post(body):
        calls.append(copy.deepcopy(body))
        state = json.loads(body["input"][0]["content"][0]["text"])
        arm = state["states"]["end_pose"]["base_link"]["right_arm"]
        return {
            "model": body["model"],
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": json.dumps(
                                {
                                    "mode": "motion",
                                    "instruction": "hold",
                                    "reason": "offline probe",
                                    "motion": {
                                        "frame": "base_link",
                                        "waypoints": [
                                            {
                                                "right_pose": arm["position"] + arm["orientation"],
                                                "grippers": [0, 0],
                                            }
                                        ],
                                    },
                                }
                            ),
                        }
                    ],
                }
            ],
        }

    settings = load_config("examples/g2_llm_recipe.json").override(
        depth=depth,
        gripper_center=center,
        grasp_geometry=False,
        height_prior=height,
        height_calibration=CALIBRATION,
    )
    assert settings.execution.need_depth is depth
    planner = EmbodiedPlanner(**settings.planner.options, post=post)
    context = {
        "execution": {
            "mode": "motion",
            "motion_frame": "base_link",
            "motion_frame_axes": ["forward", "left", "up"],
        }
    }
    before = copy.deepcopy(observation)
    try:
        decision = planner.decide(observation, context)
        state = json.loads(calls[0]["input"][0]["content"][0]["text"])
        assert ("height_prior" in state) is height
        assert ("pose_reference" in state) is center
        assert planner.use_depth is depth
        if height:
            assert state["height_prior"]["table_surface_z_m"] == pytest.approx(0.69)
            assert set(state["height_prior"]) == {"frame", "table_surface_z_m", "meaning"}
            assert artifacts["observation-options"]["height_calibration"] == CALIBRATION
            assert "pose.z - table_surface_z_m" in state["height_prior"]["meaning"]
            assert "not the object top or a contact measurement" in state["height_prior"]["meaning"]
        else:
            assert "offline synthetic calibration" not in json.dumps(calls)
            assert "table_surface_world_z_m" not in json.dumps(calls)
            assert "pose.z - table_surface_z_m" not in json.dumps(calls)
        assert CALIBRATION["source"] not in json.dumps(calls)
        wrist = observation.eef("right", "base_link")
        # Holding a model-facing center or wrist must produce the same physical wrist target.
        assert decision.motion["waypoints"][0]["right_pose"] == pytest.approx(wrist)
        if not center:
            assert state["states"]["end_pose"]["base_link"]["right_arm"]["position"] == wrist[:3]
        assert observation == before
    finally:
        planner.close()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "0.73"])
def test_invalid_heights_rejected(value):
    with pytest.raises(ContractError, match="finite"):
        HeightPrior({**CALIBRATION, "table_surface_world_z_m": value})


def test_missing_calibration_and_frame_mismatch_rejected():
    with pytest.raises(ContractError, match="requires"):
        ResponsesPlanner(model="offline", use_height_prior=True)
    with pytest.raises(ContractError, match="requires calibrated"):
        ResponsesPlanner(model="offline", use_gripper_center=True)
    prior = HeightPrior(CALIBRATION)
    with pytest.raises(ContractError, match="axes"):
        prior.context("arm_base_link", ["forward", "left", "up"])
    with pytest.raises(ContractError, match="axes"):
        prior.context("base_link", ["forward", "up", "left"])


def test_center_off_disables_legacy_geometry_tool():
    planner = EmbodiedPlanner(
        model="offline", use_gripper_center=False, gripper_center_offset_m=OFFSETS, wrist_to_center_m=OFFSETS
    )
    try:
        assert planner._tool_center is None
        assert planner.wrist_to_center_m == {}
        assert not planner._geometry_tools()
    finally:
        planner.close()

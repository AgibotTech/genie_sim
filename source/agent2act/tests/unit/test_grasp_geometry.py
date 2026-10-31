# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Verify gripper-geometry input boundaries without model credentials."""

import copy
import json

import pytest

from agent2act.adapters.planners.embodied import EmbodiedPlanner
from agent2act.adapters.planners.grasp_geometry import GraspGeometry
from agent2act.adapters.planners.motion_context import (
    FIXED_MOTION_INSTRUCTIONS,
    fixed_motion_instructions,
    motion_only_state,
)
from agent2act.adapters.planners.responses import ResponsesPlanner
from agent2act.configuration import load_config
from agent2act.domain.errors import ContractError

HEIGHT = {
    "frame": "base_link",
    "table_surface_world_z_m": 0.73,
    "base_link_world_z_m": 0.04,
    "source": "offline synthetic table",
}
GEOMETRY = {
    "frame": "tcp_approach_axis",
    "fingertip_beyond_tcp_m": 0.0103,
    "pad_contact_zone_m": {"behind_tcp_m": 0.0427, "beyond_tcp_m": 0.0103},
    "pad_width_m": 0.0604,
    "pad_thickness_m": 0.0279,
    "geometry_basis": "offline synthetic mesh bounds",
    "source": "offline synthetic gripper asset",
}
OFFSETS = {"left": [0, 0, 0.14845], "right": [0, 0, 0.14845]}


def _post(body):
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


@pytest.mark.parametrize(
    "mutation, match",
    [
        ({"fingertip_beyond_tcp_m": float("nan")}, "finite"),
        ({"fingertip_beyond_tcp_m": -0.001}, "non-negative"),
        ({"pad_width_m": 0}, "positive"),
        (
            {"pad_contact_zone_m": {"behind_tcp_m": 0, "beyond_tcp_m": 0.001}},
            "behind_tcp_m must be a positive",
        ),
        ({"fingertip_beyond_tcp_m": True}, "finite"),
        ({"frame": "base_link"}, "tcp_approach_axis"),
        ({"source": " "}, "source"),
        ({"geometry_basis": ""}, "geometry_basis"),
        ({"pad_contact_zone_m": {"behind_tcp_m": 0.01}}, "behind_tcp_m and beyond_tcp_m"),
        ({"pad_contact_zone_m": {"behind_tcp_m": 0.01, "beyond_tcp_m": 0.05}}, "cannot exceed"),
        ({"pad_width_m": "wide"}, "finite"),
        ({"pad_widths_per_finger_m": {}}, "nonempty"),
        ({"pad_widths_per_finger_m": {"inner": 0}}, "positive"),
        ({"pad_widths_per_finger_m": {"inner": "wide"}}, "positive finite"),
        ({"pad_widths_per_finger_m": {1: 0.03}}, "finger names"),
        ({"gripper_notes": " "}, "gripper_notes"),
        ({"gripper_notes": 5}, "gripper_notes"),
        ({"tcp_axes": {"approach": "+z", "open_close": "z", "pad_width": "x"}}, "exactly once"),
        ({"tcp_axes": {"approach": "+z"}}, "tcp_axes requires"),
        ({"aperture_by_command": [{"command": 0, "gap_m": 0.09}]}, "at least two"),
        (
            {
                "aperture_by_command": [
                    {
                        "command": 0,
                        "gap_m": 0.09,
                        "pad_face_from_tcp_m": [0, 1],
                        "fingertip_beyond_tcp_m": 0.03,
                    },
                    {
                        "command": -0.9,
                        "gap_m": "closed",
                        "pad_face_from_tcp_m": [0, 1],
                        "fingertip_beyond_tcp_m": 0.04,
                    },
                ]
            },
            "finite",
        ),
    ],
)
def test_invalid_geometry_rejected(mutation, match):
    with pytest.raises(ContractError, match=match):
        GraspGeometry({**GEOMETRY, **mutation})


def test_per_finger_widths_surfaced_only_when_calibrated():
    base = GraspGeometry(GEOMETRY).context(0.73)
    assert "pad_widths_per_finger_m" not in base
    per_finger = {"inner": 0.026, "outer": 0.0604}
    block = GraspGeometry({**GEOMETRY, "pad_widths_per_finger_m": per_finger}).context(0.73)
    assert block["pad_widths_per_finger_m"] == per_finger
    assert block["pad_width_m"] == pytest.approx(0.0604)


def test_gripper_notes_and_axes_surfaced_only_when_calibrated():
    base = GraspGeometry(GEOMETRY).context(0.73)
    assert not {"gripper_notes", "tcp_axes", "aperture_by_command"} & set(base)
    text = "The outer finger is the wide jaw; the inner finger is the narrow jaw."
    axes = {"approach": "+z", "open_close": "x", "pad_width": "y"}
    block = GraspGeometry({**GEOMETRY, "gripper_notes": text, "tcp_axes": axes}).context(0.73)
    assert block["gripper_notes"] == text
    assert block["tcp_axes"] == axes


def test_missing_keys_rejected():
    for key in GEOMETRY:
        broken = {k: v for k, v in GEOMETRY.items() if k != key}
        with pytest.raises(ContractError, match="requires"):
            GraspGeometry(broken)


def test_geometry_requires_center():
    with pytest.raises(ContractError, match="gripper-center"):
        ResponsesPlanner(
            model="offline",
            use_height_prior=True,
            height_calibration=HEIGHT,
            use_grasp_geometry=True,
            grasp_geometry_calibration=GEOMETRY,
        )


def test_geometry_without_table_prior_keeps_finger_data(observation):
    calls = []

    def post(body):
        calls.append(copy.deepcopy(body))
        return _post(body)

    planner = ResponsesPlanner(
        model="offline",
        gripper_center_offset_m=OFFSETS,
        use_grasp_geometry=True,
        grasp_geometry_calibration=GEOMETRY,
        use_depth=True,
        post=post,
    )
    try:
        planner.decide(observation, {"execution": {"mode": "motion"}})
        state = json.loads(calls[0]["input"][0]["content"][0]["text"])
        assert "height_prior" not in state
        assert "table_surface_z_m" not in state["grasp_geometry"]
        assert state["grasp_geometry"]["fingertip_beyond_tcp_m"] == pytest.approx(0.0103)
        assert planner.use_depth is True
    finally:
        planner.close()


def _planner(**kwargs):
    return ResponsesPlanner(
        model="offline",
        gripper_center_offset_m=OFFSETS,
        use_height_prior=True,
        height_calibration=HEIGHT,
        **kwargs,
    )


def test_geometry_block_in_state_and_artifacts(observation, monkeypatch):
    artifacts = {}
    calls = []
    monkeypatch.setattr(
        "agent2act.adapters.planners.responses.record_artifact",
        lambda kind, value: artifacts.update({kind: copy.deepcopy(value)}),
    )

    def post(body):
        calls.append(copy.deepcopy(body))
        return _post(body)

    planner = _planner(use_grasp_geometry=True, grasp_geometry_calibration=GEOMETRY, post=post)
    context = {
        "execution": {
            "mode": "motion",
            "motion_frame": "base_link",
            "motion_frame_axes": ["forward", "left", "up"],
        }
    }
    before = copy.deepcopy(observation)
    try:
        planner.decide(observation, context)
        state = json.loads(calls[0]["input"][0]["content"][0]["text"])
        block = state["grasp_geometry"]
        assert block["table_surface_z_m"] == pytest.approx(0.69)
        assert block["fingertip_beyond_tcp_m"] == pytest.approx(0.0103)
        assert block["pad_contact_zone_m"] == {"behind_tcp_m": 0.0427, "beyond_tcp_m": 0.0103}
        assert "not measured contact" in block["meaning"]
        # Facts only: provenance and strategy text stay out of the model input.
        assert "geometry_basis" not in block
        assert "source" not in block
        assert "guidance" not in block
        assert "offline synthetic gripper asset" not in json.dumps(calls)
        options = artifacts["observation-options"]
        assert options["grasp_geometry_enabled"] is True
        assert options["grasp_geometry_calibration"]["source"] == GEOMETRY["source"]
        assert options["strategy_version"]
        assert observation == before
    finally:
        planner.close()


def test_geometry_disabled_does_not_leak(observation, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "agent2act.adapters.planners.responses.record_artifact",
        lambda kind, value: None,
    )

    def post(body):
        calls.append(copy.deepcopy(body))
        return _post(body)

    planner = _planner(use_grasp_geometry=False, grasp_geometry_calibration=GEOMETRY, post=post)
    context = {
        "execution": {
            "mode": "motion",
            "motion_frame": "base_link",
            "motion_frame_axes": ["forward", "left", "up"],
        }
    }
    try:
        planner.decide(observation, context)
        state = json.loads(calls[0]["input"][0]["content"][0]["text"])
        assert "grasp_geometry" not in state
        assert "offline synthetic gripper asset" not in json.dumps(calls)
        assert "fingertip_beyond_tcp_m" not in json.dumps(calls)
    finally:
        planner.close()


def test_motion_projection_preserves_geometry_block():
    state = {
        "instruction": "probe",
        "robot_type": "probe",
        "states": {"end_pose": {}, "gripper_states": [0, 0]},
        "context": {
            "execution": {"motion_frame": "base_link", "motion_frame_axes": ["forward", "left", "up"]},
            "robot": {},
        },
        "grasp_geometry": GraspGeometry(GEOMETRY).context(0.730005),
    }
    projected = motion_only_state(copy.deepcopy(state))
    assert projected["grasp_geometry"]["table_surface_z_m"] == pytest.approx(0.730005)


def test_fixed_motion_instructions_off_path_unchanged():
    assert fixed_motion_instructions(False) == FIXED_MOTION_INSTRUCTIONS
    unrestricted = fixed_motion_instructions(True)
    assert unrestricted != FIXED_MOTION_INSTRUCTIONS
    assert "fixed_steps interpolation steps" in unrestricted
    extended = fixed_motion_instructions(False, grasp_geometry=True)
    assert extended != FIXED_MOTION_INSTRUCTIONS
    assert "grasp_geometry" in extended
    assert "finger geometry" in extended
    assert "fingertip_beyond_tcp_m" in extended
    assert "Never command TCP below table_surface_z_m + fingertip_beyond_tcp_m" in extended
    assert "open/close axis" in extended and "parallel to the table" in extended
    plain = fixed_motion_instructions(False)
    assert "grasp_geometry" not in plain
    assert "fingertip_beyond_tcp_m" not in plain
    assert "open/close axis" not in plain
    assert extended.count("Use supplied calibration") == 1
    without_table = fixed_motion_instructions(False, grasp_geometry=True, table_height_available=False)
    assert "gripper_notes" in without_table
    assert "use current depth and views" in without_table
    assert "Never command TCP below table_surface_z_m" not in without_table


def _recipe_post(observation, context, calls):
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

    return post


def test_llm_recipe_config_end_to_end(observation, monkeypatch):
    """The API recipe example carries the geometry input through the responses path."""
    monkeypatch.setattr(
        "agent2act.adapters.planners.responses.record_artifact",
        lambda kind, value: None,
    )
    calls = []
    settings = load_config("examples/g2_llm_recipe.json")
    assert settings.planner.options["reasoning_effort"] == "high"
    # The shipped recipe leaves height_prior off; enable it to exercise the combination.
    options = {
        **settings.planner.options,
        "use_height_prior": True,
        "height_calibration": {
            "frame": "base_link",
            "table_surface_world_z_m": 0.730005,
            "base_link_world_z_m": 0.0,
            "source": "offline synthetic table",
        },
    }
    planner = EmbodiedPlanner(**options, post=_recipe_post(observation, {}, calls))
    context = {
        "execution": {
            "mode": "motion",
            "motion_frame": "base_link",
            "motion_frame_axes": ["forward", "left", "up"],
            "executors": {
                "motion": {
                    "compiler": {
                        "fixed_steps": 10,
                        "retime_to_limits": True,
                        "speed_limits_enforced": False,
                        "timing": "linear position/gripper interpolation and quaternion SLERP",
                    }
                }
            },
        }
    }
    before = copy.deepcopy(observation)
    try:
        planner.decide(observation, context)
        body = calls[0]
        state = json.loads(body["input"][0]["content"][0]["text"])
        block = state["grasp_geometry"]
        assert block["table_surface_z_m"] == pytest.approx(0.730005)
        assert block["fingertip_beyond_tcp_m"] == pytest.approx(0.0103)
        assert "height_prior" in state
        developer = next(
            part["text"]
            for message in body["input"]
            if message.get("role") == "developer"
            for part in message["content"]
            if part.get("type") == "input_text"
        )
        assert "grasp_geometry" in developer
        assert "finger geometry" in developer
        assert "gripper_notes must be followed" in developer
        assert "inner 26 mm" in block["gripper_notes"] and "outer 60.4 mm" in block["gripper_notes"]
        assert block["tcp_axes"]["open_close"] == "x"
        assert block["pad_widths_per_finger_m"] == {"inner": 0.026, "outer": 0.0604}
        assert observation == before
    finally:
        planner.close()


def test_llm_recipe_joint_backend_prompt_carries_finger_guidance(observation, monkeypatch):
    """The urdf_joint recipe has no fixed_steps compiler, so it uses the motion profile prompt."""
    monkeypatch.setattr(
        "agent2act.adapters.planners.responses.record_artifact",
        lambda kind, value: None,
    )
    calls = []
    settings = load_config("examples/g2_llm_recipe.json")
    assert settings.motion_backend.type == "urdf_joint"
    planner = EmbodiedPlanner(**settings.planner.options, post=_recipe_post(observation, {}, calls))
    context = {
        "execution": {
            "mode": "motion",
            "motion_frame": "base_link",
            "motion_frame_axes": ["forward", "left", "up"],
        }
    }
    try:
        planner.decide(observation, context)
        body = calls[0]
        state = json.loads(body["input"][0]["content"][0]["text"])
        assert "inner 26 mm" in state["grasp_geometry"]["gripper_notes"]
        assert "image-left" not in state["grasp_geometry"]["gripper_notes"]
        developer = body["input"][1]["content"][0]["text"]
        assert "gripper_notes must be followed" in developer
        assert '"grippers":[-0.785,-0.785]' in developer
        assert "context.decision_history" in developer
    finally:
        planner.close()


def test_recipe_without_grasp_geometry_configured_has_no_geometry_leak(observation, monkeypatch):
    """A plain recipe that never sets use_grasp_geometry must never surface it in the prompt."""
    monkeypatch.setattr(
        "agent2act.adapters.planners.responses.record_artifact",
        lambda kind, value: None,
    )
    calls = []
    planner = ResponsesPlanner(model="test-model", post=_recipe_post(observation, {}, calls))
    context = {
        "execution": {
            "mode": "motion",
            "motion_frame": "base_link",
            "motion_frame_axes": ["forward", "left", "up"],
            "executors": {
                "motion": {
                    "compiler": {
                        "fixed_steps": 10,
                        "retime_to_limits": True,
                        "speed_limits_enforced": False,
                        "timing": "linear position/gripper interpolation and quaternion SLERP",
                    }
                }
            },
        }
    }
    try:
        planner.decide(observation, context)
        body = calls[0]
        state = json.loads(body["input"][0]["content"][0]["text"])
        assert "grasp_geometry" not in state
        assert "fingertip_beyond_tcp_m" not in json.dumps(body)
        developer = next(
            part["text"]
            for message in body["input"]
            if message.get("role") == "developer"
            for part in message["content"]
            if part.get("type") == "input_text"
        )
        assert "grasp_geometry" not in developer
    finally:
        planner.close()

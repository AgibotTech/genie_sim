# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import copy
import json
import sys
from pathlib import Path

import pytest

from agent2act.adapters.simulators.corobot import decode_request
from agent2act.domain.models import RobotSpec

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    with (ROOT / "examples/demo.toml").open("rb") as stream:
        return tomllib.load(stream)


@pytest.fixture
def request_payload():
    return json.loads((ROOT / "examples/observation.json").read_text())


@pytest.fixture
def spec(config):
    return RobotSpec(**config["robot"])


@pytest.fixture
def observation(request_payload, spec):
    return decode_request(request_payload, spec)


@pytest.fixture
def motion_decision():
    return {
        "mode": "motion",
        "instruction": "Move slightly",
        "reason": "short joint motion",
        "motion": {"frame": "base_link", "waypoints": [{"duration_s": 0.1}]},
    }


@pytest.fixture
def pi_config():
    return copy.deepcopy(
        {
            "robot_type": "demo_dual_arm",
            "checkpoint": "test-checkpoint",
            "skills": ["test"],
            "host": "127.0.0.1",
            "port": 8000,
            "camera_map": {"image": "head"},
            "image_size": [16, 16],
            "state_fields": ["arm_joint_states", "gripper_states"],
            "state_key": "state",
            "prompt_key": "prompt",
            "action_key": "actions",
            "action_dim": 4,
            "left_indices": [0],
            "right_indices": [1],
            "gripper_indices": [2, 3],
            "kind": "JOINT_ABS",
            "frame": "base_link",
            "dt_s": 0.05,
            "action_scale": [1, 1, 1, 1],
            "action_offset": [0, 0, 0, 0],
            "state_scale": [1, 1, 1, 1],
            "state_offset": [0, 0, 0, 0],
        }
    )


@pytest.fixture
def g2_pi05_settings():
    """The G2_omnipicker robot/motion config paired with the pi05 backend the archived fixtures in tests/fixtures/pi05/ were recorded against."""
    from agent2act.configuration import load_config, parse_settings

    data = load_config(ROOT / "examples/g2_llm_recipe.json").model_dump()
    # The archived fixtures were recorded against a 1.0 rad/s speed-enforced profile.
    data["robot"]["max_joint_speed"] = 1.0
    data["robot"]["enforce_speed_limits"] = True
    data["policy"] = {
        "default": "pi05",
        "backends": {
            "pi05": {
                "type": "pi05",
                "robot_type": "G2_omnipicker",
                "checkpoint": "checkpoints/instruction_and_robust_pi05",
                "skills": ["test"],
                "capabilities": [
                    {
                        "name": "local_instruction",
                        "description": "test",
                        "instruction_scope": "local_subtask",
                        "max_policy_chunks": 4,
                    }
                ],
                "host": "127.0.0.1",
                "port": 8000,
                "timeout_s": 30,
                "camera_map": {"top_head": "head", "hand_left": "hand_left", "hand_right": "hand_right"},
                "image_size": [224, 224],
                "state_fields": ["arm_joint_states", "gripper_states"],
                "state_key": "state",
                "prompt_key": "prompt",
                "action_key": "actions",
                "action_dim": 16,
                "left_indices": [0, 1, 2, 3, 4, 5, 6],
                "right_indices": [7, 8, 9, 10, 11, 12, 13],
                "gripper_indices": [14, 15],
                "body_indices": [None, None, None, None, None],
                "kind": "JOINT_ABS",
                "frame": "base_link",
                "dt_s": 1 / 30,
                "action_scale": [1] * 16,
                "action_offset": [0] * 16,
                "state_scale": [1] * 16,
                "state_offset": [0] * 16,
                "joint_delta": False,
                "retime_joint_actions": True,
                "gripper_clip_tolerance": 0.005,
            }
        },
    }
    return parse_settings(data)

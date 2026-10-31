# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Policy boundary correction remains opt-in and cannot bypass action validation."""

import logging

import pytest

from agent2act.adapters.policies.pi05 import Pi05Adapter
from agent2act.domain.models import ContractError
from agent2act.domain.validation import ActionValidator


@pytest.mark.parametrize("tolerance", [-0.1, float("inf"), float("nan"), True, "0.005"])
def test_invalid_tolerance(spec, pi_config, tolerance):
    pi_config["gripper_clip_tolerance"] = tolerance
    with pytest.raises(ContractError):
        Pi05Adapter(spec, pi_config)


def test_default_does_not_clip(spec, pi_config, observation):
    chunk = Pi05Adapter(spec, pi_config).decode_actions({"actions": [[0, 0, -0.003, 0.5]]}, observation)
    assert chunk.grippers == [[-0.003, 0.5]]
    assert "gripper_clips" not in chunk.metadata
    with pytest.raises(ContractError, match="gripper bounds"):
        ActionValidator(spec).validate(chunk, observation)


def test_clips_mapped_values_and_records_each_step(spec, pi_config, observation, caplog):
    spec.gripper_bounds = [[-0.785, 0], [-0.785, 0]]
    observation.grippers = [-0.785, 0]
    pi_config.update(gripper_clip_tolerance=0.005, action_scale=[1, 1, -1, 1])
    raw = {"actions": [[0, 0, 0.78812, 0.002], [0, 0, 0.785, 0]]}
    with caplog.at_level(logging.INFO):
        chunk = Pi05Adapter(spec, pi_config).decode_actions(raw, observation)
    assert chunk.grippers == [[-0.785, 0], [-0.785, 0]]
    assert raw["actions"][0][2] == 0.78812
    clips = chunk.metadata["gripper_clips"]
    assert len(clips) == 2
    assert clips[0]["step"] == 0 and clips[0]["side"] == "left"
    assert clips[0]["original"] == -0.78812
    assert clips[0]["corrected"] == -0.785
    assert clips[0]["delta"] == pytest.approx(0.00312)
    assert caplog.records[-1].event["corrections"] == clips
    ActionValidator(spec).validate(chunk, observation)
    chunk.left_arm[0][0] = 1
    with pytest.raises(ContractError, match="joint speed"):
        ActionValidator(spec).validate(chunk, observation)


@pytest.mark.parametrize("value", [-0.80, 0.006, float("nan"), float("inf")])
def test_large_or_nonfinite_violation_rejected(spec, pi_config, observation, value):
    spec.gripper_bounds = [[-0.785, 0], [-0.785, 0]]
    pi_config["gripper_clip_tolerance"] = 0.005
    with pytest.raises(ContractError):
        Pi05Adapter(spec, pi_config).decode_actions({"actions": [[0, 0, -0.785, value]]}, observation)


def test_gripper_speed_still_checked(spec, pi_config, observation):
    spec.gripper_bounds = [[-0.785, 0], [-0.785, 0]]
    observation.grippers = [0, 0]
    pi_config["gripper_clip_tolerance"] = 0.005
    chunk = Pi05Adapter(spec, pi_config).decode_actions({"actions": [[0, 0, -0.788, 0]]}, observation)
    with pytest.raises(ContractError, match="gripper speed"):
        ActionValidator(spec).validate(chunk, observation)


@pytest.mark.parametrize(
    "filename, clips, failed_step",
    [
        ("gripper_boundary.json", 86, 0),
        ("gripper_boundary_retry_1.json", 92, 0),
        ("gripper_boundary_retry_2.json", 92, 0),
    ],
)
def test_recorded_g2_prediction_against_actual_initial_observation(
    filename, clips, failed_step, g2_pi05_settings
):
    import json
    from pathlib import Path

    from agent2act.adapters.simulators.corobot import decode_request

    root = Path(__file__).resolve().parents[2]
    fixture = json.loads((root / "tests/fixtures/pi05" / filename).read_text())
    settings = g2_pi05_settings
    # These archived predictions were originally checked with a nominal 20 Hz profile.
    settings.robot.dt_s = 0.05
    options = {**settings.policy.backends["pi05"].options, "dt_s": 0.05, "retime_joint_actions": False}
    obs = decode_request(fixture["request"], settings.robot)
    raw = {"actions": fixture["actions"]}
    strict = Pi05Adapter(settings.robot, {**options, "gripper_clip_tolerance": 0})
    with pytest.raises(ContractError, match=f"gripper bounds exceeded at step {failed_step}"):
        ActionValidator(settings.robot).validate(strict.decode_actions(raw, obs), obs)
    chunk = Pi05Adapter(settings.robot, options).decode_actions(raw, obs)
    ActionValidator(settings.robot).validate(chunk, obs)
    assert chunk.steps == 50
    assert len(chunk.metadata["gripper_clips"]) == clips
    assert chunk.left_arm == [row[:7] for row in fixture["actions"]]
    assert chunk.right_arm == [row[7:14] for row in fixture["actions"]]
    # A larger error even in the unsent tail must still reject the whole prediction.
    fixture["actions"][-1][14] = -0.791
    with pytest.raises(ContractError, match="beyond clip tolerance at step 49"):
        Pi05Adapter(settings.robot, options).decode_actions(raw, obs)


def test_raw_prediction_archived_before_rejection(spec, pi_config, observation, tmp_path):
    import json

    import numpy as np

    from agent2act.adapters.telemetry import capture_artifacts
    from agent2act.service.trace import RequestTrace

    pi_config["gripper_clip_tolerance"] = 0.002
    raw = np.array([[0, 0, -0.1, 0.5]])
    with capture_artifacts(RequestTrace(tmp_path)):
        with pytest.raises(ContractError, match="beyond clip tolerance"):
            Pi05Adapter(spec, pi_config).decode_actions({"actions": raw}, observation)
    saved = json.loads(next(tmp_path.glob("*-policy-actions.json")).read_text())
    assert saved["actions"] == raw.tolist()

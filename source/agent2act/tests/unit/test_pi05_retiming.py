# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Retiming preserves target paths and refuses invalid or unbounded predictions."""

import json
from pathlib import Path

import pytest

from agent2act.adapters.policies.pi05 import Pi05Adapter
from agent2act.adapters.simulators.corobot import decode_request
from agent2act.domain.models import ContractError
from agent2act.domain.validation import ActionValidator


def test_retime_preserves_targets_and_joint_gripper_synchronization(spec, pi_config, observation):
    raw = {"actions": [[0.13, 0, 0.3, 0], [0.14, 0, 0.3, 0]]}
    strict = Pi05Adapter(spec, pi_config).decode_actions(raw, observation)
    with pytest.raises(ContractError, match="joint speed"):
        ActionValidator(spec).validate(strict, observation)
    pi_config["retime_joint_actions"] = True
    result = Pi05Adapter(spec, pi_config).decode_actions(raw, observation)
    ActionValidator(spec).validate(result, observation)
    assert result.steps == 6
    assert result.left_arm[-2:] == [[0.13], [0.14]]
    assert result.grippers[-2:] == [[0.3, 0], [0.3, 0]]
    assert result.metadata["retiming"]["inserted_steps"] == 4


def test_retime_checks_unsent_tail_and_horizon(spec, pi_config, observation):
    pi_config["retime_joint_actions"] = True
    adapter = Pi05Adapter(spec, pi_config)
    with pytest.raises(ContractError, match="gripper bounds"):
        adapter.decode_actions({"actions": [[0, 0, 0, 0], [0, 0, 9, 0]]}, observation)
    spec.joint_limits = [[-1, 1], [-1, 1]]
    with pytest.raises(ContractError, match="joint limit"):
        adapter.decode_actions({"actions": [[0, 0, 0, 0], [2, 0, 0, 0]]}, observation)
    spec.max_chunk_steps = 2
    with pytest.raises(ContractError, match="retimed trajectory exceeds"):
        adapter.decode_actions({"actions": [[0.2, 0, 0, 0]]}, observation)


@pytest.mark.parametrize("value", [1, "true", None])
def test_retime_requires_boolean(spec, pi_config, value):
    pi_config["retime_joint_actions"] = value
    with pytest.raises(ContractError, match="must be boolean"):
        Pi05Adapter(spec, pi_config)


def test_recorded_rejected_continuations(g2_pi05_settings):
    root = Path(__file__).resolve().parents[2]
    settings = g2_pi05_settings
    options = dict(settings.policy.backends["pi05"].options)
    for case in json.loads((root / "tests/fixtures/pi05/continuation_speed.json").read_text()):
        obs = decode_request(case["request"], settings.robot)
        raw = {"actions": case["actions"]}
        strict = Pi05Adapter(settings.robot, {**options, "retime_joint_actions": False})
        with pytest.raises(ContractError, match="joint speed exceeded at step 0"):
            ActionValidator(settings.robot).validate(strict.decode_actions(raw, obs), obs)
        result = Pi05Adapter(settings.robot, options).decode_actions(raw, obs)
        ActionValidator(settings.robot).validate(result, obs)
        assert result.metadata["retiming"]["inserted_steps"] >= 2
        assert result.left_arm[-1] == raw["actions"][-1][:7]
        assert result.right_arm[-1] == raw["actions"][-1][7:14]


@pytest.mark.parametrize("metadata", [None, {}, {"training_config": "picking"}])
def test_wrong_checkpoint_closes_client_before_inference(spec, pi_config, observation, monkeypatch, metadata):
    pi_config["expected_training_config"] = "spatial"

    class Client:
        closed = False

        def infer(self, _):
            pytest.fail("wrong checkpoint must not receive an inference")

        def close(self):
            self.closed = True

    client = Client()
    client.metadata = metadata
    adapter = Pi05Adapter(spec, pi_config, client_factory=lambda: client)
    monkeypatch.setattr(adapter, "encode_observation", lambda *_: {})
    with pytest.raises(ContractError, match="training config mismatch"):
        adapter.generate(observation, "stack")
    assert client.closed
    assert adapter.client is None


def test_matching_checkpoint_can_infer(spec, pi_config, observation, monkeypatch):
    pi_config["expected_training_config"] = "spatial"

    class Client:
        metadata = {"training_config": "spatial"}

        def infer(self, _):
            return {"actions": [[0, 0, 0.5, 0.5]]}

        def close(self):
            pass

    adapter = Pi05Adapter(spec, pi_config, client_factory=Client)
    monkeypatch.setattr(adapter, "encode_observation", lambda *_: {})
    try:
        assert adapter.generate(observation, "stack").steps == 1
    finally:
        adapter.close()

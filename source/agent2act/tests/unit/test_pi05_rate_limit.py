# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Sampled references retain their cadence without bypassing raw-target checks."""

import copy
from dataclasses import replace

import pytest

from agent2act.adapters.policies.pi05 import Pi05Adapter
from agent2act.application.proposals import PolicyProposal
from agent2act.domain.models import ContractError, Decision
from agent2act.domain.validation import ActionValidator


def test_gripper_spike_does_not_stretch_or_slow_other_channels(spec, pi_config, observation):
    pi_config["rate_limit_joint_actions"] = True
    observation.grippers = [0, 0]
    raw = {"actions": [[0.03, 0, 0.9, 0], [0.06, 0, 0, 0], [0.09, 0, 0, 0]]}
    before = copy.deepcopy(raw)
    result = Pi05Adapter(spec, pi_config).decode_actions(raw, observation)
    assert raw == before
    assert result.steps == 3
    assert result.dt_s == spec.dt_s
    assert result.left_arm == [[0.03], [0.06], [0.09]]
    assert result.grippers == [[0.1, 0], [0, 0], [0, 0]]
    assert result.metadata["conditioning"]["limited_steps"] == 1
    assert result.metadata["conditioning"]["max_target_error"][2] == pytest.approx(0.8)
    assert result.metadata["conditioning"]["endpoint_target_minus_command"] == [0, 0, 0, 0]
    ActionValidator(spec).validate(result, observation)


def test_endpoint_may_remain_unreached_and_is_reported(spec, pi_config, observation):
    pi_config["rate_limit_joint_actions"] = True
    result = Pi05Adapter(spec, pi_config).decode_actions({"actions": [[0.3, 0, 0, 0]]}, observation)
    assert result.steps == 1
    assert result.left_arm == [[spec.max_joint_speed * spec.dt_s]]
    assert result.metadata["conditioning"]["endpoint_target_minus_command"][0] == pytest.approx(0.25)
    ActionValidator(spec).validate(result, observation)


def test_review_preserves_conditioned_commands_and_diagnostics(spec, pi_config, observation):
    pi_config["rate_limit_joint_actions"] = True
    chunk = Pi05Adapter(spec, pi_config).decode_actions({"actions": [[0.3, 0, 0, 0]]}, observation)
    source = Decision("policy", observation.instruction, "review", backend="p")
    proposal = PolicyProposal(observation, 1, source, chunk, {}, 1, 10)
    description = proposal.describe()
    assert description["conditioning"] == chunk.metadata["conditioning"]
    description["conditioning"]["max_target_error"][0] = 99
    chunk.left_arm[0][0] = 99
    chunk.metadata["conditioning"]["max_target_error"][0] = 99
    accepted = replace(source, proposal_id=proposal.id)
    result = proposal.consume(observation, 1, accepted)
    assert result.left_arm == [[0.05]]
    assert result.metadata["conditioning"]["max_target_error"][0] == pytest.approx(0.25)
    with pytest.raises(ContractError, match="already consumed"):
        proposal.consume(observation, 1, accepted)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), 2.0])
def test_invalid_raw_tail_is_rejected_before_conditioning(spec, pi_config, observation, invalid):
    spec.joint_limits = [[-1, 1], [-1, 1]]
    pi_config["rate_limit_joint_actions"] = True
    with pytest.raises(ContractError):
        Pi05Adapter(spec, pi_config).decode_actions(
            {"actions": [[0, 0, 0, 0], [invalid, 0, 0, 0]]}, observation
        )
    with pytest.raises(ContractError, match="gripper bounds"):
        Pi05Adapter(spec, pi_config).decode_actions({"actions": [[0, 0, 2, 0]]}, observation)


def test_gripper_limits_start_at_previous_issued_command(spec, pi_config, observation):
    pi_config["rate_limit_joint_actions"] = True
    observation.issued_gripper_targets = [0.6, 0.5]
    result = Pi05Adapter(spec, pi_config).decode_actions({"actions": [[0, 0, 0, 0]]}, observation)
    assert result.grippers[0] == pytest.approx([0.5, 0.4])
    ActionValidator(spec).validate(result, observation)


def test_body_channels_keep_their_own_speed_limits(spec, pi_config, observation):
    spec.body_joints = ["body"]
    spec.body_limits = [[-1, 1]]
    spec.body_speed_limits = [0.5]
    observation.body = [0]
    pi_config.update(
        rate_limit_joint_actions=True,
        body_indices=[4],
        action_dim=5,
        action_scale=[1] * 5,
        action_offset=[0] * 5,
    )
    result = Pi05Adapter(spec, pi_config).decode_actions({"actions": [[0.02, 0, 0.9, 0, 0.5]]}, observation)
    assert result.left_arm == [[0.02]]
    assert result.body == [[0.025]]
    ActionValidator(spec).validate(result, observation)


@pytest.mark.parametrize("value", [None, 1, "true"])
def test_rate_limit_configuration_is_explicit(spec, pi_config, value):
    pi_config["rate_limit_joint_actions"] = value
    with pytest.raises(ContractError, match="must be boolean"):
        Pi05Adapter(spec, pi_config)


def test_rate_limit_rejects_conflicting_timing_and_eef(spec, pi_config):
    pi_config.update(rate_limit_joint_actions=True, retime_joint_actions=True)
    with pytest.raises(ContractError, match="choose either"):
        Pi05Adapter(spec, pi_config)
    pi_config.update(retime_joint_actions=False, kind="EEF_ABS")
    with pytest.raises(ContractError, match="requires JOINT_ABS"):
        Pi05Adapter(spec, pi_config)

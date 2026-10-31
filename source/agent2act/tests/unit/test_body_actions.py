# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Body channels preserve checkpoint masks and share action validation/retiming."""

from dataclasses import replace

import pytest

from agent2act.adapters.policies.pi05 import Pi05Adapter
from agent2act.adapters.simulators.corobot import encode_result
from agent2act.domain.models import ContractError
from agent2act.domain.validation import ActionValidator


def test_masked_body_hold_and_yaw_retiming(spec, observation, pi_config):
    spec = replace(
        spec, body_joints=["held", "yaw"], body_limits=[[-1, 1], [-1, 1]], body_speed_limits=[1, 1]
    )
    observation = replace(observation, body=[0.5, 0])
    pi_config.update(
        action_dim=6,
        action_scale=[1] * 6,
        action_offset=[0] * 6,
        body_indices=[None, 5],
        retime_joint_actions=True,
    )
    adapter = Pi05Adapter(spec, pi_config)
    chunk = adapter.decode_actions({"actions": [[0, 0, 0.5, 0.5, 0, 0.15]]}, observation)
    assert chunk.steps == 3
    assert chunk.body[-1] == [0.5, 0.15]
    ActionValidator(spec).validate(chunk, observation)
    wire = encode_result(chunk.prefix(2), history={}, need_depth=False)["result"]
    assert wire["waist"]["kind"] == "JOINT_ABS"
    assert len(wire["waist"]["values"]) == len(wire["left_arm"]["values"]) == 2
    with pytest.raises(ContractError, match="body joint limit"):
        adapter.decode_actions({"actions": [[0, 0, 0.5, 0.5, 0, 2]]}, observation)
    with pytest.raises(ContractError, match="body actions require"):
        ActionValidator(replace(spec, body_joints=[], body_limits=[], body_speed_limits=[])).validate(
            chunk, observation
        )


def test_body_tail_and_speed_cannot_be_hidden_by_prefix(spec, observation, pi_config):
    spec = replace(spec, body_joints=["yaw"], body_limits=[[-1, 1]], body_speed_limits=[1])
    observation = replace(observation, body=[0])
    pi_config.update(action_dim=5, action_scale=[1] * 5, action_offset=[0] * 5, body_indices=[4])
    chunk = Pi05Adapter(spec, pi_config).decode_actions(
        {"actions": [[0, 0, 0, 0, 0.01], [0, 0, 0, 0, 0.5]]}, observation
    )
    with pytest.raises(ContractError, match="body joint speed"):
        ActionValidator(spec).validate(chunk, observation)

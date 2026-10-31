# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Measured overshoot is preserved while generated commands remain feasible."""

import pytest

from agent2act.adapters.policies.pi05 import Pi05Adapter
from agent2act.domain.actions import hold_chunk
from agent2act.domain.models import ContractError
from agent2act.domain.validation import ActionValidator
from agent2act.execution.interpolation import LocalInterpolation


@pytest.mark.parametrize("kind", ["hold", "motion", "policy"])
def test_observed_overshoot_does_not_make_recovery_impossible(spec, observation, pi_config, kind):
    spec.gripper_bounds = [[-0.785, 0], [-0.785, 0]]
    observation.grippers = [-0.7855260372161865, -0.7849827408790588]
    original = list(observation.grippers)
    if kind == "hold":
        chunk = hold_chunk(observation, spec)
    elif kind == "motion":
        chunk = LocalInterpolation(spec).generate(
            observation, {"frame": "base_link", "waypoints": [{"duration_s": 0.5}]}
        )
    else:
        pi_config["retime_joint_actions"] = True
        # Retiming has to insert rows because of the joint jump. Interpolating
        # from raw overshoot would leave those rows outside the gripper bound.
        chunk = Pi05Adapter(spec, pi_config).decode_actions(
            {"actions": [[0.13, 0, -0.785, -0.785]]}, observation
        )
    ActionValidator(spec).validate(chunk, observation)
    assert observation.grippers == original
    assert chunk.metadata["gripper_start_correction"]["observed"] == original
    assert all(row[0] == -0.785 for row in chunk.grippers)


@pytest.mark.parametrize("kind", ["hold", "motion", "policy"])
def test_large_observation_error_cannot_bypass_speed(spec, observation, pi_config, kind):
    spec.gripper_bounds = [[-0.785, 0], [-0.785, 0]]
    observation.grippers = [-1.2, -0.785]
    with pytest.raises(ContractError, match="gripper speed"):
        if kind == "hold":
            chunk = hold_chunk(observation, spec)
        elif kind == "motion":
            chunk = LocalInterpolation(spec).generate(
                observation, {"frame": "base_link", "waypoints": [{"duration_s": 0.5}]}
            )
        else:
            pi_config["retime_joint_actions"] = True
            chunk = Pi05Adapter(spec, pi_config).decode_actions(
                {"actions": [[0.13, 0, -0.785, -0.785]]}, observation
            )
        ActionValidator(spec).validate(chunk, observation)


def test_invalid_requested_target_is_not_projected(spec, observation):
    chunk = LocalInterpolation(spec).generate(
        observation, {"frame": "base_link", "waypoints": [{"grippers": [-0.01, 0.5]}]}
    )
    with pytest.raises(ContractError, match="gripper bounds"):
        ActionValidator(spec).validate(chunk, observation)

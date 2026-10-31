# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""A planner completion or abandonment is issued as one long hold so the simulator can finish its checks."""

import pytest

from agent2act.application.settings import ExecutionConfig
from agent2act.bootstrap import build_session
from agent2act.domain.models import ContractError


def _decision(status):
    return {
        "mode": "motion",
        "instruction": "hold the finished state",
        "reason": "all checks look satisfied",
        "status": status,
        "motion": {"frame": "base_link", "waypoints": [{}]},
    }


@pytest.mark.parametrize("hold_steps, expected", [(None, 1), (30, 30), (1000, 1000)])
@pytest.mark.parametrize(
    "status, marker", [("complete", "planner_complete_unverified"), ("abandon", "planner_abandoned")]
)
def test_final_hold_length_is_configured(config, observation, hold_steps, expected, status, marker):
    config["execution"]["execute_steps"] = 2
    config["execution"]["complete_hold_steps"] = hold_steps
    config["planner"]["decisions"] = [_decision(status)]
    with build_session(config) as session:
        # A final hold may be longer than any generated motion chunk.
        assert session.runtime.spec.max_chunk_steps < 1000
        chunk = session.runtime.step(observation)
    assert chunk.steps == expected
    assert chunk.metadata["status"] == marker
    count = len(chunk.left_arm[0])
    assert all(row == list(observation.joints[:count]) for row in chunk.left_arm)
    assert all(row == chunk.grippers[0] for row in chunk.grippers)


def test_completion_hold_does_not_lengthen_continuing_motion(config, observation):
    config["execution"].update(execute_steps=2, complete_hold_steps=30)
    config["planner"]["decisions"] = [_decision("continue")]
    with build_session(config) as session:
        assert session.runtime.step(observation).steps <= 2


def test_completion_hold_is_validated():
    with pytest.raises(ContractError, match="complete_hold_steps"):
        ExecutionConfig(complete_hold_steps=0)


def test_long_hold_does_not_raise_the_motion_chunk_limit(config, observation):
    config["execution"].update(execute_steps=2, complete_hold_steps=1000)
    with build_session(config) as session:
        assert session.runtime.validator.spec.max_chunk_steps == config["robot"].get("max_chunk_steps", 256)

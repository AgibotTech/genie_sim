# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Contact must not turn measured finger positions into new actuator targets."""

import copy
from dataclasses import replace

import pytest

from agent2act.bootstrap import build_session
from agent2act.domain.actions import hold_chunk
from agent2act.domain.models import ContractError
from agent2act.domain.validation import ActionValidator


def decision(waypoint, status="continue"):
    return {
        "mode": "motion",
        "instruction": "maintain chosen grasp",
        "reason": "inspect retention",
        "status": status,
        "motion": {"frame": "base_link", "waypoints": [waypoint]},
    }


@pytest.mark.parametrize("ending", ["complete", "motion", "episode_done"])
def test_contact_hold_keeps_issued_target_and_measured_feedback(config, observation, ending):
    config["execution"]["execute_steps"] = 120
    config["planner"]["decisions"] = [
        decision({"duration_s": 0.5, "grippers": [0.9, 0.5]}),
        decision({}, "complete" if ending == "complete" else "continue"),
    ]
    events = []
    with build_session(config, event_sink=events.append) as session:
        runtime = session.runtime
        contexts = []
        original = runtime.planner.decide

        def capture(obs, context):
            contexts.append(copy.deepcopy(context))
            return original(obs, context)

        runtime.planner.decide = capture
        first = runtime.step(observation)
        assert first.grippers[-1] == [0.9, 0.5]
        contact = replace(observation, grippers=[0.6, 0.5], done=ending == "episode_done")
        second = runtime.step(contact)
        assert all(g == [0.9, 0.5] for g in second.grippers)
        assert contact.grippers == [0.6, 0.5]
        assert contact.issued_gripper_targets is None
        tracking = next(e for e in events if e.get("status") == "observed_tracking")
        assert tracking["grippers"]["observed"] == [0.6, 0.5]
        assert tracking["grippers"]["issued_target"] == [0.9, 0.5]
        assert "unverified" in tracking["playback_status"]
        if ending != "episode_done":
            assert contexts[1]["gripper_commands"] == [0.9, 0.5]
            assert contexts[1]["observed_grippers_in_command_units"] == [0.6, 0.5]
        if ending == "complete":
            assert second.steps == 1
            assert second.metadata["status"] == "planner_complete_unverified"


def test_continuity_uses_issued_prefix_and_resets_on_new_instruction(config, observation):
    config["execution"].update(execute_steps=2, max_motion_continuations=0)
    config["planner"]["decisions"] = [
        decision({"duration_s": 1, "grippers": [0.9, 0.5]}),
        decision({}, "complete"),
    ]
    with build_session(config) as session:
        runtime = session.runtime
        first = runtime.step(observation)
        target = first.grippers[-1]
        assert target[0] < 0.9
        second = runtime.step(replace(observation, grippers=[0.51, 0.5]))
        assert second.grippers == [target]
        # A new instruction resets the planner and the runtime-owned reference.
        new_obs = replace(
            observation, instruction="new episode", grippers=[0.2, 0.5], issued_gripper_targets=[1, 1]
        )
        new_chunk = runtime.step(new_obs)
        assert 0.2 < new_chunk.grippers[0][0] < target[0]


def test_rejected_target_does_not_replace_last_issued_command(config, observation):
    config["execution"].update(execute_steps=120, max_replans=1)
    config["planner"]["decisions"] = [
        decision({"grippers": [0.9, 0.5]}),
        decision({"grippers": [1.1, 0.5]}),
        decision({}, "complete"),
    ]
    with build_session(config) as session:
        session.runtime.step(observation)
        held = session.runtime.step(replace(observation, grippers=[0.6, 0.5]))
        assert held.grippers == [[0.9, 0.5]]


def test_gripper_speed_still_bounds_new_setpoints_with_command_reference(spec, observation):
    obs = replace(observation, grippers=[0.6, 0.5], issued_gripper_targets=[0.9, 0.5])
    held = hold_chunk(obs, spec)
    ActionValidator(spec).validate(held, obs)
    held.grippers[0][0] = 0.6  # Sudden release from the last command, despite matching measurement.
    with pytest.raises(ContractError, match="gripper speed"):
        ActionValidator(spec).validate(held, obs)
    held.grippers[0][0] = 1.1
    with pytest.raises(ContractError, match="gripper bounds"):
        ActionValidator(spec).validate(held, obs)

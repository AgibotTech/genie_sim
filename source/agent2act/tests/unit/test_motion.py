# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import pytest

from agent2act.bootstrap import build_session
from agent2act.domain.errors import ContractError
from agent2act.domain.models import Decision
from agent2act.execution.interpolation import LocalInterpolation


def test_motion_decision_requires_intent():
    with pytest.raises(ContractError, match="motion requires"):
        Decision.from_dict({"mode": "motion", "instruction": "move", "reason": "test"})


def test_motion_compiler_normalizes_pose_and_respects_limits(spec, observation):
    target = observation.eef("right", "base_link")
    target[2] += 0.05
    target[3:] = [0, 0, 0, 2]
    chunk = LocalInterpolation(spec).generate(
        observation,
        {"frame": "base_link", "waypoints": [{"duration_s": 0.05, "right_pose": target}]},
    )
    assert chunk.steps > 1
    assert chunk.right_arm[-1][3:] == pytest.approx([0, 0, 0, 1])
    for previous, current in zip(chunk.right_arm, chunk.right_arm[1:]):
        assert sum((a - b) ** 2 for a, b in zip(previous[:3], current[:3])) ** 0.5 <= (
            spec.max_eef_speed * spec.dt_s + 1e-9
        )


def test_demo_uses_single_motion_executor(config, request_payload):
    with build_session(config) as session:
        result = session.handle(request_payload)["result"]
    assert result["agent2act"]["mode"] == "motion"
    assert result["agent2act"]["compiler"] == "local_interpolation"


def test_automatic_timing_reaches_target_with_validated_speeds(spec, observation):
    from agent2act.domain.validation import ActionValidator

    target = observation.eef("left", "base_link")
    target[0] += 0.03
    request = {"frame": "base_link", "waypoints": [{"left_pose": target}]}
    chunk = LocalInterpolation(spec).generate(observation, request)
    ActionValidator(spec).validate(chunk, observation)
    assert chunk.left_arm[-1] == pytest.approx(target)
    assert chunk.steps > 1
    timed = LocalInterpolation(spec).generate(
        observation,
        {
            "frame": "base_link",
            "waypoints": [{"left_pose": target, "duration_s": chunk.steps * spec.dt_s * 2}],
        },
    )
    assert timed.steps == chunk.steps * 2


@pytest.mark.parametrize("duration", [None, 0, -1, True, "auto"])
def test_automatic_timing_does_not_accept_invalid_explicit_duration(spec, observation, duration):
    with pytest.raises(ContractError):
        LocalInterpolation(spec).generate(
            observation, {"frame": "base_link", "waypoints": [{"duration_s": duration}]}
        )

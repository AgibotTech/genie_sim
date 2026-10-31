# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Operating pace remains independent of robot identity and safety limits."""

import copy
import math

import pytest

from agent2act.bootstrap import build_session
from agent2act.configuration import parse_settings
from agent2act.domain.errors import ConfigurationError, ContractError
from agent2act.domain.geometry import quaternion_angle, smooth_cruise
from agent2act.domain.timing import MotionTiming
from agent2act.domain.validation import ActionValidator
from agent2act.execution.interpolation import LocalInterpolation


@pytest.mark.parametrize("dt", [0.01, 0.05, 0.1])
@pytest.mark.parametrize("profile", ["quintic", "smooth_cruise"])
def test_pace_changes_timing_not_target_or_safety(spec, observation, dt, profile):
    spec.dt_s = dt
    target = observation.eef("left", "base_link")
    target[0] += 0.04
    motion = {"frame": "base_link", "waypoints": [{"left_pose": target}]}
    before = copy.deepcopy(spec)
    fast = LocalInterpolation(spec, MotionTiming(profile=profile)).generate(observation, motion)
    slow = LocalInterpolation(spec, MotionTiming(speed_fraction=0.25, profile=profile)).generate(
        observation, motion
    )
    assert slow.steps >= fast.steps
    assert spec == before
    assert slow.left_arm[-1] == fast.left_arm[-1]
    last = observation.eef("left", "base_link")
    for pose in slow.left_arm:
        assert math.dist(last[:3], pose[:3]) <= spec.max_eef_speed * 0.25 * dt + 1e-9
        last = pose
    ActionValidator(spec).validate(slow, observation)


def test_arm_and_gripper_minimum_and_target_hold(spec, observation):
    target = observation.eef("left", "base_link")
    target[0] += 0.001
    grippers = observation.command_grippers(spec)
    grippers[0] += 0.01
    timing = MotionTiming(min_duration_s=0.3, min_gripper_duration_s=0.5, target_hold_s=0.2)
    compiler = LocalInterpolation(spec, timing)
    chunk = compiler.generate(
        observation,
        {"frame": "base_link", "waypoints": [{"left_pose": target, "grippers": grippers, "duration_s": 0.1}]},
    )
    assert chunk.metadata["motion_steps"] == math.ceil(0.5 / spec.dt_s)
    assert chunk.metadata["target_hold_steps"] == math.ceil(0.2 / spec.dt_s)
    assert chunk.left_arm[-1] == pytest.approx(target)
    assert chunk.grippers[-1] == pytest.approx(grippers)
    for row in chunk.left_arm[chunk.metadata["motion_steps"] :]:
        assert row == pytest.approx(target)  # Hold the target, never the old observed pose.
    ActionValidator(spec).validate(chunk, observation)
    hold = compiler.generate(observation, {"frame": "base_link", "waypoints": [{}]})
    assert hold.metadata["motion_steps"] == 1  # No automatic task-stage or settling loop.


def test_target_hold_cannot_bypass_chunk_limit(spec, observation):
    with pytest.raises(ContractError, match="target hold"):
        LocalInterpolation(spec, MotionTiming(target_hold_s=60)).generate(
            observation, {"frame": "base_link", "waypoints": [{}]}
        )


@pytest.mark.parametrize(
    "options",
    [{"speed_fraction": v} for v in [0, 1.1, True, "slow", float("nan")]]
    + [
        {key: value}
        for key in ["min_duration_s", "min_gripper_duration_s", "target_hold_s"]
        for value in [-1, None, True, 61]
    ]
    + [{"profile": value} for value in ["linear", None, True, 1, {}, []]]
    + [{"unknown": 1}],
)
def test_config_rejects_invalid_timing(config, options):
    config["motion"] = options
    with pytest.raises(ConfigurationError):
        parse_settings(config)


def test_hold_prefix_does_not_create_motion_continuation(config, request_payload):
    config["motion"] = {"target_hold_s": 0.5}
    config["execution"].update(execute_steps=2, max_motion_continuations=2)
    decision = config["planner"]["decisions"][0]
    decision["motion"]["waypoints"] = [{}]
    config["planner"]["decisions"] = [decision, copy.deepcopy(decision)]
    events = []
    with build_session(config, event_sink=events.append) as session:
        session.handle(request_payload)
        assert session.runtime._pending_motion is None
        session.handle(request_payload)
    tracking = next(e for e in events if e.get("status") == "observed_tracking")
    assert tracking["issued_steps"] == 2
    assert tracking["discarded_steps"] == 9
    assert tracking["grippers"]["observed"] == tracking["grippers"]["issued_target"]
    assert tracking["task_success"] == "unverified"
    assert "actual playback is unverified" in tracking["summary"]
    assert "successful grasp" in tracking["summary"]


def test_timing_roundtrip_and_bootstrap(config):
    config["motion"] = {
        "speed_fraction": 0.3,
        "min_duration_s": 0.2,
        "profile": "smooth_cruise",
    }
    settings = parse_settings(config)
    assert parse_settings(settings.model_dump()).motion == settings.motion
    with build_session(settings) as session:
        advertised = session.runtime.router.describe()["executors"]["motion"]["compiler"]
        assert advertised["operating_timing"]["speed_fraction"] == 0.3
        assert advertised["operating_timing"]["profile"] == "smooth_cruise"
        assert advertised["peak_speed_factor"] == 1.25


@pytest.mark.parametrize("execute_steps", [4, 17])
@pytest.mark.parametrize("motion_options", [None, {}, {"profile": "smooth_cruise"}])
def test_unlimited_default_timing_matches_execution_prefix(
    config, request_payload, execute_steps, motion_options
):
    config["robot"]["enforce_speed_limits"] = False
    config["execution"]["execute_steps"] = execute_steps
    if motion_options is not None:
        config["motion"] = motion_options
    target = [0.6, -0.2, 0.5, 0, 0, 0, 1]
    config["planner"]["decisions"][0]["motion"]["waypoints"] = [{"right_pose": target}]
    before = copy.deepcopy(config)
    settings = parse_settings(config)
    assert "default_steps" not in settings.model_dump()["motion"]
    assert parse_settings(settings.model_dump()).motion == settings.motion
    assert config == before
    events = []
    with build_session(settings, event_sink=events.append) as session:
        session.handle(request_payload)
        compiler = session.runtime.router.describe()["executors"]["motion"]["compiler"]
        assert compiler["operating_timing"]["default_steps"] == execute_steps
    issued = next(e for e in events if e.get("status") == "issued")
    assert issued["generated_steps"] == issued["issued_steps"] == execute_steps
    assert issued["discarded_steps"] == 0
    assert issued["issued_endpoint"]["right_arm"] == pytest.approx(target)


def test_removed_default_steps_configuration_is_rejected(config):
    config["motion"] = {"default_steps": 30}
    with pytest.raises(ConfigurationError, match="default_steps"):
        parse_settings(config)


def test_unlimited_explicit_duration_preserves_longer_interpolation_and_prefix(config, request_payload):
    config["robot"]["enforce_speed_limits"] = False
    config["execution"]["execute_steps"] = 4
    target = [0.6, -0.2, 0.5, 0, 0, 0, 1]
    config["planner"]["decisions"][0]["motion"]["waypoints"] = [{"right_pose": target, "duration_s": 1.5}]
    events = []
    with build_session(config, event_sink=events.append) as session:
        session.handle(request_payload)
    issued = next(e for e in events if e.get("status") == "issued")
    assert issued["generated_steps"] == 30
    assert issued["issued_steps"] == 4
    assert issued["discarded_steps"] == 26
    assert issued["issued_endpoint"]["right_arm"] != pytest.approx(target)


def test_cruise_progress_is_monotone_symmetric_and_c2():
    samples = [smooth_cruise(i / 1000) for i in range(1001)]
    assert samples[0] == 0
    assert samples[-1] == 1
    assert all(0 <= b - a <= 1.25 / 1000 + 1e-12 for a, b in zip(samples, samples[1:]))
    assert all(a + b == pytest.approx(1) for a, b in zip(samples, reversed(samples)))
    h = 1e-5
    for t in (0.2, 0.8):
        left_velocity = (smooth_cruise(t) - smooth_cruise(t - h)) / h
        right_velocity = (smooth_cruise(t + h) - smooth_cruise(t)) / h
        assert left_velocity == pytest.approx(right_velocity, abs=1e-7)
        acceleration = (smooth_cruise(t + h) - 2 * smooth_cruise(t) + smooth_cruise(t - h)) / h**2
        assert abs(acceleration) < 0.002
    assert smooth_cruise(h) / h < 1e-7
    assert (smooth_cruise(2 * h) - 2 * smooth_cruise(h)) / h**2 < 0.002


@pytest.mark.parametrize("dt", [0.01, 1 / 30, 0.05])
def test_cruise_dual_arm_rotation_gripper_and_hold_obey_limits(spec, observation, dt):
    spec.dt_s = dt
    left = observation.eef("left", "base_link")
    right = observation.eef("right", "base_link")
    left[:3] = [left[0] + 0.10, left[1] - 0.06, left[2] + 0.03]
    right[:3] = [right[0] - 0.06, right[1] + 0.03, right[2] + 0.04]
    left[3:] = [0, math.sin(0.2), 0, math.cos(0.2)]
    right[3:] = [math.sin(0.3), 0, 0, math.cos(0.3)]
    grippers = [sum(bounds) / 2 for bounds in spec.gripper_bounds]
    motion = {
        "frame": "base_link",
        "waypoints": [{"left_pose": left, "right_pose": right, "grippers": grippers}],
    }
    before = copy.deepcopy(spec)
    pace = 0.7
    cruise = LocalInterpolation(
        spec, MotionTiming(profile="smooth_cruise", speed_fraction=pace, target_hold_s=0.1)
    ).generate(observation, motion)
    quintic = LocalInterpolation(spec, MotionTiming(speed_fraction=pace, target_hold_s=0.1)).generate(
        observation, motion
    )
    assert cruise.steps < quintic.steps
    assert spec == before
    ActionValidator(spec).validate(cruise, observation)
    for arm, target in (("left", left), ("right", right)):
        rows = getattr(cruise, arm + "_arm")
        previous = observation.eef(arm, "base_link")
        for row in rows:
            assert math.dist(row[:3], previous[:3]) <= spec.max_eef_speed * pace * dt + 1e-9
            assert quaternion_angle(row[3:], previous[3:]) <= spec.max_eef_angular_speed * pace * dt + 1e-9
            previous = row
        assert rows[-1] == pytest.approx(target)
        assert all(row == pytest.approx(target) for row in rows[cruise.metadata["motion_steps"] :])
    previous_grippers = observation.command_grippers(spec)
    for row in cruise.grippers:
        assert (
            max(abs(a - b) for a, b in zip(row, previous_grippers))
            <= spec.max_gripper_speed * pace * dt + 1e-9
        )
        previous_grippers = row
    assert cruise.grippers[-1] == pytest.approx(grippers)
    explicit = LocalInterpolation(spec, MotionTiming(profile="smooth_cruise")).generate(
        observation,
        {"frame": "base_link", "waypoints": [{**motion["waypoints"][0], "duration_s": quintic.steps * dt}]},
    )
    assert explicit.steps == quintic.steps


def test_short_rotation_cruise_accounts_for_normalized_linear_peak(spec, observation):
    spec.dt_s = 0.01
    spec.max_eef_angular_speed = 0.05
    left = observation.eef("left", "base_link")
    assert left[3:] == [0, 0, 0, 1]
    left[3:] = [0, 0, math.sin(0.02), math.cos(0.02)]
    chunk = LocalInterpolation(spec, MotionTiming(profile="smooth_cruise")).generate(
        observation, {"frame": "base_link", "waypoints": [{"left_pose": left}]}
    )
    # Merely using the arc angle requires 100 steps, but NLERP then exceeds the
    # angular limit around its midpoint. The exact rate bound needs 101 steps.
    assert chunk.steps == 101
    ActionValidator(spec).validate(chunk, observation)


def test_cruise_retimes_gripper_only_motion(spec, observation):
    spec.max_gripper_speed = 0.4
    start = observation.command_grippers(spec)
    target = [
        high if value < (low + high) / 2 else low for value, (low, high) in zip(start, spec.gripper_bounds)
    ]
    request = {"frame": "base_link", "waypoints": [{"grippers": target}]}
    cruise = LocalInterpolation(spec, MotionTiming(profile="smooth_cruise")).generate(observation, request)
    quintic = LocalInterpolation(spec).generate(observation, request)
    assert cruise.steps < quintic.steps
    assert cruise.grippers[-1] == pytest.approx(target)
    ActionValidator(spec).validate(cruise, observation)


def test_cruise_cannot_bypass_chunk_limit(spec, observation):
    spec.max_chunk_steps = 2
    left = observation.eef("left", "base_link")
    left[0] += 1
    with pytest.raises(ContractError, match="speed timing"):
        LocalInterpolation(spec, MotionTiming(profile="smooth_cruise")).generate(
            observation, {"frame": "base_link", "waypoints": [{"left_pose": left}]}
        )
    with pytest.raises(ContractError, match="target hold"):
        LocalInterpolation(spec, MotionTiming(profile="smooth_cruise", target_hold_s=1)).generate(
            observation, {"frame": "base_link", "waypoints": [{}]}
        )


@pytest.mark.parametrize("profile", ["quintic", "smooth_cruise"])
def test_disabled_speed_limits_reach_arm_rotation_and_gripper_targets_in_ten_steps(
    spec, observation, profile
):
    spec.enforce_speed_limits = False
    spec.max_eef_speed = spec.max_eef_angular_speed = spec.max_gripper_speed = 1e-6
    target = observation.eef("left", "base_link")
    target[0] += 0.2
    target[3:] = [0, 0, math.sin(0.4), math.cos(0.4)]
    grippers = [high for low, high in spec.gripper_bounds]
    request = {"frame": "base_link", "waypoints": [{"left_pose": target, "grippers": grippers}]}
    compiler = LocalInterpolation(spec, MotionTiming(profile=profile, speed_fraction=0.25))
    result = compiler.generate(observation, request)
    assert result.steps == 10
    assert result.left_arm[-1] == pytest.approx(target)
    assert result.grippers[-1] == pytest.approx(grippers)
    assert compiler.describe()["speed_limits_enforced"] is False
    ActionValidator(spec).validate(result, observation)
    spec.enforce_speed_limits = True
    with pytest.raises(ContractError, match="speed"):
        ActionValidator(spec).validate(result, observation)


@pytest.mark.parametrize("duration,steps", [(None, 7), (0.15, 3)])
def test_unlimited_motion_preserves_explicit_duration_minimum_hold_and_default_override(
    spec, observation, duration, steps
):
    spec.enforce_speed_limits = False
    target = observation.eef("left", "base_link")
    target[0] += 0.1
    wp = {"left_pose": target}
    if duration is not None:
        wp["duration_s"] = duration
    compiler = LocalInterpolation(spec, MotionTiming(default_steps=7, min_duration_s=0.1, target_hold_s=0.1))
    chunk = compiler.generate(observation, {"frame": "base_link", "waypoints": [wp]})
    assert chunk.metadata["motion_steps"] == steps
    assert chunk.metadata["target_hold_steps"] == 2
    assert chunk.left_arm[-1] == pytest.approx(target)
    assert (
        compiler.generate(observation, {"frame": "base_link", "waypoints": [{}]}).metadata["motion_steps"]
        == 1
    )
    spec.max_chunk_steps = 2
    with pytest.raises(ContractError, match="max_chunk_steps|duration out of range"):
        compiler.generate(observation, {"frame": "base_link", "waypoints": [wp]})


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "10", None])
def test_default_interpolation_steps_reject_invalid_values(value):
    with pytest.raises(ContractError, match="default_steps"):
        MotionTiming(default_steps=value)

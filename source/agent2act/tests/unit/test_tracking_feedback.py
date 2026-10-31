# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Directional evidence does not grant motion or validate physical success."""

import copy
import json
import math

import pytest

from agent2act.adapters.planners.responses import ResponsesPlanner
from agent2act.application.runtime import _translation_feedback
from agent2act.bootstrap import ApplicationFactory, build_session, default_registry
from agent2act.configuration import parse_settings


@pytest.mark.parametrize(
    "commanded,observed,fraction,cross_track",
    [
        ([0.05, 0, 0], [0.03, 0, 0], 0.6, 0),
        ([0.05, 0, 0], [-0.05, 0, 0], -1, 0),
        ([0.05, 0, 0], [0, 0.05, 0], 0, 0.05),
        ([0.05, 0, 0], [0.075, 0, 0], 1.5, 0),
        ([0.05, 0, 0], [0.03, 0.04, 0], 0.6, 0.04),
        ([0.05, 0, 0], [0, 0, 0], 0, 0),
        ([0.000101, 0, 0], [0.0000505, 0, 0], 0.5, 0),
        ([-0.05, 0, 0], [-0.03, 0, 0], 0.6, 0),
        ([0.03, 0.04, 0], [-0.005, 0.035, 0], 0.5, 0.025),
    ],
)
def test_signed_projection_distinguishes_direction(commanded, observed, fraction, cross_track):
    before = [0.4, -0.2, 0.5, 0, 0, 0, 1]
    target = [a + b for a, b in zip(before, commanded)] + before[3:]
    actual = [a + b for a, b in zip(before, observed)] + before[3:]
    feedback = _translation_feedback(before, actual, target)
    assert feedback["commanded_delta_m"] == pytest.approx(commanded)
    assert feedback["observed_delta_m"] == pytest.approx(observed)
    assert feedback["remaining_delta_m"] == pytest.approx([a - b for a, b in zip(commanded, observed)])
    assert feedback["along_track_fraction"] == pytest.approx(fraction)
    assert feedback["along_track_displacement_m"] == pytest.approx(fraction * math.hypot(*commanded))
    assert feedback["cross_track_error_m"] == pytest.approx(cross_track)
    assert feedback["commanded_displacement_m"] == pytest.approx(math.hypot(*commanded))
    assert feedback["observed_displacement_m"] == pytest.approx(math.hypot(*observed))
    assert feedback["translation_error_m"] == pytest.approx(math.dist(actual[:3], target[:3]))
    json.dumps(feedback, allow_nan=False)


@pytest.mark.parametrize("distance", [0, 1e-12, 0.000099, 0.0001])
def test_tiny_translation_and_rotation_only_keep_drift_without_projection(distance):
    before = [0, 0, 0, 0, 0, 0, 1]
    target = [distance, 0, 0, 0, 0, 1, 0]
    actual = [0, 0.002, 0, 0, 0, 1, 0]
    feedback = _translation_feedback(before, actual, target)
    assert feedback["commanded_delta_m"] == [distance, 0, 0]
    assert feedback["remaining_delta_m"] == [distance, -0.002, 0]
    assert feedback["observed_displacement_m"] == 0.002
    assert feedback["along_track_displacement_m"] is None
    assert feedback["along_track_fraction"] is None
    assert feedback["cross_track_error_m"] is None
    json.dumps(feedback, allow_nan=False)


def test_issued_prefix_feedback_reaches_model_and_resets(config, request_payload):
    calls, events = [], []
    decision = copy.deepcopy(config["planner"]["decisions"][0])

    def post(body):
        calls.append(body)
        return {
            "status": "completed",
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": json.dumps(decision)}]}
            ],
        }

    registry = default_registry()
    registry.planners.register("offline", lambda _: ResponsesPlanner(model="test", post=post))
    config["planner"] = {"type": "offline"}
    config["execution"].update(execute_steps=8, max_motion_continuations=0)
    original = copy.deepcopy(request_payload)
    with ApplicationFactory(parse_settings(config), registry).create_session(events.append) as session:
        session.handle(request_payload)
        issued = copy.deepcopy(session.runtime._last_issued)
        assert issued.metadata["discarded_steps"] > 0
        assert issued.right_arm[-1][2] < decision["motion"]["waypoints"][0]["right_pose"][2]
        moved = copy.deepcopy(request_payload)
        # Motion is perpendicular to the commanded +Z prefix, despite equal magnitudes.
        commanded = issued.right_arm[-1][2] - 0.5
        moved["params"]["states"]["end_pose"]["base_link"]["right_arm"]["position"][0] += commanded
        session.handle(moved)
        context = json.loads(calls[1]["input"][0]["content"][0]["text"])["context"]
        tracking = context["tracking_feedback"]
        prior = context["recent_events"][0]
        assert prior["sequence"] == 1
        assert prior["observed_effect"]["observation_sequence"] == 2
        assert prior["observed_effect"]["measured_eef_motion"]["right"][
            "translation_delta_m"
        ] == pytest.approx([commanded, 0, 0])
        assert tracking["right"]["issued_pose"] == issued.right_arm[-1]
        assert tracking["generated_endpoint"]["right_arm"] == decision["motion"]["waypoints"][0]["right_pose"]
        assert tracking["right"]["commanded_delta_m"] == pytest.approx([0, 0, commanded])
        assert tracking["right"]["along_track_fraction"] == pytest.approx(0)
        assert tracking["right"]["cross_track_error_m"] == pytest.approx(commanded)
        assert tracking["left"]["along_track_fraction"] is None
        assert tracking["task_success"] == "unverified"
        assert "playback is unverified" in tracking["summary"]
        assert "not to the executed trajectory" in tracking["meaning"]
        assert session.runtime._pending_motion is None
        session.runtime.reset()
        session.handle(request_payload)
        moved["params"]["prompt"] = "A different instruction"
        session.handle(moved)
        for call in (calls[0], calls[2], calls[3]):
            state = json.loads(call["input"][0]["content"][0]["text"])
            assert state["context"]["tracking_feedback"] == {}
            assert state["context"]["recent_events"] == []
    assert request_payload == original
    tracking_event = next(e for e in events if e.get("status") == "observed_tracking")
    assert tracking_event["right"] == tracking["right"]
    assert not any(e.get("status") == "motion_continued" for e in events)


def test_missing_frame_does_not_invent_direction(config, request_payload, observation):
    with build_session(config) as session:
        session.handle(request_payload)
        observation.end_pose = {}
        feedback = session.runtime._tracking_feedback(observation)
    assert "unavailable" in feedback["left"]
    assert "unavailable" in feedback["right"]
    assert "along_track_fraction" not in feedback["left"]
    assert "signed endpoint-direction progress" not in feedback["summary"]


def test_fully_issued_but_stalled_motion_does_not_gain_automatic_settling(config, request_payload):
    config["execution"].update(execute_steps=40, max_motion_continuations=2)
    config["planner"]["decisions"] *= 2
    events = []
    with build_session(config, event_sink=events.append) as session:
        session.handle(request_payload)
        first = session.runtime._last_issued
        assert first.metadata["discarded_steps"] == 0
        assert session.runtime._pending_motion is None
        session.handle(request_payload)
        assert session.runtime._last_issued.steps == first.steps
    feedback = next(e for e in events if e.get("status") == "observed_tracking")
    assert feedback["right"]["along_track_fraction"] == 0
    assert feedback["right"]["remaining_delta_m"] == pytest.approx([0, 0, 0.05])
    assert not any(e.get("status") == "motion_continued" for e in events)


def test_joint_feedback_does_not_invent_eef_direction(config, request_payload, observation):
    observation = copy.deepcopy(observation)
    with build_session(config) as session:
        session.handle(request_payload)
        chunk = session.runtime._last_issued
        chunk.kind = "JOINT_ABS"
        chunk.left_arm = [[0.1]] * chunk.steps
        chunk.right_arm = [[-0.2]] * chunk.steps
        observation.end_pose["base_link"]["left_arm"]["position"][1] -= 0.04
        feedback = session.runtime._tracking_feedback(observation)
    assert feedback["joint_error_rad"] == pytest.approx([-0.1, 0.2])
    assert "left" not in feedback and "right" not in feedback
    measured = feedback["measured_eef_motion"]["left"]
    assert measured["translation_delta_m"] == pytest.approx([0, -0.04, 0])
    assert measured["rotation_delta_rad"] == 0
    assert "issued_pose" not in measured
    assert feedback["task_success"] == "unverified"


@pytest.mark.parametrize("complete", [False, True])
def test_concise_feedback_preserves_prefix_or_completion_without_history(config, request_payload, complete):
    calls = []
    decision = copy.deepcopy(config["planner"]["decisions"][0])
    if complete:
        decision.update(status="complete", motion=None)

    def post(body):
        calls.append(json.loads(body["input"][0]["content"][0]["text"]))
        return {
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": json.dumps(decision)}]}
            ]
        }

    registry = default_registry()
    registry.planners.register(
        "offline",
        lambda _: ResponsesPlanner(
            model="test",
            post=post,
            current_observation_only=True,
            concise_tracking_feedback=True,
            gripper_center_offset_m={"left": [0, 0, 0.14845], "right": [0, 0, 0.14845]},
        ),
    )
    config["planner"] = {"type": "offline"}
    config["execution"].update(mode="motion", execute_steps=8, max_motion_continuations=0)
    with ApplicationFactory(parse_settings(config), registry).create_session() as session:
        session.handle(request_payload)
        first = copy.deepcopy(session.runtime._last_issued)
        moved = copy.deepcopy(request_payload)
        moved["params"]["states"]["end_pose"]["base_link"]["right_arm"]["position"][0] += 0.01
        session.handle(moved)
        context = calls[-1]["context"]
        summary = context["tracking_feedback"]
        assert "recent_events" not in context
        assert "observed [10.0, 0.0, 0.0]" in summary
        assert "playback unverified" in summary
        assert "Grippers: issued" in summary
        if complete:
            assert first.steps == 1 and first.kind == "JOINT_ABS"
            assert "Previous planner declared complete" in summary
            assert "Cartesian target unavailable" in summary
            assert "remaining [" not in summary
            assert "generated final pose" not in summary
        else:
            assert first.metadata["discarded_steps"] > 0
            assert f"issued 8/{first.metadata['generated_steps']} generated" in summary
            assert "generated final pose" in summary
            # The model target is TCP z=.55, not the execution wrist z=.40155.
            assert "0.55" in summary
            assert "issued prefix endpoint" in summary
        session.runtime.reset()
        session.handle(request_payload)
        assert "tracking_feedback" not in calls[-1]["context"]


def test_joint_feedback_uses_compiler_fk_of_issued_prefix(config, request_payload, observation):
    observation = copy.deepcopy(observation)
    with build_session(config) as session:
        session.handle(request_payload)
        chunk = session.runtime._last_issued
        before = {arm: observation.eef(arm, chunk.frame) for arm in ("left", "right")}
        steps = chunk.steps
        chunk.kind = "JOINT_ABS"
        chunk.left_arm = [[0.1]] * steps
        chunk.right_arm = [[-0.2]] * steps
        # Two generated rows beyond the issued prefix must never become its target.
        chunk.metadata["cartesian_path"] = {
            arm: [
                before[arm][:2] + [before[arm][2] + 0.01 * i] + before[arm][3:] for i in range(1, steps + 3)
            ]
            for arm in ("left", "right")
        }
        observation.end_pose["base_link"]["left_arm"]["position"][2] += 0.01 * steps - 0.002
        feedback = session.runtime._tracking_feedback(observation)
    assert feedback["cartesian_target_source"] == "issued_joint_fk"
    assert feedback["joint_error_rad"] == pytest.approx([-0.1, 0.2])
    left = feedback["left"]
    assert left["issued_pose"] == chunk.metadata["cartesian_path"]["left"][steps - 1]
    assert left["commanded_delta_m"] == pytest.approx([0, 0, 0.01 * steps])
    assert left["remaining_delta_m"] == pytest.approx([0, 0, 0.002])
    assert feedback["generated_endpoint"]["left_arm"] == chunk.metadata["cartesian_path"]["left"][-1]
    assert "left EEF residual to issued endpoint" in feedback["summary"]
    assert "Joint residuals" in feedback["summary"]


def test_decision_history_survives_current_observation_only(config, request_payload):
    calls = []
    decision = copy.deepcopy(config["planner"]["decisions"][0])

    def post(body):
        calls.append(json.loads(body["input"][0]["content"][0]["text"]))
        decision["instruction"] = f"operation {len(calls)}"
        return {
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": json.dumps(decision)}]}
            ]
        }

    registry = default_registry()
    registry.planners.register(
        "offline",
        lambda _: ResponsesPlanner(
            model="test",
            post=post,
            current_observation_only=True,
            concise_tracking_feedback=True,
            decision_history_turns=2,
            gripper_center_offset_m={"left": [0, 0, 0.14845], "right": [0, 0, 0.14845]},
        ),
    )
    config["planner"] = {"type": "offline"}
    config["execution"].update(mode="motion", execute_steps=8, max_motion_continuations=0)
    with ApplicationFactory(parse_settings(config), registry).create_session() as session:
        for _ in range(4):
            session.handle(request_payload)
        events = copy.deepcopy(session.runtime.memory)
    assert "decision_history" not in calls[0]["context"]
    context = calls[3]["context"]
    assert "recent_events" not in context
    history = context["decision_history"]
    # Bounded to the configured turns, oldest first, each with its measured response.
    assert [entry["sequence"] for entry in history] == [2, 3]
    assert [entry["instruction"] for entry in history] == ["operation 2", "operation 3"]
    assert history[-1]["success_condition"] == decision["success_condition"]
    for entry in history:
        assert entry["measured_result"].startswith(f"Action {entry['sequence']}, XYZ in base_link")
        assert "remaining [" in entry["measured_result"]
    tracking = events[-2]["observed_effect"]["tracking"]["right"]
    assert tracking["issued_pose"] == events[-2]["issued_endpoint"]["right_arm"]


def test_decision_history_turns_validated():
    for value in (-1, 33, 1.5, True):
        with pytest.raises(Exception, match="decision_history_turns"):
            ResponsesPlanner(model="test", decision_history_turns=value)

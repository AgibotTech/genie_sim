# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Before/after comparison must survive retries and finite policy grants."""

import io
import json

import pytest
from PIL import Image

from agent2act.adapters.planners.chat_completions import ChatCompletionsPlanner
from agent2act.adapters.planners.embodied import EmbodiedPlanner
from agent2act.adapters.planners.responses import ResponsesPlanner


def camera(color):
    data = io.BytesIO()
    Image.new("RGB", (16, 16), color).save(data, "JPEG")
    return {"head": {"encoding": "JPEG", "image_data": data.getvalue()}}


@pytest.mark.parametrize("planner_class", [ResponsesPlanner, ChatCompletionsPlanner])
def test_reference_is_start_observation_not_adjacent_chunk_frame(planner_class, observation, motion_decision):
    calls = []

    def post(body):
        calls.append(body)
        if planner_class is ResponsesPlanner:
            return {
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": json.dumps(motion_decision)}],
                    }
                ]
            }
        return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(motion_decision)}}]}

    planner = planner_class(model="test", post=post, history_frames=1, use_depth=True)

    def parts():
        key = "input" if planner_class is ResponsesPlanner else "messages"
        return next(item["content"] for item in calls[-1][key] if item["role"] == "user")

    def images():
        return [p for p in parts() if p["type"] in ("input_image", "image_url")]

    observation.images = camera("red")
    planner.decide(observation, {"sequence": 1})
    red = images()[-1]
    observation.images = camera("blue")
    observation.history = [{"images": observation.images}]
    # Policy grants skip intermediate planner calls; the reference is still sequence 1.
    planner.decide(observation, {"sequence": 9})
    assert images()[0] == red
    assert images()[1] != red
    assert "previous planner observation sequence=1" in str(parts())
    assert "in-chunk" not in str(parts())  # The configured one-frame budget is respected.
    planner.decide(observation, {"sequence": 9, "planning_attempt": 1})
    assert images()[0] == red  # Replanning cannot replace the baseline with the current frame.
    planner.reset()
    observation.history = []
    planner.decide(observation, {"sequence": 1})
    assert len(images()) == 1
    planner.close()
    assert planner._last_visual == planner._visual_reference == []


def test_history_disabled_does_not_cache(observation, motion_decision):
    planner = ResponsesPlanner(
        model="test",
        history_frames=0,
        post=lambda _: {
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": json.dumps(motion_decision)}]}
            ]
        },
    )
    observation.images = camera("red")
    planner.decide(observation, {"sequence": 1})
    assert planner._last_visual == []
    planner.close()


def test_multiple_planner_references_are_bounded_and_survive_retry(observation, motion_decision):
    calls = []

    def post(body):
        calls.append(body)
        return {
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": json.dumps(motion_decision)}]}
            ]
        }

    planner = ResponsesPlanner(model="test", post=post, history_frames=2, use_depth=False)

    def image_urls():
        return [
            item["image_url"] for item in calls[-1]["input"][0]["content"] if item["type"] == "input_image"
        ]

    colors = ("red", "green", "blue", "yellow")
    current_images = []
    for sequence, color in zip((1, 4, 9, 12), colors):
        observation.images = camera(color)
        # Adjacent playback images must not displace informative planner references.
        observation.history = [{"images": camera("black")}]
        planner.decide(observation, {"sequence": sequence})
        current_images.append(image_urls()[-1])
    assert image_urls() == current_images[-3:]
    assert "previous planner observation sequence=4" in str(calls[-1])
    assert "previous planner observation sequence=9" in str(calls[-1])
    assert "sequence=1 " not in str(calls[-1])
    assert "in-chunk" not in str(calls[-1])
    planner.decide(observation, {"sequence": 12, "planning_attempt": 1})
    assert image_urls() == current_images[-3:]
    assert len(planner._visual_history) == 2
    planner.reset()
    observation.history = []
    planner.decide(observation, {"sequence": 1})
    assert len(image_urls()) == 1
    planner.close()
    assert planner._visual_history == []


@pytest.mark.parametrize("planner_class", [ResponsesPlanner, ChatCompletionsPlanner, EmbodiedPlanner])
def test_historical_robot_state_matches_images_not_targets_or_retries(
    planner_class, observation, motion_decision
):
    calls = []

    def post(body):
        calls.append(body)
        if planner_class is ChatCompletionsPlanner:
            return {
                "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(motion_decision)}}]
            }
        return {
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": json.dumps(motion_decision)}]}
            ]
        }

    def references():
        key = "messages" if planner_class is ChatCompletionsPlanner else "input"
        result = []
        for item in calls[-1][key]:
            if not isinstance(item.get("content"), list):
                continue
            for part in item.get("content", []):
                if part["type"] not in ("text", "input_text"):
                    continue
                if part["text"].startswith('{"previous_robot_observation":'):
                    result.append(json.loads(part["text"])["previous_robot_observation"])
        return result

    planner = planner_class(model="test", post=post, history_frames=1, use_depth=False)
    observation.images = camera("red")
    observation.end_pose["arm_base_link"] = {"unselected_frame": True}
    context = {
        "sequence": 1,
        "execution": {"motion_frame": "base_link"},
        "gripper_commands": [0.9, 0.9],
        "observed_grippers_in_command_units": [0.3, 0.4],
    }
    planner.decide(observation, context)
    assert references() == []
    old_states = json.loads(
        next(
            item["content"]
            for item in calls[-1]["messages" if planner_class is ChatCompletionsPlanner else "input"]
            if item["role"] == "user"
        )[0]["text"]
    )["states"]

    # Mutating the caller's observation and context cannot change the old snapshot.
    observation.end_pose["base_link"]["left_arm"]["position"][2] += 0.1
    observation.state_groups["gripper_states"][0] = 0.6
    context["observed_grippers_in_command_units"][0] = 0.6
    observation.images = camera("blue")
    context["sequence"] = 9
    planner.decide(observation, context)
    baseline = references()
    assert len(baseline) == 1
    assert baseline[0]["sequence"] == 1
    assert baseline[0]["states"] == old_states
    assert set(baseline[0]["states"]["end_pose"]) == {"base_link"}
    assert baseline[0]["observed_grippers_in_command_units"] == [0.3, 0.4]
    assert "gripper_commands" not in baseline[0]
    assert "context" not in baseline[0]
    assert not calls[-1].get("tools")
    planner.decide(observation, {**context, "planning_attempt": 1})
    assert references() == baseline
    planner.decide(observation, {**context, "sequence": 10})
    assert [r["sequence"] for r in references()] == [9]
    assert references()[0]["observed_grippers_in_command_units"] == [0.6, 0.4]
    planner.reset()
    planner.decide(observation, {**context, "sequence": 1})
    assert references() == []
    planner.close()

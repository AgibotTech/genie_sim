# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import io
import json
from pathlib import Path

import pytest
from PIL import Image

from agent2act.adapters.planners.responses import ResponsesPlanner
from agent2act.adapters.planners.static_rgb_calibration import StaticRgbCalibration
from agent2act.domain.errors import ContractError

RECIPE = Path(__file__).resolve().parents[2] / "examples/g2_llm_recipe.json"


def test_g2_rgb_calibration_reaches_current_model_input(observation, motion_decision):
    calibration = json.loads(RECIPE.read_text())["planner"]["rgb_camera_calibration"]
    rgb = io.BytesIO()
    Image.new("RGB", (640, 528), "white").save(rgb, "JPEG")
    head_rgb = io.BytesIO()
    Image.new("RGB", (640, 400), "white").save(head_rgb, "JPEG")
    observation.images = {
        camera: {"encoding": "JPEG", "image_data": rgb.getvalue()} for camera in ("hand_left", "hand_right")
    }
    observation.images["head"] = {"encoding": "JPEG", "image_data": head_rgb.getvalue()}
    observation.state_groups["waist_joint_states"] = [0.0] * 5
    observation.state_groups["head_joint_states"] = [0.0] * 3
    requests = []

    def post(body):
        requests.append(body)
        return {
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": json.dumps(motion_decision)}]}
            ]
        }

    planner = ResponsesPlanner(model="test", post=post, rgb_camera_calibration=calibration)
    try:
        planner.decide(observation, {"execution": {"mode": "motion"}})
        parts = requests[0]["input"][0]["content"]
        prompt = requests[0]["input"][1]["content"][0]["text"]
        assert "single RGB pixel defines a ray" in prompt
        assert "current measured" in prompt
        assert "metric_projection_ready=false" in prompt
        assert "use the head view for coarse scene alignment" in prompt
        assert "cite the corresponding current image and calibration" in prompt
        entries = [
            json.loads(p["text"]) for p in parts if p["type"] == "input_text" and '"calibration"' in p["text"]
        ]
        assert len(entries) == 3
        assert all(entry["calibration"]["source_intrinsics"]["width"] == 640 for entry in entries)
        assert {
            entry["camera"]: entry["calibration"]["source_intrinsics"]["height"] for entry in entries
        } == {
            "head": 400,
            "hand_left": 528,
            "hand_right": 528,
        }
        assert all(
            entry["calibration"]["optical_to_base_link"][3] == [0.0, 0.0, 0.0, 1.0] for entry in entries
        )
        assert all(entry["calibration"]["metric_projection_ready"] is False for entry in entries)
    finally:
        planner.close()


def test_wrist_extrinsic_tracks_current_observed_pose(observation):
    calibration = StaticRgbCalibration(json.loads(RECIPE.read_text())["planner"]["rgb_camera_calibration"])
    first = calibration.for_observation(observation)["hand_right"]["optical_to_base_link"]
    observation.end_pose["base_link"]["right_arm"]["position"][0] += 0.1
    second = calibration.for_observation(observation)["hand_right"]["optical_to_base_link"]
    assert second[0][3] == pytest.approx(first[0][3] + 0.1)
    head = calibration.for_observation(observation)["head"]
    assert head["mount_frame"] == "head_link3"


def test_head_extrinsic_uses_measured_head_and_waist_joints(observation):
    calibration = StaticRgbCalibration(json.loads(RECIPE.read_text())["planner"]["rgb_camera_calibration"])
    observation.state_groups["waist_joint_states"] = [0.0] * 5
    observation.state_groups["head_joint_states"] = [0.0] * 3
    zero = calibration.for_observation(observation)["head"]
    assert zero["optical_to_base_link"][3] == [0.0, 0.0, 0.0, 1.0]
    assert zero["metric_projection_ready"] is False
    observation.state_groups["head_joint_states"][2] = 0.1
    moved = calibration.for_observation(observation)["head"]
    assert moved["optical_to_base_link"] != zero["optical_to_base_link"]
    del observation.state_groups["head_joint_states"]
    unavailable = calibration.for_observation(observation)["head"]
    assert "optical_to_base_link" not in unavailable
    assert unavailable["base_extrinsic_reason"]


def test_rgb_calibration_rejects_wrong_aspect_ratio_and_nonrigid_mount():
    values = json.loads(RECIPE.read_text())["planner"]["rgb_camera_calibration"]
    calibration = StaticRgbCalibration(values)
    metadata = calibration.image_metadata({"intrinsics_size": [1280, 1056]}, [640, 400], [640, 400])
    assert metadata["available"] is False
    values["hand_right"]["camera_to_mount"][0][0] = 0.5
    with pytest.raises(ContractError):
        StaticRgbCalibration(values)

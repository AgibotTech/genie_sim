# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import io
import json
import struct
from copy import deepcopy

import pytest
from PIL import Image

from agent2act.adapters.planners.embodied import EmbodiedPlanner
from agent2act.adapters.planners.raw_depth import read_depth_pixels
from agent2act.adapters.planners.responses import ResponsesPlanner
from agent2act.domain.models import ContractError


def sensor(observation):
    output = io.BytesIO()
    Image.new("RGB", (3, 2)).save(output, "JPEG")
    observation.images = {
        "head": {
            "encoding": "JPEG",
            "image_data": output.getvalue(),
            "width": 3,
            "height": 2,
            "snapshot_id": "current",
            "calibration": {
                "available": True,
                "width": 3,
                "height": 2,
                "snapshot_id": "current",
                "rgb_depth_aligned": True,
                "fx": 3.0,
                "fy": 3.0,
                "cx": 1.0,
                "cy": 1.0,
                "distortion": [0.0] * 8,
                "depth_kind": "distance_to_image_plane",
                "depth_scale_m": 0.001,
                "frame": "base_link",
                "optical_to_frame": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            },
        },
        "head_depth": {
            "encoding": "RAW_UINT16",
            "width": 3,
            "height": 2,
            "image_data": struct.pack("<6H", 0, 1234, 65535, 1, 500, 65000),
            "snapshot_id": "current",
        },
    }
    return observation


def query(u=1, v=0):
    return {"camera": "head", "pixels": [{"label": "model choice", "u": u, "v": v}]}


def test_exact_pixels_endianness_invalid_values_and_no_mutation(observation):
    obs = sensor(observation)
    before = deepcopy(obs)
    args = {
        "camera": "head",
        "pixels": [{"label": str(v * 3 + u), "u": u, "v": v} for v in range(2) for u in range(3)],
    }
    result = read_depth_pixels(obs, args)
    assert [p["raw_uint16"] for p in result["pixels"]] == [0, 1234, 65535, 1, 500, 65000]
    assert [p["depth_mm"] for p in result["pixels"]] == [None, 1234, None, 1, 500, 65000]
    assert result["snapshot_id"] == "current"
    assert result["calibration"]["available"] is True
    assert not result["motion_executed"]
    assert "surface_point_m" not in str(result)
    result["calibration"]["fx"] = 999
    assert obs == before


@pytest.mark.parametrize("u,v", [(-1, 0), (3, 0), (0, 2), (1.5, 0), (True, 0), (1000, 1000)])
def test_source_coordinates_are_not_clamped_or_normalized(observation, u, v):
    with pytest.raises(ContractError, match="pixel"):
        read_depth_pixels(sensor(observation), query(u, v))


@pytest.mark.parametrize("damage", ["bytes", "dimensions", "encoding", "stale_rgb"])
def test_bad_or_stale_sensor_data_rejected(observation, damage):
    obs = sensor(observation)
    depth = obs.images["head_depth"]
    if damage == "bytes":
        depth["image_data"] = b""
    if damage == "dimensions":
        depth["width"] = 4
    if damage == "encoding":
        depth["encoding"] = "JPEG"
    if damage == "stale_rgb":
        obs.images["head"]["snapshot_id"] = "older"
    with pytest.raises(ContractError):
        read_depth_pixels(obs, query())


@pytest.mark.parametrize("damage", ["unavailable", "missing", "stale", "unaligned", "dimensions"])
def test_raw_reading_does_not_require_usable_extrinsics(observation, damage):
    obs = sensor(observation)
    cal = obs.images["head"]["calibration"]
    if damage == "unavailable":
        cal.update(available=False, reason="pose mismatch")
    if damage == "missing":
        obs.images["head"].pop("calibration")
    if damage == "stale":
        cal["snapshot_id"] = "older"
    if damage == "unaligned":
        cal["rgb_depth_aligned"] = False
    if damage == "dimensions":
        cal["width"] = 4
    result = read_depth_pixels(obs, query())
    assert result["pixels"][0]["depth_mm"] == 1234
    assert not result["calibration"]["available"]


def test_disabled_depth_and_options():
    planner = EmbodiedPlanner(
        model="test", use_depth=False, enable_raw_depth_query=True, send_raw_depth_image=False
    )
    assert not planner._geometry_tools()
    planner.close()
    with pytest.raises(ContractError, match="choose raw"):
        EmbodiedPlanner(model="test", enable_raw_depth_query=True, enable_depth_measurement=True)
    with pytest.raises(ContractError, match="boolean"):
        EmbodiedPlanner(model="test", enable_raw_depth_query="yes")
    with pytest.raises(ContractError, match="depth query tool"):
        ResponsesPlanner(model="test", use_depth=True, send_raw_depth_image=False)
    with pytest.raises(ContractError, match="depth query tool"):
        EmbodiedPlanner(model="test", use_depth=True, send_raw_depth_image=False)


def test_tool_budget_errors_and_reset(observation):
    requests = []
    obs = sensor(observation)

    def post(body):
        requests.append(deepcopy(body))
        return {
            "output": [
                {
                    "type": "function_call",
                    "name": "read_depth_pixels",
                    "call_id": str(len(requests)),
                    "arguments": json.dumps(query(99, 0)),
                }
            ]
        }

    planner = EmbodiedPlanner(
        model="test", use_depth=True, enable_raw_depth_query=True, send_raw_depth_image=False, post=post
    )
    try:
        with pytest.raises(ContractError, match="budget"):
            planner.decide(obs, {"sequence": 1, "execution": {"mode": "motion"}})
        assert len(requests) == 3
        first_content = requests[0]["input"][0]["content"]
        assert sum(part["type"] == "input_image" for part in first_content) == 1
        assert all("data:image/png" not in str(part) for part in first_content)
        depth = next(
            json.loads(part["text"])["depth"]
            for part in first_content
            if part.get("type") == "input_text" and '"depth"' in part["text"]
        )
        assert depth["encoding"] == "RAW_UINT16_LOCAL"
        assert {tool["name"] for tool in requests[0]["tools"]} == {"read_depth_pixels"}
        assert requests[-1]["tool_choice"] == "none"
        returned = json.loads(requests[1]["input"][-1]["output"])
        assert "outside source" in returned["error"]
        assert returned["motion_executed"] is False
        protocol = next(m for m in requests[0]["input"] if m.get("role") == "developer")["content"][0]["text"]
        assert "Do not return markdown fences, commentary, or a tool call" not in protocol
        assert "FINAL decision" in protocol
        planner.reset()
        assert planner._geometry_memory == []
    finally:
        planner.close()


def test_disabled_tool_cannot_be_called(observation):
    planner = EmbodiedPlanner(model="test", use_depth=False, enable_raw_depth_query=True)
    try:
        with pytest.raises(ContractError, match="unavailable"):
            planner._execute_geometry_call(
                {"name": "read_depth_pixels", "arguments": json.dumps(query())}, observation
            )
    finally:
        planner.close()

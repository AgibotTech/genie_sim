# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import json
import struct
from copy import deepcopy

import pytest

from agent2act.adapters.planners.embodied import EmbodiedPlanner
from agent2act.adapters.planners.geometry_tools import measure_surface_points
from agent2act.domain.models import ContractError


def calibrated(observation):
    observation.images = {
        "head": {
            "calibration": {
                "model": "pinhole",
                "frame": "base_link",
                "units": "m",
                "width": 5,
                "height": 5,
                "fx": 2.0,
                "fy": 2.0,
                "cx": 2.0,
                "cy": 2.0,
                "depth_kind": "distance_to_image_plane",
                "depth_scale_m": 0.001,
                "optical_to_frame": [
                    [1.0, 0.0, 0.0, 0.1],
                    [0.0, -1.0, 0.0, 0.2],
                    [0.0, 0.0, -1.0, 1.5],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            }
        },
        "head_depth": {
            "encoding": "RAW_UINT16",
            "width": 5,
            "height": 5,
            "image_data": struct.pack("<25H", *([500] * 25)),
        },
    }
    return observation


def query(x=500, y=500):
    return {"camera": "head", "points": [{"label": "chosen surface", "x": x, "y": y}]}


def test_surface_projection_is_sensor_only_and_does_not_mutate(observation):
    obs = calibrated(observation)
    before = deepcopy(obs)
    result = measure_surface_points(obs, query())
    assert result["points"][0]["surface_point_m"] == [0.1, 0.2, 1.0]
    assert measure_surface_points(obs, query(1000, 0))["points"][0]["surface_point_m"] == [0.6, 0.7, 1.0]
    assert obs == before


@pytest.mark.parametrize("damage", ["missing", "resolution", "distortion", "matrix", "scale", "bytes"])
def test_invalid_calibration_rejected(observation, damage):
    obs = calibrated(observation)
    c = obs.images["head"]["calibration"]
    d = obs.images["head_depth"]
    if damage == "missing":
        obs.images["head"].pop("calibration")
    elif damage == "resolution":
        d["width"] = 4
    elif damage == "distortion":
        c["model"] = "fisheye"
    elif damage == "matrix":
        c["optical_to_frame"][0][0] = 2.0
    elif damage == "scale":
        c["depth_scale_m"] = 1
    elif damage == "bytes":
        d["image_data"] = b""
    with pytest.raises(ContractError):
        measure_surface_points(obs, query())


def test_invalid_depth_and_boundaries(observation):
    obs = calibrated(observation)
    obs.images["head_depth"]["image_data"] = struct.pack("<25H", *([0] * 25))
    assert "error" in measure_surface_points(obs, query())["points"][0]
    obs.images["head_depth"]["image_data"] = struct.pack("<25H", *([500] * 12 + [800] * 13))
    assert "error" in measure_surface_points(obs, query())["points"][0]
    with pytest.raises(ContractError):
        measure_surface_points(obs, query(-1))


@pytest.mark.parametrize("send_raw_depth_image", [True, False])
def test_tool_round_preserves_reasoning_and_delivers_measured_results(observation, send_raw_depth_image):
    obs = calibrated(observation)
    requests = []
    reasoning = {"type": "reasoning", "id": "rs_test", "summary": [], "encrypted_content": "encrypted"}
    call = {
        "type": "function_call",
        "name": "measure_surface_points",
        "call_id": "call_1",
        "arguments": json.dumps(query()),
    }
    final = {
        "mode": "motion",
        "instruction": "inspect",
        "reason": "surface query returned",
        "motion": {"frame": "base_link", "waypoints": [{"grippers": [0.0, 0.0]}]},
        "status": "continue",
        "success_condition": "inspect",
    }

    def post(body):
        requests.append(deepcopy(body))
        if len(requests) == 1:
            return {"status": "completed", "output": [reasoning, call]}
        return {
            "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(final)}]}]
        }

    planner = EmbodiedPlanner(
        model="test",
        post=post,
        use_depth=True,
        enable_depth_measurement=True,
        send_raw_depth_image=send_raw_depth_image,
    )
    assert planner.decide(obs, {"sequence": 1}).mode == "motion"
    assert reasoning in requests[1]["input"]
    returned = requests[1]["input"][-1]
    assert returned["call_id"] == "call_1"
    assert json.loads(returned["output"])["points"][0]["surface_point_m"] == [0.1, 0.2, 1.0]
    if not send_raw_depth_image:
        assert all("data:image/png" not in str(part) for part in requests[0]["input"][0]["content"])
    planner.close()


def test_tool_budget_and_error_feedback(observation):
    requests = []

    def post(body):
        requests.append(deepcopy(body))
        return {
            "output": [
                {
                    "type": "function_call",
                    "name": "measure_surface_points",
                    "call_id": str(len(requests)),
                    "arguments": json.dumps(query()),
                }
            ]
        }

    planner = EmbodiedPlanner(model="test", post=post, use_depth=True, enable_depth_measurement=True)
    with pytest.raises(ContractError, match="budget"):
        planner.decide(observation, {"sequence": 1})
    assert len(requests) == 3
    assert "error" in json.loads(requests[1]["input"][-1]["output"])
    assert requests[-1]["tool_choice"] == "none"
    planner.close()


def test_inverse_distortion_recovers_off_axis_point():
    from agent2act.adapters.planners.geometry_tools import undistort_point

    x, y = 0.35, -0.27
    coeff = [-0.15, 0.03, 0.002, -0.001, 0.004, 0.01, -0.002, 0.0001]
    k1, k2, p1, p2, k3, k4, k5, k6 = coeff
    r2 = x * x + y * y
    radial = (1 + k1 * r2 + k2 * r2**2 + k3 * r2**3) / (1 + k4 * r2 + k5 * r2**2 + k6 * r2**3)
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    assert undistort_point(xd, yd, coeff) == pytest.approx((x, y), abs=1e-8)
    with pytest.raises(ContractError):
        undistort_point(xd, yd, [float("nan")] * 8)


def test_geometry_memory_is_labelled_historical_and_reset(observation):
    obs = calibrated(observation)
    requests = []

    def post(body):
        requests.append(deepcopy(body))
        if len(requests) == 1:
            return {
                "output": [
                    {
                        "type": "function_call",
                        "name": "measure_surface_points",
                        "call_id": "q",
                        "arguments": json.dumps(query()),
                    }
                ]
            }
        return {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": json.dumps(
                                {
                                    "mode": "motion",
                                    "instruction": "inspect",
                                    "reason": "geometry",
                                    "status": "complete",
                                    "success_condition": "unverified",
                                }
                            ),
                        }
                    ],
                }
            ]
        }

    planner = EmbodiedPlanner(model="test", post=post, use_depth=True, enable_depth_measurement=True)
    planner.decide(obs, {"sequence": 1})
    planner.decide(obs, {"sequence": 2})
    memory = json.loads(requests[-1]["input"][-1]["content"][0]["text"])
    assert memory["prior_geometry_queries"][0]["sequence"] == 1
    assert memory["current_sequence"] == 2
    assert "NOT current" in memory["meaning"]
    planner.reset()
    planner.decide(obs, {"sequence": 1})
    assert "prior_geometry_queries" not in str(requests[-1])
    planner.close()
    assert planner._geometry_memory == []


def test_tool_center_conversion_accounts_for_orientation():
    from agent2act.adapters.planners.geometry_tools import compose_grasp_pose

    offsets = {"left": [0, 0, 0.15], "right": [0, 0, 0.15]}
    args = {"arm": "right", "center_m": [0.5, -0.1, 0.8], "orientation_xyzw": [1.0, 0.0, 0.0, 0.0]}
    result = compose_grasp_pose(args, offsets)
    assert result["motion_pose"] == pytest.approx([0.5, -0.1, 0.95, 1, 0, 0, 0])
    args["orientation_xyzw"] = [0, 0, 0, 1]
    assert compose_grasp_pose(args, offsets)["motion_pose"][:3] == pytest.approx([0.5, -0.1, 0.65])
    with pytest.raises(ContractError):
        compose_grasp_pose(args, {})
    args["orientation_xyzw"] = [0, 0, 0, 0]
    with pytest.raises(ContractError):
        compose_grasp_pose(args, offsets)


def test_configured_tool_geometry_is_exposed_and_dispatched(observation):
    requests = []

    def post(body):
        requests.append(deepcopy(body))
        if len(requests) == 1:
            return {
                "output": [
                    {
                        "type": "function_call",
                        "name": "compose_grasp_pose",
                        "call_id": "pose1",
                        "arguments": json.dumps(
                            {"arm": "left", "center_m": [0.5, 0.1, 0.8], "orientation_xyzw": [1, 0, 0, 0]}
                        ),
                    }
                ]
            }
        return {"output": []}

    planner = EmbodiedPlanner(
        model="test",
        post=post,
        use_depth=True,
        enable_depth_measurement=True,
        wrist_to_center_m={"left": [0, 0, 0.15], "right": [0, 0, 0.15]},
        gripper_open_closed={"left": [-0.785, 0], "right": [-0.785, 0]},
    )
    planner._decision_response(planner._request_body([]), observation)
    assert {t["name"] for t in requests[0]["tools"]} == {"measure_surface_points", "compose_grasp_pose"}
    assert "robot_tool_geometry" in str(requests[0]["input"])
    reference = next(
        json.loads(content["text"])["robot_gripper_command_reference"]
        for message in requests[0]["input"]
        for content in message.get("content", [])
        if "robot_gripper_command_reference" in content.get("text", "")
    )
    assert reference["right"] == {"fully_open_command": -0.785, "fully_closed_command": 0.0}
    result = json.loads(requests[1]["input"][-1]["output"])
    assert result["motion_pose"] == pytest.approx([0.5, 0.1, 0.95, 1, 0, 0, 0])
    assert planner._geometry_memory[-1]["name"] == "compose_grasp_pose"
    planner.close()
    with pytest.raises(ContractError, match="left and right"):
        EmbodiedPlanner(model="test", post=post, wrist_to_center_m={"left": [0, 0, 0.15]})
    with pytest.raises(ContractError, match="must differ"):
        EmbodiedPlanner(model="test", post=post, gripper_open_closed={"left": [0, 0], "right": [0, 1]})


@pytest.mark.parametrize("depth_option", [None, False])
def test_depth_disabled_hides_measurements_history_and_tool(observation, monkeypatch, depth_option):
    import io

    from PIL import Image

    obs = calibrated(observation)
    rgb = io.BytesIO()
    Image.new("RGB", (5, 5), "red").save(rgb, "JPEG")
    obs.images["head"].update(encoding="JPEG", image_data=rgb.getvalue())
    obs.history = [{"images": deepcopy(obs.images)}]
    bodies = []

    def post(body):
        bodies.append(deepcopy(body))
        return {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": json.dumps(
                                {
                                    "mode": "motion",
                                    "instruction": "inspect RGB",
                                    "reason": "bounded exploration",
                                    "motion": {"frame": "base_link", "waypoints": [{}]},
                                }
                            ),
                        }
                    ],
                }
            ]
        }

    kwargs = {} if depth_option is None else {"use_depth": depth_option}
    planner = EmbodiedPlanner(model="test", post=post, history_frames=1, **kwargs)
    planner._geometry_memory = [{"name": "measure_surface_points", "result": {"depth_m": 0.123456789}}]
    planner.decide(obs, {"sequence": 1})
    planner.decide(obs, {"sequence": 2})
    for body in bodies:
        assert not body.get("tools")
        text = json.dumps(body)
        assert '"depth":' not in text
        assert '"depth_m"' not in text and "0.123456789" not in text
        assert "data:image/png" not in text
        parts = body["input"][0]["content"]
        assert sum(p["type"] == "input_image" for p in parts) == 2
        assert all(p["image_url"].startswith("data:image/jpeg") for p in parts if p["type"] == "input_image")

    def forbidden(*args):
        pytest.fail("disabled depth must never be sampled")

    monkeypatch.setattr("agent2act.adapters.planners.embodied.measure_surface_points", forbidden)
    with pytest.raises(ContractError, match="unavailable"):
        planner._execute_geometry_call(
            {"name": "measure_surface_points", "arguments": json.dumps(query())}, obs
        )
    planner.close()


def test_rgb_only_robot_pose_tool_remains_available(observation):
    planner = EmbodiedPlanner(model="test", wrist_to_center_m={"left": [0, 0, 0.15], "right": [0, 0, 0.15]})
    body = planner._with_geometry_context(planner._request_body([]), observation)
    assert [t["name"] for t in body["tools"]] == ["compose_grasp_pose"]
    protocol = next(m for m in body["input"] if m.get("role") == "developer")["content"][0]["text"]
    assert protocol.count("Depth access is disabled") == 1
    assert "Depth access is disabled" not in body["instructions"]
    result = planner._execute_geometry_call(
        {
            "name": "compose_grasp_pose",
            "call_id": "pose",
            "arguments": json.dumps(
                {"arm": "left", "center_m": [0.5, 0.2, 1], "orientation_xyzw": [0, 0, 0, 1]}
            ),
        },
        observation,
    )
    assert result["motion_pose"] == pytest.approx([0.5, 0.2, 0.85, 0, 0, 0, 1])
    planner.close()


def test_depth_information_does_not_automatically_expose_measurement_tool(observation):
    planner = EmbodiedPlanner(model="test", use_depth=True, post=lambda body: {})
    try:
        body = planner._request_body([])
        assert "tools" not in body
        assert "No tools are available" in body["instructions"]
        assert "Depth access is disabled" not in body["instructions"]
        with pytest.raises(ContractError, match="unavailable"):
            planner._execute_geometry_call(
                {"name": "measure_surface_points", "arguments": json.dumps(query()), "call_id": "forbidden"},
                calibrated(observation),
            )
    finally:
        planner.close()


@pytest.mark.parametrize("limit", [0, 2, 8])
def test_geometry_history_bound_preserves_latest_results_and_reset(observation, limit):
    planner = EmbodiedPlanner(
        model="test",
        post=lambda _: {},
        use_depth=True,
        enable_depth_measurement=True,
        geometry_history_queries=limit,
    )
    obs = calibrated(observation)
    try:
        planner.reset()
        for sequence in range(1, 11):
            planner._call_context = {"sequence": sequence}
            result = planner._execute_geometry_call(
                {
                    "name": "measure_surface_points",
                    "call_id": str(sequence),
                    "arguments": json.dumps(query()),
                },
                obs,
            )
            assert "error" not in result
        assert [entry["sequence"] for entry in planner._geometry_memory] == list(range(11 - limit, 11))
        if limit:
            assert planner._geometry_memory[-1]["result"] == result
            result.clear()
            assert planner._geometry_memory[-1]["result"]
        planner.reset()
        assert planner._geometry_memory == []
    finally:
        planner.close()


@pytest.mark.parametrize("limit", [-1, 9, True, 1.5, "2"])
def test_geometry_history_rejects_invalid_limit(limit):
    with pytest.raises(ContractError, match="geometry_history_queries"):
        EmbodiedPlanner(model="test", geometry_history_queries=limit)

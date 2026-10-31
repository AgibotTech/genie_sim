# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Passive observation, episode isolation and calibrated overlays (no live models)."""

import base64
import io
import json
from copy import deepcopy

import pytest
from PIL import Image

from agent2act.adapters.planners.responses import ResponsesPlanner
from agent2act.adapters.telemetry import capture_artifacts
from agent2act.bootstrap import ApplicationFactory
from agent2act.configuration import parse_settings
from agent2act.service.action_trace import action_log_path
from agent2act.service.observer.projection import image_projection
from agent2act.service.observer.store import TraceStore
from agent2act.service.trace import RequestTrace


def calibration():
    return {
        "available": True,
        "model": "opencv_pinhole",
        "width": 101,
        "height": 101,
        "fx": 100,
        "fy": 100,
        "cx": 50,
        "cy": 50,
        "distortion": [0] * 8,
        "optical_to_base_link": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        "metric_projection_ready": False,
    }


def jpeg():
    stream = io.BytesIO()
    Image.new("RGB", (101, 101), (80, 110, 130)).save(stream, "JPEG")
    return stream.getvalue()


def record(root, request="r1", episode=0, session="s1", sequence=1, *, done=False):
    trace = RequestTrace(root / session / request, action_log_path(root))
    trace("simulator-request", {"params": {"task_name": "pick", "episode_idx": episode, "prompt": "pick"}})
    if done:
        trace("runtime-event", {"sequence": sequence, "status": "episode_done"})
        return trace
    trace(
        "observer-input",
        {
            "sequence": sequence,
            "planning_attempt": 0,
            "reference": "gripper_center",
            "gripper_center_offset_m": {"left": [0.1, 0, 0], "right": [0.1, 0, 0]},
            "end_pose": {"base_link": {"left_arm": {"position": [0.1, 0, 1], "orientation": [0, 0, 0, 1]}}},
            "content": [
                {"type": "input_text", "text": "current camera=hand_left"},
                {
                    "type": "input_image",
                    "image_url": "data:image/jpeg;base64," + base64.b64encode(jpeg()).decode(),
                },
                {
                    "type": "input_text",
                    "text": json.dumps(
                        {"camera": "hand_left", "image_role": "current", "calibration": calibration()}
                    ),
                },
            ],
        },
    )
    trace(
        "planner-output-decision",
        {
            "decision": {
                "mode": "motion",
                "reason": "align",
                "motion": {"frame": "base_link", "waypoints": [{"left_pose": [0.2, 0, 1, 0, 0, 0, 1]}]},
            }
        },
    )
    trace(
        "observer-action",
        {
            "sequence": sequence,
            "frame": "base_link",
            "kind": "EEF_ABS",
            "steps": 3,
            "endpoint": {"left": [0.05, 0, 1, 0, 0, 0, 1]},
        },
    )
    trace("runtime-event", {"sequence": sequence, "status": "issued", "reason": "align"})
    return trace


def test_episode_isolation_feedback_reset_and_unknown(tmp_path):
    root = tmp_path / "run.trace"
    record(root)
    trace = record(root, "r2", sequence=2)
    trace("runtime-event", {"sequence": 2, "previous_sequence": 1, "status": "observed_tracking"})
    record(root, "r3", episode=1)
    record(root, "r4", session="s2")
    record(root, "r5", episode=None)
    store = TraceStore(root)
    assert len(store.catalog(None)["episodes"]) == 4
    first = store.detail("s1/r1")
    assert first["feedback"][0]["source_request"] == "s1/r2"
    assert store.detail("s1/r3")["feedback"] == []
    assert store.detail("s1/r5")["episode"]["episode"] is None
    record(root, "end", done=True)
    store.refresh()
    record(root, "restart")
    store.refresh()
    assert store.rows["s1/restart"]["episode_id"] != store.rows["s1/r1"]["episode_id"]


def test_exact_rgb_and_wrist_center_conversion(tmp_path):
    root = tmp_path / "run.trace"
    record(root)
    store = TraceStore(root)
    detail = store.detail("s1/r1")
    assert len(detail["views"]) == 1
    image = detail["views"][0]["images"][0]
    assert store.resolve(image["path"]).read_bytes() == jpeg()
    markers = {m["kind"]: m for m in image["markers"]}
    assert markers["current"]["pixel"] == [60, 50]
    assert markers["target"]["pixel"] == [70, 50]
    assert markers["issued"]["pose"][:3] == pytest.approx([0.15, 0, 1])
    assert markers["issued"]["pixel"] == pytest.approx([65, 50])
    assert markers["target"]["reference_only"] is True


def test_projection_distortion_resize_camera_pose_and_invalid_inputs():
    c = calibration()
    c["distortion"][0] = 0.5
    assert image_projection(c, [0.2, 0, 1], (101, 101))["pixel"][0] == pytest.approx(70.4)
    c = calibration()
    c["rgb_encoded_size"] = [50, 50]
    c["encoded_to_source_pixel"] = {"scale_x": 101 / 50, "scale_y": 101 / 50}
    assert image_projection(c, [0, 0, 1], (50, 50))["pixel"] == [24.5, 24.5]
    c = calibration()
    c["optical_to_base_link"][0][3] = 0.1
    assert image_projection(c, [0.1, 0, 1], (101, 101))["pixel"] == [50, 50]
    assert "behind" in image_projection(c, [0, 0, -1], (101, 101))["reason"]
    assert image_projection(c, [10, 0, 1], (101, 101))["pixel"] is None
    assert image_projection(c, [0, 0, 1], (50, 50))["pixel"] is None
    c["model"] = "fisheye"
    assert image_projection(c, [0, 0, 1], (101, 101))["pixel"] is None
    for bad in (None, [], {}, {"available": False}, {"available": True, "optical_to_base_link": []}):
        assert image_projection(bad, [0, 0, 1], (101, 101))["pixel"] is None


def test_retry_views_keep_targets_and_images_separate(tmp_path):
    root = tmp_path / "run.trace"
    trace = record(root)
    snapshot = json.loads(next(trace.directory.glob("*-observer-input.json")).read_text())
    # Reuse persisted references through the fixture writer; no new model calls.
    snapshot["planning_attempt"] = 1
    trace("observer-input", snapshot)
    trace("request-error", {"error_type": "ContractError"})
    detail = TraceStore(root).detail("s1/r1")
    assert detail["request"]["status"] == "failed"
    assert len(detail["views"]) == 2
    assert detail["views"][0]["decision"]["reason"] == "align"
    assert detail["views"][1]["decision"] is None
    assert not any(m["kind"] == "target" for m in detail["views"][1]["points"])


def test_tail_partial_json_truncate_and_symlink(tmp_path):
    root = tmp_path / "run.trace"
    record(root)
    store = TraceStore(root)
    store.refresh()
    offset = store.offset
    with action_log_path(root).open("ab") as stream:
        stream.write(b'{"broken":')
    store.refresh()
    assert store.offset == offset
    with action_log_path(root).open("ab") as stream:
        stream.write(b" true}\nnot json\n")
    store.refresh()
    assert store.malformed == 1
    action_log_path(root).write_text("")
    store.refresh()
    assert not store.rows
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    (root / "link.json").symlink_to(outside)
    with pytest.raises(ValueError):
        store.resolve("link.json")
    with pytest.raises(ValueError):
        store.resolve("../outside.json")


def test_trace_keeps_model_request_and_decision_identical(observation, motion_decision, tmp_path):
    calls = []

    def post(body):
        calls.append(deepcopy(body))
        return {
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": json.dumps(motion_decision)}]}
            ]
        }

    planner = ResponsesPlanner(model="offline", post=post)
    try:
        first = planner.decide(deepcopy(observation), {"sequence": 1})
        planner.reset()
        trace = RequestTrace(tmp_path / "s" / "r")
        with capture_artifacts(trace):
            second = planner.decide(deepcopy(observation), {"sequence": 1})
        assert calls[0] == calls[1]
        assert first == second
        assert list(trace.directory.glob("*-observer-input.json"))
    finally:
        planner.close()


def test_trace_keeps_issued_actions_identical(config, request_payload, tmp_path):
    normal = parse_settings(config)
    traced = parse_settings({**config, "logging": {"trace_dir": str(tmp_path / "run.trace")}})
    with ApplicationFactory(normal).create_session() as a, ApplicationFactory(traced).create_session() as b:
        first, second = a.handle(deepcopy(request_payload)), b.handle(deepcopy(request_payload))
    first["result"].pop("agent2act")
    second["result"].pop("agent2act")
    assert first == second
    store = TraceStore(tmp_path / "run.trace")
    catalog = store.catalog(None)
    assert len(catalog["requests"]) == 1
    assert catalog["requests"][0]["status"] == "issued"


def test_legacy_tcp_is_converted_and_joint_endpoint_uses_prefix_fk(tmp_path):
    root = tmp_path / "run.trace"
    trace = RequestTrace(root / "s" / "r", action_log_path(root))
    image = {"encoding": "JPEG", "image_data": jpeg(), "calibration": calibration()}
    trace("simulator-request", {"params": {"task_name": "t", "episode_idx": 0, "images": {"head": image}}})
    trace("runtime-event", {"status": "phase_started", "sequence": 1})
    trace(
        "planner-output-decision",
        {
            "decision": {
                "motion": {"frame": "base_link", "waypoints": [{"left_pose": [0.2, 0, 1, 0, 0, 0, 1]}]}
            }
        },
    )
    trace(
        "gripper-center-execution-decision",
        {"motion": {"frame": "base_link", "waypoints": [{"left_pose": [0.1, 0, 1, 0, 0, 0, 1]}]}},
    )
    trace(
        "runtime-event",
        {"status": "issued", "sequence": 1, "frame": "base_link", "kind": "JOINT_ABS", "issued_steps": 1},
    )
    trace(
        "simulator-response",
        {
            "result": {
                "agent2act": {"cartesian_path": {"left": [[0.05, 0, 1, 0, 0, 0, 1], [0.1, 0, 1, 0, 0, 0, 1]]}}
            }
        },
    )
    detail = TraceStore(root).detail("s/r")
    points = {p["kind"]: p for p in detail["views"][0]["points"]}
    assert points["target"]["pose"][0] == 0.1
    assert points["issued"]["pose"][0] == 0.05
    assert detail["views"][0]["reference"] == "wrist"


def test_terminal_snapshot_preserves_sequence_and_does_not_fake_success(config, request_payload, tmp_path):
    root = tmp_path / "run.trace"
    with ApplicationFactory(
        parse_settings({**config, "logging": {"trace_dir": str(root)}})
    ).create_session() as session:
        session.handle(deepcopy(request_payload))
        done = deepcopy(request_payload)
        done["params"]["episode_done"] = True
        session.handle(done)
    store = TraceStore(root)
    catalog = store.catalog(None)
    assert catalog["requests"][0]["status"] == "episode_done"
    assert catalog["requests"][0]["sequence"] == 2
    assert store.detail(catalog["requests"][1]["id"])["feedback"]


def test_raw_rgb_rejects_cross_snapshot_calibration(tmp_path):
    root = tmp_path / "run.trace"
    trace = RequestTrace(root / "s" / "r", action_log_path(root))
    image = {
        "encoding": "JPEG",
        "image_data": jpeg(),
        "snapshot_id": "current",
        "calibration": {**calibration(), "snapshot_id": "previous"},
    }
    trace("simulator-request", {"params": {"images": {"head": image}}})
    trace(
        "runtime-event",
        {
            "status": "planned",
            "sequence": 1,
            "decision": {
                "motion": {"frame": "base_link", "waypoints": [{"left_pose": [0, 0, 1, 0, 0, 0, 1]}]}
            },
        },
    )
    view = TraceStore(root).detail("s/r")["views"][0]
    marker = view["images"][0]["markers"][0]
    assert marker["pixel"] is None
    assert "snapshot_id" in marker["reason"]


def test_raw_provider_summary_is_available_without_diagnostics(tmp_path):
    root = tmp_path / "run.trace"
    trace = record(root)
    trace(
        "llm-response",
        {"output": [{"type": "reasoning", "summary": [{"type": "summary_text", "text": "public summary"}]}]},
    )
    summaries = TraceStore(root).detail("s1/r1")["summaries"]
    assert summaries[0]["reasoning_summary"] == ["public summary"]

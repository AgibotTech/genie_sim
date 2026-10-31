# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import copy
import io
import json
import logging
import struct

import pytest
from PIL import Image

from agent2act.adapters.planners.responses import ResponsesPlanner
from agent2act.adapters.planners.vision import depth_summary
from agent2act.bootstrap import ApplicationFactory, build_session, default_registry
from agent2act.configuration import parse_settings
from agent2act.domain.errors import ContractError, LocalIKConvergenceError, PlanningExhaustedError
from agent2act.execution.interpolation import LocalInterpolation
from agent2act.service.observability import HumanFilter, HumanFormatter, JsonFormatter, log_context


def response(decision):
    return {
        "status": "completed",
        "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(decision)}]}],
    }


def set_observed(request, chunk):
    request = copy.deepcopy(request)
    states = request["params"]["states"]
    for arm, values in (("left", chunk.left_arm), ("right", chunk.right_arm)):
        states["end_pose"][chunk.frame][arm + "_arm"] = {
            "position": values[-1][:3],
            "orientation": values[-1][3:],
        }
    states["gripper_states"] = chunk.grippers[-1]
    return request


def test_non_grid_duration_is_rounded_up_and_reaches_target(spec, observation):
    target = observation.eef("left", "base_link")
    target[0] += 0.01
    chunk = LocalInterpolation(spec).generate(
        observation, {"frame": "base_link", "waypoints": [{"duration_s": 0.113, "left_pose": target}]}
    )
    assert chunk.steps == 3
    assert chunk.left_arm[-1] == pytest.approx(target)
    for duration in (-1, 0, float("nan")):
        with pytest.raises(ContractError):
            LocalInterpolation(spec).generate(
                observation, {"frame": "base_link", "waypoints": [{"duration_s": duration}]}
            )


def test_backend_rejection_is_fed_back_and_raw_answer_is_saved(config, request_payload, tmp_path):
    calls = []
    decision = config["planner"]["decisions"][0]

    def post(body):
        calls.append(body)
        return response({**decision, "backend": "local_interpolation" if len(calls) == 1 else None})

    registry = default_registry()
    registry.planners.register("offline", lambda options: ResponsesPlanner(model="test", post=post))
    config["planner"] = {"type": "offline"}
    config["logging"] = {"trace_dir": str(tmp_path)}
    with ApplicationFactory(parse_settings(config), registry).create_session() as session:
        session.handle(request_payload)
    assert len(calls) == 2
    context = json.loads(calls[1]["input"][0]["content"][0]["text"])["context"]
    assert "backend=null" in context["recent_events"][-1]["error"]
    raw = sorted(tmp_path.rglob("*-llm-response.json"))
    assert len(raw) == 2
    assert "local_interpolation" in raw[0].read_text()
    assert list(tmp_path.rglob("*-simulator-request.json"))
    assert list(tmp_path.rglob("*-simulator-response.json"))


def test_exhausted_invalid_answer_still_has_artifacts(config, request_payload, tmp_path):
    # Capture the decision before replacing plugin configuration.
    decision = config["planner"]["decisions"][0]
    registry = default_registry()
    registry.planners.register(
        "offline",
        lambda options: ResponsesPlanner(
            model="test", post=lambda _: response({**decision, "backend": "wrong"})
        ),
    )
    config["planner"] = {"type": "offline"}
    config["execution"]["max_replans"] = 1
    config["logging"] = {"trace_dir": str(tmp_path)}
    with ApplicationFactory(parse_settings(config), registry).create_session() as session:
        with pytest.raises(PlanningExhaustedError):
            session.handle(request_payload)
    assert len(list(tmp_path.rglob("*-llm-response.json"))) == 2
    assert not list(tmp_path.rglob("*-simulator-response.json"))


def test_planning_exhaustion_can_end_failed_episode_without_changing_budget(
    config, request_payload, tmp_path
):
    decision = config["planner"]["decisions"][0]
    registry = default_registry()
    registry.planners.register(
        "offline",
        lambda options: ResponsesPlanner(
            model="test", post=lambda _: response({**decision, "backend": "wrong"})
        ),
    )
    config["planner"] = {"type": "offline"}
    config["execution"]["max_replans"] = 1
    config["execution"]["terminal_on_planning_exhaustion"] = True
    config["logging"] = {"trace_dir": str(tmp_path)}
    events = []
    with ApplicationFactory(parse_settings(config), registry).create_session(events.append) as session:
        result = session.handle(request_payload)["result"]
    assert result["episode_done"] is True
    assert result["agent2act"]["status"] == "planning_exhausted"
    assert result["agent2act"]["task_success"] == "unverified"
    left_count = len(config["robot"]["left_joints"])
    assert result["left_arm"]["values"] == [
        request_payload["params"]["states"]["arm_joint_states"][:left_count]
    ]
    assert len(list(tmp_path.rglob("*-llm-response.json"))) == 2
    assert list(tmp_path.rglob("*-simulator-response.json"))
    assert any(event.get("status") == "planning_exhausted" for event in events)
    assert not any(event.get("status") == "issued" for event in events)


def test_ik_exhaustion_holds_for_new_observation_without_ending_episode(config, request_payload, monkeypatch):
    config["execution"]["execute_steps"] = 10
    config["execution"]["max_replans"] = 1
    config["execution"]["terminal_on_planning_exhaustion"] = True
    config["execution"]["retry_ik_exhaustion"] = True
    decision = config["planner"]["decisions"][0]
    registry = default_registry()
    registry.planners.register(
        "offline", lambda options: ResponsesPlanner(model="test", post=lambda _: response(decision))
    )
    config["planner"] = {"type": "offline"}
    calls = []

    def reject_ik(*_args, **_kwargs):
        calls.append(1)
        raise LocalIKConvergenceError("local IK did not converge; no motion issued")

    events = []
    with ApplicationFactory(parse_settings(config), registry).create_session(events.append) as session:
        monkeypatch.setattr(session.runtime.router, "generate", reject_ik)
        first = session.handle(request_payload)["result"]
        second = session.handle(request_payload)["result"]
    assert len(calls) == 4
    for result in (first, second):
        assert "episode_done" not in result
        assert result["agent2act"]["status"] == "ik_replan_required"
        assert result["agent2act"]["task_success"] == "unverified"
        held_left = request_payload["params"]["states"]["arm_joint_states"][:1]
        held_right = request_payload["params"]["states"]["arm_joint_states"][1:]
        assert result["left_arm"]["values"] == [held_left] * 10
        assert result["right_arm"]["values"] == [held_right] * 10
        assert len(result["left_effector"]) == len(result["right_effector"]) == 10
    assert [first["agent2act"]["sequence"], second["agent2act"]["sequence"]] == [1, 2]
    assert sum(event.get("status") == "ik_replan_required" for event in events) == 2
    assert not any(event.get("status") == "issued" for event in events)


def test_model_motion_omissions_are_normalized_at_planner_boundary(observation):
    malformed = {
        "mode": "motion",
        "instruction": "move",
        "reason": "approach",
        "backend": None,
        "status": "continue",
        "motion": {
            "frame": "base_link",
            "waypoints": [{"duration_s": 0.1, "left_pose": [0.25, -0.1, 0.4], "right_pose": None}],
        },
    }
    planner = ResponsesPlanner(model="test", post=lambda _: response(malformed))
    decision = planner.decide(observation, {})
    waypoint = decision.motion["waypoints"][0]
    assert waypoint["left_pose"] == pytest.approx(
        [0.25, -0.1, 0.4, *observation.eef("left", "base_link")[3:]]
    )
    assert "right_pose" not in waypoint
    planner.close()


def test_continuation_finishes_target_without_second_planner_call(config, request_payload):
    config["execution"].update(execute_steps=8, max_motion_continuations=2)
    events = []
    with build_session(config, event_sink=events.append) as session:
        session.handle(request_payload)
        first = session.runtime._last_issued
        second_request = set_observed(request_payload, first)
        session.handle(second_request)  # The scripted planner only has ONE decision.
        second = session.runtime._last_issued
        session.handle(set_observed(second_request, second))
        last = session.runtime._last_issued
        assert last.right_arm[-1][2] == pytest.approx(0.55)
        assert last.steps == 4
        assert session.runtime._pending_motion is None
    assert len([e for e in events if e.get("status") == "motion_continued"]) == 2
    assert all(e["task_success"] == "unverified" for e in events if e.get("status") == "observed_tracking")


def test_tracking_error_prevents_continuation(config, request_payload):
    config["execution"].update(
        execute_steps=8, max_motion_continuations=2, continuation_translation_tolerance_m=0.001
    )
    config["planner"]["decisions"] *= 2
    events = []
    with build_session(config, event_sink=events.append) as session:
        session.handle(request_payload)
        session.handle(request_payload)  # Actual robot has not moved.
    assert not any(e.get("status") == "motion_continued" for e in events)
    tracking = next(e for e in events if e.get("status") == "observed_tracking")
    assert tracking["right"]["observed_displacement_m"] == 0
    assert tracking["right"]["translation_error_m"] > 0.001


def test_reset_discards_pending_motion(config, request_payload):
    config["execution"]["max_motion_continuations"] = 2
    with build_session(config) as session:
        session.handle(request_payload)
        assert session.runtime._pending_motion is not None
        session.runtime.reset()
        assert session.runtime._pending_motion is None
        assert session.runtime._last_issued is None


def test_depth_invalid_and_saturated_values_are_not_invented():
    item = {
        "encoding": "RAW_UINT16",
        "width": 2,
        "height": 2,
        "image_data": struct.pack("<4H", 0, 500, 65535, 1500),
    }
    summary = depth_summary(item, grid=2)
    assert summary["grid"] == [[None, 500], [None, 1500]]
    assert summary["camera_to_robot_calibration"].startswith("not determined by this depth summary")
    assert not depth_summary({**item, "width": 10})["available"]
    assert not depth_summary(None)["available"]


def test_rgb_compression_depth_and_trace_references(observation, tmp_path, caplog):
    from agent2act.adapters.telemetry import capture_artifacts
    from agent2act.service.trace import RequestTrace

    buf = io.BytesIO()
    Image.new("RGB", (1280, 720), "red").save(buf, "JPEG", quality=95)
    observation.images = {
        "head": {"encoding": "JPEG", "image_data": buf.getvalue()},
        "head_depth": {
            "encoding": "RAW_UINT16",
            "width": 8,
            "height": 8,
            "image_data": struct.pack("<64H", *([1200] * 64)),
        },
    }
    calls = []
    planner = ResponsesPlanner(
        model="test",
        image_max_side=640,
        use_depth=True,
        post=lambda body: (
            calls.append(body)
            or response(
                {
                    "mode": "motion",
                    "instruction": "hold",
                    "reason": "test",
                    "motion": {"frame": "base_link", "waypoints": [{"duration_s": 0.1}]},
                }
            )
        ),
    )
    with caplog.at_level(logging.INFO), capture_artifacts(RequestTrace(tmp_path)):
        planner.decide(observation, {})
    body = calls[0]
    content = body["input"][0]["content"]
    depth = json.loads(content[-1]["text"])["depth"]
    assert depth["encoding"] == "PNG_UINT16"
    assert "grid" not in depth
    artifact = json.loads(next(tmp_path.glob("*-llm-request.json")).read_text())
    ref = artifact["input"][0]["content"][2]["image_url"]
    with Image.open(tmp_path / ref["$artifact"]) as image:
        assert image.size == (640, 360)
    metrics = next(r.event for r in caplog.records if r.getMessage() == "camera_input")
    assert metrics["encoded_bytes"] < metrics["source_bytes"]
    assert metrics["depth_encoded_bytes"] > 0
    assert "data:image/jpeg;base64," not in json.dumps(artifact)


def test_human_log_is_readable_and_json_preserves_context():
    record = logging.LogRecord("test", logging.INFO, "", 1, "runtime_event", (), None)
    record.event = {
        "status": "issued",
        "issued_steps": 20,
        "discarded_steps": 3,
        "reason": "Sponge beside fingers.\nShift left.",
    }
    with log_context(inference_id="abcdefgh1234"):
        assert "abcdefgh issued" in HumanFormatter().format(record)
        assert "reason=Sponge beside fingers. Shift left." in HumanFormatter().format(record)
        assert json.loads(JsonFormatter().format(record))["inference_id"] == "abcdefgh1234"
    record.event = {"status": "phase_started"}
    assert not HumanFilter().filter(record)


def test_human_log_shows_model_object_target_and_issued_endpoint():
    record = logging.LogRecord("test", logging.INFO, "", 1, "model_object_location", (), None)
    record.event = {
        "location": {
            "label": "apple",
            "x_m": None,
            "y_m": None,
            "z_m": None,
            "frame": None,
            "basis": "visual_estimate",
            "camera": "head",
            "u_norm": 430,
            "v_norm": 570,
        }
    }
    assert "XYZ=unknown" in HumanFormatter().format(record)
    assert "head(430,570)/1000" in HumanFormatter().format(record)
    record.msg = "model_motion_target"
    record.event = {"frame": "base_link", "right_pose": [0.5, 0.1, 0.8, 0, 0, 0, 1]}
    assert "right_target=[0.500,0.100,0.800]m" in HumanFormatter().format(record)
    record.msg = "runtime_event"
    record.event = {
        "status": "issued",
        "kind": "EEF_ABS",
        "frame": "base_link",
        "issued_endpoint": {"right_arm": [0.48, 0.1, 0.7, 0, 0, 0, 1]},
    }
    assert "right_issued=[0.480,0.100,0.700]m" in HumanFormatter().format(record)


def test_continuation_limit_and_done_are_honored(config, request_payload):
    config["execution"].update(execute_steps=2, max_motion_continuations=1)
    config["planner"]["decisions"] *= 2
    events = []
    with build_session(config, event_sink=events.append) as session:
        session.handle(request_payload)
        obs = set_observed(request_payload, session.runtime._last_issued)
        session.handle(obs)
        obs = set_observed(obs, session.runtime._last_issued)
        session.handle(obs)  # Budget forces a fresh planner call.
        assert len([e for e in events if e.get("status") == "motion_continued"]) == 1
        obs = set_observed(obs, session.runtime._last_issued)
        obs["params"]["episode_done"] = True
        session.handle(obs)
        assert session.runtime._pending_motion is None
        assert events[-1]["status"] == "episode_done"
        assert events[-2]["status"] == "observed_tracking"


def test_factory_rolls_back_planner_if_later_construction_fails(config, monkeypatch):
    from agent2act.domain.errors import ConfigurationError

    closed = []

    class Planner:
        def close(self):
            closed.append(True)

    registry = default_registry()
    registry.planners.register("offline", lambda options: Planner())
    config["planner"] = {"type": "offline"}

    def fail(spec):
        raise ContractError("construction failed")

    monkeypatch.setattr("agent2act.bootstrap.MotionCompiler", fail)
    with pytest.raises(ConfigurationError):
        ApplicationFactory(parse_settings(config), registry).create_session()
    assert closed == [True]


def test_http_phases_and_credentials_are_not_in_artifacts(observation, tmp_path, monkeypatch, caplog):
    import urllib.request

    from agent2act.adapters.telemetry import capture_artifacts
    from agent2act.service.trace import RequestTrace

    class Reply(io.BytesIO):
        status = 200

    decision = {
        "mode": "motion",
        "instruction": "hold",
        "reason": "test",
        "motion": {"frame": "base_link", "waypoints": [{"duration_s": 0.1}]},
    }

    def urlopen(request, timeout):
        assert request.get_header("Authorization") == "Bearer TEST_SECRET"
        return Reply(json.dumps(response(decision)).encode())

    monkeypatch.setenv("OFFLINE_TEST_KEY", "TEST_SECRET")
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    with capture_artifacts(RequestTrace(tmp_path)), caplog.at_level(logging.INFO):
        ResponsesPlanner(model="test", api_key_env="OFFLINE_TEST_KEY").decide(observation, {})
    event = next(r.event for r in caplog.records if r.getMessage() == "llm_http_finished")
    assert event["headers_wait_s"] >= 0
    assert event["body_read_s"] >= 0
    assert all("TEST_SECRET" not in p.read_text() for p in tmp_path.glob("*.json"))


def test_report_handles_old_logs_and_escapes_model_text(tmp_path):
    import importlib.util
    from pathlib import Path

    module_spec = importlib.util.spec_from_file_location(
        "report", Path(__file__).parents[2] / "scripts/render_agent_report.py"
    )
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    log = tmp_path / "agent.jsonl"
    log.write_text(
        "not-json\n"
        + json.dumps(
            {
                "inference_id": "test",
                "status": "issued",
                "instruction": "<script>untrusted</script>",
                "episode": ["test", 0, "pick"],
            }
        )
    )
    output = tmp_path / "report.html"
    module.render(log, output)
    text = output.read_text()
    assert "1 malformed lines" in text
    assert "&lt;script&gt;" in text
    assert "<script>" not in text
    assert "unfinished" in text


def test_readable_policy_diagnostics_keep_full_json():
    record = logging.LogRecord("agent2act", logging.INFO, "", 0, "pi05_gripper_clipped", (), None)
    corrections = [{"delta": 0.0042, "original": -0.7892, "corrected": -0.785}] * 92
    record.event = {"corrections": corrections, "tolerance": 0.005}
    readable = HumanFormatter().format(record)
    assert "92" in readable and "0.004200" in readable
    assert "original" not in readable
    assert json.loads(JsonFormatter().format(record))["corrections"] == corrections
    record.msg = "llm_http_started"
    record.event = {"sequence": 1, "planning_attempt": 2, "timeout_s": 300}
    assert "no new action issued this round" in HumanFormatter().format(record)
    assert "planning_attempt=2" in HumanFormatter().format(record)
    record.msg = "runtime_event"
    record.event = {"status": "rejected", "error": "bounds exceeded"}
    assert "not issued" in HumanFormatter().format(record)


def test_readable_filter_preserves_request_start_and_warnings():
    record = logging.LogRecord("agent2act", logging.INFO, "", 0, "phase_started", (), None)
    record.event = {"phase": "inference_request"}
    assert HumanFilter().filter(record)
    record.event = {"phase": "observation_decode"}
    assert not HumanFilter().filter(record)
    record.levelno = logging.WARNING
    assert HumanFilter().filter(record)

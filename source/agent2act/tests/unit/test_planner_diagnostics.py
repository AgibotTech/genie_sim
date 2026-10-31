# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import base64
import hashlib
import io
import json
from copy import deepcopy

import pytest
from PIL import Image

from agent2act.adapters.planners.chat_completions import ChatCompletionsPlanner
from agent2act.adapters.planners.diagnostics import (
    evidence_warnings,
    extract_diagnostics,
    response_summary,
)
from agent2act.adapters.planners.embodied import EmbodiedPlanner
from agent2act.adapters.planners.responses import ResponsesPlanner
from agent2act.adapters.telemetry import capture_artifacts
from agent2act.domain.errors import ContractError


@pytest.fixture
def report():
    return {
        "rgb_use": "used",
        "depth_use": "not_used",
        "evidence": [
            {
                "modality": "rgb",
                "source": "input:2 head region [450,500]",
                "finding": "Target silhouette is visible near the open gripper.",
            }
        ],
        "estimates": [
            {
                "quantity": "approach distance",
                "value": 0.03,
                "units": "m",
                "frame": "base_link",
                "source": "input:2",
                "basis": "visual_estimate",
            }
        ],
        "object_location": {
            "label": "target object",
            "frame": None,
            "x_m": None,
            "y_m": None,
            "z_m": None,
            "camera": "head",
            "u_norm": 450,
            "v_norm": 500,
            "source": "input:2",
            "basis": "visual_estimate",
        },
        "camera_projections": [],
        "uncertainties": ["Depth tile mixes target and background; contact is unconfirmed."],
        "expected_effect": "Reduce visible gripper-to-target separation.",
        "verification": "Compare target and fingertip positions in the next observation.",
    }


@pytest.mark.parametrize("planner_type", [ResponsesPlanner, ChatCompletionsPlanner, EmbodiedPlanner])
@pytest.mark.parametrize("enabled", [False, True])
def test_diagnostics_extension_keeps_actions_and_baseline_wire(
    planner_type, enabled, report, observation, motion_decision
):
    calls, artifacts = [], []
    payload = {**motion_decision, **({"diagnostics": report} if enabled else {})}

    def post(body):
        calls.append(body)
        if "messages" in body:
            return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(payload)}}]}
        return {
            "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(payload)}]}]
        }

    planner = planner_type(
        model="test",
        post=post,
        diagnostic_mode=enabled,
        compact_motion_context=True,
        current_observation_only=True,
    )
    try:
        with capture_artifacts(lambda kind, value: artifacts.append((kind, deepcopy(value)))):
            decision = planner.decide(
                observation, {"sequence": 7, "planning_attempt": 2, "execution": {"mode": "motion"}}
            )
        assert decision.reason == motion_decision["reason"]
        assert not hasattr(decision, "diagnostics")
        assert ("DIAGNOSTIC OUTPUT EXTENSION" in json.dumps(calls)) == enabled
        reports = [value for kind, value in artifacts if kind == "decision-diagnostics"]
        if enabled:
            assert reports[0]["report"] == report
            assert reports[0]["sequence"] == 7
            assert reports[0]["planning_attempt"] == 2
            assert "input_manifest" in json.dumps(calls)
        else:
            assert not reports
            assert "input_manifest" not in json.dumps(calls)
    finally:
        planner.close()


@pytest.mark.parametrize("bad", [None, {}, "text", {"unexpected": 1}])
def test_invalid_report_is_removed_without_corrupting_decision(bad, motion_decision):
    decision, diagnostic = extract_diagnostics({**motion_decision, "diagnostics": bad})
    assert decision == motion_decision
    assert diagnostic["status"] == "invalid"
    assert diagnostic["report"] is None
    assert extract_diagnostics(motion_decision)[1]["status"] == "missing"


@pytest.mark.parametrize("bad", [True, float("nan"), float("inf"), 10**1000])
def test_invalid_estimates_not_recorded_as_measured_numbers(bad, report, motion_decision):
    report["estimates"][0]["value"] = bad
    assert extract_diagnostics({**motion_decision, "diagnostics": report})[1]["status"] == "invalid"


def test_object_location_requires_finite_numeric_coordinates(report, motion_decision):
    report["object_location"]["x_m"] = float("nan")
    assert extract_diagnostics({**motion_decision, "diagnostics": report})[1]["status"] == "invalid"


def test_camera_projection_is_model_reported_and_validated(report, motion_decision):
    projection = {
        "camera": "hand_right",
        "arm": "right",
        "u_norm": 610,
        "v_norm": 470,
        "image_source": "input:2",
        "calibration_source": "input:3",
        "method": "calibration_projection",
        "note": "Projected first waypoint.",
    }
    report["camera_projections"] = [projection]
    assert extract_diagnostics({**motion_decision, "diagnostics": report})[1]["report"] == report
    assert any("input:3" in warning for warning in evidence_warnings(report, [{"id": "input:2"}]))
    projection["u_norm"] = float("nan")
    assert extract_diagnostics({**motion_decision, "diagnostics": report})[1]["status"] == "invalid"


def test_older_diagnostics_without_camera_projections_remain_readable(report, motion_decision):
    del report["camera_projections"]
    assert extract_diagnostics({**motion_decision, "diagnostics": report})[1]["report"] == report


@pytest.mark.parametrize("depth", [False, True])
def test_input_inventory_matches_actual_encoded_images_and_depth_availability(
    depth, observation, motion_decision
):
    image = io.BytesIO()
    Image.new("RGB", (256, 256), "blue").save(image, "JPEG")
    observation.images = {
        "head": {"encoding": "JPEG", "image_data": image.getvalue()},
        "head_depth": {
            "encoding": "RAW_UINT16",
            "width": 256,
            "height": 256,
            "image_data": b"\xe8\x03" * 256 * 256,
        },
    }
    calls, artifacts = [], {}

    def post(body):
        calls.append(body)
        return {
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": json.dumps(motion_decision)}]}
            ]
        }

    planner = ResponsesPlanner(
        model="test", post=post, use_depth=depth, diagnostic_mode=True, image_max_side=128
    )
    try:
        with capture_artifacts(lambda kind, value: artifacts.update({kind: value})):
            planner.decide(observation, {"execution": {"mode": "motion"}})
        content = calls[0]["input"][0]["content"]
        manifest = json.loads(content[-1]["text"])["input_manifest"]
        full_manifest = artifacts["diagnostic-input"]["input_manifest"]
        assert [item["id"] for item in manifest] == [item["id"] for item in full_manifest]
        assert all(
            "sha256" not in item and "calibration" not in item and "depth" not in item for item in manifest
        )
        assert len(json.dumps(manifest)) < len(json.dumps(full_manifest))
        images = [item for item in full_manifest if item["type"] == "input_image"]
        assert len(images) == (2 if depth else 1)
        assert images[0]["media_type"] == "image/jpeg"
        for item in images:
            index = int(item["id"].split(":")[1])
            encoded = base64.b64decode(content[index]["image_url"].split(",")[1])
            assert item["sha256"] == hashlib.sha256(encoded).hexdigest()
        assert any("depth" in item for item in full_manifest) == depth
        assert artifacts["decision-diagnostics"]["status"] == "missing"
    finally:
        planner.close()


def test_responses_summary_option_and_public_only_extraction(observation, motion_decision):
    calls, artifacts = [], {}

    def post(body):
        calls.append(body)
        return {
            "output": [
                {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "Public evidence summary"}],
                    "content": ["do not extract"],
                    "encrypted_content": "do not decrypt",
                },
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": json.dumps(motion_decision)}],
                },
            ]
        }

    planner = ResponsesPlanner(model="test", post=post, reasoning_summary="auto")
    try:
        with capture_artifacts(lambda kind, value: artifacts.update({kind: value})):
            planner.decide(observation, {})
        assert calls[0]["reasoning"] == {"summary": "auto"}
        summary = artifacts["reasoning-summary"]
        assert summary["summary"] == ["Public evidence summary"]
        assert "do not" not in json.dumps(summary)
        assert "DIAGNOSTIC OUTPUT EXTENSION" not in json.dumps(calls)
        assert response_summary({"output": 123})["status"] == "not_provided"
    finally:
        planner.close()


@pytest.mark.parametrize("options", [{"diagnostic_mode": "yes"}, {"reasoning_summary": "full"}])
def test_invalid_settings_rejected(options):
    with pytest.raises(ContractError):
        ResponsesPlanner(model="test", **options)


def test_chat_completions_rejects_unsupported_summary_option():
    with pytest.raises(ContractError, match="reasoning_summary"):
        ChatCompletionsPlanner(model="test", reasoning_summary="auto")


def test_modality_use_claims_are_checked_only_against_input_presence(report):
    report["depth_use"] = "used"
    warnings = evidence_warnings(report, [{"id": "input:0", "type": "input_text"}])
    assert any("claims depth" in warning for warning in warnings)
    assert any("claims rgb" in warning for warning in warnings)
    assert any("input:2" in warning for warning in warnings)
    manifest = [
        {"id": "input:2", "media_type": "image/jpeg"},
        {"id": "input:3", "depth": {"available": True}},
    ]
    assert (
        evidence_warnings(report, manifest) == []
    )  # Absence of warnings is not evidence of correct perception.


def test_api_tool_round_summaries_and_final_diagnostics_are_all_preserved(
    observation, motion_decision, report, monkeypatch
):
    calls, artifacts = [], []

    def post(body):
        calls.append(deepcopy(body))
        summary = {"type": "reasoning", "summary": [{"type": "summary_text", "text": f"round {len(calls)}"}]}
        item = (
            {
                "type": "function_call",
                "name": "measure_surface_points",
                "call_id": "depth-call",
                "arguments": "{}",
            }
            if len(calls) == 1
            else {
                "type": "message",
                "content": [
                    {"type": "output_text", "text": json.dumps({**motion_decision, "diagnostics": report})}
                ],
            }
        )
        return {"output": [summary, item]}

    planner = EmbodiedPlanner(
        model="test",
        post=post,
        use_depth=True,
        enable_depth_measurement=True,
        reasoning_summary="auto",
        reasoning_effort="high",
        diagnostic_mode=True,
    )
    monkeypatch.setattr(
        planner, "_execute_geometry_call", lambda call, obs: {"status": "fixture_measurement"}
    )
    try:
        with capture_artifacts(lambda kind, value: artifacts.append((kind, deepcopy(value)))):
            result = planner.decide(observation, {"execution": {"mode": "motion"}})
        assert result.reason == motion_decision["reason"]
        assert len(calls) == 2
        assert all(body["reasoning"] == {"effort": "high", "summary": "auto"} for body in calls)
        assert all("DIAGNOSTIC OUTPUT EXTENSION" in json.dumps(body) for body in calls)
        summaries = [value for kind, value in artifacts if kind == "reasoning-summary"]
        assert [value["summary"] for value in summaries] == [["round 1"], ["round 2"]]
        assert [value["round"] for value in summaries] == [0, 1]
        assert next(value for kind, value in artifacts if kind == "decision-diagnostics")["report"] == report
    finally:
        planner.close()

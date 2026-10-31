# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import importlib.util
import json
from pathlib import Path


def test_export_pairs_next_observation_with_prior_action_and_preserves_model_evidence(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "export_action_reasoning",
        Path(__file__).resolve().parents[2] / "scripts/export_action_reasoning.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    request = tmp_path / "request.json"
    response = tmp_path / "response.json"
    request.write_text(json.dumps({"model": "requested"}))
    response.write_text(
        json.dumps(
            {
                "model": "other",
                "output": [
                    {
                        "type": "reasoning",
                        "summary": [{"type": "summary_text", "text": "Public summary"}],
                        "encrypted_content": "DO_NOT_EXPORT",
                        "content": [{"type": "text", "text": "DO_NOT_EXPORT"}],
                    }
                ],
            }
        )
    )
    base = {"session_id": "s", "episode": ["task", 0, "pick"]}
    events = [
        {"inference_id": "a", "sequence": 1, "status": "phase_started"},
        {
            "inference_id": "a",
            "message": "artifact_saved",
            "artifact_kind": "llm-request",
            "artifact_path": str(request),
        },
        {
            "inference_id": "a",
            "message": "artifact_saved",
            "artifact_kind": "llm-response",
            "artifact_path": str(response),
        },
        {"inference_id": "a", "sequence": 1, "status": "issued", "reason": "Approach"},
        {"inference_id": "b", "sequence": 2, "status": "observed_tracking", "previous_sequence": 1},
        {"inference_id": "b", "sequence": 2, "status": "episode_done"},
        # Same episode label after reset must not inherit the old action's feedback.
        {"inference_id": "c", "sequence": 2, "status": "observed_tracking", "previous_sequence": 1},
    ]
    log, output = tmp_path / "run.jsonl", tmp_path / "reasons.jsonl"
    log.write_text("".join(json.dumps({**base, **event}) + "\n" for event in events))
    rows = module.export(log, output)
    assert rows[0]["issued"][0]["reason"] == "Approach"
    assert rows[0]["feedback_after_action"][0]["inference_id"] == "b"
    assert len(rows[0]["feedback_after_action"]) == 1
    assert rows[1]["feedback_after_action"] == []
    assert rows[2]["export_warnings"]
    assert rows[0]["responses"][0]["model_identity_status"] == "mismatch"
    assert rows[0]["responses"][0]["reasoning_summary"] == ["Public summary"]
    assert "DO_NOT_EXPORT" not in output.read_text()
    assert module.response_metadata({"text": "{}"})["reasoning_summary_status"] == "not_provided"


def test_diagnostic_exports_and_html_include_public_evidence_and_images(tmp_path):
    def load(name):
        spec = importlib.util.spec_from_file_location(
            name, Path(__file__).resolve().parents[2] / f"scripts/{name}.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    data = {
        "diagnostic-input": {
            "input_manifest": [{"id": "input:2", "label": "current head depth", "sha256": "hash"}]
        },
        "decision-diagnostics": {
            "status": "available",
            "report": {"expected_effect": "<script>UNTRUSTED</script>"},
        },
        "reasoning-summary": {"summary": ["Public summary"]},
        "llm-request": {"input": [{"type": "image", "url": {"$artifact": "depth.png", "sha256": "hash"}}]},
    }
    events = []
    for kind, value in data.items():
        path = tmp_path / f"{kind}.json"
        path.write_text(json.dumps(value))
        events.append(
            {
                "session_id": "s",
                "inference_id": "a",
                "message": "artifact_saved",
                "artifact_kind": kind,
                "artifact_path": str(path),
            }
        )
    log = tmp_path / "run.jsonl"
    log.write_text("".join(json.dumps(event) + "\n" for event in events))
    rows = load("export_action_reasoning").export(log, tmp_path / "export.jsonl")
    assert rows[0]["decision_diagnostics"][0]["status"] == "available"
    assert rows[0]["reasoning_summary"][0]["summary"] == ["Public summary"]
    html_path = tmp_path / "report.html"
    load("render_agent_report").render(log, html_path)
    html = html_path.read_text()
    assert "input:2 current head depth" in html
    assert "Provider reasoning summary" in html
    assert "<script>UNTRUSTED</script>" not in html
    assert "&lt;script&gt;" in html


def test_html_compares_model_object_and_control_targets_without_conflating_issued_prefix(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "render_agent_report", Path(__file__).resolve().parents[2] / "scripts/render_agent_report.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    diagnostic = tmp_path / "diagnostic.json"
    diagnostic.write_text(
        json.dumps(
            {
                "status": "available",
                "report": {
                    "object_location": {
                        "label": "apple",
                        "frame": "base_link",
                        "x_m": 0.5,
                        "y_m": 0.1,
                        "z_m": 0.8,
                        "camera": "head",
                        "u_norm": 430,
                        "v_norm": 570,
                        "source": "input:2",
                        "basis": "visual_estimate",
                    }
                },
            }
        )
    )
    model = tmp_path / "model.json"
    model.write_text(
        json.dumps(
            {
                "decision": {
                    "motion": {
                        "frame": "base_link",
                        "waypoints": [{"right_pose": [0.51, 0.11, 0.82, 0, 0, 0, 1]}],
                    }
                }
            }
        )
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "input_manifest": [
                    {
                        "id": "input:2",
                        "sha256": "current-rgb",
                        "label": "current camera=head",
                        "media_type": "image/jpeg",
                    },
                    {
                        "id": "input:4",
                        "sha256": "history-rgb",
                        "label": "history camera=head",
                        "media_type": "image/jpeg",
                    },
                    {
                        "id": "input:6",
                        "sha256": "current-depth",
                        "label": "current camera=head raw depth",
                        "media_type": "image/png",
                    },
                ]
            }
        )
    )
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {
                "input": [
                    {"url": {"$artifact": "head.jpg", "sha256": "current-rgb"}},
                    {"url": {"$artifact": "history.jpg", "sha256": "history-rgb"}},
                    {"url": {"$artifact": "depth.png", "sha256": "current-depth"}},
                ]
            }
        )
    )
    base = {"inference_id": "one", "session_id": "session"}
    events = [
        {
            **base,
            "message": "artifact_saved",
            "artifact_kind": "diagnostic-input",
            "artifact_path": str(manifest),
        },
        {
            **base,
            "message": "artifact_saved",
            "artifact_kind": "llm-request",
            "artifact_path": str(request),
        },
        {
            **base,
            "message": "artifact_saved",
            "artifact_kind": "decision-diagnostics",
            "artifact_path": str(diagnostic),
        },
        {
            **base,
            "message": "artifact_saved",
            "artifact_kind": "planner-output-decision",
            "artifact_path": str(model),
        },
        {
            **base,
            "status": "planned",
            "decision": {
                "motion": {
                    "frame": "base_link",
                    "waypoints": [{"right_pose": [0.51, 0.11, 0.67, 0, 0, 0, 1]}],
                }
            },
        },
        {
            **base,
            "status": "issued",
            "kind": "EEF_ABS",
            "frame": "base_link",
            "issued_endpoint": {"right_arm": [0.49, 0.10, 0.68, 0, 0, 0, 1]},
        },
    ]
    log = tmp_path / "run.jsonl"
    log.write_text("".join(json.dumps(event) + "\n" for event in events))
    output = tmp_path / "report.html"
    module.render(log, output)
    html = output.read_text()
    for value in (
        "apple",
        "head image (430, 570)",
        "0.510, 0.110, 0.820 m",
        "0.510, 0.110, 0.670 m",
        "0.490, 0.100, 0.680 m",
    ):
        assert value in html
    assert html.count('class="model-marker"') == 1
    assert 'style="left:43.0%;top:57.0%"' in html
    assert "The red hollow circle is the object image position reported by the LLM" in html
    assert "LLM estimate: apple" in html
    assert "Current head camera (input:2)" in html


def test_image_marker_requires_cited_current_rgb_and_valid_normalized_pixel():
    spec = importlib.util.spec_from_file_location(
        "render_agent_report", Path(__file__).resolve().parents[2] / "scripts/render_agent_report.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    claim = {"camera": "head", "u_norm": 430, "v_norm": 570, "source": "input:2"}
    image = {"id": "input:2", "label": "current camera=head", "media_type": "image/jpeg"}
    assert module.image_marker(claim, image) == [430, 570]
    assert module.image_marker({**claim, "source": "input:20"}, image) is None
    assert module.image_marker({**claim, "u_norm": 1001}, image) is None
    assert module.image_marker(claim, {**image, "label": "history camera=head"}) is None
    assert module.image_marker(claim, {**image, "media_type": "image/png"}) is None


def test_html_projects_model_target_only_with_matching_current_camera_calibration(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "render_agent_report", Path(__file__).resolve().parents[2] / "scripts/render_agent_report.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calibration = {
        "available": True,
        "metric_projection_ready": False,
        "source_intrinsics": {
            "fx": 100,
            "fy": 100,
            "cx": 50,
            "cy": 50,
            "width": 101,
            "height": 101,
            "distortion": [0] * 8,
        },
        "optical_to_base_link": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    }
    assert module.project_base_point(calibration, [0.1, 0, 1]) == [600, 500]
    assert module.project_base_point(calibration, [0, 0, -1]) is None
    assert module.project_base_point(calibration, [10, 0, 1]) is None
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "input_manifest": [
                    {
                        "id": "input:2",
                        "sha256": "head",
                        "label": "current camera=head",
                        "media_type": "image/jpeg",
                    },
                    {
                        "id": "input:3",
                        "camera": "head",
                        "image_role": "current",
                        "calibration": {"available": True},
                    },
                    {
                        "id": "input:5",
                        "sha256": "wrist",
                        "label": "current camera=hand_right",
                        "media_type": "image/jpeg",
                    },
                    {
                        "id": "input:6",
                        "camera": "hand_right",
                        "image_role": "current",
                        "calibration": calibration,
                    },
                ]
            }
        )
    )
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {
                "input": [
                    {"url": {"$artifact": "head.jpg", "sha256": "head"}},
                    {"url": {"$artifact": "wrist.jpg", "sha256": "wrist"}},
                ]
            }
        )
    )
    diagnostic = tmp_path / "diagnostic.json"
    diagnostic.write_text(
        json.dumps(
            {
                "status": "available",
                "report": {
                    "object_location": {
                        "label": "mouse",
                        "camera": "head",
                        "u_norm": 400,
                        "v_norm": 600,
                        "source": "input:2",
                        "frame": None,
                        "x_m": None,
                        "y_m": None,
                        "z_m": None,
                    },
                    "camera_projections": [
                        {
                            "camera": "hand_right",
                            "arm": "right",
                            "u_norm": 620,
                            "v_norm": 480,
                            "image_source": "input:5",
                            "calibration_source": "input:6",
                            "method": "calibration_projection",
                            "note": "I projected the first waypoint using the camera matrix.",
                        }
                    ],
                },
            }
        )
    )
    model = tmp_path / "model.json"
    model.write_text(
        json.dumps(
            {
                "decision": {
                    "motion": {"frame": "base_link", "waypoints": [{"right_pose": [0.1, 0, 1, 0, 0, 0, 1]}]}
                }
            }
        )
    )
    events = [
        {
            "inference_id": "one",
            "message": "artifact_saved",
            "artifact_kind": kind,
            "artifact_path": str(path),
        }
        for kind, path in (
            ("diagnostic-input", manifest),
            ("llm-request", request),
            ("decision-diagnostics", diagnostic),
            ("planner-output-decision", model),
        )
    ]
    log = tmp_path / "run.jsonl"
    log.write_text("".join(json.dumps(event) + "\n" for event in events))
    output = tmp_path / "report.html"
    module.render(log, output)
    rendered = output.read_text()
    assert rendered.count('class="model-marker"') == 1
    assert rendered.count('class="projected-marker target-marker') == 1
    assert rendered.count('class="projected-marker llm-target-marker') == 1
    assert 'style="left:60.0%;top:50.0%"' in rendered
    assert 'style="left:62.0%;top:48.0%"' in rendered
    assert "LLM/code landing offset" in rendered
    assert "The purple hollow square is the position the LLM itself reports" in rendered
    assert "this image lacks same-round base_link camera extrinsics" in rendered
    assert "so projections are for troubleshooting only" in rendered


def test_llm_waypoint_marker_requires_matching_current_image_calibration_and_waypoint():
    spec = importlib.util.spec_from_file_location(
        "render_agent_report", Path(__file__).resolve().parents[2] / "scripts/render_agent_report.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    image = {"id": "input:5", "label": "current camera=hand_right", "media_type": "image/jpeg"}
    manifest = [
        image,
        {
            "id": "input:6",
            "camera": "hand_right",
            "image_role": "current",
            "calibration": {"available": True},
        },
    ]
    decision = {"motion": {"frame": "base_link", "waypoints": [{"right_pose": [0, 0, 1, 0, 0, 0, 1]}]}}
    claim = {
        "camera": "hand_right",
        "arm": "right",
        "u_norm": 620,
        "v_norm": 480,
        "image_source": "input:5",
        "calibration_source": "input:6",
        "method": "calibration_projection",
    }
    assert module.llm_waypoint_marker(claim, manifest, image, decision) == [620, 480]
    assert module.llm_waypoint_marker({**claim, "image_source": "input:4"}, manifest, image, decision) is None
    assert (
        module.llm_waypoint_marker({**claim, "calibration_source": "input:7"}, manifest, image, decision)
        is None
    )
    assert module.llm_waypoint_marker({**claim, "arm": "left"}, manifest, image, decision) is None
    assert (
        module.llm_waypoint_marker(claim, manifest, {**image, "label": "history camera=hand_right"}, decision)
        is None
    )

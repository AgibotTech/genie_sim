# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import json
from copy import deepcopy

from agent2act.adapters.planners.review_context import compact_review_input


def test_compact_history_preserves_authority_evidence_and_current_precision():
    assessment = {
        "execution_status": "failed",
        "execution_evidence": "Object remained on support",
        "intent_status": "misaligned",
        "intent_evidence": "Candidate would undo correction",
        "remaining": ["future work"],
    }
    old = {
        "sequence": 2,
        "status": "issued",
        "mode": "motion",
        "instruction": "test",
        "reason": "evidence",
        "success_condition": "object follows",
        "issued_steps": 7,
        "discarded_steps": 8,
        "frame": "base_link",
        "kind": "JOINT_ABS",
        "issued_endpoint": {"grippers": [0.123456789012345, 0], "left_arm": [0.9]},
        "assessment": assessment,
        "observed_effect": {"measured_eef_motion": {"left": {"translation_delta_m": [0, -0.04, 0]}}},
    }
    latest = {**old, "sequence": 3}
    rejection = {"status": "rejected", "error": "fixed mode forbids motion", "sequence": 3}
    state = {
        "instruction": "original task",
        "states": {"joints": [0.123456789012345]},
        "context": {
            "proposal": {"proposal_id": "exact", "trajectory": [[0.123456789012345]]},
            "execution": {"mode": "policy"},
            "previous_assessment": assessment,
            "recent_events": [old, latest, rejection],
            "tracking_feedback": {"joint_error_rad": [0.0000001234567]},
        },
    }
    parts = [{"type": "input_text", "text": json.dumps(state)}, {"type": "input_image", "image_url": "rgb"}]
    before = deepcopy(parts)
    result = compact_review_input(parts)
    assert compact_review_input(result) == result
    new = json.loads(result[0]["text"])
    assert parts == before
    assert result[1:] == parts[1:]
    assert new["states"] == state["states"]
    for key in ("proposal", "execution", "previous_assessment", "tracking_feedback"):
        assert new["context"][key] == state["context"][key]
    older, current, rejected = new["context"]["recent_events"]
    assert rejected == rejection
    assert current["issued_endpoint"] == latest["issued_endpoint"]
    assert current["assessment_reference"] == "context.previous_assessment"
    assert older["prior_action_assessment"]["execution_status"] == "failed"
    assert older["observed_effect"] == old["observed_effect"]
    assert older["issued_gripper_commands"] == old["issued_endpoint"]["grippers"]
    assert older["discarded_steps"] == 8
    assert older["sequence"] == 2


def test_non_review_context_is_unchanged():
    parts = [{"type": "input_text", "text": json.dumps({"context": {"recent_events": []}})}]
    assert compact_review_input(parts) is parts


def test_compact_does_not_invent_reference_for_missing_assessment():
    state = {"context": {"proposal": {}, "recent_events": [{"status": "issued", "sequence": 1}]}}
    parts = [{"type": "input_text", "text": json.dumps(state)}]
    result = compact_review_input(parts)
    event = json.loads(result[0]["text"])["context"]["recent_events"][0]
    assert "assessment_reference" not in event
    assert compact_review_input(result) == result


def test_long_exact_historical_instruction_is_referenced_without_changing_current_data():
    instruction = "Original instruction with exact punctuation and spacing. " * 8
    old = {"status": "issued", "sequence": 1, "instruction": instruction, "reason": "evidence"}
    local = {**old, "sequence": 2, "instruction": instruction + " "}
    latest = {**old, "sequence": 3}
    state = {
        "instruction": instruction,
        "context": {
            "proposal": {"instruction": instruction, "proposal_id": "exact"},
            "recent_events": [old, local, latest],
        },
    }
    content = [{"type": "input_text", "text": json.dumps(state)}]
    before = deepcopy(content)
    result = compact_review_input(content)
    decoded = json.loads(result[0]["text"])
    events = decoded["context"]["recent_events"]
    assert content == before
    assert events[0] == {
        "status": "issued",
        "sequence": 1,
        "reason": "evidence",
        "instruction_reference": "instruction",
    }
    assert decoded[events[0]["instruction_reference"]] == old["instruction"]
    assert events[1]["instruction"] == local["instruction"]
    assert events[2] == latest
    assert decoded["context"]["proposal"] == state["context"]["proposal"]
    assert compact_review_input(result) == result
    assert len(result[0]["text"]) < len(json.dumps(state, separators=(",", ":")))


def test_short_duplicate_instruction_does_not_add_reference_overhead():
    state = {
        "instruction": "test",
        "context": {
            "proposal": {},
            "recent_events": [
                {"status": "issued", "sequence": 1, "instruction": "test"},
                {"status": "issued", "sequence": 2, "instruction": "test"},
            ],
        },
    }
    result = compact_review_input([{"type": "input_text", "text": json.dumps(state)}])
    decoded = json.loads(result[0]["text"])
    assert decoded["context"]["recent_events"][0]["instruction"] == "test"
    assert "instruction_reference" not in result[0]["text"]

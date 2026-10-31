# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Bounded review presentation. Never changes observations, decisions, or action validation."""

import json
from copy import deepcopy
from typing import Any

COMPACT_REVIEW_INSTRUCTIONS = """
REVIEW ECONOMY AND ACTION EFFECTS
Give one short evidence sentence per field; do not repeat the task or the same evidence across fields.
Historical issued events are summaries: command endpoints and old candidate speculation may be
omitted. The latest command, current proposal and current assessment remain available in full.
An assessment stored at sequence N evaluates the action BEFORE N, not the action issued at N.
observed_effect is measured robot motion after the associated issued action, never object success.
Compare signed measured changes across corrections AND intervening policy actions. A subsequent
candidate may undo an alignment correction; evaluate the path through closure/contact, not just
the starting pose. After repeated failed engagement, a changed wrist pose or projected overlap
alone is not improved evidence. Revise the failed geometric/contact hypothesis using independent
views or a bounded informative change; do not recycle the same correction/closure loop.
Choose the next useful observation boundary. Combine already-supported free-space corrections
when intermediate feedback is unnecessary. If consecutive policy actions are supported without
an intervening contact/retention decision, consider a bounded policy_chunks grant. Keep one chunk
for unresolved contact or commitment; never grant extra chunks merely to save inference cost.
Shorter command duration alone is not lower wall time: include another planner wait in the cost
of splitting an operation. Preserve all existing routing, evidence and action constraints.
"""


def compact_review_input(content: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize older issued events only; keep latest targets, failures and sensors exact.

    Rejections and unknown events pass through unchanged so constraints cannot disappear."""
    if not content or content[0].get("type") != "input_text":
        return content
    state = json.loads(content[0]["text"])
    context = state.get("context", {})
    events = context.get("recent_events")
    if not isinstance(events, list) or not isinstance(context.get("proposal"), dict):
        return content
    issued_indices = [i for i, event in enumerate(events) if event.get("status") == "issued"]
    latest = issued_indices[-1] if issued_indices else None
    keep = {
        "sequence",
        "status",
        "mode",
        "backend",
        "instruction",
        "instruction_reference",
        "reason",
        "success_condition",
        "planner_status",
        "frame",
        "kind",
        "issued_steps",
        "discarded_steps",
        "issued_duration_s",
        "attempt",
        "observed_effect",
        "prior_action_assessment",
        "issued_gripper_commands",
    }
    instruction = state.get("instruction")
    reference_note = " instruction_reference=instruction denotes the exact top-level instruction."
    # Reference only sufficiently long, exact duplicates. The saved characters must
    # pay for the explanatory note even when there is only one matching old event.
    reference_profitable = isinstance(instruction, str) and (
        len(json.dumps({"instruction": instruction}, separators=(",", ":")))
        - len(json.dumps({"instruction_reference": "instruction"}, separators=(",", ":")))
        > len(reference_note)
    )
    for i in issued_indices:
        event = events[i]
        if i != latest:
            summary = {key: value for key, value in event.items() if key in keep}
            endpoint = event.get("issued_endpoint")
            if isinstance(endpoint, dict):
                summary["issued_gripper_commands"] = endpoint.get("grippers")
            assessment = event.get("assessment")
            if isinstance(assessment, dict):
                # Keep the retrospective evidence. Old predictions are not observations.
                summary["prior_action_assessment"] = {
                    key: assessment[key]
                    for key in ("execution_status", "execution_evidence")
                    if key in assessment
                }
            if reference_profitable and summary.get("instruction") == instruction:
                summary.pop("instruction")
                summary["instruction_reference"] = "instruction"
            events[i] = summary
        elif isinstance(event.get("assessment"), dict) and event["assessment"] == context.get(
            "previous_assessment"
        ):
            event.pop("assessment", None)
            event["assessment_reference"] = "context.previous_assessment"
    context["history_presentation"] = (
        "Older issued events retain action rationale, expected result and retrospective evidence; "
        "old command arrays and prospective assessments omitted. Full events remain in trace. "
        "Assessment at sequence N concerns the preceding action; observed_effect concerns action N."
    )
    if any(events[i].get("instruction_reference") == "instruction" for i in issued_indices):
        context["history_presentation"] += reference_note
    result = [deepcopy(content[0]), *content[1:]]
    result[0]["text"] = json.dumps(state, allow_nan=False, separators=(",", ":"))
    return result

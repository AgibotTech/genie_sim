# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Public evidence reports and input provenance; never reconstruct hidden reasoning."""

import base64
import hashlib
import json
import math
import re
from copy import deepcopy
from typing import Any


def _object(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


TEXT = {"type": "string"}
DIAGNOSTIC_SCHEMA = _object(
    {
        "rgb_use": {"type": "string", "enum": ["used", "not_used", "unavailable", "uncertain"]},
        "depth_use": {"type": "string", "enum": ["used", "not_used", "unavailable", "uncertain"]},
        "evidence": {
            "type": "array",
            "maxItems": 8,
            "items": _object(
                {
                    "modality": {
                        "type": "string",
                        "enum": ["rgb", "depth", "robot_state", "tool"],
                    },
                    "source": TEXT,
                    "finding": TEXT,
                }
            ),
        },
        "estimates": {
            "type": "array",
            "maxItems": 8,
            "items": _object(
                {
                    "quantity": TEXT,
                    "value": {"type": ["number", "null"]},
                    "units": TEXT,
                    "frame": {"type": ["string", "null"]},
                    "source": TEXT,
                    "basis": {
                        "type": "string",
                        "enum": ["measured", "visual_estimate", "assumed", "unknown"],
                    },
                }
            ),
        },
        "object_location": _object(
            {
                "label": TEXT,
                "frame": {"type": ["string", "null"]},
                "x_m": {"type": ["number", "null"]},
                "y_m": {"type": ["number", "null"]},
                "z_m": {"type": ["number", "null"]},
                "camera": {"type": ["string", "null"]},
                "u_norm": {"type": ["number", "null"]},
                "v_norm": {"type": ["number", "null"]},
                "source": TEXT,
                "basis": {"type": "string", "enum": ["measured", "visual_estimate", "assumed", "unknown"]},
            }
        ),
        "camera_projections": {
            "type": "array",
            "maxItems": 6,
            "items": _object(
                {
                    "camera": {"type": "string", "enum": ["head", "hand_left", "hand_right"]},
                    "arm": {"type": "string", "enum": ["left", "right"]},
                    "u_norm": {"type": ["number", "null"]},
                    "v_norm": {"type": ["number", "null"]},
                    "image_source": TEXT,
                    "calibration_source": TEXT,
                    "method": {
                        "type": "string",
                        "enum": ["calibration_projection", "visual_estimate", "unavailable"],
                    },
                    "note": TEXT,
                }
            ),
        },
        "uncertainties": {"type": "array", "maxItems": 8, "items": TEXT},
        "expected_effect": TEXT,
        "verification": TEXT,
    }
)

DIAGNOSTIC_INSTRUCTIONS = """
DIAGNOSTIC OUTPUT EXTENSION
Add one top-level diagnostics object alongside the normal decision fields. Keep the normal
reason concise; diagnostics is a short, checkable evidence report, not private chain of thought.
Report rgb_use and depth_use as used/not_used/unavailable/uncertain. List evidence entries with
modality (rgb/depth/robot_state/tool), source, and finding. Cite the input_manifest
entry ID plus camera/region or state field; cite actual tool call IDs for tool evidence. Image
locations use normalized 0..1000 pixel coordinates. A source citation is a claim, not validation.
Each estimates entry has exactly quantity, value, units, frame, source, basis. The key is
value, never numeric_value; value must be a single number or null, never an array. Split XYZ
vectors into separate entries. frame is a string or null. basis is measured/visual_estimate/
assumed/unknown. Include only numbers pertinent to
this action. Never invent readings, calibration, tool calls or certainty. A depth tile median
is not a measured target-object coordinate; grayscale intensity is not metric depth. Distinguish
camera depth from robot-frame position. Explain non-use of available depth in uncertainties.
In object_location, identify the object this action targets. Give its image location as camera
and normalized u_norm/v_norm (0..1000) when visible, and its XYZ in metres only when you have
a defensible estimate in the stated frame. Use null for unknown coordinates or frame; RGB alone
does not establish a measured 3D location. source must cite the observation or tool result.
If no object is targeted, use label "none", null coordinates, and basis "unknown".
camera_projections reports YOUR OWN expected image location of each moving arm's FIRST
base_link waypoint, for each current RGB camera with usable calibration. Use the
waypoint XYZ, the exact image's optical_to_base_link transform and source_intrinsics;
invert the transform to get optical XYZ, apply lens distortion, then convert source
pixels to normalized 0..1000 image coordinates. Report camera, arm, u_norm, v_norm,
image_source and calibration_source (their exact input_manifest IDs), method and a
short note. Use method calibration_projection only if you actually applied those
parameters; use visual_estimate for an image-based guess. For missing calibration,
no arm waypoint, an off-image/behind-camera result, or uncertainty, use null pixels
and method unavailable with a short reason. Never copy object_location pixels into
camera_projections as if they were the waypoint. No precomputed waypoint pixels are
given to you. An empty array is valid when there is no calibrated current RGB view.
Give uncertainties as an array of strings. expected_effect and verification must each be
a single string, not a list; verification describes observable next-step checks. At most
8 evidence/estimate/uncertainty entries and 6 camera projections; use short factual findings.
No exhaustive reasoning narrative. Empty lists and
unknown values are valid. Do not claim internal attention or prove causality from this self-report.
"""


def diagnostic_schema(schema: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(schema)
    result["properties"]["diagnostics"] = deepcopy(DIAGNOSTIC_SCHEMA)
    result["required"].append("diagnostics")
    return result


def input_manifest(content: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Hash exact attached image bytes, retaining camera/history labels and depth metadata."""
    result = []
    label = ""
    for index, part in enumerate(content):
        entry: dict[str, Any] = {"id": f"input:{index}", "type": part["type"]}
        if part["type"] == "input_text":
            text = part["text"]
            try:
                value = json.loads(text)
            except ValueError:
                value = None
            if isinstance(value, dict):
                entry["fields"] = list(value)
                if "calibration" in value:
                    entry.update(
                        {key: value[key] for key in ("camera", "image_role", "calibration") if key in value}
                    )
                if "depth" in value:
                    entry.update(
                        {key: value[key] for key in ("camera", "image_role", "depth") if key in value}
                    )
                    label = f"{value.get('image_role')} camera={value.get('camera')} depth"
            else:
                entry["label"] = text
                label = text
        elif part["type"] == "input_image":
            url = part["image_url"]
            if url.startswith("data:") and ";base64," in url:
                header, encoded = url.split(",", 1)
                data = base64.b64decode(encoded, validate=True)
                entry.update(
                    label=label,
                    media_type=header[5:].split(";")[0],
                    sha256=hashlib.sha256(data).hexdigest(),
                    encoded_bytes=len(data),
                )
        result.append(entry)
    return result


def model_input_manifest(manifest: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Index existing content for citations; full metadata belongs in the trace."""
    keys = ("id", "type", "fields", "label", "camera", "image_role")
    return [{key: entry[key] for key in keys if key in entry} for entry in manifest]


def _valid(value: Any, schema: dict[str, Any]) -> bool:
    kind = schema["type"]
    if isinstance(kind, list):
        return any(_valid(value, {**schema, "type": option}) for option in kind)
    if kind == "null":
        return value is None
    if kind == "string":
        return (
            isinstance(value, str)
            and len(value) <= 2000
            and ("enum" not in schema or value in schema["enum"])
        )
    if kind == "number":
        try:
            return type(value) in (int, float) and math.isfinite(value)
        except OverflowError:
            return False
    if kind == "array":
        return (
            isinstance(value, list)
            and len(value) <= schema["maxItems"]
            and all(_valid(item, schema["items"]) for item in value)
        )
    return (
        isinstance(value, dict)
        and value.keys() == schema["properties"].keys()
        and all(_valid(value[key], definition) for key, definition in schema["properties"].items())
    )


def extract_diagnostics(payload: Any) -> tuple[Any, dict[str, Any]]:
    """Bad/missing diagnostic prose never causes action retries or enters core Decision."""
    if not isinstance(payload, dict):
        return payload, {"status": "missing", "report": None}
    decision = dict(payload)
    if "diagnostics" not in decision:
        return decision, {"status": "missing", "report": None}
    report = decision.pop("diagnostics")
    # Preserve older diagnostic reports while the new prompt requests projections.
    candidate = (
        {**report, "camera_projections": []}
        if isinstance(report, dict) and "camera_projections" not in report
        else report
    )
    valid = _valid(candidate, DIAGNOSTIC_SCHEMA)
    return decision, {
        "status": "available" if valid else "invalid",
        "report": report if valid else None,
        "meaning": "Model-reported evidence; source citations and modality-use claims are unverified.",
    }


def evidence_warnings(report: dict[str, Any], manifest: list[dict[str, Any]]) -> list[str]:
    """Check only input presence/references; never certify perception or claimed modality use."""
    warnings = []
    available = {
        "rgb": any(item.get("media_type") == "image/jpeg" for item in manifest),
        "depth": any(item.get("depth", {}).get("available") is True for item in manifest),
    }
    for modality, present in available.items():
        if report[f"{modality}_use"] == "used" and not present:
            warnings.append(
                f"Model claims {modality} use but no such input is present in the initial manifest."
            )
    ids = {item["id"] for item in manifest}
    claims: list[tuple[str, ...]] = [
        (claim["source"],) for claim in report["evidence"] + report["estimates"] + [report["object_location"]]
    ]
    claims += [
        (claim["image_source"], claim["calibration_source"]) for claim in report.get("camera_projections", [])
    ]
    for sources in claims:
        for reference in re.findall(r"input:\d+", " ".join(sources)):
            if reference not in ids:
                warnings.append(f"Unknown initial-input reference: {reference}")
    return list(dict.fromkeys(warnings))


def response_summary(response: dict[str, Any]) -> dict[str, Any]:
    output = response.get("output")
    texts = [
        part["text"]
        for item in (output if isinstance(output, list) else [])
        if isinstance(item, dict) and item.get("type") == "reasoning"
        for part in (item["summary"] if isinstance(item.get("summary"), list) else [])
        if isinstance(part, dict)
        and part.get("type") == "summary_text"
        and isinstance(part.get("text"), str)
        and part["text"]
    ]
    return {
        "status": "available" if texts else "not_provided",
        "summary": texts,
        "response_status": response.get("status", "not_reported"),
        "meaning": "Public provider summary, not a complete internal reasoning trace.",
    }

# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Read exact model-selected sensor pixels without projection or spatial filtering."""

import struct
from copy import deepcopy
from typing import Any

from agent2act.domain.models import ContractError, Observation

RAW_DEPTH_TOOL = {
    "type": "function",
    "name": "read_depth_pixels",
    "description": (
        "Read exact raw millimetre depth at pixels YOU select in the CURRENT observation. "
        "u,v are zero-based integer SOURCE depth pixels, not normalized 0..1000 or resized RGB "
        "coordinates. Use source_size and encoded_to_source_pixel from the image metadata. "
        "Returns original uint16 samples, validity, snapshot ID and camera calibration. "
        "No averaging, fallback pixels, projection, object recognition or robot motion."
    ),
    "strict": True,
    "parameters": {
        "type": "object",
        "properties": {
            "camera": {"type": "string", "enum": ["head", "hand_left", "hand_right"]},
            "pixels": {
                "type": "array",
                "minItems": 1,
                "maxItems": 8,
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string"},
                        "u": {"type": "integer", "minimum": 0},
                        "v": {"type": "integer", "minimum": 0},
                    },
                    "required": ["label", "u", "v"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["camera", "pixels"],
        "additionalProperties": False,
    },
}

RAW_DEPTH_INSTRUCTIONS = (
    "\nFor metric depth use read_depth_pixels: select relevant pixels yourself from CURRENT RGB "
    "and read their exact sensor values as text, rather than inferring numbers from a depth "
    "image display. u,v are integer SOURCE depth pixels (zero-based), NOT normalized coordinates. "
    "Use source_size and the RGB encoded_to_source_pixel mapping before selecting a pixel. "
    "You may query up to 8 pixels per call, at most 2 tool rounds per decision. Queries use the "
    "same current observation, do not move the robot and do not acquire new frames. Returned "
    "valid depth_mm values are measured optical depths when calibration declares optical z. "
    "You interpret the readings and supplied intrinsics, distortion and optical_to_frame "
    "extrinsics; the tool does not compute 3D positions or choose actions. Use only available, "
    "aligned calibration for RGB correspondence and robot-frame calculations. An unavailable "
    "calibration does not prevent reading raw depth, but does prevent a reliable robot-frame "
    "projection. 0 and 65535 are invalid; do not replace them with guessed values. Labels are "
    "your hypotheses, not verified object identities. Cite the tool call and actual pixel/value "
    "when using depth evidence. A sensor sample is not an object center or proof of contact."
)


def read_depth_pixels(observation: Observation, arguments: Any) -> dict[str, Any]:
    if not isinstance(arguments, dict) or set(arguments) != {"camera", "pixels"}:
        raise ContractError("expected camera and pixels")
    camera = arguments["camera"]
    if camera not in ("head", "hand_left", "hand_right"):
        raise ContractError("unknown depth camera")
    pixels = arguments["pixels"]
    if not isinstance(pixels, list) or not 1 <= len(pixels) <= 8:
        raise ContractError("provide 1 to 8 source pixels")
    depth = observation.images.get(camera + "_depth")
    if not isinstance(depth, dict):
        raise ContractError("raw depth not supplied")
    width, height, raw = depth.get("width"), depth.get("height"), depth.get("image_data")
    if (
        depth.get("encoding") != "RAW_UINT16"
        or type(width) is not int
        or type(height) is not int
        or not 1 <= width <= 8192
        or not 1 <= height <= 8192
        or width * height > 16_000_000
        or not isinstance(raw, bytes)
        or len(raw) != width * height * 2
    ):
        raise ContractError("invalid RAW_UINT16 dimensions or data")
    image = observation.images.get(camera, {})
    if not isinstance(image, dict):
        image = {}
    snapshot = depth.get("snapshot_id")
    if image.get("snapshot_id") is not None and snapshot != image["snapshot_id"]:
        raise ContractError("RGB/depth snapshot mismatch")
    calibration = deepcopy(image.get("calibration"))
    if not isinstance(calibration, dict):
        calibration = {"available": False, "reason": "camera calibration not supplied"}
    if calibration.get("available") is True and (
        not snapshot
        or calibration.get("snapshot_id") != snapshot
        or image.get("snapshot_id") != snapshot
        or [calibration.get("width"), calibration.get("height")] != [width, height]
        or calibration.get("rgb_depth_aligned") is not True
    ):
        calibration.update(available=False, reason="unaligned or mismatched calibration snapshot/dimensions")
    results = []
    for pixel in pixels:
        if not isinstance(pixel, dict) or set(pixel) != {"label", "u", "v"}:
            raise ContractError("each pixel requires label, u and v")
        u, v = pixel["u"], pixel["v"]
        if not isinstance(pixel["label"], str) or len(pixel["label"]) > 200:
            raise ContractError("pixel label must be a string of at most 200 characters")
        if type(u) is not int or type(v) is not int or not 0 <= u < width or not 0 <= v < height:
            raise ContractError("pixel outside source dimensions or not an integer")
        value = struct.unpack_from("<H", raw, 2 * (v * width + u))[0]
        valid = value not in (0, 65535)
        results.append({**pixel, "raw_uint16": value, "valid": valid, "depth_mm": value if valid else None})
    return {
        "camera": camera,
        "snapshot_id": snapshot,
        "source_size": [width, height],
        "coordinate_system": "zero-based source depth pixels; u right, v down",
        "units": "mm",
        "pixels": results,
        "calibration": calibration,
        "motion_executed": False,
        "meaning": "Exact little-endian sensor samples, no filtering or projection. Labels are model-selected; "
        "object identity, contact, clearance and task success are not verified.",
    }

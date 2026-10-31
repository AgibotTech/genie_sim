# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Read-only crops of current RGB bytes; never synthesized visual evidence."""

import base64
import math
from io import BytesIO
from typing import Any

from PIL import Image, UnidentifiedImageError

from agent2act.domain.models import ContractError, Observation, number

RGB_INSPECTION_TOOL = {
    "type": "function",
    "name": "inspect_rgb_region",
    "description": "Inspect a region of a CURRENT RGB image. Select camera and left,top,right,bottom bounds in 0..1000 whole-image coordinates. Returns a labeled RGB crop, not depth, segmentation, new detail, contact or coordinates in metres. Use for ambiguous visible detail before deciding; cannot reveal occluded surfaces. No action is executed.",
    "parameters": {
        "type": "object",
        "properties": {
            "camera": {"type": "string", "enum": ["head", "hand_left", "hand_right"]},
            **{
                key: {"type": "number", "minimum": 0, "maximum": 1000}
                for key in ("left", "top", "right", "bottom")
            },
        },
        "required": ["camera", "left", "top", "right", "bottom"],
        "additionalProperties": False,
    },
}


def inspect_rgb_region(observation: Observation, arguments: Any) -> dict[str, Any]:
    if not isinstance(arguments, dict) or set(arguments) != {"camera", "left", "top", "right", "bottom"}:
        raise ContractError("RGB inspection requires camera and four bounds")
    camera = arguments["camera"]
    if camera not in ("head", "hand_left", "hand_right"):
        raise ContractError("RGB inspection requires an allowed current RGB camera")
    left, top, right, bottom = [number(arguments[k], k) for k in ("left", "top", "right", "bottom")]
    if not (0 <= left < right <= 1000 and 0 <= top < bottom <= 1000):
        raise ContractError("RGB inspection bounds must be ordered within 0..1000")
    entry = observation.images.get(camera, {})
    raw = entry.get("image_data")
    if entry.get("encoding") != "JPEG" or not isinstance(raw, bytes):
        raise ContractError("Current RGB JPEG unavailable")
    try:
        with Image.open(BytesIO(raw)) as source:
            if source.format != "JPEG" or source.width * source.height > 16_000_000:
                raise ContractError("RGB inspection requires a bounded JPEG")
            width, height = source.size
            box = (
                math.floor(left * width / 1000),
                math.floor(top * height / 1000),
                math.ceil(right * width / 1000),
                math.ceil(bottom * height / 1000),
            )
            if min(box[2] - box[0], box[3] - box[1]) < 8:
                raise ContractError("RGB inspection crop must span at least 8 source pixels per axis")
            crop = source.convert("RGB").crop(box)
            crop.thumbnail((768, 768), Image.Resampling.LANCZOS)
            output = BytesIO()
            crop.save(output, format="PNG")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ContractError("Current RGB JPEG cannot be decoded") from exc
    return {
        "camera": camera,
        "source_size": [width, height],
        "source_box_pixels": list(box),
        "crop_size": list(crop.size),
        "motion_executed": False,
        "meaning": "Current RGB crop; source box uses half-open pixel edges. No upscaling, synthesized detail, depth or contact verification. Coordinates for other tools remain relative to the original whole image.",
        "image_url": "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii"),
    }

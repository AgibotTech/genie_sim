# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Bounded RGB encoding and measured camera-depth summaries for planning."""

import base64
import io
import json
import logging
import struct
import sys
from array import array
from collections import Counter
from statistics import median
from typing import Any

from PIL import Image

from agent2act.adapters.planners.camera_calibration import camera_metadata
from agent2act.adapters.planners.static_rgb_calibration import StaticRgbCalibration
from agent2act.domain.errors import ContractError

logger = logging.getLogger(__name__)


def depth_summary(item: Any, grid: int = 8) -> dict[str, Any]:
    """Summarize sampled valid uint16 millimetres; never infer object or base poses."""
    if not isinstance(item, dict):
        return {"available": False, "reason": "not supplied"}
    width, height, data = item.get("width"), item.get("height"), item.get("image_data")
    if (
        item.get("encoding") != "RAW_UINT16"
        or type(width) is not int
        or type(height) is not int
        or not 1 <= width <= 8192
        or not 1 <= height <= 8192
        or width * height > 16_000_000
        or not isinstance(data, bytes)
        or len(data) != width * height * 2
    ):
        return {"available": False, "reason": "invalid RAW_UINT16 dimensions or data"}
    # CoRobot's x86 simulator emits little-endian uint16 millimetres.
    rows = []
    for y in range(grid):
        row = []
        for x in range(grid):
            x0, x1 = x * width // grid, (x + 1) * width // grid
            y0, y1 = y * height // grid, (y + 1) * height // grid
            values = []
            for py in range(y0, y1, max(1, (y1 - y0) // 8)):
                for px in range(x0, x1, max(1, (x1 - x0) // 8)):
                    v = struct.unpack_from("<H", data, 2 * (py * width + px))[0]
                    if 0 < v < 65535:
                        values.append(v)
            row.append(int(median(values)) if values else None)
        rows.append(row)
    if not any(value is not None for row in rows for value in row):
        return {"available": False, "reason": "no valid sampled depth pixels"}
    return {
        "available": True,
        "units": "mm",
        "source_size": [width, height],
        "grid": rows,
        "coordinates": "equal image tiles, rows top-to-bottom, columns left-to-right",
        "meaning": "sampled median surface depth; mixed surfaces, NOT object coordinates or grasp clearance",
        "camera_to_robot_calibration": "not determined by this depth summary; consult camera metadata or registered geometry tools",
    }


def encode_raw_depth(item: Any, include_png: bool = True) -> tuple[dict[str, Any], bytes | None]:
    """Describe valid source depth, optionally attaching a lossless full-resolution PNG."""
    if not isinstance(item, dict):
        return {"available": False, "reason": "not supplied"}, None
    width, height, data = item.get("width"), item.get("height"), item.get("image_data")
    if (
        item.get("encoding") != "RAW_UINT16"
        or type(width) is not int
        or type(height) is not int
        or not 1 <= width <= 8192
        or not 1 <= height <= 8192
        or width * height > 16_000_000
        or not isinstance(data, bytes)
        or len(data) != width * height * 2
    ):
        return {"available": False, "reason": "invalid RAW_UINT16 dimensions or data"}, None
    metadata = {
        "available": True,
        "units": "mm",
        "source_size": [width, height],
        "encoding": "PNG_UINT16" if include_png else "RAW_UINT16_LOCAL",
        "invalid_values": [0, 65535],
        "snapshot_id": item.get("snapshot_id"),
        "meaning": (
            "The preceding depth PNG preserves every original uint16 millimetre sample; "
            "provider image preprocessing may not preserve numeric precision. "
            if include_png
            else "Raw uint16 millimetre samples are retained locally for read_depth_pixels; "
            "no depth PNG is attached. "
        )
        + "No resizing, normalization, clipping or tile aggregation is applied. "
        "Pixel axes are u right, v down. Use associated camera calibration for depth semantics and transforms.",
    }
    if not include_png:
        return metadata, None
    image = Image.frombytes("I;16", (width, height), data, "raw", "I;16")
    output = io.BytesIO()
    image.save(output, "PNG")
    return metadata, output.getvalue()


def depth_render(item: Any, max_side: int | None) -> tuple[dict[str, Any], bytes | None]:
    """Render measured depth as a bounded lossless PNG preserving local spatial structure.

    The percentile scale is per frame and included in metadata. Not a calibrated robot-coordinate image."""
    summary = depth_summary(item)
    if not summary["available"]:
        return summary, None
    values = array("H", item["image_data"])
    if sys.byteorder != "little":
        values.byteswap()
    counts = Counter(values)
    counts.pop(0, None)
    counts.pop(65535, None)
    total = sum(counts.values())
    if not total:
        return {**summary, "available": False, "reason": "no valid depth pixels"}, None
    ordered = sorted(counts)
    low, high = ordered[0], ordered[-1]
    cumulative = 0
    low_found = False
    for value in ordered:
        cumulative += counts[value]
        if not low_found and cumulative >= max(1, total * 0.02):
            low, low_found = value, True
        if cumulative >= total * 0.98:
            high = value
            break
    high = max(high, low + 1)
    palette = bytes(
        0 if value in (0, 65535) else max(64, min(255, round(255 - 191 * (value - low) / (high - low))))
        for value in range(65536)
    )
    pixels = bytes(palette[value] for value in values)
    image = Image.frombytes("L", (item["width"], item["height"]), pixels)
    # Nearest-neighbor avoids blending invalid pixels or inventing surface depths.
    bound = min(max_side or 1024, 1024)
    image.thumbnail((bound, bound), Image.Resampling.NEAREST)
    output = io.BytesIO()
    image.save(output, "PNG")
    center = values[(item["height"] // 2) * item["width"] + item["width"] // 2]
    summary["render"] = {
        "format": "PNG",
        "encoded_size": list(image.size),
        "near_mm": low,
        "far_mm": high,
        "valid_fraction": total / len(values),
        "center_mm": center if 0 < center < 65535 else None,
        "scale": "per-frame 2nd-98th percentiles; clipped outside range; near=255, far=64, invalid=0",
        "interpretation": "same source pixel axes as depth sensor; RGB alignment is not verified; compare metric labels across frames, not brightness alone",
    }
    return summary, output.getvalue()


def append_camera_inputs(
    content: list[dict[str, Any]],
    images: dict[str, Any],
    label: str,
    max_side: int | None,
    quality: int,
    use_depth: bool,
    static_rgb_calibration: dict[str, dict[str, Any]] | None = None,
    send_raw_depth_image: bool = True,
) -> None:
    for camera in ("head", "hand_left", "hand_right"):
        item = images.get(camera)
        if not isinstance(item, dict) or item.get("encoding") != "JPEG":
            continue
        data = item.get("image_data")
        if not isinstance(data, bytes):
            raise ContractError("CoRobot JPEG image_data must be bytes")
        original_bytes = len(data)
        try:
            with Image.open(io.BytesIO(data)) as im:
                original_size = list(im.size)
                if im.width * im.height > 16_000_000:
                    raise ContractError("camera image exceeds 16 megapixels")
                if max_side is not None:
                    converted = im.convert("RGB")
                    converted.thumbnail((max_side, max_side))
                    output = io.BytesIO()
                    converted.save(output, "JPEG", quality=quality)
                    data = output.getvalue()
                encoded_size = list(converted.size) if max_side is not None else original_size
        except (OSError, ValueError) as exc:
            raise ContractError("invalid JPEG camera image") from exc
        content.append({"type": "input_text", "text": f"{label} camera={camera}"})
        content.append(
            {
                "type": "input_image",
                "image_url": "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii"),
            }
        )
        # Preserve sensor samples; RGB resizing must never resize metric depth.
        depth, depth_png = (
            encode_raw_depth(images.get(camera + "_depth"), include_png=send_raw_depth_image)
            if use_depth
            else (None, None)
        )
        calibration = camera_metadata(item, images.get(camera + "_depth"), original_size, encoded_size)
        if "calibration" not in item and static_rgb_calibration and camera in static_rgb_calibration:
            calibration = StaticRgbCalibration.image_metadata(
                static_rgb_calibration[camera], original_size, encoded_size
            )
        if (
            use_depth
            or "calibration" in item
            or (static_rgb_calibration and camera in static_rgb_calibration)
        ):
            content.append(
                {
                    "type": "input_text",
                    "text": json.dumps(
                        {
                            "camera": camera,
                            "image_role": label,
                            "calibration": calibration,
                        }
                    ),
                }
            )
        if depth_png is not None:
            content.append({"type": "input_text", "text": f"{label} camera={camera} raw depth (uint16 mm)"})
            content.append(
                {
                    "type": "input_image",
                    "image_url": "data:image/png;base64," + base64.b64encode(depth_png).decode("ascii"),
                }
            )
        if depth is not None:
            content.append(
                {
                    "type": "input_text",
                    "text": json.dumps({"camera": camera, "image_role": label, "depth": depth}),
                }
            )
        logger.info(
            "camera_input",
            extra={
                "event": {
                    "camera": camera,
                    "image_role": label,
                    "source_bytes": original_bytes,
                    "encoded_bytes": len(data),
                    "source_size": original_size,
                    "encoded_size": encoded_size,
                    "depth_available": bool(depth and depth["available"]),
                    "depth_encoded_bytes": len(depth_png) if depth_png else 0,
                }
            },
        )

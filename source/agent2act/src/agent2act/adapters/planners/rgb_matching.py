# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Local RGB correspondence with abstention; no metric geometry or semantic inference."""

from io import BytesIO
from typing import Any

from PIL import Image

from agent2act.domain.errors import ContractError
from agent2act.domain.models import number

RGB_MATCH_TOOL = {
    "type": "function",
    "name": "match_rgb_patch",
    "description": (
        "Match a textured patch YOU select in the previous planner RGB image to the current image "
        "of the SAME camera. x,y are 0..1000 across that previous whole image. Returns a local "
        "image correspondence hypothesis or abstains. Use distinct visible surface texture, not an occlusion "
        "edge. No depth, contact, object identity or world motion is inferred. Camera motion also "
        "moves image patches. No new observation or action is obtained."
    ),
    "strict": True,
    "parameters": {
        "type": "object",
        "properties": {
            "previous_sequence": {"type": "integer"},
            "camera": {"type": "string", "enum": ["head", "hand_left", "hand_right"]},
            "x": {"type": "number", "minimum": 0, "maximum": 1000},
            "y": {"type": "number", "minimum": 0, "maximum": 1000},
        },
        "required": ["previous_sequence", "camera", "x", "y"],
        "additionalProperties": False,
    },
}


def decode_rgb_pair(previous: bytes, current: bytes) -> tuple[Any, Any]:
    """Decode equal-size bounded RGB JPEGs to bounded grayscale analysis arrays."""
    import numpy as np

    arrays = []
    sizes = []
    for encoded in (previous, current):
        with Image.open(BytesIO(encoded)) as image:
            if image.format != "JPEG" or image.width * image.height > 16_000_000:
                raise ContractError("RGB matching requires bounded JPEG images")
            sizes.append(image.size)
            gray = image.convert("L")
            gray.thumbnail((640, 640), Image.Resampling.BILINEAR)
            arrays.append(np.asarray(gray, dtype=np.uint8))
    if sizes[0] != sizes[1]:
        raise ContractError("RGB matching requires unchanged camera image dimensions")
    return arrays[0], arrays[1]


def match_rgb_patch(previous: bytes, current: bytes, x: float, y: float) -> dict[str, Any]:
    """Return a conservative translation match in whole-image normalized coordinates.

    Fixed search bounds keep work independent of resolution; only small translations are supported."""
    import numpy as np
    from numpy.lib.stride_tricks import sliding_window_view

    x, y = number(x, "x"), number(y, "y")
    if not 0 <= x <= 1000 or not 0 <= y <= 1000:
        raise ContractError("RGB coordinates must be in 0..1000")
    before, after = (array.astype(np.float32) for array in decode_rgb_pair(previous, current))
    height, width = before.shape
    px, py = round(x * (width - 1) / 1000), round(y * (height - 1) / 1000)
    half, radius = 7, 32

    def search(source: Any, target: Any, cx: int, cy: int) -> dict[str, Any]:
        if not half <= cx < width - half or not half <= cy < height - half:
            return {"status": "unavailable", "reason": "patch crosses image boundary"}
        patch = source[cy - half : cy + half + 1, cx - half : cx + half + 1]
        if float(patch.std()) < 8:
            return {"status": "unavailable", "reason": "insufficient texture"}
        left, right = max(half, cx - radius), min(width - half - 1, cx + radius)
        top, bottom = max(half, cy - radius), min(height - half - 1, cy + radius)
        window = target[top - half : bottom + half + 1, left - half : right + half + 1]
        candidates = sliding_window_view(window, patch.shape)
        centered = candidates - candidates.mean(axis=(-2, -1), keepdims=True)
        centered_patch = patch - patch.mean()
        denominator = np.sqrt(np.sum(centered**2, axis=(-2, -1)) * np.sum(centered_patch**2))
        scores = np.sum(centered * centered_patch, axis=(-2, -1)) / np.maximum(denominator, 1e-9)
        iy, ix = (int(i) for i in np.unravel_index(int(scores.argmax()), scores.shape))
        best = float(scores[iy, ix])
        alternatives = scores.copy()
        alternatives[max(0, iy - 3) : iy + 4, max(0, ix - 3) : ix + 4] = -1
        margin = best - float(alternatives.max())
        if best < 0.85 or margin < 0.08:
            return {"status": "unavailable", "reason": "weak or ambiguous match", "correlation": best}
        if ix in (0, scores.shape[1] - 1) or iy in (0, scores.shape[0] - 1):
            return {"status": "unavailable", "reason": "best match reaches search boundary"}
        return {"status": "matched", "pixel": [int(left + ix), int(top + iy)], "correlation": best}

    result = search(before, after, px, py)
    if result["status"] == "matched":
        qx, qy = result["pixel"]
        reverse = search(after, before, qx, qy)
        if (
            reverse["status"] != "matched"
            or sum((a - b) ** 2 for a, b in zip(reverse["pixel"], (px, py))) > 4
        ):
            result = {"status": "unavailable", "reason": "forward/backward mismatch"}
        else:
            result.update(
                previous_xy=[1000 * px / (width - 1), 1000 * py / (height - 1)],
                current_xy=[1000 * qx / (width - 1), 1000 * qy / (height - 1)],
                delta_xy=[1000 * (qx - px) / (width - 1), 1000 * (qy - py) / (height - 1)],
            )
            del result["pixel"]
    return {
        **result,
        "analysis_size": [width, height],
        "motion_executed": False,
        "meaning": "Local RGB translation hypothesis only; not semantic identity, 3D motion, contact or retention. Camera motion and mixed-surface patches can invalidate object interpretation.",
    }

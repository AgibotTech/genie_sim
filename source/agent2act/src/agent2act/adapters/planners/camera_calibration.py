# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Model-visible camera calibration and explicit source/encoded pixel mapping."""

from copy import deepcopy
from typing import Any


def camera_metadata(
    item: dict[str, Any], depth: Any, source_size: list[int], encoded_size: list[int]
) -> dict[str, Any]:
    calibration = item.get("calibration")
    if not isinstance(calibration, dict):
        return {"available": False, "reason": "camera calibration not supplied"}
    result = deepcopy(calibration)
    if calibration.get("available") is not True:
        result["available"] = False
        return result
    width, height = source_size
    if [calibration.get("width"), calibration.get("height")] != source_size:
        result.update(available=False, reason="calibration and decoded RGB resolution mismatch")
    elif not isinstance(depth, dict) or [depth.get("width"), depth.get("height")] != source_size:
        result.update(available=False, reason="RGB/depth resolution mismatch")
    elif (
        not calibration.get("snapshot_id")
        or item.get("snapshot_id") != calibration["snapshot_id"]
        or depth.get("snapshot_id") != calibration["snapshot_id"]
    ):
        result.update(available=False, reason="RGB/depth/calibration snapshot mismatch")
    result["rgb_source_size"] = source_size
    result["rgb_encoded_size"] = encoded_size
    result["encoded_to_source_pixel"] = {
        "scale_x": width / encoded_size[0],
        "scale_y": height / encoded_size[1],
        "formula": "source_u=(encoded_u+0.5)*scale_x-0.5; source_v=(encoded_v+0.5)*scale_y-0.5",
    }
    result["usage"] = (
        "fx/fy/cx/cy are SOURCE pixels, not encoded image pixels. "
        "Depth is optical z, not Euclidean range. Undistort rays before backprojection. "
        "Do not substitute depth tile medians for target-object depth. "
        "Extrinsics belong only to this snapshot; never reuse wrist transforms across frames."
    )
    return result

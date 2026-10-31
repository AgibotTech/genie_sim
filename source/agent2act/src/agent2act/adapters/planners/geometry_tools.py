# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Read-only geometry queries on the current sensor observation."""

import math
import struct
from statistics import median
from typing import Any

from agent2act.domain.models import ContractError, Observation, number, pose, vector

SURFACE_TOOL = {
    "type": "function",
    "name": "measure_surface_points",
    "description": "Measure sensor surface points selected by YOU in the CURRENT RGB image. Coordinates x,y are 0..1000 across the entire image (left/top=0,right/bottom=1000). Returns base_link metres from aligned depth and calibration. These are visible surfaces, NOT object centers, grasp poses, collision clearance or semantic labels. No motion is executed.",
    "strict": True,
    "parameters": {
        "type": "object",
        "properties": {
            "camera": {"type": "string", "enum": ["head", "hand_left", "hand_right"]},
            "points": {
                "type": "array",
                "minItems": 1,
                "maxItems": 8,
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string"},
                        "x": {"type": "number"},
                        "y": {"type": "number"},
                    },
                    "required": ["label", "x", "y"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["camera", "points"],
        "additionalProperties": False,
    },
}

GRASP_POSE_TOOL = {
    "type": "function",
    "name": "compose_grasp_pose",
    "strict": True,
    "description": "Convert YOUR desired gripper-center position and orientation in base_link into the motion executor's wrist/EEF pose using calibrated robot geometry. You choose the target and orientation. This performs rigid geometry only, not IK, collision checking, planning, or movement. Local +z points from wrist toward fingers; local x is finger closing direction for this configured gripper.",
    "parameters": {
        "type": "object",
        "properties": {
            "arm": {"type": "string", "enum": ["left", "right"]},
            "center_m": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
            "orientation_xyzw": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
        },
        "required": ["arm", "center_m", "orientation_xyzw"],
        "additionalProperties": False,
    },
}


def rotate_vector(q: list[float], v: list[float]) -> list[float]:
    x, y, z, w = q
    matrix = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    return [sum(a * b for a, b in zip(row, v)) for row in matrix]


def compose_grasp_pose(arguments: Any, offsets: dict[str, list[float]]) -> dict[str, Any]:
    if not isinstance(arguments, dict) or set(arguments) != {"arm", "center_m", "orientation_xyzw"}:
        raise ContractError("expected arm, center_m, orientation_xyzw")
    arm = arguments["arm"]
    if not isinstance(arm, str) or arm not in offsets:
        raise ContractError("arm tool geometry unavailable")
    center = vector(arguments["center_m"], 3, "center_m")
    orientation = vector(arguments["orientation_xyzw"], 4, "orientation")
    pose(center + orientation)
    offset = rotate_vector(orientation, offsets[arm])
    return {
        "frame": "base_link",
        "arm": arm,
        "motion_pose": [round(c - o, 7) for c, o in zip(center, offset)] + orientation,
        "grasp_center_m": center,
        "wrist_to_center_local_m": offsets[arm],
        "meaning": "Your chosen pose with wrist offset removed. Reachability, finger extent, object center and collision clearance NOT verified. No motion executed.",
    }


def measure_surface_points(observation: Observation, arguments: Any) -> dict[str, Any]:
    if not isinstance(arguments, dict) or set(arguments) != {"camera", "points"}:
        raise ContractError("expected camera and points")
    camera, points = arguments["camera"], arguments["points"]
    if camera not in ("head", "hand_left", "hand_right"):
        raise ContractError("unsupported camera")
    if not isinstance(points, list) or not 1 <= len(points) <= 8:
        raise ContractError("query needs 1..8 points")
    image = observation.images.get(camera, {})
    calibration = image.get("calibration", {})
    depth = observation.images.get(camera + "_depth", {})
    if (
        calibration.get("model") not in ("pinhole", "opencv_pinhole")
        or calibration.get("frame") != "base_link"
        or calibration.get("depth_kind") != "distance_to_image_plane"
        or calibration.get("units") != "m"
    ):
        raise ContractError("calibrated pinhole optical depth unavailable")
    if "available" in calibration:
        if (
            calibration["available"] is not True
            or calibration.get("rgb_depth_aligned") is not True
            or calibration.get("pose_checked_against_render") is not True
            or not calibration.get("snapshot_id")
            or image.get("snapshot_id") != calibration["snapshot_id"]
            or depth.get("snapshot_id") != calibration["snapshot_id"]
        ):
            raise ContractError("unavailable, unaligned or mismatched calibration snapshot")
        if [image.get("width"), image.get("height")] != [calibration.get("width"), calibration.get("height")]:
            raise ContractError("RGB calibration resolution mismatch")
    width, height = calibration.get("width"), calibration.get("height")
    if (
        type(width) is not int
        or type(height) is not int
        or min(width, height) < 2
        or width * height > 16_000_000
    ):
        raise ContractError("invalid calibration resolution")
    if (
        depth.get("encoding") != "RAW_UINT16"
        or depth.get("width") != width
        or depth.get("height") != height
        or calibration.get("depth_scale_m") != 0.001
    ):
        raise ContractError("depth format or alignment mismatch")
    raw = depth.get("image_data")
    if not isinstance(raw, bytes) or len(raw) != width * height * 2:
        raise ContractError("invalid depth payload")
    fx, fy, cx, cy = [number(calibration.get(k), k) for k in ("fx", "fy", "cx", "cy")]
    if min(fx, fy) <= 0:
        raise ContractError("invalid focal length")
    matrix = calibration.get("optical_to_frame")
    if (
        not isinstance(matrix, list)
        or len(matrix) != 4
        or any(not isinstance(row, list) or len(row) != 4 for row in matrix)
    ):
        raise ContractError("invalid optical transform")
    matrix = [[number(v, "transform") for v in row] for row in matrix]
    if any(abs(a - b) > 1e-6 for a, b in zip(matrix[3], [0, 0, 0, 1])):
        raise ContractError("non-affine optical transform")
    for i in range(3):
        for j in range(3):
            if abs(sum(matrix[k][i] * matrix[k][j] for k in range(3)) - (i == j)) > 1e-4:
                raise ContractError("non-rigid optical transform")
    determinant = sum(
        matrix[0][i]
        * (matrix[1][(i + 1) % 3] * matrix[2][(i + 2) % 3] - matrix[1][(i + 2) % 3] * matrix[2][(i + 1) % 3])
        for i in range(3)
    )
    if abs(determinant - 1) > 1e-4:
        raise ContractError("reflected optical transform")
    results = []
    for point in points:
        if (
            not isinstance(point, dict)
            or set(point) != {"label", "x", "y"}
            or not isinstance(point["label"], str)
            or len(point["label"]) > 160
        ):
            raise ContractError("point needs label, x and y")
        x, y = number(point["x"], "x"), number(point["y"], "y")
        if not (0 <= x <= 1000 and 0 <= y <= 1000):
            raise ContractError("normalized image coordinates must be within 0..1000")
        u, v = round(x * (width - 1) / 1000), round(y * (height - 1) / 1000)
        samples = [
            struct.unpack_from("<H", raw, 2 * (yy * width + xx))[0]
            for yy in range(max(0, v - 1), min(height, v + 2))
            for xx in range(max(0, u - 1), min(width, u + 2))
        ]
        valid = [d for d in samples if 0 < d < 65535]
        result: dict[str, Any] = {"label": point["label"], "pixel": [u, v], "valid_samples": len(valid)}
        if len(valid) < math.ceil(len(samples) / 2) or max(valid) - min(valid) > 50:
            result["error"] = "insufficient depth or depth edge; select an interior surface pixel"
        else:
            z = median(valid) * 0.001
            nx, ny = (u - cx) / fx, (v - cy) / fy
            if calibration["model"] == "opencv_pinhole":
                nx, ny = undistort_point(nx, ny, calibration.get("distortion"))
            optical = [nx * z, ny * z, z, 1]
            result["surface_point_m"] = [
                round(sum(row[k] * optical[k] for k in range(4)), 6) for row in matrix[:3]
            ]
            result["depth_m"] = z
            result["spread_m"] = (max(valid) - min(valid)) * 0.001
        results.append(result)
    return {
        "camera": camera,
        "frame": "base_link",
        "points": results,
        "meaning": "visible 3x3 median surfaces; labels supplied by planner, not verified; no grasp center or clearance inferred",
    }


def undistort_point(xd: float, yd: float, coefficients: Any) -> tuple[float, float]:
    """Invert OpenCV's rational Brown-Conrady model with bounded Newton steps."""
    if not isinstance(coefficients, list) or len(coefficients) != 8:
        raise ContractError("expected 8 OpenCV distortion coefficients")
    k1, k2, p1, p2, k3, k4, k5, k6 = [number(v, "distortion") for v in coefficients]

    def distort(x: float, y: float) -> tuple[float, float]:
        r2 = x * x + y * y
        denominator = 1 + k4 * r2 + k5 * r2**2 + k6 * r2**3
        if abs(denominator) < 1e-10 or r2 > 100:
            raise ContractError("distortion inversion outside calibrated range")
        radial = (1 + k1 * r2 + k2 * r2**2 + k3 * r2**3) / denominator
        return (
            x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x),
            y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y,
        )

    x, y = xd, yd
    for _ in range(20):
        px, py = distort(x, y)
        ex, ey = px - xd, py - yd
        if max(abs(ex), abs(ey)) < 1e-10:
            return x, y
        step = 1e-6
        ax, ay = distort(x + step, y)
        bx, by = distort(x, y + step)
        a, b, c, d = (ax - px) / step, (bx - px) / step, (ay - py) / step, (by - py) / step
        det = a * d - b * c
        if not math.isfinite(det) or abs(det) < 1e-10:
            raise ContractError("singular distortion inversion")
        x -= (d * ex - b * ey) / det
        y -= (a * ey - c * ex) / det
    raise ContractError("distortion inversion did not converge")

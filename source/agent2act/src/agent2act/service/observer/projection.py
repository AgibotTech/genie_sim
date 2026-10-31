# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Projection of recorded targets; never used to plan or execute actions."""

import math
from typing import Any


def project_base_point(calibration: Any, position: Any) -> list[float] | None:
    """Project a base_link point through the exact current-image calibration."""
    if not isinstance(calibration, dict) or calibration.get("available") is not True:
        return None
    if not isinstance(position, (list, tuple)) or len(position) < 3:
        return None
    transform = calibration.get("optical_to_base_link") or calibration.get("optical_to_frame")
    if calibration.get("optical_to_frame") and calibration.get("frame") != "base_link":
        return None
    intrinsics = calibration.get("source_intrinsics", calibration)
    if not isinstance(intrinsics, dict) or not isinstance(transform, list) or len(transform) != 4:
        return None
    try:
        xyz = [float(v) for v in position[:3]]
        matrix = [[float(v) for v in row] for row in transform]
        if any(len(row) != 4 for row in matrix):
            return None
        fx, fy, cx, cy = [float(intrinsics[key]) for key in ("fx", "fy", "cx", "cy")]
        width, height = [int(intrinsics[key]) for key in ("width", "height")]
        raw_distortion = intrinsics.get("distortion", calibration.get("distortion", [0] * 8))
        if not isinstance(raw_distortion, (list, tuple)):
            return None
        distortion = [float(v) for v in raw_distortion]
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    values = xyz + [v for row in matrix for v in row] + [fx, fy, cx, cy] + distortion
    if (
        not all(math.isfinite(v) for v in values)
        or width < 2
        or height < 2
        or min(fx, fy) <= 0
        or len(distortion) != 8
        or any(abs(a - b) > 1e-6 for a, b in zip(matrix[3], [0, 0, 0, 1]))
    ):
        return None
    delta = [xyz[i] - matrix[i][3] for i in range(3)]
    x, y, z = [sum(matrix[i][j] * delta[i] for i in range(3)) for j in range(3)]
    if z <= 0:
        return None
    x, y = x / z, y / z
    r2 = x * x + y * y
    if not math.isfinite(r2) or r2 > 100:
        return None
    k1, k2, p1, p2, k3, k4, k5, k6 = distortion
    denominator = 1 + k4 * r2 + k5 * r2**2 + k6 * r2**3
    if not math.isfinite(denominator) or abs(denominator) < 1e-9:
        return None
    radial = (1 + k1 * r2 + k2 * r2**2 + k3 * r2**3) / denominator
    u = fx * (x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)) + cx
    v = fy * (y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y) + cy
    if (
        not all(math.isfinite(value) for value in (u, v))
        or not 0 <= u <= width - 1
        or not 0 <= v <= height - 1
    ):
        return None
    return [1000 * u / (width - 1), 1000 * v / (height - 1)]


def image_projection(calibration: Any, position: Any, size: tuple[int, int]) -> dict[str, Any]:
    """Map source pixel centers to the actual encoded RGB, with explicit unavailability."""
    unavailable = {"pixel": None, "reason": "camera calibration unavailable"}
    if not isinstance(calibration, dict):
        return unavailable
    if calibration.get("available") is not True:
        return {**unavailable, "reason": calibration.get("reason", unavailable["reason"])}
    if calibration.get("model", "pinhole") not in ("pinhole", "opencv_pinhole"):
        return {**unavailable, "reason": "unsupported camera distortion model"}
    transform = calibration.get("optical_to_base_link") or calibration.get("optical_to_frame")
    if transform is None:
        return {**unavailable, "reason": "missing camera extrinsics for this frame"}
    try:
        matrix = [[float(v) for v in row] for row in transform]
        if len(matrix) != 4 or any(len(row) != 4 for row in matrix):
            return unavailable
        # Rigid camera transforms only: the inverse below is R transpose.
        if any(
            abs(sum(matrix[k][i] * matrix[k][j] for k in range(3)) - (i == j)) > 1e-4
            for i in range(3)
            for j in range(3)
        ):
            return {**unavailable, "reason": "invalid camera rotation matrix"}
        z = sum(matrix[i][2] * (position[i] - matrix[i][3]) for i in range(3))
        if z <= 0:
            return {**unavailable, "reason": "target is behind the camera"}
        pixel = project_base_point(calibration, position)
        if pixel is None:
            return {
                **unavailable,
                "reason": "target is outside the image or projection parameters are invalid",
            }
        intrinsics = calibration.get("source_intrinsics", calibration)
        source = [intrinsics["width"], intrinsics["height"]]
        encoded = calibration.get("rgb_encoded_size", source)
        if list(size) != encoded or calibration.get("rgb_source_size", source) != source:
            return {**unavailable, "reason": "image size does not match calibration"}
        scale = [source[i] / size[i] for i in range(2)]
        mapping = calibration.get("encoded_to_source_pixel", {})
        if any(abs(mapping.get(k, scale[i]) - scale[i]) > 1e-6 for i, k in enumerate(("scale_x", "scale_y"))):
            return {**unavailable, "reason": "unsupported image resize or crop mapping"}
        mapped = [(pixel[i] / 1000 * (source[i] - 1) + 0.5) / scale[i] - 0.5 for i in range(2)]
        if any(not math.isfinite(v) or not 0 <= v <= size[i] - 1 for i, v in enumerate(mapped)):
            return {**unavailable, "reason": "target is outside the encoded image"}
        return {
            "pixel": mapped,
            "percent": [100 * (v + 0.5) / size[i] for i, v in enumerate(mapped)],
            "reference_only": calibration.get("metric_projection_ready") is not True,
            "reason": "code projection; does not verify occlusion, contact, or success",
        }
    except (TypeError, ValueError, KeyError, IndexError, OverflowError, ZeroDivisionError):
        return unavailable

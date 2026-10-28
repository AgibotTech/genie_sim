# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Pure helpers for inline-OVRtx FreeCam optical zoom."""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

MIN_ZOOM_FACTOR = 0.25
MAX_ZOOM_FACTOR = 8.0


def clamp_zoom_factor(value: float) -> float:
    """Return a finite optical zoom factor inside the supported range."""
    zoom = float(value)
    if not math.isfinite(zoom):
        raise ValueError("camera zoom factor must be finite")
    return min(max(zoom, MIN_ZOOM_FACTOR), MAX_ZOOM_FACTOR)


def scaled_intrinsics(
    *,
    base_fx: float,
    base_fy: float,
    cx: float,
    cy: float,
    zoom: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build CameraInfo K/P arrays for a focal-length zoom.

    Optical zoom changes focal length while leaving the principal point at
    the image centre. OVRtx receives the same scale through the USD camera's
    ``focalLength`` attribute.
    """
    factor = clamp_zoom_factor(zoom)
    fx = float(base_fx) * factor
    fy = float(base_fy) * factor
    K = np.asarray(
        [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0],
        dtype=np.float64,
    )
    P = np.asarray(
        [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0],
        dtype=np.float64,
    )
    return K, P


__all__ = [
    "MAX_ZOOM_FACTOR",
    "MIN_ZOOM_FACTOR",
    "clamp_zoom_factor",
    "scaled_intrinsics",
]

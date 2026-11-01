# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0
"""Geometry helpers for the scene block shipped with each inference payload.

Pure numpy: no USD, no Isaac imports, so it can be exercised outside the
simulator. api_core reads raw 4x4 poses and world bounding boxes off the stage;
this turns them into world position/rpy and pixel indices.

Pixel convention throughout: origin at the image's top-left corner, u to the
right, v down, integer indices.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as R

DIST_KEYS = ("k1", "k2", "p1", "p2", "k3")

# A point sitting on the optical centre plane blows the distortion polynomial
# up to inf, which then casts to a garbage int. Callers already gate on depth,
# so the magnitude past this is meaningless — it only has to survive the cast.
_UV_LIMIT = 1.0e6


def parse_intrinsics(raw: dict) -> dict:
    """USD lens-distortion attributes (stringified by USD) -> plain floats.

    ``width``/``height`` come from the ``imageSize`` attribute, i.e. the
    resolution the intrinsics were calibrated at, which is not necessarily the
    resolution we render at — see scale_intrinsics.
    """
    out = {k: float(raw[k]) for k in ("fx", "fy", "cx", "cy")}
    out.update({k: float(raw.get(k) or 0.0) for k in DIST_KEYS})
    size = raw.get("imageSize")
    if isinstance(size, str):
        size = [int(float(v)) for v in size.strip("()[] ").split(",")]
    out["width"], out["height"] = (int(size[0]), int(size[1])) if size else (0, 0)
    out["model"] = str(raw.get("model", "opencvPinhole"))
    return out


def scale_intrinsics(intr: dict, width: int, height: int) -> dict:
    """Rescale to the rendered resolution when it differs from imageSize.

    Distortion coefficients are dimensionless (they act on normalized image
    coordinates), so only the focal lengths and principal point move.
    """
    if not (width and height and intr["width"] and intr["height"]):
        return intr
    if (width, height) == (intr["width"], intr["height"]):
        return intr
    sx, sy = width / intr["width"], height / intr["height"]
    out = dict(intr)
    out["fx"] *= sx
    out["cx"] *= sx
    out["fy"] *= sy
    out["cy"] *= sy
    out["width"], out["height"] = width, height
    return out


def pose_to_pos_rpy(pose) -> tuple[list, list]:
    """4x4 world transform -> (position, rpy).

    rpy is the fixed-axis roll/pitch/yaw triple, i.e. R = Rz(yaw) Ry(pitch)
    Rx(roll) — scipy's extrinsic "xyz" — matching how EEF_ABS actions are
    already expressed on this wire.
    """
    T = np.asarray(pose, dtype=float).reshape(4, 4)
    return T[:3, 3].tolist(), R.from_matrix(T[:3, :3]).as_euler("xyz").tolist()


def project(points_world, cam_to_world, intr) -> tuple[np.ndarray, np.ndarray]:
    """World points -> (pixel indices Nx2 int, camera-frame depth N).

    ``cam_to_world`` is OpenCV-convention (x right, y down, z forward), which is
    what api_core.get_obj_world_pose_matrix(..., camera=True) returns. Depth <= 0
    means the point is behind the camera and its uv is meaningless.
    """
    pts = np.atleast_2d(np.asarray(points_world, dtype=float))
    T = np.asarray(cam_to_world, dtype=float).reshape(4, 4)
    # Row-vector form of R.T @ (p - t).
    cam = (pts - T[:3, 3]) @ T[:3, :3]
    z = cam[:, 2]
    safe = np.where(np.abs(z) < 1e-9, 1e-9, z)
    x, y = cam[:, 0] / safe, cam[:, 1] / safe
    r2 = x * x + y * y
    radial = 1.0 + intr["k1"] * r2 + intr["k2"] * r2**2 + intr["k3"] * r2**3
    xd = x * radial + 2 * intr["p1"] * x * y + intr["p2"] * (r2 + 2 * x * x)
    yd = y * radial + intr["p1"] * (r2 + 2 * y * y) + 2 * intr["p2"] * x * y
    uv = np.stack([intr["fx"] * xd + intr["cx"], intr["fy"] * yd + intr["cy"]], axis=1)
    uv = np.clip(np.nan_to_num(uv, nan=0.0, posinf=_UV_LIMIT, neginf=-_UV_LIMIT), -_UV_LIMIT, _UV_LIMIT)
    return np.rint(uv).astype(np.int64), z


def project_track(points_world, cam_to_world, intr) -> list:
    """Per-point pixel index, or None where the point is behind the camera.

    Off-frame indices are kept as-is (they can be negative or past the edge):
    an accumulated trajectory is still useful once it leaves the field of view,
    and the consumer has the resolution to clip with.
    """
    if len(points_world) == 0:
        return []
    uv, depth = project(points_world, cam_to_world, intr)
    return [[int(u), int(v)] if d > 0 else None for (u, v), d in zip(uv, depth)]


def envelope_2d(corners_world, cam_to_world, intr, width, height):
    """Axis-aligned pixel envelope of a projected 3D box.

    Returns (bbox, corners_uv). bbox is [u_min, v_min, u_max, v_max] clipped to
    the frame, or None when every corner is behind the camera or the box misses
    the image entirely. corners_uv is the raw unclipped projection of all 8
    corners, so a consumer wanting the exact quad can rebuild it.
    """
    uv, depth = project(corners_world, cam_to_world, intr)
    corners_uv = uv.tolist()
    front = depth > 0
    if not front.any():
        return None, corners_uv
    u, v = uv[front, 0], uv[front, 1]
    u0, v0, u1, v1 = int(u.min()), int(v.min()), int(u.max()), int(v.max())
    if u1 < 0 or v1 < 0 or u0 > width - 1 or v0 > height - 1:
        return None, corners_uv
    return [max(u0, 0), max(v0, 0), min(u1, width - 1), min(v1, height - 1)], corners_uv


def hull_points(points, max_input: int = 65536, max_output: int = 512) -> np.ndarray:
    """Convex-hull vertices of a point cloud, thinned at both ends.

    The 2D envelope of a projected shape depends only on its convex hull, so
    hulling shrinks a mesh to the few points worth transforming every frame
    without changing the answer. ``max_input`` bounds the one-off hull cost on
    a huge mesh; ``max_output`` bounds the per-frame cost on a mesh whose hull
    is itself dense (a sphere has no interior points to drop) and is the only
    lossy step -- it can pull the envelope in by the gap between neighbouring
    hull vertices, i.e. well under a pixel at any normal viewing distance.

    Degenerate clouds (flat, collinear, fewer than 4 points) have no hull and
    come back as-is.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(pts) > max_input:
        pts = pts[:: len(pts) // max_input + 1]
    if len(pts) >= 4:
        try:
            pts = pts[ConvexHull(pts).vertices]
        except Exception:
            pass
    if len(pts) > max_output:
        pts = pts[:: len(pts) // max_output + 1]
    return pts


def object_name_from_prim(prim_path: str, known_names) -> str:
    """The object-name path segment, e.g. /Workspace/Objects/<name>/entity.

    ``known_names`` is the semantics keyset; matching against it rather than
    guessing a fixed depth keeps this working for both /Workspace/Objects and
    the /World/Objects layout. Falls back to the last non-"entity" segment.
    """
    segments = [s for s in prim_path.split("/") if s]
    for segment in segments:
        if segment in known_names:
            return segment
    for segment in reversed(segments):
        if segment != "entity":
            return segment
    return prim_path

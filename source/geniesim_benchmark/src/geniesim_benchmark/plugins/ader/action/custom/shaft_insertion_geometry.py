# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# License: Mozilla Public License Version 2.0
"""Finite circular bore/shaft geometry in metres, independent of simulator I/O."""

from __future__ import annotations

import numpy as np

from .precise_geometry import rigid_pose, transformed


def segment(value: list) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (2, 3) or not np.isfinite(result).all():
        raise ValueError("Expected two distinct finite three-dimensional endpoints")
    length = np.linalg.norm(result[1] - result[0])
    if not np.isfinite(length) or length < 1e-9:
        raise ValueError("Expected a finite nonzero segment length")
    return result


def pose_from_wxyz(position, quaternion) -> np.ndarray:
    """Reject unavailable API poses (including the missing-prim zero quaternion)."""
    p, q = np.asarray(position, dtype=float), np.asarray(quaternion, dtype=float)
    if (
        p.shape != (3,)
        or q.shape != (4,)
        or not np.isfinite(p).all()
        or not np.isfinite(q).all()
    ):
        raise ValueError("Invalid measured position/quaternion")
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm < 1e-9:
        raise ValueError("Invalid measured quaternion")
    w, x, y, z = q / norm
    pose = np.eye(4)
    pose[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    pose[:3, 3] = p
    return pose


def insertion(
    relative_pose: np.ndarray,
    hole: np.ndarray,
    shaft: np.ndarray,
    hole_radius: float,
    shaft_radius: float,
    min_overlap: float,
    max_angle: float,
    radial_tolerance: float = 0.0,
    shaft_sections: list | None = None,
) -> dict:
    """Bound the shaft cross-section throughout the overlapping bore interval.

    Endpoints are local to the workpiece and reference respectively. Either axis
    direction is accepted for a through bore. The oblique cylinder cross-section
    is conservatively bounded by radius / cos(angle); axial hover is not overlap.
    radial_tolerance is an explicit measurement/contact allowance, not bore size.
    This measures insertion geometry, not release, contact force or bottom seating.
    """
    if not rigid_pose(relative_pose):
        raise ValueError("Invalid measured relative pose")
    bore = transformed(hole, relative_pose)
    u = bore[1] - bore[0]
    length = float(np.linalg.norm(u))
    if not np.isfinite(length) or length < 1e-9:
        raise ValueError("Degenerate measured bore")
    u /= length
    v = shaft[1] - shaft[0]
    v /= np.linalg.norm(v)
    dot = float(np.dot(u, v))
    angle = float(np.arccos(np.clip(abs(dot), 0, 1)))
    projected = (shaft - bore[0]) @ u
    lo, hi = max(0.0, float(projected.min())), min(length, float(projected.max()))
    overlap = max(0.0, hi - lo)
    tests = {"angle": angle <= max_angle, "overlap": overlap >= min_overlap}
    radial, envelope = None, None
    if overlap > 0 and abs(dot) > 1e-9:
        ends = shaft[0] + np.outer((np.array([lo, hi]) - projected[0]) / dot, v)
        on_axis = bore[0] + np.outer([lo, hi], u)
        radial = float(np.linalg.norm(ends - on_axis, axis=1).max())
        envelope = radial + shaft_radius / abs(dot)
    tests["radial_clearance"] = (
        envelope is not None and envelope <= hole_radius + radial_tolerance
    )
    section_results = None
    if shaft_sections is not None:
        section_results = []
        for section in shaft_sections:
            points = shaft[0] + np.outer([section["start_m"], section["end_m"]], v)
            projection = (points - bore[0]) @ u
            a = max(0.0, float(projection.min()))
            b = min(length, float(projection.max()))
            if b <= a:
                continue
            section_radial = section_envelope = None
            if abs(dot) > 1e-9:
                axial = (np.array([a, b]) - projected[0]) / dot
                shaft_axis = shaft[0] + np.outer(axial, v)
                bore_axis = bore[0] + np.outer([a, b], u)
                section_radial = float(
                    np.linalg.norm(shaft_axis - bore_axis, axis=1).max()
                )
                section_envelope = section_radial + section["radius_m"] / abs(dot)
            section_results.append(
                {
                    **section,
                    "overlap_m": b - a,
                    "radial_offset_m": section_radial,
                    "radial_envelope_m": section_envelope,
                    "pass": section_envelope is not None
                    and section_envelope <= hole_radius + radial_tolerance,
                }
            )
        tests["radial_clearance"] = bool(section_results) and all(
            s["pass"] for s in section_results
        )
        valid = [s for s in section_results if s["radial_envelope_m"] is not None]
        radial = max((s["radial_offset_m"] for s in valid), default=None)
        envelope = max((s["radial_envelope_m"] for s in valid), default=None)
    result = {
        "pass": bool(all(tests.values())),
        "tests": tests,
        "overlap_m": overlap,
        "angle_rad": angle,
        "radial_offset_m": radial,
        "radial_envelope_m": envelope,
        "hole_radius_m": hole_radius,
        "radial_tolerance_m": radial_tolerance,
    }
    if section_results is not None:
        result["shaft_sections"] = section_results
    return result


def validated_sections(
    value: list | None, shaft: np.ndarray, max_radius: float
) -> list[dict[str, float]] | None:
    """Validate an explicit conservative union of coaxial finite cylinders.

    Distances run from shaft[0] toward shaft[1]. Overlapping intervals are
    allowed (both constraints apply); uncovered axial gaps are rejected.
    """
    if value is None:
        return None
    if not isinstance(value, list) or not value:
        raise ValueError("shaft_sections must be a nonempty list")
    length = float(np.linalg.norm(shaft[1] - shaft[0]))
    covered = 0.0
    previous_start = -1.0
    result = []
    for section in value:
        if not isinstance(section, dict) or set(section) != {
            "start_m",
            "end_m",
            "radius_m",
        }:
            raise ValueError("Expected start_m, end_m and radius_m per section")
        start, end, radius = (section[k] for k in ["start_m", "end_m", "radius_m"])
        if any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v)
            for v in [start, end, radius]
        ):
            raise ValueError("Section geometry must contain finite numbers")
        if not (
            0 <= start < end <= length + 1e-9
            and previous_start <= start <= covered + 1e-9
            and 0 < radius <= max_radius
        ):
            raise ValueError("Invalid, unordered or uncovered shaft section")
        covered = max(covered, end)
        previous_start = start
        result.append({k: float(section[k]) for k in section})
    if abs(covered - length) > 1e-9:
        raise ValueError("Shaft sections must cover the complete shaft")
    return result

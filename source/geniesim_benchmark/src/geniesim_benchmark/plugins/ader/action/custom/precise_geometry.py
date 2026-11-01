# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# License: Mozilla Public License Version 2.0
"""Asset-independent point containment and projected surface coverage, in metres."""

from __future__ import annotations

import numpy as np


def points(value) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.ndim != 2 or result.shape[1] != 3 or len(result) == 0 or not np.isfinite(result).all():
        raise ValueError("points must be a nonempty finite N x 3 array")
    return result


def rigid_pose(value) -> bool:
    pose = np.asarray(value, dtype=float)
    return bool(
        pose.shape == (4, 4)
        and np.isfinite(pose).all()
        and np.allclose(pose[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
        and np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-5, rtol=0)
        and np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-5, rtol=0)
    )


def transformed(value: np.ndarray, pose: np.ndarray) -> np.ndarray:
    return value @ pose[:3, :3].T + pose[:3, 3]


def hull(value: np.ndarray) -> list[list[float]]:
    vertices = sorted(set(map(tuple, value.tolist())))
    if len(vertices) < 3:
        return [list(p) for p in vertices]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    low, high = [], []
    for p in vertices:
        while len(low) >= 2 and cross(low[-2], low[-1], p) <= 0:
            low.pop()
        low.append(p)
    for p in reversed(vertices):
        while len(high) >= 2 and cross(high[-2], high[-1], p) <= 0:
            high.pop()
        high.append(p)
    return [list(p) for p in low[:-1] + high[:-1]]


def area(poly: list[list[float]]) -> float:
    return (
        abs(sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(poly, poly[1:] + poly[:1]))) / 2
        if len(poly) >= 3
        else 0.0
    )


def intersection(subject: list[list[float]], clip: list[list[float]]) -> list[list[float]]:
    result = subject
    for a, b in zip(clip, clip[1:] + clip[:1]):
        old, result = result, []
        if not old:
            break

        def side(p):
            return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])

        prev = old[-1]
        for cur in old:
            sp, sc = side(prev), side(cur)
            if (sp >= -1e-12) != (sc >= -1e-12):
                t = sp / (sp - sc)
                result.append([prev[i] + t * (cur[i] - prev[i]) for i in range(2)])
            if sc >= -1e-12:
                result.append(cur)
            prev = cur
    return result


def containment(value: np.ndarray, lower: list, upper: list, quantifier: str = "all") -> dict:
    """None disables a boundary; any/all refers to the supplied sample points."""
    tests = np.ones(len(value), dtype=bool)
    for axis in range(3):
        if lower[axis] is not None:
            tests &= value[:, axis] >= lower[axis]
        if upper[axis] is not None:
            tests &= value[:, axis] <= upper[axis]
    return {
        "pass": bool(tests.all() if quantifier == "all" else tests.any()),
        "minimum": value.min(axis=0).tolist(),
        "maximum": value.max(axis=0).tolist(),
    }


def coverage(
    source: np.ndarray,
    target: np.ndarray,
    up_axis: int,
    threshold: float,
    max_gap: float | None = None,
    normal_dot: float | None = None,
    min_normal_dot: float | None = None,
) -> dict:
    """Convex projected coverage of target by source, plus optional surface gates."""
    axes = [i for i in range(3) if i != up_axis]
    source_polygon, target_polygon = hull(source[:, axes]), hull(target[:, axes])
    denominator = area(target_polygon)
    ratio = area(intersection(target_polygon, source_polygon)) / denominator if denominator > 0 else 0.0
    gap = float(source[:, up_axis].max() - target[:, up_axis].max())
    tests = {"projection": denominator > 0 and area(source_polygon) > 0 and ratio >= threshold}
    if max_gap is not None:
        tests["height"] = gap <= max_gap
    if min_normal_dot is not None:
        tests["facing"] = normal_dot is not None and normal_dot > min_normal_dot
    return {
        "pass": bool(all(tests.values())),
        "tests": tests,
        "coverage": ratio,
        "highest_gap": gap,
        "normal_dot": normal_dot,
    }

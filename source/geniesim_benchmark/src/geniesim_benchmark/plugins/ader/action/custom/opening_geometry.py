# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# License: Mozilla Public License, v. 2.0
"""Convex object sections through a moving reference-local opening, in metres."""

from __future__ import annotations

import numpy as np

from .precise_geometry import area, hull


def affine_pose(value: np.ndarray) -> bool:
    pose = np.asarray(value, dtype=float)
    return bool(
        pose.shape == (4, 4)
        and np.isfinite(pose).all()
        and np.allclose(pose[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
        and abs(np.linalg.det(pose[:3, :3])) > 1e-12
    )


def convex_polygon(value: list) -> np.ndarray:
    polygon = np.asarray(value, dtype=float)
    if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3 or not np.isfinite(polygon).all():
        raise ValueError("An opening polygon requires at least three finite 2D vertices")
    edges = np.roll(polygon, -1, axis=0) - polygon
    crosses = np.cross(edges, np.roll(edges, -1, axis=0))
    if not (np.all(crosses > 0) or np.all(crosses < 0)) or len(hull(polygon)) != len(polygon):
        raise ValueError("Opening vertices must describe a strictly convex perimeter")
    polygon = polygon if np.all(crosses > 0) else polygon[::-1]
    canonical = np.asarray(hull(polygon))
    start = next(i for i, p in enumerate(polygon) if np.array_equal(p, canonical[0]))
    if not np.allclose(np.roll(polygon, -start, axis=0), canonical, atol=1e-12, rtol=0):
        raise ValueError("Opening perimeter must not self-intersect")
    return polygon


def section(vertices: np.ndarray, axis: int, coordinate: float) -> np.ndarray:
    """Intersect the convex hull with a plane using its complete vertex chords.

    Every edge is a chord; extra interior chords cannot enlarge the section's
    convex hull. This also handles bevels without assuming a circular end cap.
    """
    axes = [i for i in range(3) if i != axis]
    distance = vertices[:, axis] - coordinate
    first, second = np.triu_indices(len(vertices), 1)
    crossing = (distance[first] < 0) & (distance[second] > 0) | (distance[first] > 0) & (distance[second] < 0)
    first, second = first[crossing], second[crossing]
    t = distance[first] / (distance[first] - distance[second])
    intersected = vertices[first] + t[:, None] * (vertices[second] - vertices[first])
    on_plane = vertices[np.abs(distance) <= 1e-12]
    if len(on_plane):
        intersected = np.concatenate([intersected, on_plane])
    if not len(intersected):
        return np.empty((0, 2))
    return np.asarray(hull(intersected[:, axes]), dtype=float)


def opening_slabs(slices: list) -> list:
    """Supporting side planes of the convex loft between aperture polygons."""
    slabs = []
    for first, second in zip(slices, slices[1:]):
        normals = []
        for polygon in (first["polygon"], second["polygon"]):
            edges = np.roll(polygon, -1, axis=0) - polygon
            normals.append(
                np.column_stack([edges[:, 1], -edges[:, 0]]) / np.linalg.norm(edges, axis=1)[:, None]
            )
        normals = np.concatenate(normals)
        slabs.append(
            {
                "lower": first["coordinate"],
                "upper": second["coordinate"],
                "normals": normals,
                "lower_support": (first["polygon"] @ normals.T).max(axis=0),
                "upper_support": (second["polygon"] @ normals.T).max(axis=0),
            }
        )
    return slabs


def through_opening(
    vertices: np.ndarray, axis: int, slices: list, tolerance: float, slabs: list | None = None
) -> dict:
    """Require material across both faces and every sampled section in its hole.

    The source point hull must match the intended convex collision geometry.
    The opening is given as parallel convex aperture sections in its rigid
    reference frame. No world-coordinate target, preferred object end, release,
    grasp, contact, or extra angle requirement is inferred here.
    """
    minimum, maximum = float(vertices[:, axis].min()), float(vertices[:, axis].max())
    lower, upper = slices[0]["coordinate"], slices[-1]["coordinate"]
    spans = minimum < lower - 1e-12 and maximum > upper + 1e-12
    measurements = []
    for plane in slices:
        cut = section(vertices, axis, plane["coordinate"])
        valid_section = len(cut) >= 3 and area(cut.tolist()) > 1e-15
        clearance = None
        if valid_section:
            polygon = plane["polygon"]
            edges = np.roll(polygon, -1, axis=0) - polygon
            relative = cut[:, None, :] - polygon[None, :, :]
            distances = (
                edges[None, :, 0] * relative[:, :, 1] - edges[None, :, 1] * relative[:, :, 0]
            ) / np.linalg.norm(edges, axis=1)[None, :]
            clearance = float(distances.min())
        measurements.append(
            {
                "coordinate": plane["coordinate"],
                "section_vertices": len(cut),
                "clearance_m": clearance,
                "pass": valid_section and clearance >= -tolerance,
            }
        )
    # Clipped hull vertices are the face cuts plus source vertices inside the
    # slab. The latter must be checked too: material can bulge between faces.
    # Minkowski-sum supports interpolate linearly; the two polygons' combined
    # edge normals give every side plane of this convex loft.
    interior_clearance = None
    interior_count = 0
    axes = [i for i in range(3) if i != axis]
    for slab in opening_slabs(slices) if slabs is None else slabs:
        inside = vertices[(vertices[:, axis] > slab["lower"]) & (vertices[:, axis] < slab["upper"])]
        if not len(inside):
            continue
        interior_count += len(inside)
        fraction = (inside[:, axis] - slab["lower"]) / (slab["upper"] - slab["lower"])
        support = (
            slab["lower_support"][None, :]
            + fraction[:, None] * (slab["upper_support"] - slab["lower_support"])[None, :]
        )
        clearance = float((support - inside[:, axes] @ slab["normals"].T).min())
        interior_clearance = clearance if interior_clearance is None else min(interior_clearance, clearance)
    return {
        "pass": bool(
            spans
            and all(m["pass"] for m in measurements)
            and (interior_clearance is None or interior_clearance >= -tolerance)
        ),
        "spans_opening": bool(spans),
        "source_interval": [minimum, maximum],
        "penetration_beyond_inner_m": lower - minimum,
        "sections": measurements,
        "interior_vertices": interior_count,
        "interior_vertex_clearance_m": interior_clearance,
    }

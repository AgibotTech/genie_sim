# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Containment in an open bucket's actual collision shell, in bucket coordinates."""

import numpy as np


def transform(points, matrix):
    """USD matrices use row vectors, including translation in the last row."""
    points = np.asarray(points, dtype=float)
    return (np.c_[points, np.ones(len(points))] @ np.asarray(matrix))[:, :3]


class BucketCavity:
    """Vertical rays distinguish the cavity, solid walls and exterior.

    A cavity point has a surface below it, no surface above it, and is below
    the rim. Signed crossings also identify the solid shell so a grain in a
    wall is never counted as either a received grain or a tabletop spill.
    The supplied closed, consistently wound collision mesh must be Z-up;
    callers transform live points into its rigid frame before testing.
    """

    def __init__(self, triangles):
        triangles = np.asarray(triangles, dtype=float)
        if triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or not np.isfinite(triangles).all():
            raise ValueError("Bucket collision mesh must contain finite triangles")
        if not len(triangles):
            raise ValueError("Bucket collision mesh is empty")
        self.lower = triangles.min(axis=(0, 1))
        self.upper = triangles.max(axis=(0, 1))
        a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
        e1, e2 = b - a, c - a
        det = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
        valid = np.abs(det) > 1e-14
        self.a, self.e1, self.e2, self.det = a[valid], e1[valid], e2[valid], det[valid]
        if not valid.any():
            raise ValueError("Bucket collision mesh has no horizontal projection")

    def classify(self, points):
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("Popcorn positions must be finite N x 3 points")
        inside, envelope = np.zeros(len(points), dtype=bool), np.zeros(len(points), dtype=bool)
        candidates = np.flatnonzero(((points >= self.lower) & (points <= self.upper)).all(axis=1))
        for start in range(0, len(candidates), 32):
            indices = candidates[start : start + 32]
            p = points[indices]
            dx, dy = p[:, None, 0] - self.a[:, 0], p[:, None, 1] - self.a[:, 1]
            u = (dx * self.e2[:, 1] - dy * self.e2[:, 0]) / self.det
            v = (self.e1[:, 0] * dy - self.e1[:, 1] * dx) / self.det
            hit = (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9)
            z = self.a[:, 2] + u * self.e1[:, 2] + v * self.e2[:, 2]
            above = hit & (z > p[:, None, 2] + 1e-7)
            below = hit & (z <= p[:, None, 2] + 1e-7)
            cavity = below.any(axis=1) & ~above.any(axis=1)
            solid = np.abs((above * np.sign(self.det)).sum(axis=1)) > 0.5
            inside[indices] = cavity
            envelope[indices] = cavity | solid
        return inside, envelope


def score_counts(inside, spilled):
    """Both conditions are from the same observation; the second depends on the first."""
    return 0.0 if inside == 0 else (0.5 if spilled else 1.0)

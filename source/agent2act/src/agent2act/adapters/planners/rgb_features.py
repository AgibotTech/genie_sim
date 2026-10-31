# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Bounded local affine RGB correspondence; neither object tracking nor contact sensing."""

from typing import Any

from agent2act.domain.models import ContractError, number

from .rgb_matching import decode_rgb_pair


def match_rgb_feature_patch(previous: bytes, current: bytes, x: float, y: float) -> dict[str, Any]:
    """Match a patch under scale/rotation using local feature consensus and an observed patch check.

    No global motion model is fitted. The point must lie inside the convex hull of nearby,
    mutually distinctive features in both frames."""
    import cv2
    import numpy as np

    x, y = number(x, "x"), number(y, "y")
    if not 0 <= x <= 1000 or not 0 <= y <= 1000:
        raise ContractError("RGB coordinates must be in 0..1000")
    before, after = decode_rgb_pair(previous, current)
    height, width = before.shape
    point = np.array([x * (width - 1) / 1000, y * (height - 1) / 1000])
    common = {
        "analysis_size": [width, height],
        "method": "affine_features",
        "motion_executed": False,
        "meaning": "Local RGB affine correspondence hypothesis; not semantic identity, 3D motion, contact or retention. Camera motion and mixed surfaces can invalidate object interpretation. Fit residual is not a calibrated uncertainty bound.",
    }

    def unavailable(reason: str) -> dict[str, Any]:
        return {**common, "status": "unavailable", "reason": reason}

    def inside_image(p: Any, margin: float = 8) -> bool:
        return bool(margin <= p[0] < width - margin and margin <= p[1] < height - margin)

    if not inside_image(point):
        return unavailable("patch crosses image boundary")
    sift = cv2.SIFT.create(nfeatures=3000)
    full_image = np.full(before.shape, 255, dtype=np.uint8)
    keys1, descriptors1 = sift.detectAndCompute(before, full_image)
    keys2, descriptors2 = sift.detectAndCompute(after, full_image)
    if descriptors1 is None or descriptors2 is None or min(len(keys1), len(keys2)) < 6:
        return unavailable("insufficient distinctive features")
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    def distinctive(a: Any, b: Any) -> dict[int, int]:
        return {
            m.queryIdx: m.trainIdx
            for pair in matcher.knnMatch(a, b, k=2)
            if len(pair) == 2
            for m, n in [pair]
            if m.distance < 0.7 * n.distance
        }

    forward = distinctive(descriptors1, descriptors2)
    backward = distinctive(descriptors2, descriptors1)
    pairs = []
    seen_source, seen_target = set(), set()
    for i, j in forward.items():
        a, b = np.array(keys1[i].pt), np.array(keys2[j].pt)
        # SIFT can assign multiple orientations to one location. These are not
        # independent support points and must not inflate geometric evidence.
        source_key, target_key = tuple(np.round(a * 4)), tuple(np.round(b * 4))
        if (
            backward.get(j) == i
            and np.linalg.norm(a - point) <= 48
            and source_key not in seen_source
            and target_key not in seen_target
        ):
            pairs.append((a, b))
            seen_source.add(source_key)
            seen_target.add(target_key)
    if len(pairs) < 6:
        return unavailable("insufficient local mutual features")
    source, target = (np.asarray(values, dtype=np.float32) for values in zip(*pairs))
    transform, mask = cv2.estimateAffine2D(
        source, target, method=cv2.RANSAC, ransacReprojThreshold=2, maxIters=1000, confidence=0.99
    )
    if transform is None or mask is None or not np.isfinite(transform).all():
        return unavailable("local affine fit unavailable")
    selected = mask.ravel().astype(bool)
    if selected.sum() < 6 or selected.mean() < 0.65:
        return unavailable("inconsistent local feature motion")
    source, target = source[selected], target[selected]
    scales = np.linalg.svd(transform[:, :2], compute_uv=False)
    if (
        np.linalg.det(transform[:, :2]) <= 0
        or min(scales) < 0.5
        or max(scales) > 2
        or max(scales) / min(scales) > 1.5
    ):
        return unavailable("unsupported local deformation")
    mapped = transform @ np.append(point, 1)

    def supported(points: Any, query: Any) -> bool:
        hull = cv2.convexHull(points)
        return bool(
            np.linalg.eigvalsh(np.cov(points.T)).min() >= 1
            and cv2.pointPolygonTest(hull, (float(query[0]), float(query[1])), False) >= 0
        )

    if not inside_image(mapped) or not supported(source, point) or not supported(target, mapped):
        return unavailable("point lacks enclosing two-dimensional support")
    reverse, _ = cv2.estimateAffine2D(
        target, source, method=cv2.RANSAC, ransacReprojThreshold=2, maxIters=1000, confidence=0.99
    )
    if (
        reverse is None
        or not np.isfinite(reverse).all()
        or np.linalg.norm(reverse @ np.append(mapped, 1) - point) > 2
    ):
        return unavailable("forward/backward mismatch")
    residuals = np.linalg.norm(source @ transform[:, :2].T + transform[:, 2] - target, axis=1)
    if not np.isfinite(residuals).all() or np.median(residuals) > 1:
        return unavailable("local affine residual too large")
    # A supported point can still be occluded while surrounding features survive.
    # Check its actual observed texture after warping; do not return extrapolated
    # coordinates solely because a neighboring surface model can be fitted.
    gx, gy = np.meshgrid(np.arange(-7, 8), np.arange(-7, 8))
    grid = np.stack([gx + point[0], gy + point[1]], axis=-1).astype(np.float32)
    warped = (grid @ transform[:, :2].T + transform[:, 2]).astype(np.float32)
    if np.any(warped < 0) or np.any(warped[..., 0] >= width - 1) or np.any(warped[..., 1] >= height - 1):
        return unavailable("warped patch crosses image boundary")
    patch1 = cv2.remap(before, grid[..., 0], grid[..., 1], cv2.INTER_LINEAR).astype(float)
    patch2 = cv2.remap(after, warped[..., 0], warped[..., 1], cv2.INTER_LINEAR).astype(float)
    if min(patch1.std(), patch2.std()) < 8:
        return unavailable("insufficient selected-point texture")
    correlation = float(np.corrcoef(patch1.ravel(), patch2.ravel())[0, 1])
    if not np.isfinite(correlation) or correlation < 0.85:
        return unavailable("selected point occluded or appearance inconsistent")
    current_xy = 1000 * mapped / [width - 1, height - 1]
    return {
        **common,
        "status": "matched",
        "previous_xy": [x, y],
        "current_xy": current_xy.tolist(),
        "delta_xy": (current_xy - [x, y]).tolist(),
        "correlation": correlation,
        "support_points": len(source),
        "median_fit_error_px": float(np.median(residuals)),
    }

# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Camera intrinsics and extrinsics published next to the inference images.

Renderer matrices use USD row vectors. Published transforms use column vectors,
metres, and optical axes (+x right, +y down, +z forward), so a client can
backproject a depth pixel into the robot frame without guessing conventions.
Every failure returns ``available=False`` with a reason instead of a transform.
"""

import uuid

import numpy as np

from geniesim_benchmark.plugins.logger import Logger

logger = Logger()


def camera_calibration(stage, camera_path, base_path, params, rgb, depth, snapshot_id):
    """Intrinsics/extrinsics for one render product, validated against its render."""
    from pxr import UsdGeom

    result = {
        "available": False,
        "snapshot_id": snapshot_id,
        "camera_path": camera_path,
        "base_path": base_path,
        "source": "same render product camera_params + USD",
    }
    try:
        if rgb is None or depth is None or tuple(rgb.shape[:2]) != tuple(depth.shape):
            raise ValueError("RGB/depth dimensions differ or sensor missing")
        height, width = depth.shape
        if list(params["renderProductResolution"]) != [width, height]:
            raise ValueError("render product resolution mismatch")
        prim, base = stage.GetPrimAtPath(camera_path), stage.GetPrimAtPath(base_path)
        if not prim.IsValid() or not base.IsValid():
            raise ValueError("camera/base prim unavailable")
        cache = UsdGeom.XformCache(0.0)
        camera_world = np.asarray(cache.GetLocalToWorldTransform(prim), dtype=float).T
        base_world = np.asarray(cache.GetLocalToWorldTransform(base), dtype=float).T
        rendered_world = np.linalg.inv(np.asarray(params["cameraViewTransform"], dtype=float).reshape(4, 4).T)
        # A stale render must never silently use a newer wrist pose.
        if not np.allclose(camera_world, rendered_world, atol=1e-5, rtol=0):
            result["pose_mismatch"] = {
                "usd_camera_world": camera_world.tolist(),
                "rendered_camera_world": rendered_world.tolist(),
                "max_abs_matrix_difference": float(np.max(np.abs(camera_world - rendered_world))),
                "translation_difference_scene_units": float(
                    np.linalg.norm(camera_world[:3, 3] - rendered_world[:3, 3])
                ),
            }
            raise ValueError("rendered camera pose differs from observation USD pose")
        transform = np.linalg.inv(base_world) @ rendered_world @ np.diag([1.0, -1.0, -1.0, 1.0])
        meters = float(params["metersPerSceneUnit"])
        if not np.isfinite(meters) or meters <= 0:
            raise ValueError("invalid scene unit scale")
        transform[:3, 3] *= meters
        if (
            not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(transform[:3, :3]), 1, atol=1e-5)
        ):
            raise ValueError("non-rigid optical-to-base transform")

        def attr(name, default=None):
            value = prim.GetAttribute(name).Get()
            return default if value is None else value

        lens = attr("omni:lensdistortion:model", "pinhole")
        if attr("projection", "perspective") != "perspective":
            raise ValueError("unsupported non-perspective projection")
        if lens == "opencvPinhole":
            prefix = "omni:lensdistortion:opencvPinhole:"
            size = list(attr(prefix + "imageSize"))
            sx, sy = width / size[0], height / size[1]
            fx, fy = float(attr(prefix + "fx")) * sx, float(attr(prefix + "fy")) * sy
            cx, cy = float(attr(prefix + "cx")) * sx, float(attr(prefix + "cy")) * sy
            distortion = [
                float(attr(prefix + key, 0)) for key in ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")
            ]
            model = "opencv_pinhole"
        elif lens in ("pinhole", "") and params["cameraModel"] == "pinhole":
            if attr("cameraProjectionType", "pinhole") != "pinhole":
                raise ValueError("unsupported legacy lens projection")
            # Actual renderer projection includes aperture offsets/aspect conformance.
            projection = np.asarray(params["cameraProjection"], dtype=float).reshape(4, 4).T
            fx, fy = projection[0, 0] * width / 2, projection[1, 1] * height / 2
            cx, cy = (1 - projection[0, 2]) * width / 2, (1 + projection[1, 2]) * height / 2
            distortion, model = [0.0] * 8, "pinhole"
        else:
            raise ValueError("unsupported lens model: " + str(lens))
        if not np.isfinite([fx, fy, cx, cy, *distortion]).all() or min(fx, fy) <= 0:
            raise ValueError("invalid intrinsics")
        result.update(
            available=True,
            model=model,
            width=int(width),
            height=int(height),
            fx=float(fx),
            fy=float(fy),
            cx=float(cx),
            cy=float(cy),
            distortion=distortion,
            frame="base_link",
            units="m",
            optical_to_frame=transform.tolist(),
            depth_kind="distance_to_image_plane",
            depth_scale_m=0.001,
            rgb_depth_aligned=True,
            pose_checked_against_render=True,
            pixel_convention="u right, v down; intrinsics at source resolution",
            optical_axes="+x right, +y down, +z forward",
            transform_convention="column vectors: p_base = optical_to_frame @ [x,y,z,1]",
        )
    except (KeyError, TypeError, ValueError, AttributeError, np.linalg.LinAlgError, ArithmeticError) as exc:
        result["reason"] = str(exc)
    return result


def capture_calibrations(stage, images, depths, paths, annotators, base_path):
    """One calibration per camera, sharing the snapshot ID of this capture."""
    snapshot_id = uuid.uuid4().hex
    results = {}
    for name, path in paths.items():
        try:
            params = annotators[name].get_data()
        except (KeyError, RuntimeError) as exc:
            results[name] = {"available": False, "snapshot_id": snapshot_id, "reason": str(exc)}
            continue
        results[name] = camera_calibration(
            stage, path, base_path, params, images.get(name), depths.get(name), snapshot_id
        )
        if not results[name]["available"]:
            logger.warning(f"[camera_calibration] {name}: {results[name].get('reason')}")
    return results

# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Temporal RGB ray geometry conditional on a stationary material point."""

import itertools
import math
from typing import Any

from agent2act.domain.models import ContractError, number, pose, vector

from .geometry_tools import rotate_vector
from .rgb_matching import RGB_MATCH_TOOL

STATIC_RGB_TOOL = {
    **RGB_MATCH_TOOL,
    "name": "locate_static_rgb_patch",
    "description": (
        "Track ONE distinctive interior material point from the previous planner image to the current "
        "SAME wrist camera, then intersect rays using measured robot motion and explicit robot camera "
        "calibration. x,y are 0..1000 on the previous whole image. Use only to test a STATIC surface "
        "hypothesis when visible support makes stationarity plausible, before contact/manipulation. "
        "Returns a conditional visible-point position or abstains. An epipolar-moving object can "
        "produce a wrong position with zero residual. No stationary-state, object center, contact, "
        "clearance, retention or action is verified. Do not use to locate a carried or moving object."
    ),
}


class RgbRayGeometry:
    """Explicit static wrist-camera calibration, never scene or sensor depth."""

    def __init__(self, calibration: dict[str, Any]) -> None:
        import numpy as np

        if (
            not isinstance(calibration, dict)
            or not calibration
            or set(calibration) - {"hand_left", "hand_right"}
        ):
            raise ContractError("RGB geometry requires explicit wrist-camera calibration")
        self.cameras: dict[str, Any] = {}
        for name, entry in calibration.items():
            if not isinstance(entry, dict) or set(entry) != {"camera_to_wrist", "intrinsics"}:
                raise ContractError("RGB camera requires camera_to_wrist and intrinsics")
            try:
                transform = np.array(entry["camera_to_wrist"], dtype=float)
            except (TypeError, ValueError) as exc:
                raise ContractError("invalid RGB camera transform") from exc
            if (
                transform.shape != (4, 4)
                or not np.isfinite(transform).all()
                or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
                or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-5, rtol=0)
                or abs(np.linalg.det(transform[:3, :3]) - 1) > 1e-5
            ):
                raise ContractError("camera_to_wrist must be rigid SE(3), column vectors, USD optical axes")
            intrinsic = entry["intrinsics"]
            if not isinstance(intrinsic, dict) or set(intrinsic) != {
                "fx",
                "fy",
                "cx",
                "cy",
                "width",
                "height",
                "k1",
                "k2",
                "k3",
                "p1",
                "p2",
            }:
                raise ContractError(
                    "explicit focal lengths, principal point, dimensions and distortion required"
                )
            k = {key: number(value, key) for key, value in intrinsic.items()}
            if min(k["fx"], k["fy"]) <= 0 or any(
                not 2 <= k[key] <= 8192 or k[key] != int(k[key]) for key in ("width", "height")
            ):
                raise ContractError("invalid RGB camera intrinsics")
            self.cameras[name] = {"transform": transform, "k": k}

    @staticmethod
    def _distort(xy: Any, k: dict[str, float]) -> Any:
        import numpy as np

        x, y = xy
        r2 = x * x + y * y
        radial = 1 + k["k1"] * r2 + k["k2"] * r2**2 + k["k3"] * r2**3
        return np.array(
            [
                x * radial + 2 * k["p1"] * x * y + k["p2"] * (r2 + 2 * x * x),
                y * radial + k["p1"] * (r2 + 2 * y * y) + 2 * k["p2"] * x * y,
            ]
        )

    @staticmethod
    def _wrist(values: list[float]) -> Any:
        import numpy as np

        wrist = pose(values, "RGB observation wrist pose")
        transform = np.eye(4)
        transform[:3, :3] = np.array(
            [rotate_vector(wrist[3:], axis) for axis in ([1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0])]
        ).T
        transform[:3, 3] = wrist[:3]
        return transform

    def locate_static(
        self,
        camera: str,
        previous_pose: list[float],
        current_pose: list[float],
        previous_xy: list[float],
        current_xy: list[float],
        image_size: tuple[int, int],
    ) -> dict[str, Any]:
        """Intersect tracked rays, rejecting insufficient or inconsistent geometry.

        Stationarity is an assumption, not observable from a two-view residual; pixel perturbation is
        sensitivity analysis, not calibrated confidence."""
        import numpy as np

        if camera not in self.cameras:
            raise ContractError("RGB camera is not calibrated")
        entry = self.cameras[camera]
        k = entry["k"]
        if tuple(image_size) != (k["width"], k["height"]):
            raise ContractError("RGB image dimensions disagree with calibration")
        coordinates = [vector(xy, 2, "RGB point") for xy in (previous_xy, current_xy)]
        if any(not 0 <= v <= 1000 for xy in coordinates for v in xy):
            raise ContractError("RGB point coordinates must be in 0..1000")
        wrists = [self._wrist(values) for values in (previous_pose, current_pose)]
        cameras = [wrist @ entry["transform"] for wrist in wrists]
        baseline = float(np.linalg.norm(cameras[0][:3, 3] - cameras[1][:3, 3]))
        if baseline < 0.02:
            raise ContractError("insufficient measured camera translation")
        pixels = np.asarray(coordinates) / 1000 * [k["width"] - 1, k["height"] - 1]

        def ray(pixel: Any, transform: Any) -> Any:
            distorted = (pixel - [k["cx"], k["cy"]]) / [k["fx"], k["fy"]]
            xy = distorted.copy()
            for _ in range(20):
                residual = self._distort(xy, k) - distorted
                jac = np.column_stack(
                    [
                        (self._distort(xy + delta, k) - self._distort(xy, k)) / 1e-6
                        for delta in (np.array([1e-6, 0]), np.array([0, 1e-6]))
                    ]
                )
                if not np.isfinite(jac).all() or np.linalg.det(jac) < 0.05:
                    raise ContractError("RGB distortion inverse is ill-conditioned")
                if np.linalg.norm(residual) < 1e-9:
                    break
                xy -= np.linalg.solve(jac, residual)
            if not np.isfinite(xy).all() or np.linalg.norm(self._distort(xy, k) - distorted) > 1e-7:
                raise ContractError("RGB distortion inverse did not converge")
            direction = transform[:3, :3] @ np.array([xy[0], -xy[1], -1])
            return direction / np.linalg.norm(direction)

        def intersect(selected_pixels: Any) -> tuple[Any, float, float]:
            directions = [ray(pixel, t) for pixel, t in zip(selected_pixels, cameras)]
            d1, d2 = directions
            angle = math.degrees(math.acos(min(1, abs(float(d1 @ d2)))))
            if angle < 3:
                raise ContractError("insufficient RGB ray parallax")
            o1, o2 = (t[:3, 3] for t in cameras)
            distances = np.linalg.lstsq(np.column_stack([d1, -d2]), o2 - o1, rcond=None)[0]
            if min(distances) <= 0:
                raise ContractError("RGB intersection is behind a camera")
            p1, p2 = o1 + distances[0] * d1, o2 + distances[1] * d2
            return (p1 + p2) / 2, float(np.linalg.norm(p1 - p2)), angle

        point, gap, angle = intersect(pixels)
        if gap > 0.01:
            raise ContractError("RGB rays inconsistent with a static point")
        errors = []
        for transform, pixel in zip(cameras, pixels):
            local = np.linalg.inv(transform) @ np.append(point, 1)
            if local[2] >= -1e-5:
                raise ContractError("RGB intersection is behind or too close to a camera")
            normalized = np.array([local[0], -local[1]]) / -local[2]
            projected = self._distort(normalized, k) * [k["fx"], k["fy"]] + [k["cx"], k["cy"]]
            errors.append(float(np.linalg.norm(projected - pixel)))
        if not all(math.isfinite(error) for error in errors) or max(errors) > 3:
            raise ContractError("RGB reprojection inconsistent with a static point")
        shifts = [
            float(np.linalg.norm(intersect(pixels + np.array(signs).reshape(2, 2) * 2)[0] - point))
            for signs in itertools.product((-1, 1), repeat=4)
        ]
        return {
            "status": "hypothesis",
            "position_hypothesis_m": point.tolist(),
            "frame": "base_link",
            "position_in_current_wrist_m": (np.linalg.inv(wrists[1]) @ np.append(point, 1))[:3].tolist(),
            "camera_translation_m": baseline,
            "ray_angle_deg": angle,
            "ray_gap_m": gap,
            "reprojection_error_px": errors,
            "pixel_perturbation_px": 2,
            "max_sampled_perturbation_m": max(shifts),
            "stationarity_verified": False,
            "motion_executed": False,
            "meaning": "Position conditional on the SAME material point staying STATIC in base_link between measured poses and valid camera calibration. Small residual cannot verify stationarity or identity. Two-pixel corner perturbations measure sensitivity, not a confidence bound; calibration and pose errors are not included. This is a visible surface point, not an object/grasp center, contact, retention or task success.",
        }

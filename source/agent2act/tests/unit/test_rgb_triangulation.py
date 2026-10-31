# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import copy

import cv2
import numpy as np
import pytest

from agent2act.adapters.planners.rgb_triangulation import RgbRayGeometry
from agent2act.domain.models import ContractError


def calibration():
    return {
        "hand_left": {
            "camera_to_wrist": np.eye(4).tolist(),
            "intrinsics": {
                "fx": 400,
                "fy": 410,
                "cx": 319.5,
                "cy": 239.5,
                "width": 640,
                "height": 480,
                "k1": 0,
                "k2": 0,
                "k3": 0,
                "p1": 0,
                "p2": 0,
            },
        }
    }


def pixels(point, translation, intrinsic):
    # Independent OpenCV forward projection: USD optical y up/z back -> OpenCV y down/z forward.
    p = (np.array(point, dtype=float) - translation) * [1, -1, -1]
    k = intrinsic
    matrix = np.array([[k["fx"], 0, k["cx"]], [0, k["fy"], k["cy"]], [0, 0, 1]], float)
    distortion = np.array([k[key] for key in ("k1", "k2", "p1", "p2", "k3")], float)
    projected, _ = cv2.projectPoints(p.reshape(1, 3), np.zeros(3), np.zeros(3), matrix, distortion)
    return (projected.ravel() / [639, 479] * 1000).tolist()


def locate(cal=None, point=(0.03, 0.02, -0.5), after_point=None, translation=(0.08, 0, 0)):
    cal = cal or calibration()
    k = cal["hand_left"]["intrinsics"]
    return RgbRayGeometry(cal).locate_static(
        "hand_left",
        [0, 0, 0, 0, 0, 0, 1],
        [*translation, 0, 0, 0, 1],
        pixels(point, [0, 0, 0], k),
        pixels(after_point or point, translation, k),
        (640, 480),
    )


@pytest.mark.parametrize("distorted", [False, True])
def test_known_static_surface_and_pixel_sensitivity(distorted):
    cal = calibration()
    if distorted:
        cal["hand_left"]["intrinsics"].update(k1=-0.04, k2=0.01, k3=0.001, p1=0.002, p2=-0.001)
    result = locate(cal)
    assert result["position_hypothesis_m"] == pytest.approx([0.03, 0.02, -0.5], abs=1e-6)
    assert result["position_in_current_wrist_m"] == pytest.approx([-0.05, 0.02, -0.5], abs=1e-6)
    assert max(result["reprojection_error_px"]) < 1e-5
    assert 0 < result["max_sampled_perturbation_m"] < 0.05
    assert result["stationarity_verified"] is False and result["motion_executed"] is False


def test_epipolar_motion_is_unobservable_and_must_remain_conditional():
    # A real point moving parallel to camera translation has a different, perfectly
    # reprojecting static solution. Never certify stationarity from small residual.
    result = locate(point=(0, 0, -0.5), after_point=(0.02, 0, -0.5))
    assert abs(result["position_hypothesis_m"][2] + 0.5) > 0.1
    assert max(result["reprojection_error_px"]) < 1e-5
    assert result["stationarity_verified"] is False
    assert "cannot verify stationarity" in result["meaning"]


def test_measured_wrist_rotation_and_fixed_camera_lever_arm():
    cal = calibration()
    k = cal["hand_left"]["intrinsics"]
    offset = np.array([0.01, 0.02, 0.03])
    for row, value in zip(cal["hand_left"]["camera_to_wrist"], offset):
        row[3] = value
    rotation, _ = cv2.Rodrigues(np.array([0.0, 0.0, np.pi / 2]))
    point = np.array([0.03, 0.02, -0.5])
    current = np.array([0.08, 0, 0])
    first_pixel = pixels(point - offset, [0, 0, 0], k)
    second_pixel = pixels(rotation.T @ (point - current) - offset, [0, 0, 0], k)
    result = RgbRayGeometry(cal).locate_static(
        "hand_left",
        [0, 0, 0, 0, 0, 0, 1],
        [*current, 0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)],
        first_pixel,
        second_pixel,
        (640, 480),
    )
    assert result["position_hypothesis_m"] == pytest.approx(point, abs=1e-6)


@pytest.mark.parametrize(
    "kwargs,reason",
    [
        ({"translation": (0, 0, 0)}, "translation"),
        ({"point": (0, 0, -10)}, "parallax"),
        ({"after_point": (0.03, 0.12, -0.5)}, "inconsistent"),
        ({"after_point": (0.2, 0.02, -0.5)}, "behind"),
    ],
)
def test_uninformative_and_inconsistent_rays_abstain(kwargs, reason):
    with pytest.raises(ContractError, match=reason):
        locate(**kwargs)


def test_calibration_is_explicit_validated_and_copied():
    cal = calibration()
    geometry = RgbRayGeometry(cal)
    cal["hand_left"]["intrinsics"]["fx"] = 0
    assert geometry.cameras["hand_left"]["k"]["fx"] == 400
    cases = [calibration() for _ in range(4)]
    cases[0]["hand_left"]["camera_to_wrist"][0][0] = -1
    cases[1]["hand_left"]["intrinsics"]["fx"] = float("nan")
    del cases[2]["hand_left"]["intrinsics"]["k1"]
    cases[3]["head"] = cases[3].pop("hand_left")
    for bad in [{}, *cases]:
        with pytest.raises(ContractError):
            RgbRayGeometry(bad)


def test_dimensions_coordinates_and_measured_pose_are_required():
    g = RgbRayGeometry(calibration())
    arguments = [
        "hand_left",
        [0, 0, 0, 0, 0, 0, 1],
        [0.08, 0, 0, 0, 0, 0, 1],
        [500, 500],
        [400, 500],
        (640, 480),
    ]
    for index, bad in [(0, "head"), (1, [0] * 7), (3, [True, 500]), (4, [1001, 500]), (5, (320, 240))]:
        modified = copy.deepcopy(arguments)
        modified[index] = bad
        with pytest.raises(ContractError):
            g.locate_static(*modified)

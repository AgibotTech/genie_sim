# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Collection-calibrated policy cameras; independent inspection views."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np


def calibration():
    return json.loads(Path(__file__).with_name("camera_calibration.json").read_text())


def policy_cameras():
    return deepcopy(calibration()["cameras"])


def inspection_cameras():
    # Retain the original engineering views for humans, outside the policy payload.
    return {
        "inspect_hand_left": {
            "body": "l_wrist",
            "position": [0.04, 0.08, 0.07],
            "forward": [1, 0, 0.3],
            "up": [0, 0, -1],
            "fovy": 75,
            "width": 1280,
            "height": 1056,
        },
        "inspect_hand_right": {
            "body": "r_wrist",
            "position": [0.04, -0.08, 0.07],
            "forward": [1, 0, 0.3],
            "up": [0, 0, -1],
            "fovy": 75,
            "width": 1280,
            "height": 1056,
        },
    }


def camera_state(model, data, camera):
    rotation = data.xmat[model.body(camera["body"]).id].reshape(3, 3)
    eye = data.xpos[model.body(camera["body"]).id] + rotation @ np.asarray(camera["position"])
    forward = rotation @ np.asarray(camera["forward"], dtype=float)
    up = rotation @ np.asarray(camera["up"], dtype=float)
    forward /= np.linalg.norm(forward)
    up -= forward * np.dot(forward, up)
    up /= np.linalg.norm(up)
    if not np.isfinite(np.r_[eye, forward, up]).all():
        raise ValueError("Invalid camera basis")
    return {"eye": eye.tolist(), "forward": forward.tolist(), "up": up.tolist()}


def projection_parameters(camera, focal=24.0):
    """USD physical apertures and lens shifts for a pixel-centred pinhole K."""
    width, height = camera["width"], camera["height"]
    for value in (width, height):
        if type(value) is not int or not 1 <= value <= 2048:
            raise ValueError("Camera dimensions must be integers in 1..2048")
    intr = camera.get("intrinsics")
    if intr is None:
        fy = height / (2 * np.tan(np.deg2rad(camera["fovy"]) / 2))
        intr = {"fx": fy, "fy": fy, "cx": width / 2, "cy": height / 2}
    if set(intr) != {"fx", "fy", "cx", "cy"} or not np.isfinite(list(intr.values())).all():
        raise ValueError("Camera requires finite fx, fy, cx, cy")
    fx, fy, cx, cy = (float(intr[key]) for key in ("fx", "fy", "cx", "cy"))
    if min(fx, fy) <= 0 or not 0 <= cx <= width or not 0 <= cy <= height:
        raise ValueError("Invalid camera focal length or principal point")
    horizontal, vertical = width * focal / fx, height * focal / fy
    return {
        "focal": focal,
        "horizontal": horizontal,
        "vertical": vertical,
        "horizontal_offset": (width / 2 - cx) * horizontal / width,
        "vertical_offset": (cy - height / 2) * vertical / height,
        "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
    }

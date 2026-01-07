# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import os

import omni.kit.commands
import omni.usd
from isaacsim.core.prims import SingleXFormPrim as XFormPrim
from pxr import Gf, Sdf, UsdLux, UsdShade

import numpy as np
from scipy.spatial.transform import Rotation


def batch_matrices_to_quaternions_scipy(pose_matrices):
    """
    Convert batch 4x4 pose matrices to quaternions using Scipy's Rotation library

    Args:
        pose_matrices: numpy array of shape (n, 4, 4)

    Returns:
        numpy array of shape (n, 4), each row is a quaternion [x, y, z, w] (Scipy default order)
    """
    # Extract rotation matrix part (n, 3, 3)
    rotation_matrices = pose_matrices[:, :3, :3]

    # Create Rotation object and convert to quaternion
    rot = Rotation.from_matrix(rotation_matrices)
    quaternions = rot.as_quat()  # Returns (n, 4) array, order is [x, y, z, w]

    return quaternions


def batch_matrices_to_quaternions_scipy_w_first(pose_matrices):
    """
    Use Scipy's Rotation library, return quaternions in [w, x, y, z] order

    Args:
        pose_matrices: numpy array of shape (n, 4, 4)

    Returns:
        numpy array of shape (n, 4), each row is a quaternion [w, x, y, z]
    """
    quaternions = batch_matrices_to_quaternions_scipy(pose_matrices)
    # Adjust order to [w, x, y, z]
    quaternions_w_first = np.zeros_like(quaternions)
    quaternions_w_first[:, 0] = quaternions[:, 3]  # w
    quaternions_w_first[:, 1:] = quaternions[:, :3]  # x, y, z
    return quaternions_w_first


class Light:
    def __init__(self, prim_path, stage, light_type, intensity, color, orientation, texture_file):
        self.prim_path = prim_path
        self.light_type = light_type
        self.stage = stage
        self.intensity = intensity
        self.color = color
        self.orientation = orientation
        base_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/data/" + texture_file
        for file in os.listdir(base_folder):
            if file.endswith(".hdr"):
                self.texture_file = os.path.join(base_folder, file)

    def initialize(self):
        # selection between different light types
        if self.light_type == "Dome":
            light = UsdLux.DomeLight.Define(self.stage, Sdf.Path(self.prim_path))
            light.CreateIntensityAttr(self.intensity)
            light.CreateColorTemperatureAttr(self.color)
            light.CreateTextureFileAttr().Set(Sdf.AssetPath(self.texture_file))
        elif self.light_type == "Sphere":
            light = UsdLux.SphereLight.Define(self.stage, Sdf.Path(self.prim_path))
            light.CreateIntensityAttr(self.intensity)
            light.CreateColorTemperatureAttr(self.color)
        elif self.light_type == "Disk":
            light = UsdLux.DiskLight.Define(self.stage, Sdf.Path(self.prim_path))
            light.CreateIntensityAttr(self.intensity)
            light.CreateColorTemperatureAttr(self.color)
        elif self.light_type == "Rect":
            light = UsdLux.RectLight.Define(self.stage, Sdf.Path(self.prim_path))
            light.CreateIntensityAttr(self.intensity)
            light.CreateColorTemperatureAttr(self.color)
        elif self.light_type == "Distant":
            light = UsdLux.DistantLight.Define(self.stage, Sdf.Path(self.prim_path))
            light.CreateIntensityAttr(self.intensity)
            light.CreateColorTemperatureAttr(self.color)

        light.CreateEnableColorTemperatureAttr().Set(True)
        lightPrim = XFormPrim(self.prim_path, orientation=self.orientation)

        return lightPrim

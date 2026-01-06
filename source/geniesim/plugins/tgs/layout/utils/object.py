# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import os
import trimesh
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import RegularGridInterpolator

from geniesim.utils.object import OmniObject

from .sdf import compute_sdf_from_obj, compute_sdf_from_obj_surface
from geniesim.utils.transform_utils import (
    farthest_point_sampling,
    get_bott_up_point,
    random_point,
)


def load_and_prepare_mesh(obj_path, up_axis):
    if not os.path.exists(obj_path):
        return None
    mesh = trimesh.load(obj_path, force="mesh")
    if "z" in up_axis:
        align_rotation = R.from_euler("xyz", [0, 180, 0], degrees=True).as_matrix()
    elif "y" in up_axis:
        align_rotation = R.from_euler("xyz", [-90, 180, 0], degrees=True).as_matrix()
    elif "x" in up_axis:
        align_rotation = R.from_euler("xyz", [0, 0, 90], degrees=True).as_matrix()
    else:
        align_rotation = R.from_euler("xyz", [-90, 180, 0], degrees=True).as_matrix()

    align_transform = np.eye(4)
    align_transform[:3, :3] = align_rotation
    mesh.apply_transform(align_transform)
    mesh.apply_scale(1000)
    return mesh


def setup_sdf(mesh):
    _, sdf_voxels = compute_sdf_from_obj_surface(mesh)
    # create callable sdf function with interpolation

    min_corner = mesh.bounds[0]
    max_corner = mesh.bounds[1]

    x = np.linspace(min_corner[0], max_corner[0], sdf_voxels.shape[0])
    y = np.linspace(min_corner[1], max_corner[1], sdf_voxels.shape[1])
    z = np.linspace(min_corner[2], max_corner[2], sdf_voxels.shape[2])
    sdf_func = RegularGridInterpolator((x, y, z), sdf_voxels, bounds_error=False, fill_value=0)
    return sdf_func


class LayoutObject(OmniObject):
    def __init__(self, obj_info, use_sdf=False, N_collision_points=60, **kwargs):
        super().__init__(name=obj_info["object_id"], **kwargs)

        obj_dir = obj_info["obj_path"]
        up_aixs = obj_info["upAxis"]
        if len(up_aixs) == 0:
            up_aixs = ["y"]

        self.mesh = load_and_prepare_mesh(obj_dir, up_aixs)

        if use_sdf:
            self.sdf = setup_sdf(self.mesh)

        if self.mesh is not None:
            mesh_points, _ = trimesh.sample.sample_surface(self.mesh, 2000)
            if mesh_points.shape[0] > N_collision_points:
                self.collision_points = farthest_point_sampling(mesh_points, N_collision_points)

            self.anchor_points = {}
            self.anchor_points["top"] = get_bott_up_point(mesh_points, 1.5, descending=False)
            self.anchor_points["buttom"] = get_bott_up_point(mesh_points, 1.5, descending=True)

            self.anchor_points["top"] = random_point(self.anchor_points["top"], 3)[np.newaxis, :]
            self.anchor_points["buttom"] = random_point(self.anchor_points["buttom"], 3)[np.newaxis, :]

            self.size = self.mesh.extents.copy()
        self.up_axis = up_aixs[0]

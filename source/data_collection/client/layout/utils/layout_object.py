# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import os

import numpy as np
import trimesh
from pxr import Usd, UsdGeom
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial.transform import Rotation as R

from client.layout.object import OmniObject
from client.layout.utils.func import get_bott_up_point, random_point
from client.layout.utils.sdf import compute_sdf_from_obj_surface
from common.base_utils.logger import logger
from common.base_utils.transform_utils import farthest_point_sampling


def load_mesh_from_usd(usd_path):
    """
    Extract all meshes from USD file and merge into a single trimesh object.

    Args:
        usd_path: USD file path

    Returns:
        trimesh.Trimesh object, or None if no mesh is found
    """
    stage = Usd.Stage.Open(usd_path)
    if not stage:
        return None

    all_vertices = []
    all_faces = []
    vertex_offset = 0
    adjusted_scale = 1.0

    def traverse_prims(prim):
        nonlocal vertex_offset, all_vertices, all_faces, adjusted_scale

        if prim.IsA(UsdGeom.Mesh):
            usd_mesh = UsdGeom.Mesh(prim)
            if prim.HasAttribute("xformOp:transform:transform1"):
                transform = prim.GetAttribute("xformOp:transform:transform1").Get()
                adjusted_scale = transform[0, 0]
            points = usd_mesh.GetPointsAttr().Get()
            indices = usd_mesh.GetFaceVertexIndicesAttr().Get()
            face_counts = usd_mesh.GetFaceVertexCountsAttr().Get()

            if points is None or indices is None or face_counts is None:
                return

            # Add vertices
            for point in points:
                all_vertices.append([point[0], point[1], point[2]])

            # Add faces (fan-style triangulation, consistent with generate_obj.py logic)
            idx = 0
            for face_count in face_counts:
                for i in range(1, face_count - 1):
                    all_faces.append(
                        [
                            indices[idx] + vertex_offset,
                            indices[idx + i] + vertex_offset,
                            indices[idx + i + 1] + vertex_offset,
                        ]
                    )
                idx += face_count

            vertex_offset += len(points)

        for child in prim.GetChildren():
            traverse_prims(child)

    root_prim = stage.GetPseudoRoot()
    for child in root_prim.GetChildren():
        traverse_prims(child)

    if not all_vertices or not all_faces:
        return None

    return (
        trimesh.Trimesh(vertices=np.array(all_vertices), faces=np.array(all_faces)),
        adjusted_scale,
    )


def load_and_prepare_mesh(usd_path, up_axis, scale=1.0):
    if not os.path.exists(usd_path):
        return None
    mesh, adjusted_scale = load_mesh_from_usd(usd_path)
    if mesh is None:
        return None
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
    mesh.apply_scale(scale * adjusted_scale)
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
        data_info_dir = obj_info["data_info_dir"]
        usd_path = os.path.join(os.environ.get("SIM_ASSETS"), data_info_dir, "Aligned.usd")
        up_aixs = obj_info["upAxis"]
        if len(up_aixs) == 0:
            up_aixs = ["y"]
        logger.info(f"usd_path: {usd_path}")
        mesh_scale = obj_info.get("scale", 0.001) * 1000
        self.up_side_down = obj_info.get("up_side_down", False)
        self.object_type = obj_info.get("type", "rigid_body")
        self.mesh = load_and_prepare_mesh(usd_path, up_aixs, mesh_scale)
        if use_sdf:
            self.sdf = setup_sdf(self.mesh)

        if (self.mesh is not None) and (self.object_type == "rigid_body"):
            mesh_points, _ = trimesh.sample.sample_surface(self.mesh, 2000)  # Surface sampling
            if mesh_points.shape[0] > N_collision_points:
                self.collision_points = farthest_point_sampling(
                    mesh_points, N_collision_points
                )  # Collision detection points
            self.anchor_points = {}
            self.anchor_points["top"] = get_bott_up_point(mesh_points, 1.5, descending=False)
            self.anchor_points["buttom"] = get_bott_up_point(mesh_points, 1.5, descending=True)
            self.anchor_points["top"] = random_point(self.anchor_points["top"], 3)[np.newaxis, :]
            self.anchor_points["buttom"] = random_point(self.anchor_points["buttom"], 3)[
                np.newaxis, :
            ]
            self.size = self.mesh.extents.copy()
        self.up_axis = up_aixs[0]

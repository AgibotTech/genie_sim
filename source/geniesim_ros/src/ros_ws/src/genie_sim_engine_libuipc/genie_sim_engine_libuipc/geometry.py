# Copyright (c) 2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Reusable USD mesh and robot-collision helpers for SolverIPC scenes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


def _matrix_to_quat_xyzw(rotation: np.ndarray) -> np.ndarray:
    """Convert a row-vector rotation matrix to an xyzw quaternion."""
    m = np.asarray(rotation, dtype=np.float64).T
    trace = float(np.trace(m))
    if trace > 0.0:
        s = 0.5 / math.sqrt(trace + 1.0)
        quat = [(m[2, 1] - m[1, 2]) * s, (m[0, 2] - m[2, 0]) * s, (m[1, 0] - m[0, 1]) * s, 0.25 / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        quat = [0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        quat = [(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
    else:
        s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        quat = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s]
    result = np.asarray(quat, dtype=np.float64)
    return result / np.linalg.norm(result)


@dataclass(frozen=True)
class IPCMesh:
    """Triangulated USD mesh and its local-to-world transform."""

    path: str
    points: np.ndarray
    triangles: np.ndarray
    xform: np.ndarray
    tets: np.ndarray | None = None

    def world_points(self) -> np.ndarray:
        """Return world-space points."""
        return np.hstack((self.points, np.ones((len(self.points), 1)))) @ self.xform[:, :3]

    def decompose(self) -> tuple[np.ndarray, np.ndarray, float]:
        """Return translation, xyzw rotation, and uniform scale."""
        position, rotation, scales = self.decompose_axis_scaled()
        if not np.allclose(scales, scales[0], rtol=1.0e-6, atol=1.0e-9):
            raise ValueError(f"{self.path} does not have uniform scale: {scales}")
        return position, rotation, float(scales[0])

    def decompose_axis_scaled(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return translation [m], xyzw rotation and positive orthogonal axis scales."""
        scales = np.linalg.norm(self.xform[:3, :3], axis=1)
        if not np.isfinite(self.xform).all() or np.any(scales <= 0):
            raise ValueError(f"{self.path} has a degenerate or non-finite transform")
        rotation = self.xform[:3, :3] / scales[:, None]
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1.0e-6) or np.linalg.det(rotation) <= 0:
            raise ValueError(f"{self.path} has an unsupported shear or reflection")
        return self.xform[3, :3].copy(), _matrix_to_quat_xyzw(rotation), scales


def read_usd_mesh(stage: Any, path: str) -> IPCMesh:
    """Read and triangulate a USD Mesh at ``path``."""
    from pxr import Usd, UsdGeom

    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid() or not prim.IsA(UsdGeom.Mesh):
        raise KeyError(f"{path} is not a Mesh in {stage.GetRootLayer().identifier}")
    mesh = UsdGeom.Mesh(prim)
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
    triangles = []
    offset = 0
    for count in counts:
        face = indices[offset : offset + count]
        offset += int(count)
        triangles.extend((face[0], face[index], face[index + 1]) for index in range(1, int(count) - 1))
    xform = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()), dtype=np.float64)
    tets = None
    tet_values = prim.GetAttribute("tetVertexIndices").Get()
    if tet_values is not None:
        tets = np.asarray(tet_values)
        if tets.ndim != 2 or tets.shape[1] != 4 or not len(tets):
            raise ValueError(f"{path} tetVertexIndices must be a nonempty array of four-index tetrahedra")
        if not np.issubdtype(tets.dtype, np.integer):
            raise ValueError(f"{path} tetVertexIndices must contain integer indices")
        if np.any(tets < 0) or np.any(tets >= len(points)):
            raise ValueError(f"{path} tetVertexIndices references points outside the mesh")
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError(f"{path} tet mesh points must be finite three-vectors")
        ordered = np.sort(tets, axis=1)
        if np.any(np.diff(ordered, axis=1) == 0) or len(np.unique(ordered, axis=0)) != len(tets):
            raise ValueError(f"{path} has repeated tet vertices or duplicate tetrahedra")
        if len(np.unique(tets)) != len(points):
            raise ValueError(f"{path} has points unused by any tetrahedron")
        edges = points[tets[:, 1:]] - points[tets[:, 0, None]]
        volume6 = np.linalg.det(edges)
        if not np.isfinite(volume6).all() or np.any(volume6 <= 0):
            raise ValueError(f"{path} has nonpositive or nonfinite tetrahedron volume")
        tets = tets.astype(np.int32)
    return IPCMesh(path, points, np.asarray(triangles, dtype=np.int32), xform, tets)


def select_robot_collision_bodies(
    body_labels: Iterable[str], *, robot_prefix: str, collision_link_names: Iterable[str]
) -> set[int]:
    """Return robot body rows whose link suffix is enabled for collision."""
    prefix = f"/{robot_prefix}/" if robot_prefix else ""
    active = set(str(name) for name in collision_link_names)
    return {
        index
        for index, label in enumerate(body_labels)
        if (not prefix or str(label).startswith(prefix)) and str(label).rsplit("/", 1)[-1] in active
    }


def configure_kinematic_robot_collision(
    builder: Any, *, robot_prefix: str, collision_link_names: Iterable[str], logger: Any
) -> None:
    """Make robot bodies kinematic and keep collision only on selected links."""
    import newton

    prefix = f"/{robot_prefix}/" if robot_prefix else ""
    robot_bodies = {
        index for index, label in enumerate(builder.body_label) if not prefix or str(label).startswith(prefix)
    }
    active_bodies = select_robot_collision_bodies(
        builder.body_label, robot_prefix=robot_prefix, collision_link_names=collision_link_names
    )
    for index in robot_bodies:
        builder.body_flags[index] = int(newton.BodyFlags.KINEMATIC)
    collision_mask = int(newton.ShapeFlags.COLLIDE_SHAPES | newton.ShapeFlags.COLLIDE_PARTICLES)
    active_shapes = []
    for shape, body in enumerate(builder.shape_body):
        if body not in robot_bodies:
            continue
        if body not in active_bodies:
            builder.shape_flags[shape] = int(builder.shape_flags[shape]) & ~collision_mask
        elif int(builder.shape_flags[shape]) & int(newton.ShapeFlags.COLLIDE_SHAPES):
            active_shapes.append(shape)
    if active_shapes:
        builder.approximate_meshes(shape_indices=active_shapes, method="convex_hull")
    logger.info(
        f"[ipc-scene] robot bodies={len(robot_bodies)} collision bodies={len(active_bodies)} hull shapes={len(active_shapes)}"
    )


__all__ = ["IPCMesh", "configure_kinematic_robot_collision", "read_usd_mesh", "select_robot_collision_bodies"]

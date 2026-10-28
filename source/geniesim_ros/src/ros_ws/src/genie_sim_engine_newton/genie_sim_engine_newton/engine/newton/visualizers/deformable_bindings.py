# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Validation and coordinate helpers for authored deformable Mesh bindings."""

from __future__ import annotations

from typing import Any, Mapping, NamedTuple, Sequence

import numpy as np


class DeformableRenderBinding(NamedTuple):
    """One authored Mesh and its fixed Newton-particle reconstruction map."""

    prim_path: str
    global_particle_indices: np.ndarray
    barycentric_weights: np.ndarray


def world_points_to_mesh_local(
    world_points: Any,
    *,
    parent_pose_world: Any,
    mesh_pose_parent: Any,
) -> np.ndarray:
    """Convert world-space vertices to a USD Mesh's local coordinates.

    Matrices use the ROS/dataset column-vector convention::

        p_world = parent_pose_world @ mesh_pose_parent @ p_mesh_local

    Newton's FEM particles and the OVRtx reconstruction are world-space, while
    a USD Mesh ``points`` attribute is local to the Mesh prim.  Keeping this
    conversion beside the shared binding validation gives rendering and
    recording one explicit coordinate contract.
    """
    points = np.asarray(world_points, dtype=np.float64)
    parent_to_world = np.asarray(parent_pose_world, dtype=np.float64)
    mesh_to_parent = np.asarray(mesh_pose_parent, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"world_points must have shape (N, 3); got {points.shape}")
    if parent_to_world.shape != (4, 4) or mesh_to_parent.shape != (4, 4):
        raise ValueError("parent_pose_world and mesh_pose_parent must both have shape (4, 4)")
    if not (np.isfinite(points).all() and np.isfinite(parent_to_world).all() and np.isfinite(mesh_to_parent).all()):
        raise ValueError("deformable points and transforms must be finite")

    mesh_to_world = parent_to_world @ mesh_to_parent
    try:
        world_to_mesh = np.linalg.inv(mesh_to_world)
    except np.linalg.LinAlgError as exc:
        raise ValueError("deformable Mesh transform must be invertible") from exc
    local = points @ world_to_mesh[:3, :3].T + world_to_mesh[:3, 3]
    return np.asarray(local, dtype=np.float32)


def prepare_deformable_bindings(
    bindings: Sequence[Mapping[str, Any]], particle_count: int
) -> list[DeformableRenderBinding]:
    """Validate plugin bindings and convert local particle IDs to global IDs."""
    particle_count = int(particle_count)
    if particle_count < 0:
        raise ValueError("model particle count must be non-negative")

    prepared = []
    seen_paths = set()
    for binding in bindings:
        prim_path = str(binding.get("mesh_prim_path", ""))
        if not prim_path.startswith("/"):
            raise ValueError(f"deformable Mesh prim path must be absolute: {prim_path!r}")
        if prim_path in seen_paths:
            raise ValueError(f"duplicate deformable Mesh prim path: {prim_path}")
        seen_paths.add(prim_path)

        if binding.get("mapping_kind") != "tet_barycentric_v1":
            raise ValueError(
                f"deformable binding {prim_path} has unsupported mapping kind " f"{binding.get('mapping_kind')!r}"
            )
        start = int(binding.get("particle_start", -1))
        count = int(binding.get("particle_count", -1))
        vertex_count = int(binding.get("vertex_count", -1))
        if start < 0 or count <= 0 or vertex_count <= 0:
            raise ValueError(f"deformable binding {prim_path} has invalid counts")
        if start + count > particle_count:
            raise ValueError(
                f"deformable binding {prim_path} exceeds model particle count " f"({start}+{count}>{particle_count})"
            )

        local_indices = np.asarray(binding.get("tet_particle_indices"), dtype=np.int32)
        weights = np.asarray(binding.get("barycentric_weights"), dtype=np.float32)
        expected_shape = (vertex_count, 4)
        if local_indices.shape != expected_shape or weights.shape != expected_shape:
            raise ValueError(
                f"deformable binding {prim_path} mapping arrays must have shape "
                f"{expected_shape}; got {local_indices.shape} and {weights.shape}"
            )
        if np.any(local_indices < 0) or np.any(local_indices >= count):
            raise ValueError(f"deformable binding {prim_path} indices lie outside its particle slice")
        if not np.isfinite(weights).all() or np.any(weights < -1.0e-6):
            raise ValueError(f"deformable binding {prim_path} has invalid barycentric weights")
        if not np.allclose(weights.sum(axis=1), 1.0, atol=1.0e-5):
            raise ValueError(f"deformable binding {prim_path} has invalid barycentric weights")

        prepared.append(
            DeformableRenderBinding(
                prim_path=prim_path,
                global_particle_indices=np.ascontiguousarray(local_indices + start),
                barycentric_weights=np.ascontiguousarray(weights),
            )
        )
    return prepared

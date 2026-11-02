# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Bind authored USD object roots to runtime body names, without the recording stack.

`IPCYamlScene.on_stage` needs `body_bindings` to record each task object's authored
root prim and the mesh->root pose offset. On the donor branch that function lives in
`genie_sim_engine_newton/common/recording_schema.py`, whose other contents (canonical
replay frames, delivery/merge helpers) belong to the recording subsystem that this
migration deliberately leaves behind.

`body_bindings` and its one helper `rigid_matrix` are copied verbatim from that module
(donor `2dee24bab1`, `common/recording_schema.py:30-61`) so the bindings stay correct
rather than silently empty. Nothing here depends on recording.

If `common/recording_schema.py` is migrated later, reconcile these two definitions
instead of keeping two copies: import them from there and delete this module.
"""

from __future__ import annotations

import numpy as np


def rigid_matrix(matrix):
    """Remove authored scale while retaining a USD prim's world pose."""
    result = np.asarray(matrix, dtype=float).copy()
    scale = np.linalg.norm(result[:3, :3], axis=0)
    if np.any(scale < 1e-12):
        raise ValueError("Singular recorded object transform")
    result[:3, :3] /= scale
    if not np.allclose(result[:3, :3].T @ result[:3, :3], np.eye(3), atol=1e-5):
        raise ValueError("Sheared recorded object transform")
    return result


def body_bindings(stage, objects):
    """Bind runtime body names to the authored object root and pose offset."""
    from pxr import UsdGeom

    cache = UsdGeom.XformCache()
    bindings = {}
    for obj in objects:
        if obj["kind"] != "rigid_mesh":
            continue
        mesh_path = obj["mesh_prim_path"]
        root_path = obj.get("body_prim_path", mesh_path)
        mesh, root = stage.GetPrimAtPath(mesh_path), stage.GetPrimAtPath(root_path)
        if not mesh or not root:
            raise ValueError(f"Missing recording body/mesh prim: {root_path}, {mesh_path}")
        if not mesh.GetPath().HasPrefix(root.GetPath()):
            raise ValueError(f"body_prim_path must be an ancestor of its mesh: {root_path}, {mesh_path}")
        mesh_world = rigid_matrix(np.asarray(cache.GetLocalToWorldTransform(mesh)).T)
        root_world = rigid_matrix(np.asarray(cache.GetLocalToWorldTransform(root)).T)
        bindings[obj["name"]] = (root_path, (np.linalg.inv(mesh_world) @ root_world).tolist())
    return bindings

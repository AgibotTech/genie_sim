# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Read composed USD mesh geometry without launching Isaac Sim."""

from __future__ import annotations

from pathlib import Path


def inspect_usd(path: str | Path) -> dict:
    from pxr import Usd, UsdGeom

    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    stage = Usd.Stage.Open(str(source))
    if stage is None:
        raise ValueError(f"Cannot open USD stage: {source}")
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy], useExtentsHint=False)
    bounds = []
    mesh_paths = []
    vertex_count = face_count = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        mesh_paths.append(str(prim.GetPath()))
        vertex_count += len(mesh.GetPointsAttr().Get() or [])
        face_count += len(mesh.GetFaceVertexCountsAttr().Get() or [])
        box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if not box.IsEmpty():
            bounds.append(box)
    result = {
        "usd": str(source), "mesh_count": len(mesh_paths), "mesh_paths": mesh_paths,
        "vertex_count": vertex_count, "face_count": face_count,
        "size_m": None, "bbox_min_m": None, "bbox_max_m": None,
    }
    if bounds:
        minimum = [min(float(box.GetMin()[axis]) for box in bounds) for axis in range(3)]
        maximum = [max(float(box.GetMax()[axis]) for box in bounds) for axis in range(3)]
        scale = float(UsdGeom.GetStageMetersPerUnit(stage))
        result["bbox_min_m"] = [round(x * scale, 6) for x in minimum]
        result["bbox_max_m"] = [round(x * scale, 6) for x in maximum]
        result["size_m"] = [round((maximum[i] - minimum[i]) * scale, 6) for i in range(3)]
    return result

# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Locate horizontal placement surfaces in cabinets, shelves, and tables."""

from __future__ import annotations

import json
import os
from pathlib import Path

from pxr import Gf, Usd, UsdGeom


def _round_bounds(corners):
    xs = [point[0] for point in corners]
    ys = [point[1] for point in corners]
    return [[round(min(xs), 6), round(min(ys), 6)],
            [round(max(xs), 6), round(max(ys), 6)]]


def _find_layers_json(usd_path: Path, prim_path: str) -> Path | None:
    asset = usd_path
    if "Objects" in prim_path and (usd_path.parent / "scene_info.json").is_file():
        info = json.loads((usd_path.parent / "scene_info.json").read_text(encoding="utf-8"))
        object_id = prim_path.split("/World/Objects/", 1)[-1].split("/", 1)[0]
        asset_path = info.get("layout", {}).get(object_id, {}).get("asset")
        if asset_path:
            asset = Path(asset_path)
    for parent in asset.parents:
        if parent.name == "assets":
            candidate = parent / "interaction" / asset.parent.name / "layers.json"
            if candidate.is_file():
                return candidate
            break
    return None


def _from_layers_json(stage, prim_path: str, layers_path: Path) -> list[dict]:
    data = json.loads(layers_path.read_text(encoding="utf-8"))
    reference = data.get("reference_prim", "/World/entity")
    if not reference.startswith("/World/"):
        raise ValueError(f"Unsupported layer reference_prim: {reference}")
    mapped = prim_path.rstrip("/") + reference[len("/World"):]
    entity = stage.GetPrimAtPath(mapped)
    if not entity:
        raise ValueError(f"Layer reference prim not found: {mapped}")
    matrix = UsdGeom.XformCache().GetLocalToWorldTransform(entity)
    surfaces = []
    for layer in data.get("layers", []):
        corners = layer.get("corners_local", [])
        if len(corners) != 4:
            raise ValueError(f"Layer {layer.get('index')} needs four corners")
        world = [matrix.Transform(Gf.Vec3d(*map(float, corner))) for corner in corners]
        points = [[round(float(point[axis]), 6) for axis in range(3)] for point in world]
        height = sum(point[2] for point in points) / 4
        if max(abs(point[2] - height) for point in points) > 0.02:
            raise ValueError(f"Layer {layer.get('index')} is not horizontal in world coordinates")
        surfaces.append({"index": int(layer.get("index", len(surfaces))),
                         "kind": layer.get("kind", "unknown"),
                         "height_m": round(height, 6),
                         "bounds_xy_m": _round_bounds(points),
                         "corners_world_m": points,
                         "area_m2": round(float(layer.get("length", 0)) * float(layer.get("width", 0)), 6)})
    return sorted(surfaces, key=lambda item: item["height_m"])


def _from_mesh_faces(stage, prim, *, height_tolerance: float, min_area: float) -> list[dict]:
    cache = UsdGeom.XformCache()
    candidates = []
    for child in Usd.PrimRange(prim):
        if not child.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(child)
        points = mesh.GetPointsAttr().Get() or []
        counts = mesh.GetFaceVertexCountsAttr().Get() or []
        indices = mesh.GetFaceVertexIndicesAttr().Get() or []
        transform = cache.GetLocalToWorldTransform(child)
        vertices = [transform.Transform(Gf.Vec3d(*map(float, point))) for point in points]
        offset = 0
        for count in counts:
            face = indices[offset:offset + count]
            offset += count
            if count < 3:
                continue
            anchor = vertices[face[0]]
            for j in range(1, count - 1):
                a, b, c = anchor, vertices[face[j]], vertices[face[j + 1]]
                if max(a[2], b[2], c[2]) - min(a[2], b[2], c[2]) > height_tolerance:
                    continue
                cross = Gf.Cross(b - a, c - a)
                area = float(cross.GetLength()) / 2
                if area < 1e-8 or float(cross[2]) / (2 * area) < 0.85:
                    continue
                candidates.append({"z": float((a[2] + b[2] + c[2]) / 3),
                                   "area": area, "points": [a, b, c]})
    groups: list[dict] = []
    for face in sorted(candidates, key=lambda item: item["z"]):
        if groups and abs(groups[-1]["z"] - face["z"]) <= height_tolerance:
            group = groups[-1]
            group["area"] += face["area"]
            group["points"].extend(face["points"])
        else:
            groups.append({"z": face["z"], "area": face["area"], "points": list(face["points"])})
    surfaces = []
    for group in groups:
        points = group["points"]
        bounds = _round_bounds(points)
        width = bounds[1][0] - bounds[0][0]
        depth = bounds[1][1] - bounds[0][1]
        if group["area"] < min_area or width < 0.05 or depth < 0.05:
            continue
        height = round(group["z"], 6)
        x0, y0 = bounds[0]
        x1, y1 = bounds[1]
        surfaces.append({"index": len(surfaces), "kind": "candidate_plank",
                         "height_m": height, "bounds_xy_m": bounds,
                         "corners_world_m": [[x0, y0, height], [x1, y0, height],
                                             [x1, y1, height], [x0, y1, height]],
                         "area_m2": round(group["area"], 6)})
    return surfaces


def inspect_supports(usd_path: str | Path, prim_path: str | None = None, *,
                     layers_json: str | Path | None = None, auto_layers: bool = True,
                     height_tolerance: float = 0.01, min_area: float = 0.01) -> dict:
    """Prefer reviewed layers.json; otherwise report upward horizontal mesh faces."""
    source = Path(usd_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    stage = Usd.Stage.Open(str(source), load=Usd.Stage.LoadAll)
    if stage is None:
        raise ValueError(f"Cannot open USD: {source}")
    prim_path = prim_path or str(stage.GetDefaultPrim().GetPath())
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        raise ValueError(f"Prim not found: {prim_path}")
    layers_path = Path(layers_json).resolve() if layers_json else (_find_layers_json(source, prim_path) if auto_layers else None)
    if layers_path is not None:
        if not layers_path.is_file():
            raise FileNotFoundError(layers_path)
        surfaces = _from_layers_json(stage, prim_path, layers_path)
        kind = "layers_json"
    else:
        surfaces = _from_mesh_faces(stage, prim, height_tolerance=height_tolerance, min_area=min_area)
        kind = "mesh_faces"
    return {"usd": str(source), "prim": prim_path, "source": kind,
            "layers_json": str(layers_path) if layers_path else None,
            "surfaces": surfaces}


def write_support_markers(source_usd: str | Path, surfaces: list[dict], output_usd: str | Path) -> Path:
    """Create a separate preview layer; never edit the scene or source asset."""
    source = Path(source_usd).resolve()
    output = Path(output_usd).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    world.GetPrim().GetReferences().AddReference(os.path.relpath(source, output.parent))
    colors = ((1, 0, 0), (0, 0.7, 0), (0, 0.3, 1), (1, 0.7, 0), (0.7, 0, 1))
    for index, surface in enumerate(surfaces):
        corners = surface["corners_world_m"]
        for corner_index, position in enumerate(corners):
            marker = UsdGeom.Cube.Define(stage, f"/World/SupportMarkers/surface_{index:03d}/corner_{corner_index}")
            marker.CreateSizeAttr(0.03)
            marker.CreateDisplayColorAttr([Gf.Vec3f(*colors[index % len(colors)])])
            marker.AddTranslateOp().Set(Gf.Vec3d(*map(float, position)))
    stage.GetRootLayer().Save()
    return output

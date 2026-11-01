# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Validate final scene artifacts without changing the scene stage."""

from __future__ import annotations

import json
import math
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from pxr import Usd, UsdGeom


def validate_scene(folder: str | Path, *, require_renders: bool = False) -> dict:
    folder = Path(folder).resolve()
    errors: list[str] = []
    warnings: list[str] = []
    scene_path = folder / "scene.usda"
    info_path = folder / "scene_info.json"
    if not scene_path.is_file():
        errors.append("scene.usda is missing")
    if not info_path.is_file():
        errors.append("scene_info.json is missing")
    if errors:
        return {"ok": False, "errors": errors, "warnings": warnings, "object_count": 0}
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"ok": False, "errors": [f"Invalid scene_info.json: {exc}"], "warnings": [], "object_count": 0}
    stage = Usd.Stage.Open(str(scene_path), load=Usd.Stage.LoadAll)
    if stage is None:
        return {"ok": False, "errors": ["Cannot open scene.usda"], "warnings": [], "object_count": 0}
    if stage.GetDefaultPrim().GetPath().pathString != "/World":
        errors.append("defaultPrim must be /World")
    if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
        errors.append("Stage up axis must be Z")
    if not math.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1.0):
        errors.append("Stage must use meters")
    objects = stage.GetPrimAtPath("/World/Objects")
    actual_ids = {child.GetName() for child in objects.GetChildren()} if objects else set()
    # An empty ink container is runtime state, rather than a scene asset.
    auxiliary_ids = set()
    auxiliary_paths = info.get("auxiliary_prims", [])
    if not isinstance(auxiliary_paths, list):
        errors.append("auxiliary_prims must be a list of direct /World/Objects child paths")
        auxiliary_paths = []
    for path in auxiliary_paths:
        if not isinstance(path, str) or not path.startswith("/World/Objects/") or path.count("/") != 3:
            errors.append(f"Invalid auxiliary prim path: {path}")
            continue
        prim = stage.GetPrimAtPath(path)
        if not prim or not (prim.IsA(UsdGeom.Xform) or prim.IsA(UsdGeom.Scope)):
            errors.append(f"Missing or invalid auxiliary container: {path}")
            continue
        if any(child.IsA(UsdGeom.Boundable) for child in Usd.PrimRange(prim)):
            errors.append(f"Auxiliary container must be empty of geometry: {path}")
            continue
        auxiliary_ids.add(prim.GetName())
    layout = info.get("layout", {})
    expected_ids = set(layout)
    if actual_ids - auxiliary_ids != expected_ids or auxiliary_ids & expected_ids:
        errors.append(f"Object IDs differ: stage={sorted(actual_ids)}, layout={sorted(expected_ids)}")
    background = stage.GetPrimAtPath("/World/Background")
    actual_background = {child.GetName() for child in background.GetChildren()} if background else set()
    expected_background = set(info.get("background", {}))
    if actual_background != expected_background:
        errors.append(f"Background IDs differ: stage={sorted(actual_background)}, metadata={sorted(expected_background)}")
    for background_id, entry in info.get("background", {}).items():
        if not Path(entry.get("asset", "")).is_file():
            errors.append(f"Missing background asset for {background_id}")
    bounds = {}
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy])
    for object_id, entry in layout.items():
        asset = entry.get("asset")
        if asset and not Path(asset).is_file():
            errors.append(f"Missing asset payload for {object_id}: {asset}")
        prim = stage.GetPrimAtPath(f"/World/Objects/{object_id}")
        if not prim:
            continue
        pose = entry.get("xyz", []) + entry.get("xyzw", [])
        if len(pose) != 7 or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in pose):
            errors.append(f"Invalid pose for {object_id}")
        else:
            translate = prim.GetAttribute("xformOp:translate").Get()
            orient = prim.GetAttribute("xformOp:orient").Get()
            if translate is None or any(abs(float(translate[axis]) - pose[axis]) > 1e-3 for axis in range(3)):
                errors.append(f"Position in scene_info differs from USD pose for {object_id}")
            if orient is None:
                errors.append(f"Orientation missing from USD pose for {object_id}")
            else:
                usd_quat = [float(value) for value in orient.GetImaginary()] + [float(orient.GetReal())]
                dot = abs(sum(usd_quat[axis] * pose[axis + 3] for axis in range(4)))
                if abs(dot - 1.0) > 1e-3:
                    errors.append(f"Orientation in scene_info differs from USD pose for {object_id}")
        if not any(child.IsA(UsdGeom.Boundable) for child in Usd.PrimRange(prim)):
            errors.append(f"No loaded geometry for {object_id}; check payload")
            continue
        box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if box.IsEmpty():
            warnings.append(f"Empty world bounds for {object_id}")
        else:
            bounds[object_id] = {"min": list(map(float, box.GetMin())), "max": list(map(float, box.GetMax()))}
    if require_renders:
        for view in ("front", "side", "top"):
            image_path = folder / "renders" / f"{view}.png"
            if not image_path.is_file():
                errors.append(f"Missing Isaac Sim render: {view}.png")
                continue
            try:
                with Image.open(image_path) as image:
                    extrema = image.convert("RGB").getextrema()
                    if all(maximum - minimum <= 2 for minimum, maximum in extrema):
                        errors.append(f"Blank Isaac Sim render: {view}.png")
            except (OSError, UnidentifiedImageError) as exc:
                errors.append(f"Unreadable Isaac Sim render: {view}.png: {exc}")
    return {"ok": not errors, "errors": errors, "warnings": warnings,
            "object_count": len(actual_ids), "bounds": bounds}

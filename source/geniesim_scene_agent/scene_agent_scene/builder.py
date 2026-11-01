# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Canonical foreground scene builder and artifact writer."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics


def canonical_object_name(name: str, index: int) -> str:
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9 _-]*", name) or index < 0:
        raise ValueError(f"Invalid object name or index: {name!r}, {index}")
    stem = re.sub(r"[_ -]+", "_", name.strip()).lower().strip("_")
    return f"{stem}_{index:03d}"


def validate_scene_id(scene_id: str) -> str:
    if not re.fullmatch(r"[a-z][a-z0-9_]*", scene_id):
        raise ValueError("scene_id must be lowercase snake_case")
    return scene_id


class SceneBuilder:
    """Add foreground objects to one stage; object IDs are stable by role."""

    def __init__(self, output_dir: str | Path, scene_id: str, seed: int):
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.scene_id = validate_scene_id(scene_id)
        self.seed = int(seed)
        self.stage = Usd.Stage.CreateNew(str(self.output_dir / "scene.usda"))
        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(self.stage, 1.0)
        world = UsdGeom.Xform.Define(self.stage, "/World")
        self.stage.SetDefaultPrim(world.GetPrim())
        UsdGeom.Scope.Define(self.stage, "/World/Objects")
        self.layout: dict[str, dict] = {}
        self.background: dict[str, dict] = {}
        self._counts: dict[str, int] = {}
        self._background_counts: dict[str, int] = {}

    def _object_id(self, name: str) -> str:
        stem = canonical_object_name(name, 0)[:-4]
        count = self._counts.get(stem, 0)
        self._counts[stem] = count + 1
        return canonical_object_name(name, count)

    def _record(self, object_id: str, asset_id: str, xyz, xyzw, keywords, description, asset_path: str | None):
        self.layout[object_id] = {
            "usd": asset_id,
            "xyz": [float(value) for value in xyz],
            "xyzw": [float(value) for value in xyzw],
            "tags": ["root_scene", object_id],
            "keywords": list(keywords or []),
            "description": description or {},
        }
        if asset_path:
            self.layout[object_id]["asset"] = asset_path

    def add_asset(self, name: str, asset_usd: str | Path, *, xyz=(0, 0, 0),
                  xyzw=(0, 0, 0, 1), scale=(1, 1, 1), keywords=(), description=None) -> str:
        asset = Path(asset_usd).expanduser().resolve()
        if not asset.is_file():
            raise FileNotFoundError(asset)
        if len(xyz) != 3 or len(xyzw) != 4 or len(scale) != 3:
            raise ValueError("Expected xyz[3], xyzw[4], scale[3]")
        object_id = self._object_id(name)
        xform = UsdGeom.Xform.Define(self.stage, f"/World/Objects/{object_id}")
        relative = os.path.relpath(asset, self.output_dir)
        xform.GetPrim().GetPayloads().AddPayload(relative)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(*map(float, xyz)))
        xform.AddOrientOp().Set(Gf.Quatf(float(xyzw[3]), *map(float, xyzw[:3])))
        xform.AddScaleOp().Set(Gf.Vec3f(*map(float, scale)))
        self._record(object_id, asset.parent.name, xyz, xyzw, keywords, description, str(asset))
        return object_id

    def add_box(self, name: str, *, size=(0.1, 0.1, 0.1), xyz=(0, 0, 0),
                xyzw=(0, 0, 0, 1), keywords=(), description=None) -> str:
        """Author simple geometry when the DSL or asset library cannot express it."""
        if len(size) != 3 or any(value <= 0 for value in size):
            raise ValueError("size must be three positive dimensions")
        object_id = self._object_id(name)
        xform = UsdGeom.Xform.Define(self.stage, f"/World/Objects/{object_id}")
        xform.AddTranslateOp().Set(Gf.Vec3d(*map(float, xyz)))
        xform.AddOrientOp().Set(Gf.Quatf(float(xyzw[3]), *map(float, xyzw[:3])))
        cube = UsdGeom.Cube.Define(self.stage, f"/World/Objects/{object_id}/mesh")
        cube.CreateSizeAttr(1.0)
        cube.AddScaleOp().Set(Gf.Vec3f(*map(float, size)))
        self._record(object_id, f"generated:{object_id}", xyz, xyzw, keywords, description, None)
        return object_id

    def add_background(self, name: str, asset_usd: str | Path, *, xyz=(0, 0, 0),
                       xyzw=(0, 0, 0, 1)) -> str:
        """Add requested background geometry, separate from foreground roles."""
        asset = Path(asset_usd).expanduser().resolve()
        if not asset.is_file():
            raise FileNotFoundError(asset)
        stem = canonical_object_name(name, 0)[:-4]
        index = self._background_counts.get(stem, 0)
        self._background_counts[stem] = index + 1
        object_id = canonical_object_name(name, index)
        prim = UsdGeom.Xform.Define(self.stage, f"/World/Background/{object_id}")
        prim.GetPrim().GetPayloads().AddPayload(os.path.relpath(asset, self.output_dir))
        prim.ClearXformOpOrder()
        prim.AddTranslateOp().Set(Gf.Vec3d(*map(float, xyz)))
        prim.AddOrientOp().Set(Gf.Quatf(float(xyzw[3]), *map(float, xyzw[:3])))
        self.background[object_id] = {"asset": str(asset), "xyz": list(map(float, xyz)),
                                      "xyzw": list(map(float, xyzw))}
        return object_id

    def place_asset_on_surface(self, name: str, asset_usd: str | Path, surface: dict,
                               *, x: float, y: float, margin: float = 0.02,
                               clearance: float = 0.0, keywords=(), description=None) -> str:
        """Place an asset with its true lower bound on a horizontal support polygon."""
        if surface.get("kind") == "basket":
            raise ValueError("Basket placement needs an interior support height")
        if margin < 0 or clearance < 0:
            raise ValueError("margin and clearance must be nonnegative")
        source = Usd.Stage.Open(str(Path(asset_usd).resolve()), load=Usd.Stage.LoadAll)
        if source is None or not source.GetDefaultPrim():
            raise ValueError(f"Cannot read asset bounds: {asset_usd}")
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy])
        box = cache.ComputeWorldBound(source.GetDefaultPrim()).ComputeAlignedRange()
        if box.IsEmpty():
            raise ValueError(f"Asset has empty bounds: {asset_usd}")
        minimum = box.GetMin()
        maximum = box.GetMax()
        center_x = (float(minimum[0]) + float(maximum[0])) / 2
        center_y = (float(minimum[1]) + float(maximum[1])) / 2
        origin_x = float(x) - center_x
        origin_y = float(y) - center_y
        footprint = [(origin_x + local_x, origin_y + local_y)
                     for local_x in (float(minimum[0]), float(maximum[0]))
                     for local_y in (float(minimum[1]), float(maximum[1]))]
        polygon = [(float(point[0]), float(point[1])) for point in surface["corners_world_m"]]
        signed_area = sum(polygon[i][0] * polygon[(i + 1) % len(polygon)][1]
                          - polygon[(i + 1) % len(polygon)][0] * polygon[i][1]
                          for i in range(len(polygon)))
        sign = 1.0 if signed_area >= 0 else -1.0
        for point_x, point_y in footprint:
            for index, (ax, ay) in enumerate(polygon):
                bx, by = polygon[(index + 1) % len(polygon)]
                edge_length = ((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5
                if edge_length == 0:
                    raise ValueError("Degenerate support polygon")
                inward_distance = sign * ((bx - ax) * (point_y - ay) - (by - ay) * (point_x - ax)) / edge_length
                if inward_distance < margin - 1e-6:
                    raise ValueError(f"Asset footprint outside support surface by margin {margin}")
        z = float(surface["height_m"]) - float(minimum[2]) + float(clearance)
        return self.add_asset(name, asset_usd, xyz=(origin_x, origin_y, z),
                              keywords=keywords, description=description)

    def save(self) -> None:
        self.stage.GetRootLayer().Save()
        nodes = [{"id": "root_scene"}] + [{"id": object_id} for object_id in self.layout]
        links = [{"source": "root_scene", "target": object_id} for object_id in self.layout]
        scene_info = {"scene_id": self.scene_id, "seed": self.seed,
                      "layout": self.layout, "background": self.background,
                      "relations": {"graph": {"directed": True, "multigraph": False,
                                              "graph": {}, "nodes": nodes, "links": links}},
                      "object_states": {object_id: entry["states"] for object_id, entry in self.layout.items()
                                        if "states" in entry}}
        (self.output_dir / "scene_info.json").write_text(json.dumps(scene_info, indent=2), encoding="utf-8")

    def set_object_state(self, object_id: str, key: str, value) -> None:
        if object_id not in self.layout:
            raise KeyError(object_id)
        states = self.layout[object_id].setdefault("states", {})
        states[key] = value

    def set_revolute_joint_state(self, object_id: str, relative_joint_path: str,
                                 angle_deg: float, *, damping: float = 1.0,
                                 friction: float | None = None) -> None:
        """Set an instance joint's initial angle and drive target in degrees."""
        if object_id not in self.layout:
            raise KeyError(object_id)
        parts = relative_joint_path.split("/")
        if not parts or any(not part or part in (".", "..") for part in parts):
            raise ValueError("relative_joint_path must stay inside the object")
        path = f"/World/Objects/{object_id}/{relative_joint_path}"
        joint = self.stage.GetPrimAtPath(path)
        if not joint or joint.GetTypeName() != "PhysicsRevoluteJoint":
            raise ValueError(f"Not a revolute joint: {path}")
        angle = float(angle_deg)
        for attribute, predicate in (("physics:lowerLimit", lambda value: angle >= value),
                                     ("physics:upperLimit", lambda value: angle <= value)):
            attr = joint.GetAttribute(attribute)
            limit = attr.Get() if attr else None
            if limit is not None and not predicate(float(limit)):
                raise ValueError(f"{angle} degrees violates {attribute}={limit}")
        joint.AddAppliedSchema("PhysicsJointStateAPI:angular")
        joint.CreateAttribute("state:angular:physics:position", Sdf.ValueTypeNames.Float).Set(angle)
        joint.CreateAttribute("state:angular:physics:velocity", Sdf.ValueTypeNames.Float).Set(0.0)
        drive = UsdPhysics.DriveAPI.Apply(joint, "angular")
        drive.CreateTargetPositionAttr().Set(angle)
        drive.CreateTargetVelocityAttr().Set(0.0)
        drive.CreateStiffnessAttr().Set(0.0)
        drive.CreateDampingAttr().Set(float(damping))
        if friction is not None:
            joint.AddAppliedSchema("PhysxJointAPI")
            joint.CreateAttribute("physxJoint:jointFriction", Sdf.ValueTypeNames.Float).Set(float(friction))
        self.set_object_state(object_id, relative_joint_path,
                              {"angle_deg": angle, "damping": float(damping), "friction": friction})

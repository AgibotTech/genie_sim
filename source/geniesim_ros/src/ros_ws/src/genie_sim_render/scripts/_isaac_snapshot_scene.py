# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""USD authoring for trusted Newton viewer snapshots; no Kit/physics imports.

The caller creates SimulationApp first. CPU tests can use an in-memory USD
stage without creating an Isaac application or using a GPU.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np


def _array(value: Any, columns: int, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.ndim != 2 or result.shape[1] != columns or not np.isfinite(result).all():
        raise ValueError(f"{label} must be a finite Nx{columns} array")
    return result


def _vector(value: Any, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{label} must be a finite three-vector")
    return result


def _name(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("snapshot names must be nonempty strings")
    return value


def _key(name: str) -> str:
    # Viewer paths contain slashes and can collide after USD identifier repair.
    return "n_" + hashlib.sha256(name.encode("utf-8")).hexdigest()[:24]


def _pose_matrix(value: Any, label: str) -> Any:
    from pxr import Gf

    row = np.asarray(value, dtype=np.float64)
    if row.shape != (7,) or not np.isfinite(row).all() or np.linalg.norm(row[3:]) < 1e-9:
        raise ValueError(f"{label} requires finite xyz+xyzw and a nonzero quaternion")
    quat = row[3:] / np.linalg.norm(row[3:])
    matrix = Gf.Matrix4d(1)
    matrix.SetRotate(Gf.Quatd(float(quat[3]), Gf.Vec3d(*quat[:3])))
    matrix.SetTranslateOnly(Gf.Vec3d(*row[:3]))
    return matrix


def validate_cameras(cameras: Any, width: Any, height: Any) -> list[dict[str, Any]]:
    for label, value in (("width", width), ("height", height)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError(f"{label} must be a positive integer")
    if not isinstance(cameras, list) or not cameras:
        raise ValueError("cameras must be a nonempty list")
    result = []
    seen: set[str] = set()
    for camera in cameras:
        name = _name(camera["name"])
        if name in seen:
            raise ValueError(f"duplicate camera name: {name}")
        seen.add(name)
        eye, target, up = (_vector(camera[k], f"camera {name}.{k}") for k in ("eye", "target", "up"))
        front = target - eye
        if np.linalg.norm(front) < 1e-9 or np.linalg.norm(np.cross(front, up)) < 1e-9:
            raise ValueError(f"camera {name} has a degenerate look-at basis")
        fov = float(camera["fov_degrees"])
        if not math.isfinite(fov) or not 0 < fov < 179:
            raise ValueError(f"camera {name}.fov_degrees must be between 0 and 179")
        row = dict(name=name, eye=eye, target=target, up=up, fov_degrees=fov)
        for key, default in (("width", width), ("height", height)):
            value = camera.get(key, default)
            if type(value) is not int or value <= 0:
                raise ValueError(f"camera {name}.{key} must be a positive integer")
            row[key] = value
        calibration = camera.get("calibration")
        if calibration is not None:
            if not isinstance(calibration, dict) or calibration.get("model") not in ("opencvPinhole", "opencvFisheye"):
                raise ValueError(f"camera {name} has an unsupported calibration model")
            calibration = dict(calibration)
            for key in ("fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4", "p1", "p2"):
                calibration[key] = (
                    float(calibration[key]) if key in ("fx", "fy", "cx", "cy") else float(calibration.get(key, 0))
                )
                if not math.isfinite(calibration[key]):
                    raise ValueError(f"camera {name} has nonfinite calibration")
            if min(calibration["fx"], calibration["fy"]) <= 0:
                raise ValueError(f"camera {name} has invalid focal lengths")
            row["calibration"] = calibration
        result.append(row)
    return result


class SnapshotScene:
    """Keep mesh deltas and apply complete instance batches onto one USD stage."""

    def __init__(self, stage: Any) -> None:
        from pxr import UsdGeom, UsdLux

        self.stage = stage
        self.meshes: dict[str, dict[str, Any]] = {}
        self._batches: set[str] = set()
        self._materials: dict[tuple[Any, ...], Any] = {}
        self._texture_dir: TemporaryDirectory | None = None
        self._robot_descriptor: tuple[str, str] | None = None
        self._robot_paths: dict[str, Any] = {}
        self._robot_scales: dict[str, tuple[float, ...]] = {}
        self._robot_audit_logged = False
        self._scene_descriptor: tuple[Any, ...] | None = None
        self._scene_visuals: dict[str, tuple[str, Any]] = {}
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        world = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world.GetPrim())
        for path in (
            "/World/Instances",
            "/World/Direct",
            "/World/Materials",
            "/World/Cameras",
        ):
            UsdGeom.Scope.Define(stage, path)
        # Parent visibility is not copied by an internal reference to a child.
        # Prototype meshes themselves remain inherited so visible references work.
        UsdGeom.Scope.Define(stage, "/World/Prototypes")
        UsdGeom.Imageable(stage.GetPrimAtPath("/World/Prototypes")).MakeInvisible()
        dome = UsdLux.DomeLight.Define(stage, "/World/Light")
        dome.CreateIntensityAttr(800.0)
        dome.CreateColorAttr((0.85, 0.9, 1.0))
        sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
        sun.CreateIntensityAttr(1800.0)
        sun.CreateAngleAttr(1.0)
        UsdGeom.Xformable(sun).AddRotateXYZOp().Set((30.0, -35.0, -20.0))

    def mesh_path(self, name: str) -> str:
        return "/World/Prototypes/" + _key(name)

    def instance_path(self, name: str, index: int = 0) -> str:
        return f"/World/Instances/{_key(name)}/i_{index:06d}"

    def direct_path(self, name: str) -> str:
        return "/World/Direct/" + _key(name)

    def _texture_path(self, texture: Any) -> str:
        if isinstance(texture, str):
            path = Path(texture).expanduser().resolve(strict=True)
            if not path.is_file():
                raise ValueError(f"texture must be a local image file: {path}")
            return str(path)
        pixels = np.asarray(texture)
        if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] not in (3, 4) or not pixels.size:
            raise ValueError("texture must be a uint8 HxWx3/4 image or an existing image path")
        from PIL import Image

        if self._texture_dir is None:
            self._texture_dir = TemporaryDirectory(prefix="genie-isaac-textures-")
        digest = hashlib.sha256(str(pixels.shape).encode() + pixels.tobytes()).hexdigest()
        path = Path(self._texture_dir.name) / f"{digest}.png"
        if not path.exists():
            Image.fromarray(pixels).save(path, format="PNG")
        return str(path)

    def _material(
        self,
        color: Any,
        roughness: float,
        metallic: float,
        texture_path: str | None = None,
        opacity: float = 1.0,
    ) -> Any:
        from pxr import Gf, Sdf, UsdShade

        rgb = _vector(color, "color")
        values = (*rgb, float(roughness), float(metallic), float(opacity))
        if not np.isfinite(values).all() or any(not 0 <= x <= 1 for x in values):
            raise ValueError("material color/roughness/metallic/opacity must be finite values in [0,1]")
        key = (*[float(x) for x in values], texture_path)
        if key not in self._materials:
            path = "/World/Materials/" + _key(repr(key))
            material = UsdShade.Material.Define(self.stage, path)
            shader = UsdShade.Shader.Define(self.stage, path + "/Surface")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*key[:3]))
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(key[3])
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(key[4])
            shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(opacity))
            if texture_path is not None:
                reader = UsdShade.Shader.Define(self.stage, path + "/PrimvarReader")
                reader.CreateIdAttr("UsdPrimvarReader_float2")
                reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
                reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
                texture = UsdShade.Shader.Define(self.stage, path + "/Texture")
                texture.CreateIdAttr("UsdUVTexture")
                texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(texture_path))
                texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
                texture.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
                texture.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
                texture.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*key[:3], float(opacity)))
                texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
                texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
                texture.CreateOutput("a", Sdf.ValueTypeNames.Float)
                shader.GetInput("diffuseColor").ConnectToSource(texture.ConnectableAPI(), "rgb")
                shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).ConnectToSource(texture.ConnectableAPI(), "a")
            shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            self._materials[key] = material
        return self._materials[key]

    def _bind_material(
        self, prim: Any, mesh: dict[str, Any], color: Any = None, material: Any = None, opacity: float | None = None
    ) -> None:
        from pxr import UsdShade

        opacity = float(mesh.get("opacity", 1.0)) if opacity is None else opacity
        roughness = mesh.get("roughness")
        metallic = mesh.get("metallic")
        roughness = 0.5 if roughness is None else float(roughness)
        metallic = 0.0 if metallic is None else float(metallic)
        texture_path = mesh.get("_texture_path")
        if material is not None:
            # Newton ViewerBase materials: roughness, metallic, checker, texture.
            roughness, metallic = float(material[0]), float(material[1])
            if float(material[3]) < 0.5:
                texture_path = None
        color = mesh.get("color") if color is None else color
        if color is None:
            color = (1.0, 1.0, 1.0) if texture_path else (0.7, 0.7, 0.7)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(self._material(color, roughness, metallic, texture_path, opacity))

    def _write_mesh(self, name: str, patch: dict[str, Any]) -> None:
        from pxr import Sdf, UsdGeom, Vt

        data = {**self.meshes.get(name, {}), **patch}
        points = _array(data["points"], 3, f"mesh {name}.points")
        indices = np.asarray(data["indices"])
        if indices.ndim != 1 or indices.dtype.kind not in "iu" or indices.size % 3:
            raise ValueError(f"mesh {name}.indices must be a flat integer triangle array")
        if indices.size and (indices.min() < 0 or indices.max() >= len(points)):
            raise ValueError(f"mesh {name}.indices are outside its vertex array")
        normals = data.get("normals")
        if normals is not None:
            normals = _array(normals, 3, f"mesh {name}.normals")
            if len(normals) != len(points):
                raise ValueError(f"mesh {name}.normals do not match its points")
        data.update(points=points, indices=indices.astype(np.int32), normals=normals)
        uvs = data.get("uvs")
        if uvs is not None:
            uvs = _array(uvs, 2, f"mesh {name}.uvs")
            if len(uvs) != len(points):
                raise ValueError(f"mesh {name}.uvs do not match its points")
        if data.get("texture") is not None:
            if uvs is None:
                raise ValueError(f"mesh {name} has a texture but no UV coordinates")
            if "texture" in patch or "_texture_path" not in data:
                data["_texture_path"] = self._texture_path(data["texture"])
        else:
            data.pop("_texture_path", None)
        mesh = UsdGeom.Mesh.Define(self.stage, self.mesh_path(name))
        mesh.CreatePointsAttr().Set(Vt.Vec3fArray.FromNumpy(points))
        mesh.CreateFaceVertexCountsAttr().Set(Vt.IntArray([3] * (indices.size // 3)))
        mesh.CreateFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(data["indices"]))
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateDoubleSidedAttr(not data.get("backface_culling", True))
        extent = np.array([points.min(axis=0), points.max(axis=0)]) if len(points) else np.zeros((2, 3), np.float32)
        mesh.CreateExtentAttr().Set(Vt.Vec3fArray.FromNumpy(extent.astype(np.float32)))
        if normals is not None:
            mesh.CreateNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(normals))
            mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        else:
            mesh.GetNormalsAttr().Clear()
        primvars = UsdGeom.PrimvarsAPI(mesh)
        if uvs is not None:
            primvars.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex).Set(
                Vt.Vec2fArray.FromNumpy(uvs)
            )
        elif primvars.HasPrimvar("st"):
            primvars.GetPrimvar("st").GetAttr().Clear()
        self._bind_material(mesh.GetPrim(), data)
        self.meshes[name] = data

    def _reference(self, path: str, mesh_name: str) -> Any:
        from pxr import UsdGeom

        prim = self.stage.GetPrimAtPath(path)
        if not prim:
            prim = self.stage.DefinePrim(path, "Mesh")
        wanted = self.mesh_path(mesh_name)
        if prim.GetCustomDataByKey("snapshotMesh") != mesh_name:
            prim.GetReferences().ClearReferences()
            prim.GetReferences().AddInternalReference(wanted)
            prim.SetCustomDataByKey("snapshotMesh", mesh_name)
        # Never copy prototype-container visibility onto its references.
        UsdGeom.Imageable(prim).CreateVisibilityAttr().Set(UsdGeom.Tokens.inherited)
        return prim

    def apply(self, snapshot: dict[str, Any]) -> None:
        from pxr import Gf, UsdGeom

        self._apply_authored_scene(snapshot)
        self._apply_authored_robot(snapshot)
        patches, batches = snapshot.get("meshes", {}), snapshot["instances"]
        if not isinstance(patches, dict) or not isinstance(batches, dict):
            raise ValueError("scene.meshes and scene.instances must be dictionaries")
        for name, patch in patches.items():
            self._write_mesh(_name(name), patch)
        referenced: set[str] = set()
        for name, batch in batches.items():
            _name(name)
            mesh_name = _name(batch["mesh"])
            if mesh_name not in self.meshes:
                raise ValueError(f"instance {name} references unknown mesh {mesh_name}")
            referenced.add(mesh_name)
            root_path = "/World/Instances/" + _key(name)
            root = UsdGeom.Xform.Define(self.stage, root_path)
            hidden = batch.get("hidden", False)
            root.CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible if hidden else UsdGeom.Tokens.inherited)
            if hidden:
                # ViewerBase skips shapes.update() for hidden batches, so their
                # world_xforms may be uninitialized. Keep geometry hidden and
                # referenced, without inventing or applying a transform.
                continue
            mesh = self.meshes[mesh_name]
            xforms = _array(batch["xforms"], 7, f"instance {name}.xforms")
            scales = _array(batch["scales"], 3, f"instance {name}.scales")
            colors, materials = batch.get("colors"), batch.get("materials")
            colors = None if colors is None else _array(colors, 3, f"instance {name}.colors")
            materials = None if materials is None else _array(materials, 4, f"instance {name}.materials")
            opacities = batch.get("opacities")
            opacities = None if opacities is None else _array(opacities, 1, f"instance {name}.opacities")
            if opacities is not None and np.any((opacities < 0) | (opacities > 1)):
                raise ValueError("instance opacities must be in [0,1]")
            if any(a is not None and len(a) != len(xforms) for a in (scales, colors, materials, opacities)):
                raise ValueError(f"instance {name} attribute lengths do not match")
            if len(xforms) and np.any(np.linalg.norm(xforms[:, 3:], axis=1) < 1e-9):
                raise ValueError(f"instance {name} contains a zero quaternion")
            for child in root.GetPrim().GetChildren():
                if int(child.GetName()[2:]) >= len(xforms):
                    self.stage.RemovePrim(child.GetPath())
            for i, row in enumerate(xforms):
                path = self.instance_path(name, i)
                parent = UsdGeom.Xform.Define(self.stage, path)
                if not parent.GetOrderedXformOps():
                    parent.AddTranslateOp()
                    parent.AddOrientOp()
                    parent.AddScaleOp()
                t, q, s = parent.GetOrderedXformOps()
                t.Set(Gf.Vec3d(*[float(x) for x in row[:3]]))
                quat = row[3:] / np.linalg.norm(row[3:])
                q.Set(Gf.Quatf(float(quat[3]), Gf.Vec3f(*[float(x) for x in quat[:3]])))
                s.Set(Gf.Vec3f(*[float(x) for x in scales[i]]))
                prim = self._reference(path + "/Geometry", mesh_name)
                self._bind_material(
                    prim,
                    mesh,
                    None if colors is None else colors[i],
                    None if materials is None else materials[i],
                    None if opacities is None else float(opacities[i, 0]),
                )
        for name in self._batches - batches.keys():
            self.stage.RemovePrim("/World/Instances/" + _key(name))
        self._batches = set(batches)
        for name, mesh in self.meshes.items():
            direct = self._reference(self.direct_path(name), name)
            visible = not mesh.get("hidden", False) and name not in referenced
            UsdGeom.Imageable(direct).CreateVisibilityAttr().Set(
                UsdGeom.Tokens.inherited if visible else UsdGeom.Tokens.invisible
            )
        self._ground_if_missing(batches)

    @staticmethod
    def _prepare_authored_scene(root: Any, *, keep_lights: bool) -> None:
        """Disable simulation/native sensors while retaining visual material data."""
        from pxr import Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

        # Deinstance visual references so physics and sensor descendants can
        # be disabled in this worker layer without editing their source USD.
        for prim in list(Usd.PrimRange(root)):
            if prim.IsInstance():
                prim.SetInstanceable(False)
            for variant_name in prim.GetVariantSets().GetNames():
                if variant_name.lower() not in ("physics", "sensor"):
                    continue
                variant = prim.GetVariantSet(variant_name)
                disabled = next((name for name in variant.GetVariantNames() if name.lower() == "none"), None)
                if disabled is not None:
                    variant.SetVariantSelection(disabled)
        for prim in list(Usd.PrimRange(root)):
            for schema in list(prim.GetAppliedSchemas()):
                if schema.startswith(("Physics", "Physx", "OmniPhysics", "Newton")):
                    prim.RemoveAppliedSchema(schema)
            if not prim.HasAPI(UsdShade.MaterialBindingAPI) and any(
                relationship.GetName().startswith("material:binding") and relationship.GetTargets()
                for relationship in prim.GetRelationships()
            ):
                UsdShade.MaterialBindingAPI.Apply(prim)
            if (
                prim.IsA(UsdPhysics.Joint)
                or prim.IsA(UsdPhysics.Scene)
                or prim.IsA(UsdGeom.Camera)
                or prim.GetName().lower().startswith(("collider", "collision"))
                or (not keep_lights and prim.HasAPI(UsdLux.LightAPI))
            ):
                prim.SetActive(False)

    def _apply_authored_scene(self, snapshot: dict[str, Any]) -> None:
        """Reference a static authored scene and independently drive selected visuals.

        Source object origins may be far from their Newton rigid-body frame.
        Keep the source's entire world placement on the reference child and
        apply only reference_pose^-1 * current_pose on its parent (Gf rows).
        """
        from pxr import Sdf, Usd, UsdGeom, UsdLux, UsdShade

        descriptor = snapshot.get("authored_scene")
        root_path = "/World/AuthoredScene"
        key = self._scene_descriptor

        def source_path(value: Any, label: str) -> Any:
            path = Sdf.Path(_name(value))
            if not path.IsAbsolutePath() or not path.IsPrimPath() or path == Sdf.Path.absoluteRootPath:
                raise ValueError(f"{label} must be an absolute USD prim path")
            return path

        if descriptor is not None:
            if not isinstance(descriptor, dict):
                raise ValueError("authored_scene must be an asset descriptor")
            asset = Path(_name(descriptor["asset_path"]))
            if not asset.is_absolute():
                raise ValueError("authored_scene asset_path must be absolute")
            asset_path = str(asset.resolve(strict=True))
            if not Path(asset_path).is_file():
                raise ValueError("authored_scene asset_path must be a local USD file")
            source_root = source_path(descriptor.get("root_prim", "/World"), "authored_scene.root_prim")
            hidden = descriptor.get("hide_prims", [])
            if not isinstance(hidden, list):
                raise ValueError("authored_scene.hide_prims must be a list")
            hidden = tuple(str(source_path(path, "authored_scene.hide_prims")) for path in hidden)
            if any(not Sdf.Path(path).HasPrefix(source_root) or Sdf.Path(path) == source_root for path in hidden):
                raise ValueError("authored_scene hidden prims must be below its root")
            use_asset_lights = descriptor.get("use_asset_lights", False)
            if type(use_asset_lights) is not bool:
                raise ValueError("authored_scene.use_asset_lights must be boolean")
            visual_specs = descriptor.get("body_visuals", [])
            if not isinstance(visual_specs, list):
                raise ValueError("authored_scene.body_visuals must be a list")
            visuals = []
            seen = set()
            for visual in visual_specs:
                if not isinstance(visual, dict):
                    raise ValueError("authored_scene body_visuals entries must be mappings")
                body = _name(visual["body"])
                if body in seen:
                    raise ValueError(f"duplicate authored scene visual body: {body}")
                seen.add(body)
                path = source_path(visual["source_prim"], f"visual body {body}.source_prim")
                if not path.HasPrefix(source_root):
                    raise ValueError(f"visual body {body} source must be within the authored scene")
                _pose_matrix(visual["reference_pose"], f"visual body {body}.reference_pose")
                reference = tuple(float(x) for x in visual["reference_pose"])
                visuals.append((body, str(path), reference))
            key = (asset_path, str(source_root), hidden, use_asset_lights, tuple(visuals))
            if self._scene_descriptor is not None and key != self._scene_descriptor:
                raise ValueError("authored_scene asset/configuration cannot change within a snapshot worker")
        if key is None:
            if snapshot.get("visual_body_poses"):
                raise ValueError("visual_body_poses require an authored_scene descriptor")
            return
        asset_path, source_root, hidden, use_asset_lights, visuals = key
        poses = snapshot.get("visual_body_poses", {})
        if not isinstance(poses, dict):
            raise ValueError("authored scene requires complete visual_body_poses on every snapshot")
        expected = {body for body, _, _ in visuals}
        if poses.keys() != expected:
            raise ValueError(f"authored scene visual_body_poses must match its drivers: {sorted(expected)}")
        matrices = {body: _pose_matrix(poses[body], f"visual body {body}") for body in expected}
        if self._scene_descriptor is None:
            source = Usd.Stage.Open(asset_path)
            source_prim = source.GetPrimAtPath(source_root)
            if not source_prim:
                raise ValueError(f"authored_scene has no valid root prim: {source_root}")
            for path in hidden:
                if not source.GetPrimAtPath(path):
                    raise ValueError(f"authored_scene cannot hide missing prim: {path}")
            for body, path, _ in visuals:
                if not UsdGeom.Xformable(source.GetPrimAtPath(path)):
                    raise ValueError(f"authored_scene visual body {body} has no transformable source: {path}")
            root = UsdGeom.Xform.Define(self.stage, root_path).GetPrim()
            if not root.GetReferences().AddReference(asset_path, Sdf.Path(source_root)):
                raise ValueError(f"cannot reference authored scene: {asset_path}")
            for path in hidden:
                target = Sdf.Path(path).ReplacePrefix(Sdf.Path(source_root), Sdf.Path(root_path))
                self.stage.GetPrimAtPath(target).SetActive(False)
            self._prepare_authored_scene(root, keep_lights=use_asset_lights)
            source_xforms = UsdGeom.XformCache()
            for body, path, reference in visuals:
                parent_path = "/World/AuthoredVisualBodies/" + _key(body)
                parent = UsdGeom.Xform.Define(self.stage, parent_path)
                parent.AddTransformOp()
                child = UsdGeom.Xform.Define(self.stage, parent_path + "/Reference")
                if not child.GetPrim().GetReferences().AddReference(asset_path, Sdf.Path(path)):
                    raise ValueError(f"cannot reference authored visual body {body}: {path}")
                # Referencing a subtree omits its ancestors. Preserve the
                # complete authored world matrix, including ancestor offsets,
                # while leaving all visual children/UV/materials untouched.
                transform = child.AddTransformOp(opSuffix="authoredWorld")
                transform.Set(source_xforms.GetLocalToWorldTransform(source.GetPrimAtPath(path)))
                child.SetXformOpOrder([transform], resetXformStack=False)
                self._prepare_authored_scene(child.GetPrim(), keep_lights=False)
                self._scene_visuals[body] = (parent_path, _pose_matrix(reference, body).GetInverse())
            if use_asset_lights:
                for path in ("/World/Light", "/World/Sun"):
                    self.stage.GetPrimAtPath(path).SetActive(False)
            self._scene_descriptor = key
            prims = list(Usd.PrimRange(root))
            meshes = [prim for prim in prims if prim.IsA(UsdGeom.Mesh)]
            materials = sum(prim.IsA(UsdShade.Material) for prim in prims)
            bound = sum(bool(UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]) for prim in meshes)
            lights = sum(prim.HasAPI(UsdLux.LightAPI) for prim in prims)
            print(
                f"[isaacsim-snapshot] authored_scene asset={asset_path} source_root={source_root} "
                f"hidden={list(hidden)} body_drivers={sorted(expected)} materials={materials} "
                f"bound_meshes={bound}/{len(meshes)} asset_lights={use_asset_lights} active_asset_lights={lights}",
                file=sys.stderr,
                flush=True,
            )
            for body, path, reference in visuals:
                parent_path = self._scene_visuals[body][0]
                body_prims = list(Usd.PrimRange(self.stage.GetPrimAtPath(parent_path)))
                body_meshes = [prim for prim in body_prims if prim.IsA(UsdGeom.Mesh)]
                body_materials = sum(prim.IsA(UsdShade.Material) for prim in body_prims)
                body_bound = sum(
                    bool(UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]) for prim in body_meshes
                )
                print(
                    f"[isaacsim-snapshot] authored_body body={body} source_prim={path} reference_pose={reference} "
                    f"materials={body_materials} bound_meshes={body_bound}/{len(body_meshes)}",
                    file=sys.stderr,
                    flush=True,
                )
        for body, (path, reference_inverse) in self._scene_visuals.items():
            UsdGeom.Xformable(self.stage.GetPrimAtPath(path)).GetOrderedXformOps()[0].Set(
                reference_inverse * matrices[body]
            )

    def _apply_authored_robot(self, snapshot: dict[str, Any]) -> None:
        """Reference the complete visual USD; only body Xforms are overridden.

        The parent excludes its Newton robot instance rows, retaining scene
        shapes from mixed batches. Looks/MDL/texture/subset bindings stay in the
        authored reference, including targets outside individual link subtrees.
        """
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

        descriptor = snapshot.get("authored_robot")
        root_path = "/World/AuthoredRobot"
        if descriptor is not None:
            if not isinstance(descriptor, dict):
                raise ValueError("authored_robot must be an asset descriptor")
            asset_path = str(Path(_name(descriptor["asset_path"])).expanduser().resolve(strict=True))
            source_root = descriptor.get("root_prim") or ""
            key = (asset_path, source_root)
            if self._robot_descriptor is not None and key != self._robot_descriptor:
                raise ValueError("authored_robot asset cannot change within a snapshot worker")
            if self._robot_descriptor is None:
                source = Usd.Stage.Open(asset_path)
                source_prim = source.GetPrimAtPath(source_root) if source_root else source.GetDefaultPrim()
                if not source_prim or not source_prim.IsValid():
                    raise ValueError(f"authored_robot has no valid root/default prim: {asset_path} {source_root}")
                root = UsdGeom.Xform.Define(self.stage, root_path).GetPrim()
                if not root.GetReferences().AddReference(asset_path, source_prim.GetPath()):
                    raise ValueError(f"cannot reference authored robot: {asset_path}")
                # Assets spell these variants None/none differently. Disable
                # only physics and native sensor payloads; retain Robot/Looks.
                for variant_name in root.GetVariantSets().GetNames():
                    if variant_name.lower() not in ("physics", "sensor"):
                        continue
                    variant = root.GetVariantSet(variant_name)
                    disabled = next(
                        (name for name in variant.GetVariantNames() if name.lower() == "none"),
                        None,
                    )
                    if disabled is not None:
                        variant.SetVariantSelection(disabled)
                for prim in list(Usd.PrimRange(root, Usd.TraverseInstanceProxies())):
                    if prim.IsInstanceProxy():
                        continue
                    for schema in list(prim.GetAppliedSchemas()):
                        if schema.startswith(("Physics", "Physx", "OmniPhysics")):
                            prim.RemoveAppliedSchema(schema)
                    if not prim.HasAPI(UsdShade.MaterialBindingAPI) and any(
                        relationship.GetName().startswith("material:binding") and relationship.GetTargets()
                        for relationship in prim.GetRelationships()
                    ):
                        # Some shipped Wuji visual payloads authored the real
                        # binding relationship without declaring its API. Apply
                        # only the schema; preserve every original target/graph.
                        UsdShade.MaterialBindingAPI.Apply(prim)
                    if prim.IsA(UsdPhysics.Joint) or prim.IsA(UsdGeom.Camera) or prim.GetName() == "collisions":
                        prim.SetActive(False)
                self._robot_descriptor = key
        if self._robot_descriptor is None:
            return
        poses = snapshot.get("body_poses")
        if not isinstance(poses, dict) or not poses:
            raise ValueError("authored robot requires complete body_poses on every snapshot")
        root = self.stage.GetPrimAtPath(root_path)
        if not self._robot_paths:
            prims = list(Usd.PrimRange(root, Usd.TraverseInstanceProxies()))
            for body in poses:
                _name(body)
                flat = self.stage.GetPrimAtPath(root_path + "/" + body)
                candidates = (
                    [flat]
                    if flat and flat.IsA(UsdGeom.Xform)
                    else [prim for prim in prims if prim.GetName() == body and prim.IsA(UsdGeom.Xform)]
                )
                if len(candidates) > 1:
                    raise ValueError(f"ambiguous authored robot body: {body}")
                if candidates:
                    prim = candidates[0]
                    if prim.IsInstanceProxy():
                        raise ValueError(f"authored robot body cannot be an instance proxy: {body}")
                    self._robot_paths[body] = prim.GetPath()
                    scale = Gf.Transform(UsdGeom.Xformable(prim).GetLocalTransformation()).GetScale()
                    self._robot_scales[body] = tuple(float(x) for x in scale)
            if not self._robot_paths:
                raise ValueError("authored robot has no matching body_poses")
            # A visible asset link without a Newton driver is an error; leaving
            # it at its authored rest pose would silently detach robot parts.
            driven = list(self._robot_paths.values())
            for prim in prims:
                if (
                    not prim.IsA(UsdGeom.Gprim)
                    or UsdGeom.Imageable(prim).ComputeVisibility() == UsdGeom.Tokens.invisible
                ):
                    continue
                if not any(prim.GetPath().HasPrefix(path) for path in driven):
                    raise ValueError(f"visible authored robot geometry has no body pose: {prim.GetPath()}")
        missing = self._robot_paths.keys() - poses.keys()
        if missing:
            raise ValueError(f"authored robot body_poses missing: {', '.join(sorted(missing))}")
        for body, path in self._robot_paths.items():
            row = np.asarray(poses[body], dtype=np.float64)
            if row.shape != (7,) or not np.isfinite(row).all() or np.linalg.norm(row[3:]) < 1e-9:
                raise ValueError(f"authored robot body {body} requires finite xyz+xyzw and a nonzero quaternion")
            prim = self.stage.GetPrimAtPath(path)
            xform = UsdGeom.Xformable(prim)
            translation = (
                prim.GetAttribute("xformOp:translate")
                or xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).GetAttr()
            )
            orient = prim.GetAttribute("xformOp:orient") or xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).GetAttr()
            translation.Set(
                Gf.Vec3f(*row[:3]) if translation.GetTypeName() == Sdf.ValueTypeNames.Float3 else Gf.Vec3d(*row[:3])
            )
            quat = row[3:] / np.linalg.norm(row[3:])
            rotation = Gf.Quatd(float(quat[3]), Gf.Vec3d(*quat[:3]))
            orient.Set(Gf.Quatf(rotation) if orient.GetTypeName() == Sdf.ValueTypeNames.Quatf else rotation)
            ops = [UsdGeom.XformOp(translation), UsdGeom.XformOp(orient)]
            if not np.allclose(self._robot_scales[body], (1, 1, 1)):
                scale = prim.GetAttribute("xformOp:scale") or xform.AddScaleOp().GetAttr()
                scale.Set(Gf.Vec3f(*self._robot_scales[body]))
                ops.append(UsdGeom.XformOp(scale))
            wanted = ["!resetXformStack!", *[op.GetOpName() for op in ops]]
            if list(prim.GetAttribute("xformOpOrder").Get() or []) != wanted:
                xform.SetXformOpOrder(ops, resetXformStack=True)
        if not self._robot_audit_logged:
            prims = list(Usd.PrimRange(root, Usd.TraverseInstanceProxies()))
            materials = sum(prim.IsA(UsdShade.Material) for prim in prims)
            meshes = [prim for prim in prims if prim.IsA(UsdGeom.Mesh)]
            bound = sum(
                bool(UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0])
                or any(
                    child.IsA(UsdGeom.Subset) and bool(UsdShade.MaterialBindingAPI(child).ComputeBoundMaterial()[0])
                    for child in prim.GetChildren()
                )
                for prim in meshes
            )
            print(
                f"[isaacsim-snapshot] authored robot asset={self._robot_descriptor[0]} "
                f"root={root_path} mapped_links={len(self._robot_paths)} "
                f"materials={materials} bound_meshes={bound}/{len(meshes)}",
                file=sys.stderr,
                flush=True,
            )
            self._robot_audit_logged = True

    def close(self) -> None:
        """Release only worker-owned texture files, never authored assets."""
        if self._texture_dir is not None:
            self._texture_dir.cleanup()
            self._texture_dir = None

    def _ground_if_missing(self, batches: dict[str, Any]) -> None:
        from pxr import UsdGeom

        def ground_name(name: str) -> bool:
            return any(word in name.lower() for word in ("ground", "floor", "plane"))

        present = self._scene_descriptor is not None or any(
            ground_name(name) and not mesh.get("hidden", False) for name, mesh in self.meshes.items()
        )
        present = present or any(
            (ground_name(name) or ground_name(batch["mesh"]))
            and not batch.get("hidden", False)
            and len(batch["xforms"])
            for name, batch in batches.items()
        )
        path = "/World/FallbackGround"
        if present:
            if self.stage.GetPrimAtPath(path):
                self.stage.RemovePrim(path)
        elif not self.stage.GetPrimAtPath(path):
            ground = UsdGeom.Mesh.Define(self.stage, path)
            ground.CreatePointsAttr([(-50, -50, 0), (50, -50, 0), (50, 50, 0), (-50, 50, 0)])
            ground.CreateFaceVertexCountsAttr([4])
            ground.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
            ground.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            ground.CreateDoubleSidedAttr(True)
            self._bind_material(ground.GetPrim(), {"color": (0.16, 0.16, 0.18), "roughness": 0.8})

    def camera(self, camera: dict[str, Any], width: int, height: int) -> str:
        from pxr import Gf, UsdGeom

        path = "/World/Cameras/" + _key(camera["name"])
        cam = UsdGeom.Camera.Define(self.stage, path)
        eye, target, up = camera["eye"], camera["target"], camera["up"]
        front = (target - eye) / np.linalg.norm(target - eye)
        right = np.cross(front, up)
        right /= np.linalg.norm(right)
        true_up = np.cross(right, front)
        rotation = Gf.Matrix3d(*[float(x) for x in np.array([right, true_up, -front]).reshape(-1)])
        xform = UsdGeom.Xformable(cam)
        if not xform.GetOrderedXformOps():
            xform.AddTranslateOp()
            xform.AddOrientOp()
        t, q = xform.GetOrderedXformOps()
        t.Set(Gf.Vec3d(*[float(x) for x in eye]))
        q.Set(Gf.Quatf(rotation.ExtractRotation().GetQuat()))
        # Newton's Camera.fov is vertical (perspective_projection / fovy).
        vertical = 2.0 * math.tan(math.radians(camera["fov_degrees"]) * 0.5)
        cam.CreateFocalLengthAttr(1.0)
        cam.CreateVerticalApertureAttr(vertical)
        cam.CreateHorizontalApertureAttr(vertical * width / height)
        calibration = camera.get("calibration")
        if calibration is not None:
            from genie_sim_engine_core.common.render_layer import author_opencv_fisheye, author_opencv_pinhole

            fx, fy, cx, cy = (calibration[key] for key in ("fx", "fy", "cx", "cy"))
            cam.CreateHorizontalApertureAttr(width / fx)
            cam.CreateVerticalApertureAttr(height / fy)
            cam.CreateHorizontalApertureOffsetAttr((cx - width / 2) / fx)
            cam.CreateVerticalApertureOffsetAttr((cy - height / 2) / fy)
            author = author_opencv_fisheye if calibration["model"] == "opencvFisheye" else author_opencv_pinhole
            author(cam.GetPrim(), calibration, width, height)
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 1000.0))
        return path

# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Reproducible HDR selection and render-only background replacement."""
import hashlib
import math
import random
from pathlib import Path

HDR_DIRECTORY = "background/common/HDR"

# Visually reviewed office/home interiors suitable for both tabletop tasks.
# New assets require explicit review before entering this pool.
INDOOR_TABLETOP_HDRS = frozenset(
    {
        "0006_home.hdr",
        "0007_home.hdr",
        "0010_office.hdr",
        "0014_office_remove_chair_01.hdr",
        "0014_office_remove_chair_02.hdr",
        "010_office_move_table.hdr",
        "011_office.hdr",
        "011_office_move_chair.hdr",
        "012_office_move_table.hdr",
        "013_office_move_table.hdr",
        "015_office.hdr",
    }
)


def select_hdr(assets_root, identity, seed=0, directory=HDR_DIRECTORY):
    """Select a reviewed indoor tabletop HDR; retain content hashes for replay."""
    root = Path(assets_root).resolve()
    pool = (root / directory).resolve()
    if not pool.is_relative_to(root):
        raise ValueError("HDR directory must be inside the configured assets")
    files = sorted(
        (p for p in pool.iterdir() if p.is_file() and p.name.lower() in INDOOR_TABLETOP_HDRS),
        key=lambda p: p.name,
    )
    if not files:
        raise FileNotFoundError("No approved indoor tabletop HDR files in " + str(pool))
    for path in files:
        if not path.resolve().is_relative_to(root):
            raise ValueError("HDR files must remain inside the configured assets")
    token = hashlib.sha256(f"{seed}:{identity}".encode()).digest()
    selected = random.Random(int.from_bytes(token, "big")).choice(files)
    return {
        "schema": "geniesim.hdr-background.v1",
        "identity": str(identity),
        "seed": int(seed),
        "directory": str(pool.relative_to(root)),
        "texture_file": str(selected),
        "relative_path": str(selected.relative_to(root)),
        "sha256": hashlib.sha256(selected.read_bytes()).hexdigest(),
        "pool_sha256": hashlib.sha256("\n".join(p.name for p in files).encode()).hexdigest(),
    }


def apply_hdr(stage, selection, *, world_root="/World"):
    """Replace the visible panorama and DomeLight maps, retaining task geometry."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade

    texture_file = Path(selection["texture_file"])
    if hashlib.sha256(texture_file.read_bytes()).hexdigest() != selection["sha256"]:
        raise ValueError("Selected HDR changed after selection")
    root = stage.GetPrimAtPath(world_root)
    if not root:
        raise ValueError("Missing background root: " + world_root)
    for suffix in ["/hdr_backdrop", "/Background"]:
        old = stage.GetPrimAtPath(world_root + suffix)
        if old:
            UsdGeom.Imageable(old).MakeInvisible()
    lights = [UsdLux.DomeLight(p) for p in Usd.PrimRange(root) if p.IsA(UsdLux.DomeLight)]
    if not lights:
        light = UsdLux.DomeLight.Define(stage, world_root + "/HDRBackground/Dome")
        light.CreateIntensityAttr(1000.0)
        lights.append(light)
    for light in lights:
        light.CreateTextureFileAttr(str(texture_file))
        light.CreateTextureFormatAttr("latlong")
    path = world_root + "/HDRBackground/Panorama"
    mesh = UsdGeom.Mesh.Define(stage, path)
    if not mesh.GetPointsAttr().HasAuthoredValueOpinion():
        rings, columns, radius = 16, 32, 50.0
        points = []
        uv = []
        for ring in range(rings + 1):
            theta = math.pi * ring / rings
            for column in range(columns + 1):
                phi = 2 * math.pi * column / columns
                points.append(
                    (
                        radius * math.sin(theta) * math.cos(phi),
                        radius * math.sin(theta) * math.sin(phi),
                        radius * math.cos(theta),
                    )
                )
                uv.append((column / columns, 1 - ring / rings))
        faces = []
        for ring in range(rings):
            for column in range(columns):
                a = ring * (columns + 1) + column
                faces.extend([a, a + columns + 1, a + columns + 2, a + 1])
        mesh.CreatePointsAttr(points)
        mesh.CreateFaceVertexCountsAttr([4] * rings * columns)
        mesh.CreateFaceVertexIndicesAttr(faces)
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateOrientationAttr("leftHanded")
        mesh.CreateDoubleSidedAttr(True)
        UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "vertex").Set(uv)
        mesh.GetPrim().CreateAttribute("primvars:doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(True)
    material = UsdShade.Material.Define(stage, world_root + "/HDRBackground/Material")
    surface = UsdShade.Shader.Define(stage, str(material.GetPath()) + "/Surface")
    surface.CreateIdAttr("UsdPreviewSurface")
    surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0))
    texture = UsdShade.Shader.Define(stage, str(material.GetPath()) + "/Texture")
    texture.CreateIdAttr("UsdUVTexture")
    texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(texture_file))
    texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
    texture.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    texture.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
    reader = UsdShade.Shader.Define(stage, str(material.GetPath()) + "/UV")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
    surface.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(texture.ConnectableAPI(), "rgb")
    material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    mesh.GetPrim().CreateAttribute("geniesim:hdrSha256", Sdf.ValueTypeNames.String).Set(selection["sha256"])
    return selection

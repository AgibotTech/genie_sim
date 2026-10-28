# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Per-visual PBR isolation, including composed AS3 instances (no Kit app)."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
from xml.etree import ElementTree as ET

import pytest

from genie_sim_engine_isaacsim.assemble import _robot_impl as impl


def _xml(values):
    visuals = []
    for value in values:
        block = f"<material_override><roughness>{value}</roughness></material_override>" if value is not None else ""
        visuals.append(f'<visual><geometry><box size="1 1 1"/></geometry>{block}</visual>')
    return '<robot name="robot"><link name="link">' + "".join(visuals) + "</link></robot>"


def test_repeated_visuals_get_unique_converter_names_without_modifying_source():
    source = _xml([0.2, 0.8, None])
    prepared, specs = impl._prepare_material_override_urdf(source)
    root = ET.fromstring(prepared)
    assert [v.get("name") for v in root.findall("link/visual")] == ["geniesim_visual_0", "geniesim_visual_1", None]
    assert [s["target"] for s in specs] == ["geniesim_visual_0", "geniesim_visual_1"]
    assert impl._parse_material_override_blocks(source) == specs
    assert "material_override" not in prepared
    assert ET.fromstring(source).find("link/visual").get("name") is None


def test_generated_names_do_not_collide_with_existing_visuals_or_links():
    source = _xml([0.2]).replace("<visual>", '<visual name="geniesim_visual_0">')
    source = source.replace("</robot>", '<link name="geniesim_visual_0_"/></robot>')
    _, specs = impl._prepare_material_override_urdf(source)
    assert specs[0]["target"] == "geniesim_visual_0__"


@pytest.mark.parametrize("value,expected", [("-0.2", 0), ("1.7", 1), ("nan", None), ("inf", None), ("bad", None)])
def test_pbr_fields_warn_and_clamp_or_ignore_nonfinite_values(value, expected):
    with pytest.warns(RuntimeWarning):
        _, specs = impl._prepare_material_override_urdf(_xml([value]))
    assert (specs[0]["entry"]["roughness"] if specs else None) == expected


@pytest.fixture
def usd():
    pytest.importorskip("pxr.Usd")
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    return SimpleNamespace(Gf=Gf, Sdf=Sdf, Usd=Usd, UsdGeom=UsdGeom, UsdPhysics=UsdPhysics, UsdShade=UsdShade)


def _material(usd, stage, path):
    mat = usd.UsdShade.Material.Define(stage, path)
    mat.CreateInput("roughness", usd.Sdf.ValueTypeNames.Float).Set(0.5)
    mat.CreateInput("metallic", usd.Sdf.ValueTypeNames.Float).Set(0.1)
    shader = usd.UsdShade.Shader.Define(stage, path + "/surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness", usd.Sdf.ValueTypeNames.Float).ConnectToSource(mat.GetInput("roughness"))
    shader.CreateInput("metallic", usd.Sdf.ValueTypeNames.Float).ConnectToSource(mat.GetInput("metallic"))
    texture = usd.UsdShade.Shader.Define(stage, path + "/texture")
    texture.CreateIdAttr("UsdUVTexture")
    texture.CreateInput("file", usd.Sdf.ValueTypeNames.Asset).Set("albedo.png")
    texture.CreateOutput("rgb", usd.Sdf.ValueTypeNames.Float3)
    shader.CreateInput("diffuseColor", usd.Sdf.ValueTypeNames.Color3f).ConnectToSource(texture.ConnectableAPI(), "rgb")
    shader.CreateOutput("surface", usd.Sdf.ValueTypeNames.Token)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def _stage(usd):
    s = usd.Usd.Stage.CreateInMemory()
    root = usd.UsdGeom.Xform.Define(s, "/robot").GetPrim()
    s.SetDefaultPrim(root)
    link = usd.UsdGeom.Xform.Define(s, "/robot/link").GetPrim()
    usd.UsdPhysics.RigidBodyAPI.Apply(link)
    return s


def _bound(usd, stage, path):
    return usd.UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(path)).ComputeBoundMaterial()[0]


def _assert_material(usd, mat, roughness):
    assert mat.GetInput("roughness").Get() == pytest.approx(roughness)
    assert mat.GetInput("metallic").Get() == pytest.approx(0.1)
    surface = mat.GetSurfaceOutput().GetConnectedSource()[0]
    assert surface.GetInput("roughness").GetConnectedSource()[0].GetPath() == mat.GetPath()
    texture = surface.GetInput("diffuseColor").GetConnectedSource()[0]
    assert texture.GetInput("file").Get().path == "albedo.png"
    assert texture.GetPath().HasPrefix(mat.GetPath())


def test_shared_material_and_repeated_shapes_are_isolated_with_textures_preserved(usd):
    s = _stage(usd)
    original = _material(usd, s, "/robot/Materials/shared")
    _, specs = impl._prepare_material_override_urdf(_xml([0.2, 0.8, None]))
    for name in ("geniesim_visual_0", "geniesim_visual_1", "unmodified"):
        mesh = usd.UsdGeom.Cube.Define(s, "/robot/link/" + name).GetPrim()
        usd.UsdShade.MaterialBindingAPI.Apply(mesh).Bind(original)
    for _ in range(2):  # Reapplying must not nest clones or mutate originals.
        assert impl._apply_material_overrides(s, specs, "robot") == 2
    mats = [_bound(usd, s, "/robot/link/" + n) for n in ("geniesim_visual_0", "geniesim_visual_1", "unmodified")]
    for mat, expected in zip(mats, [0.2, 0.8, 0.5]):
        _assert_material(usd, mat, expected)
    assert len({mat.GetPath() for mat in mats}) == 3
    assert len([p for p in s.Traverse() if p.IsA(usd.UsdShade.Material)]) == 3
    _assert_material(usd, original, 0.5)


def test_as3_instances_are_patched_locally_and_survive_reopen_without_payload_edits(usd, tmp_path):
    payload = tmp_path / "instances.usda"
    source = usd.Usd.Stage.CreateNew(str(payload))
    usd.UsdGeom.Xform.Define(source, "/Visual")
    mat = _material(usd, source, "/Visual/Materials/shared")
    mesh = usd.UsdGeom.Mesh.Define(source, "/Visual/mesh").GetPrim()
    usd.UsdShade.MaterialBindingAPI.Apply(mesh).Bind(mat)
    source.GetRootLayer().Save()
    payload_bytes = payload.read_bytes()
    s = _stage(usd)
    for name in ("geniesim_visual_0", "geniesim_visual_1", "unmodified"):
        wrapper = usd.UsdGeom.Xform.Define(s, "/robot/link/" + name).GetPrim()
        wrapper.GetReferences().AddReference(str(payload), "/Visual")
        wrapper.SetInstanceable(True)
    _, specs = impl._prepare_material_override_urdf(_xml([0.2, 0.8]))
    assert impl._apply_material_overrides(s, specs, "robot") == 2
    assert s.GetPrimAtPath("/robot/link/unmodified").IsInstance()
    assert payload.read_bytes() == payload_bytes
    output = tmp_path / "robot.usda"
    s.GetRootLayer().Export(str(output))
    reopened = usd.Usd.Stage.Open(str(output))
    for name, value in [("geniesim_visual_0", 0.2), ("geniesim_visual_1", 0.8), ("unmodified", 0.5)]:
        _assert_material(usd, _bound(usd, reopened, "/robot/link/" + name + "/mesh"), value)


def test_multiple_embedded_materials_keep_physics_and_other_visual_bindings(usd):
    s = _stage(usd)
    first = _material(usd, s, "/robot/Materials/first")
    second = _material(usd, s, "/robot/Materials/second")
    _, specs = impl._prepare_material_override_urdf(_xml([0.2]))
    wrapper = "/robot/link/geniesim_visual_0"
    usd.UsdGeom.Xform.Define(s, wrapper)
    mesh = usd.UsdGeom.Mesh.Define(s, wrapper + "/mesh").GetPrim()
    binding = usd.UsdShade.MaterialBindingAPI.Apply(mesh)
    binding.Bind(first)
    binding.Bind(second, materialPurpose="physics")
    subset = usd.UsdGeom.Subset.Define(s, wrapper + "/mesh/subset").GetPrim()
    usd.UsdShade.MaterialBindingAPI.Apply(subset).Bind(second)
    other = usd.UsdGeom.Mesh.Define(s, "/robot/link/unmodified").GetPrim()
    usd.UsdShade.MaterialBindingAPI.Apply(other).Bind(first)
    assert impl._apply_material_overrides(s, specs, "robot") == 2
    for path in (wrapper + "/mesh", wrapper + "/mesh/subset"):
        _assert_material(usd, _bound(usd, s, path), 0.2)
    assert binding.GetDirectBindingRel("physics").GetTargets() == [second.GetPath()]
    assert _bound(usd, s, "/robot/link/unmodified").GetPath() == first.GetPath()
    _assert_material(usd, first, 0.5)
    _assert_material(usd, second, 0.5)


def test_missing_visual_never_falls_back_to_unrelated_geometry(usd):
    s = _stage(usd)
    usd.UsdGeom.Cube.Define(s, "/robot/link/unrelated")
    with pytest.raises(ValueError, match="regenerate robot USD"):
        impl._apply_material_overrides(s, impl._parse_material_override_blocks(_xml([0.2])), "robot")


def test_shader_only_material_and_defensive_value_validation(usd):
    s = _stage(usd)
    mat = usd.UsdShade.Material.Define(s, "/robot/Materials/m")
    shader = usd.UsdShade.Shader.Define(s, "/robot/Materials/m/surface")
    shader.CreateInput("roughness", usd.Sdf.ValueTypeNames.Float).Set(0.5)
    shader.CreateInput("metallic", usd.Sdf.ValueTypeNames.Float).Set(0.1)
    shader.CreateOutput("surface", usd.Sdf.ValueTypeNames.Token)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    with pytest.warns(RuntimeWarning):
        assert impl._set_pbr_inputs(mat.GetPrim(), {"roughness": 2, "metallic": float("nan")})
    assert shader.GetInput("roughness").Get() == 1
    assert shader.GetInput("metallic").Get() == pytest.approx(0.1)


def test_postprocess_failure_cannot_write_success_stamp(monkeypatch, tmp_path, capsys):
    import importlib.metadata

    usd_path = tmp_path / "robot.usda"
    usd_path.write_text("x" * 300)  # Converter output exists even when its overlay fails.
    stamp = tmp_path / "urdf.sha256"
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "6.2.0.0")
    monkeypatch.setitem(sys.modules, "isaacsim", SimpleNamespace(SimulationApp=lambda _: object()))
    monkeypatch.setitem(
        sys.modules, "isaacsim.asset.importer.urdf", SimpleNamespace(URDFImporterConfig=SimpleNamespace)
    )
    monkeypatch.setitem(
        sys.modules, "genie_sim_engine_isaacsim.kit.launch_config", SimpleNamespace(prepare_launch_config=lambda x: x)
    )

    def fail(*args, **kwargs):
        raise ValueError("bad material target")

    monkeypatch.setattr(impl, "_convert_urdf_to_usd_as3", fail)
    monkeypatch.setattr(impl.os, "_exit", lambda code: (_ for _ in ()).throw(SystemExit(code)))
    with pytest.raises(SystemExit) as error:
        impl._convert_urdf_to_usd("unused.urdf", str(usd_path), success_stamp=(str(stamp), "hash"))
    assert error.value.code == 1
    assert not stamp.exists()
    stderr = capsys.readouterr().err
    assert "Traceback (most recent call last)" in stderr
    assert "ValueError: bad material target" in stderr


@pytest.mark.parametrize("geometry", ["box", "mesh"])
def test_installed_isaac62_converter_preserves_visual_identity(usd, tmp_path, geometry):
    if importlib.util.find_spec("urdf_usd_converter") is None:
        pytest.skip("Isaac 6.2 converter not installed")
    from urdf_usd_converter import Converter

    xml = _xml([0.2, 0.8]).replace("</geometry>", '</geometry><material name="red"><color rgba="1 0 0 1"/></material>')
    if geometry == "mesh":
        mesh = tmp_path / "shared.obj"
        mesh.write_text("o shared\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
        xml = xml.replace('<box size="1 1 1"/>', f'<mesh filename="{mesh}"/>')
    prepared, specs = impl._prepare_material_override_urdf(xml)
    source = tmp_path / "robot.urdf"
    source.write_text(prepared)
    asset = Converter(layer_structure=False, scene=False).convert(str(source), str(tmp_path / "usd"))
    s = usd.Usd.Stage.Open(asset.path)
    prefix = s.GetDefaultPrim().GetName()
    assert impl._apply_material_overrides(s, specs, prefix) == 2
    link = impl._find_link_prim(s, prefix, "link")
    for spec in specs:
        wrapper = link.GetChild(spec["target"])
        mats = impl._collect_wrapper_materials(wrapper)
        assert mats
        for path in mats:
            mat = usd.UsdShade.Material(s.GetPrimAtPath(path))
            assert mat.GetInput("roughness").Get() == pytest.approx(spec["entry"]["roughness"])

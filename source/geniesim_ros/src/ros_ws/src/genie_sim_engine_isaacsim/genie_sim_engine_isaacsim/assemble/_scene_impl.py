# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Isaac Sim scene assembler — impl module.

Exposes :func:`assemble_scene` for :class:`IsaacSceneStage`. Isaac Sim / pxr imports
remain gated inside the individual helper functions so importing this
module does NOT require a running Kit."""

from __future__ import annotations

import os
import sys

import yaml

# Shared render_layer.usda authoring (pose / synthesized intrinsics / fisheye /
# RenderProduct / FreeCam) — identical to the MuJoCo assembler. Safe to import at
# module scope: render_layer gates its own pxr imports, so this stays Kit-free.
from genie_sim_engine_core.common import render_layer as _render_layer
from genie_sim_engine_core.schema import camera_mode_rank


def _copy_xform_ops(src_prim, dst_prim):
    from pxr import UsdGeom

    src_xformable = UsdGeom.Xformable(src_prim)
    if not src_xformable:
        return

    dst_xformable = UsdGeom.Xformable(dst_prim)

    for op in src_xformable.GetOrderedXformOps():
        attr = op.GetAttr()
        op_type = op.GetOpType()
        precision = op.GetPrecision()

        full_name = attr.GetName()
        suffix = ""
        parts = full_name.split(":")
        if len(parts) > 2:
            suffix = parts[-1]

        new_op = dst_xformable.AddXformOp(op_type, precision, suffix)
        if attr.HasValue():
            new_op.Set(attr.Get())

    order_attr = src_prim.GetAttribute("xformOpOrder")
    if order_attr and order_attr.HasValue():
        dst_order = dst_prim.GetAttribute("xformOpOrder")
        if dst_order:
            dst_order.Set(order_attr.Get())


def _copy_mesh_attrs(src_prim, dst_prim):
    mesh_attrs = [
        "points",
        "faceVertexIndices",
        "faceVertexCounts",
        "normals",
        "primvars:st",
        "primvars:st:indices",
        "subdivisionScheme",
        "doubleSided",
    ]
    for name in mesh_attrs:
        src_attr = src_prim.GetAttribute(name)
        if src_attr and src_attr.HasValue():
            val = src_attr.Get()
            dst_attr = dst_prim.CreateAttribute(name, src_attr.GetTypeName())
            dst_attr.Set(val)
            if name.startswith("primvars:"):
                interp = src_attr.GetMetadata("interpolation")
                if interp:
                    dst_attr.SetMetadata("interpolation", interp)

    extent_attr = src_prim.GetAttribute("extent")
    if extent_attr and extent_attr.HasValue():
        dst_prim.CreateAttribute("extent", extent_attr.GetTypeName()).Set(extent_attr.Get())


def _copy_material_binding(src_stage, dst_stage, src_prim, dst_prim):
    from pxr import UsdShade

    binding_api = UsdShade.MaterialBindingAPI(src_prim)
    bound = binding_api.ComputeBoundMaterial()
    if not bound or not bound[0]:
        return

    mat = bound[0]
    mat_path = mat.GetPath().pathString

    if dst_stage.GetPrimAtPath(mat_path):
        UsdShade.MaterialBindingAPI.Apply(dst_prim).Bind(UsdShade.Material(dst_stage.GetPrimAtPath(mat_path)))
        return

    _export_material(src_stage, dst_stage, mat)

    dst_mat = dst_stage.GetPrimAtPath(mat_path)
    if dst_mat:
        UsdShade.MaterialBindingAPI.Apply(dst_prim).Bind(UsdShade.Material(dst_mat))


def _export_material(src_stage, dst_stage, mat):
    from pxr import Sdf, UsdShade

    mat_path = mat.GetPath().pathString

    parent = str(Sdf.Path(mat_path).GetParentPath())
    if parent and parent != "/" and not dst_stage.GetPrimAtPath(parent):
        dst_stage.DefinePrim(parent, "Scope")

    dst_mat_prim = dst_stage.DefinePrim(mat_path, "Material")
    dst_mat = UsdShade.Material(dst_mat_prim)

    for src_output in mat.GetOutputs():
        dst_mat.CreateOutput(src_output.GetBaseName(), src_output.GetTypeName())
        connected = src_output.GetConnectedSources()
        if connected and connected[0]:
            for conn_info in connected[0]:
                shader_path = conn_info.source.GetPath().pathString
                _export_shader(src_stage, dst_stage, shader_path)
                dst_output = dst_mat.GetOutput(src_output.GetBaseName())
                dst_shader = UsdShade.Shader(dst_stage.GetPrimAtPath(shader_path))
                if dst_shader:
                    dst_output.ConnectToSource(dst_shader.ConnectableAPI(), conn_info.sourceName)


def _export_shader(src_stage, dst_stage, shader_path):
    from pxr import Sdf, UsdShade

    if dst_stage.GetPrimAtPath(shader_path):
        return

    src_prim = src_stage.GetPrimAtPath(shader_path)
    if not src_prim:
        return

    parent = str(Sdf.Path(shader_path).GetParentPath())
    if parent and parent != "/" and not dst_stage.GetPrimAtPath(parent):
        dst_stage.DefinePrim(parent, "Scope")

    dst_prim = dst_stage.DefinePrim(shader_path, "Shader")
    src_shader = UsdShade.Shader(src_prim)
    dst_shader = UsdShade.Shader(dst_prim)

    for attr in src_prim.GetAttributes():
        name = attr.GetName()
        if name.startswith("info:"):
            if attr.HasValue():
                dst_prim.CreateAttribute(name, attr.GetTypeName()).Set(attr.Get())

    for inp in src_shader.GetInputs():
        val = inp.GetAttr().Get() if inp.GetAttr().HasValue() else None
        dst_inp = dst_shader.CreateInput(inp.GetBaseName(), inp.GetTypeName())
        if val is not None:
            dst_inp.Set(val)
        connected = inp.GetConnectedSources()
        if connected and connected[0]:
            for conn_info in connected[0]:
                sub_shader_path = conn_info.source.GetPath().pathString
                _export_shader(src_stage, dst_stage, sub_shader_path)
                sub_shader = UsdShade.Shader(dst_stage.GetPrimAtPath(sub_shader_path))
                if sub_shader:
                    dst_inp.ConnectToSource(sub_shader.ConnectableAPI(), conn_info.sourceName)

    for outp in src_shader.GetOutputs():
        dst_shader.CreateOutput(outp.GetBaseName(), outp.GetTypeName())


def _export_prim_recursive(src_stage, dst_stage, src_prim):
    from pxr import UsdGeom

    path = src_prim.GetPath().pathString

    if "collision" in path.lower():
        return

    if src_prim.IsA(UsdGeom.Xform) or src_prim.GetTypeName() == "Xform":
        dst_prim = dst_stage.DefinePrim(path, "Xform")
        _copy_xform_ops(src_prim, dst_prim)
    elif src_prim.IsA(UsdGeom.Mesh):
        dst_prim = dst_stage.DefinePrim(path, "Mesh")
        _copy_mesh_attrs(src_prim, dst_prim)
        _copy_xform_ops(src_prim, dst_prim)
        _copy_material_binding(src_stage, dst_stage, src_prim, dst_prim)
        for child in src_prim.GetChildren():
            if child.GetTypeName() == "GeomSubset":
                _export_geom_subset(src_stage, dst_stage, child)
        return
    elif src_prim.IsA(UsdGeom.Camera):
        return
    elif src_prim.IsA(UsdGeom.Scope):
        dst_stage.DefinePrim(path, "Scope")
    else:
        if not src_prim.GetChildren():
            return
        dst_stage.DefinePrim(path, src_prim.GetTypeName() or "Xform")
        if UsdGeom.Xformable(src_prim):
            dst_prim = dst_stage.GetPrimAtPath(path)
            _copy_xform_ops(src_prim, dst_prim)

    for child in src_prim.GetChildren():
        _export_prim_recursive(src_stage, dst_stage, child)


def _export_geom_subset(src_stage, dst_stage, src_prim):
    path = src_prim.GetPath().pathString
    dst_prim = dst_stage.DefinePrim(path, "GeomSubset")

    for attr_name in ["elementType", "familyName"]:
        src_attr = src_prim.GetAttribute(attr_name)
        if src_attr and src_attr.HasValue():
            dst_prim.CreateAttribute(attr_name, src_attr.GetTypeName()).Set(src_attr.Get())

    indices_attr = src_prim.GetAttribute("indices")
    if indices_attr and indices_attr.HasValue():
        dst_prim.CreateAttribute("indices", indices_attr.GetTypeName()).Set(indices_attr.Get())

    _copy_material_binding(src_stage, dst_stage, src_prim, dst_prim)


def _relativize_layer_paths(layer_path: str) -> None:
    """Rewrite absolute asset references in ``layer_path`` to ``./relpath``.

    Wraps :func:`common.usd_path_helpers.make_layer_asset_paths_relative`
    so all three asset-dump sites in this module (``robot_visual.usda``,
    ``render_layer.usda``, ``newton_scene.usda``) share the same
    portability story as the engine's ``robot_runtime.usda`` dump.

    Best-effort: if the helper isn't importable (direct python invocation
    outside the colcon overlay) or raises on a specific path, the dump
    keeps its absolute references and the function returns quietly.
    Without this rewrite the dump becomes non-portable: moving the
    ``/geniesim_assets`` tree breaks every saved reference.
    """
    if not layer_path or not os.path.isfile(layer_path):
        return
    try:
        from genie_sim_engine_newton.common.usd_path_helpers import (  # noqa: PLC0415
            make_layer_asset_paths_relative,
        )

        make_layer_asset_paths_relative(layer_path)
    except Exception as exc:  # noqa: BLE001
        print(
            f"[assemble_scene] WARN: path relativizer failed on " f"{layer_path}: {exc!r}; absolute paths left in.",
            flush=True,
        )


def _export_visual_robot(src_stage, robot_prefix, out_path):
    """Write a thin USDA that references the source robot stage.

    Rather than inlining all geometry (which produces a huge ASCII file),
    we emit a single-prim USDA that references the original robot.usda.
    The renderer resolves the reference at load time, keeping the file small.
    """
    from pxr import Usd, Sdf

    src_path = src_stage.GetRootLayer().realPath

    root_prim = src_stage.GetDefaultPrim()
    if not root_prim or not root_prim.IsValid():
        root_prim = src_stage.GetPrimAtPath(f"/{robot_prefix}")
    if not root_prim or not root_prim.IsValid():
        candidates = [p for p in src_stage.GetPseudoRoot().GetChildren() if p.IsValid()]
        if candidates:
            root_prim = candidates[0]

    if not root_prim or not root_prim.IsValid():
        print(
            f"[assemble_scene] WARN: cannot determine robot root for visual export "
            f"(default_prim missing, /{robot_prefix} missing, no top-level prims)",
            file=sys.stderr,
        )
        # write empty stage as fallback
        stage = Usd.Stage.CreateNew(out_path)
        stage.GetRootLayer().Save()
        return

    prim_path = root_prim.GetPath().pathString
    print(f"[assemble_scene] visual robot root: {prim_path}")

    if os.path.exists(out_path):
        os.remove(out_path)
    dst_stage = Usd.Stage.CreateNew(out_path)
    dst_stage.SetMetadata("upAxis", "Z")
    dst_stage.SetMetadata("metersPerUnit", 1.0)

    ref_prim = dst_stage.DefinePrim(prim_path)
    ref_prim.GetReferences().AddReference(src_path, primPath=Sdf.Path(prim_path))
    dst_stage.SetDefaultPrim(ref_prim)

    dst_stage.GetRootLayer().Save()
    # ``AddReference(src_path, ...)`` above authors the source robot.usda
    # path as ABSOLUTE.  Rewrite to ``@./<rel>@`` (or ``@../<rel>@``) so
    # the dump survives a relocation of the assets tree — same convention
    # as the engine's ``robot_runtime.usda`` dump.
    _relativize_layer_paths(out_path)
    print(f"[assemble_scene] robot visual usda written: {out_path}")


def author_render_layer(
    output_dir,
    robot_usd_path,
    robot_stage,
    robot_prefix,
    cameras,
    lidars,
    free_cam_w,
    free_cam_h,
    camera_mode="cam3",
):
    """Author ``render_layer.usda`` (camera + FreeCam + lidar RenderProducts
    under ``/RenderOVRTX``) and return what the manifest build needs.

    Extracted verbatim from ``assemble_scene`` so the rlinf runtime task
    loader can produce the same render layer for a benchmark task without
    re-running the whole assemble pipeline. Behaviour is identical; the only
    change is that the former locals are now parameters.

    Returns ``(render_layer_path, authored_cam_indices, lidar_manifest_entries)``.
    """
    from pxr import Gf, Sdf, Usd, UsdGeom, Vt  # noqa: F401

    out_path = os.path.join(output_dir, "render_layer.usda")
    if os.path.exists(out_path):
        os.remove(out_path)
    stage = Usd.Stage.CreateNew(out_path)

    root = UsdGeom.Scope.Define(stage, "/RenderOVRTX")
    stage.SetDefaultPrim(root.GetPrim())

    UsdGeom.Xform.Define(stage, "/RenderOVRTX/Cameras")

    # Track which YAML camera entries are actually authored on the
    # render layer. The set gates manifest emission below so no render
    # product is advertised without a backing prim — without this gate
    # OVRTX aborts with ``Invalid USD RenderProduct Prim:
    # /RenderOVRTX/Cam_0``.
    authored_cam_indices: set = set()
    mode_rank = camera_mode_rank(camera_mode)

    # Cameras are looked up *inside* the source ``robot.usda``. That stage's
    # actual root prim path is dictated by its own ``defaultPrim`` (commonly
    # ``/robot``), NOT by the scene YAML's ``robot_prefix`` (which only
    # controls where the robot is *placed* on the composed render layer /
    # rendering scene). Resolving the in-source lookup root here fixes the
    # case where ``robot_prefix`` is e.g. ``ur5`` but the source authored
    # everything under ``/robot``.
    src_default_prim = robot_stage.GetDefaultPrim()
    if src_default_prim and src_default_prim.IsValid():
        src_root_path = src_default_prim.GetPath().pathString
    else:
        candidate = robot_stage.GetPrimAtPath(f"/{robot_prefix}")
        if candidate and candidate.IsValid():
            src_root_path = f"/{robot_prefix}"
        else:
            top_children = [p for p in robot_stage.GetPseudoRoot().GetChildren() if p.IsValid()]
            src_root_path = top_children[0].GetPath().pathString if top_children else f"/{robot_prefix}"

    for i, cam_cfg in enumerate(cameras):
        if not isinstance(cam_cfg, dict):
            print(
                f"[assemble_scene] WARN: cameras[{i}] is not a mapping, skipping",
                file=sys.stderr,
            )
            continue
        if not cam_cfg.get("enabled", True):
            continue
        if mode_rank < camera_mode_rank(cam_cfg.get("camera_group", "cam3")):
            continue
        cam_rel_path = cam_cfg.get("prim_path")
        if not cam_rel_path:
            print(
                f"[assemble_scene] WARN: cameras[{i}] missing 'prim_path', skipping",
                file=sys.stderr,
            )
            continue
        sensor = cam_cfg.get("sensor") or {}
        cam_w = int(sensor.get("width", 1280))
        cam_h = int(sensor.get("height", 800))
        topic_block = cam_cfg.get("topic") or {}
        want_depth = bool(topic_block.get("depth", "").strip())
        src_path = f"{src_root_path}/{cam_rel_path}"
        src_prim = robot_stage.GetPrimAtPath(src_path)
        # Two authoring modes:
        #   * src_prim exists AND is a UsdGeom.Camera
        #       → "mirror" mode: copy intrinsics/xform from the source prim
        #         (preferred — robot.usda was authored with cameras baked in).
        #   * src_prim missing or not a Camera
        #       → "synthesize" mode: author a fresh UsdGeom.Camera under
        #         /RenderOVRTX/Cameras/<rel_path>, deriving intrinsics from
        #         the YAML ``intrinsic`` block and pose from the YAML
        #         ``extrinsic`` block. Required for the URDF→USD route
        #         (URDFs cannot describe USD Camera prims), so scenes like
        #         scene_flat_acone / scene_flat_ur5 still get a working head
        #         camera even though their robot.usda has none.
        synthesize_camera = not (src_prim and src_prim.IsA(UsdGeom.Camera))
        if synthesize_camera:
            print(
                f"[assemble_scene] cameras[{i}]: source prim {src_path} not a Camera "
                f"(robot_prefix={robot_prefix!r}, src_root_path={src_root_path!r}); "
                f"synthesizing from YAML intrinsic/extrinsic"
            )

        parts = cam_rel_path.split("/")
        parent_path = "/RenderOVRTX/Cameras"
        for part in parts[:-1]:
            parent_path += f"/{part}"
            if not stage.GetPrimAtPath(parent_path):
                xf = UsdGeom.Xform.Define(stage, parent_path)
                src_parent = robot_stage.GetPrimAtPath(f"{src_root_path}/{'/'.join(parts[: parts.index(part) + 1])}")
                if src_parent:
                    _copy_xform_ops(src_parent, xf.GetPrim())

        cam_dst_path = parent_path + "/" + parts[-1] if len(parts) > 1 else f"/RenderOVRTX/Cameras/{parts[-1]}"
        dst_cam = UsdGeom.Camera.Define(stage, cam_dst_path)

        if not synthesize_camera:
            for attr_name in [
                "focalLength",
                "horizontalAperture",
                "verticalAperture",
                "clippingRange",
            ]:
                src_attr = src_prim.GetAttribute(attr_name)
                if src_attr and src_attr.HasValue():
                    dst_cam.GetPrim().CreateAttribute(attr_name, src_attr.GetTypeName()).Set(src_attr.Get())
            cam_extrinsic = cam_cfg.get("extrinsic")
            if cam_extrinsic:
                # YAML extrinsic overrides the baked robot.usda pose. Written verbatim (no frame
                # conversion) — the value must already be in the USD/Isaac graphics camera convention.
                _render_layer.author_camera_pose_from_extrinsic(dst_cam.GetPrim(), cam_extrinsic)
                print(f"[assemble_scene] camera[{i}]: pose from YAML extrinsic (override robot.usda)")
            else:
                _copy_xform_ops(src_prim, dst_cam.GetPrim())
        else:
            # ----- Synthesize intrinsics from YAML --------------------------------
            # Shared with the MuJoCo assembler (genie_sim_engine_core.common.render_layer):
            # pins horizontalAperture to 20.955 so rendered fx matches yaml.fx, derives
            # verticalAperture from fy so non-square pixels are preserved. These USD attrs only
            # steer the renderer's projection — the published CameraInfo is built from the
            # manifest's verbatim YAML intrinsics, not from here.
            intrinsic = cam_cfg.get("intrinsic") or {}
            _render_layer.synthesize_camera_intrinsics(
                dst_cam.GetPrim(),
                fx=float(intrinsic.get("fx", 610.0)),
                fy=float(intrinsic.get("fy", 610.0)),
                width=cam_w,
                height=cam_h,
                min_range=float(sensor.get("min_range", 0.01)),
                max_range=float(sensor.get("max_range", 10000.0)),
            )

            # ----- Synthesize pose from YAML extrinsic ----------------------------
            # ``extrinsic.xyz`` (metres) and ``extrinsic.wxyz`` (unit quaternion, w-first) describe
            # the camera pose relative to its parent body link, written verbatim (USD graphics
            # convention). Default is identity (camera coincident with the parent link).
            _render_layer.author_camera_pose_from_extrinsic(dst_cam.GetPrim(), cam_cfg.get("extrinsic") or {})

        # Lens distortion: when the yaml marks this camera as a fisheye, author the
        # OpenCvFisheye schema so ovrtx renders the barrel distortion. Pose/aperture come from
        # the mirror/synthesize block above; this only adds the omni:lensdistortion:* attrs.
        # ``opencvPinhole`` is the same deal for lenses that calibrate as an OpenCV pinhole
        # (the head stereo pair) -- it authors the full rational k1..k6 + p1/p2. Note the
        # intrinsic dict handed over here is the RAW yaml mapping, not the CameraIntrinsic
        # dataclass, which is why k5/k6 survive (the dataclass, and hence the manifest and
        # the published CameraInfo, still stop at k4).
        cam_model = (cam_cfg.get("model") or "").strip().lower()
        if cam_model == "opencvfisheye":
            _render_layer.author_opencv_fisheye(dst_cam.GetPrim(), cam_cfg.get("intrinsic") or {}, cam_w, cam_h)
            print(f"[assemble_scene] camera[{i}]: applied OpenCvFisheye lens distortion")
        elif cam_model == "opencvpinhole":
            _render_layer.author_opencv_pinhole(dst_cam.GetPrim(), cam_cfg.get("intrinsic") or {}, cam_w, cam_h)
            print(f"[assemble_scene] camera[{i}]: applied OpenCvPinhole lens distortion")
        elif cam_model:
            print(
                f"[assemble_scene] WARN: camera[{i}]: unknown lens model {cam_model!r}, "
                f"rendering as an undistorted pinhole",
                file=sys.stderr,
            )

        cam_name = f"Cam_{i}"
        rp_path = f"/RenderOVRTX/{cam_name}"
        _render_layer.define_render_product(
            stage,
            rp_path=rp_path,
            cam_path=cam_dst_path,
            width=cam_w,
            height=cam_h,
            want_depth=want_depth,
        )

        authored_cam_indices.add(i)
        depth_note = "  +depth" if want_depth else ""
        print(f"[assemble_scene] camera[{i}]: {cam_dst_path}  ({cam_w}x{cam_h}){depth_note}")

    free_cam_path = "/RenderOVRTX/Cameras/FreeCam"
    _render_layer.define_free_cam(stage, width=free_cam_w, height=free_cam_h)
    print(f"[assemble_scene] free cam:   {free_cam_path}  ({free_cam_w}x{free_cam_h})")

    # ----- Lidar render products ---------------------------------------------
    # OmniLidar sensors are baked into the robot USDA (e.g. /genie/base_link/livox_back).
    # ovrtx only resolves a RenderProduct's ``rel camera`` to prims inside the render layer's
    # own subtree, so each OmniLidar is *referenced* into /RenderOVRTX/Lidars/<body>/<leaf>
    # (mirroring how cameras are copied) and the RenderProduct targets that copy. The render
    # node mirrors <body>'s world transform onto /RenderOVRTX/Lidars/<body> each frame.
    lidar_manifest_entries: list = []
    if lidars:
        UsdGeom.Scope.Define(stage, "/RenderOVRTX/Lidars")
    for i, lidar_cfg in enumerate(lidars):
        if not isinstance(lidar_cfg, dict):
            print(
                f"[assemble_scene] WARN: lidars[{i}] is not a mapping, skipping",
                file=sys.stderr,
            )
            continue
        lidar_rel_path = lidar_cfg.get("prim_path")
        if not lidar_rel_path:
            print(
                f"[assemble_scene] WARN: lidars[{i}] missing 'prim_path', skipping",
                file=sys.stderr,
            )
            continue
        src_lidar_path = f"{src_root_path}/{lidar_rel_path}"
        src_lidar_prim = robot_stage.GetPrimAtPath(src_lidar_path)
        if not src_lidar_prim or not src_lidar_prim.IsValid():
            print(
                f"[assemble_scene] WARN: lidars[{i}] source prim {src_lidar_path} not found " f"in robot USD; skipping",
                file=sys.stderr,
            )
            continue
        if src_lidar_prim.GetTypeName() != "OmniLidar":
            print(
                f"[assemble_scene] WARN: lidars[{i}] {src_lidar_path} is "
                f"'{src_lidar_prim.GetTypeName()}', not OmniLidar; skipping",
                file=sys.stderr,
            )
            continue

        parts = lidar_rel_path.split("/")
        body = parts[0]
        leaf = parts[-1]
        lidars_body_path = f"/RenderOVRTX/Lidars/{body}"
        if not stage.GetPrimAtPath(lidars_body_path):
            UsdGeom.Xform.Define(stage, lidars_body_path)
        lidar_dst_path = f"{lidars_body_path}/{leaf}"

        # Reference the baked OmniLidar into the render layer (keeps the RenderProduct camera
        # target inside /RenderOVRTX). Its local xform (relative to <body>) comes with it.
        ref_prim = stage.DefinePrim(lidar_dst_path)
        ref_prim.GetReferences().AddReference(robot_usd_path, primPath=Sdf.Path(src_lidar_path))

        # Apply the s001 emitter-state schema (prepend, keep the referenced Core API).
        emitter_schema = Sdf.TokenListOp()
        emitter_schema.prependedItems = ["OmniSensorGenericLidarCoreEmitterStateAPI:s001"]
        ref_prim.SetMetadata("apiSchemas", emitter_schema)

        # Cartesian xyz output; emit only on full-scan completion (gives ~scanRate Hz output).
        ref_prim.CreateAttribute("omni:sensor:Core:elementsCoordsType", Sdf.ValueTypeNames.Token).Set("CARTESIAN")
        ref_prim.CreateAttribute("omni:sensor:Core:partialOutputs", Sdf.ValueTypeNames.Bool).Set(False)
        # Capture the whole 360 deg as one instantaneous snapshot at a single pose. The default
        # rotary scan spreads a full scan across several 30 Hz render steps at different robot
        # poses (NONCOMPENSATED), which skews straight walls during rotation; instantLidar removes
        # the time dimension so output matches the rmagine ray-cast (one frozen-pose scan).
        ref_prim.CreateAttribute("omni:sensor:Core:instantLidar", Sdf.ValueTypeNames.Bool).Set(True)
        # Drop non-returning rays at the source so Counts is the valid-hit count; otherwise
        # ovrtx keeps invalid points (here =1) and they show up as cone/fan noise at far range.
        ref_prim.CreateAttribute("omni:sensor:Core:skipDroppingInvalidPoints", Sdf.ValueTypeNames.Bool).Set(False)

        # Clamp fireTimeNs into ovrtx's firing window (~27.7us); larger spans yield zero
        # points. Pure time rescale — does not change point XYZ.
        ft_attr = src_lidar_prim.GetAttribute("omni:sensor:Core:emitterState:s001:fireTimeNs")
        ft = list(ft_attr.Get()) if (ft_attr and ft_attr.HasValue()) else []
        if ft:
            ft_max = max(ft)
            ft_scale = min(1.0, 26000.0 / ft_max) if ft_max > 0 else 1.0
            if ft_scale < 1.0:
                ref_prim.CreateAttribute(
                    "omni:sensor:Core:emitterState:s001:fireTimeNs",
                    Sdf.ValueTypeNames.UIntArray,
                ).Set(Vt.UIntArray([int(x * ft_scale) for x in ft]))

        # RenderProduct + PointCloud RenderVar.
        rp_path = f"/RenderOVRTX/Lidar_{i}"
        rp_prim = stage.DefinePrim(rp_path, "RenderProduct")
        rp_prim.CreateRelationship("camera").SetTargets([Sdf.Path(lidar_dst_path)])
        pc_path = f"{rp_path}/PointCloud"
        pc_prim = stage.DefinePrim(pc_path, "RenderVar")
        pc_prim.CreateAttribute("sourceName", Sdf.ValueTypeNames.String, False, Sdf.VariabilityUniform).Set(
            "PointCloud"
        )
        pc_prim.CreateAttribute("channels", Sdf.ValueTypeNames.StringArray).Set(
            ["Coordinates", "Intensity", "Counts", "TimeOffsetNs", "Flags"]
        )
        rp_prim.CreateRelationship("orderedVars").SetTargets([Sdf.Path(pc_path)])

        # Static base_link<-sensor transform so the render node can publish in the body frame.
        # XformCache.GetLocalToWorldTransform returns identity on this instanceable stage, so
        # accumulate authored local transforms from the lidar up to (excluding) the body link.
        base_link_path = f"{src_root_path}/{body}"
        mat = Gf.Matrix4d(1.0)
        walk = src_lidar_prim
        while walk and walk.IsValid() and walk.GetPath().pathString != base_link_path:
            local = UsdGeom.Xformable(walk).GetLocalTransformation()
            local_mat = local[0] if isinstance(local, tuple) else local
            mat = mat * local_mat
            walk = walk.GetParent()

        # ovrtx's OmniLidar spins about its prim-local +Y axis (the ovrtx lidar example uses
        # rotateXYZ(90,0,-90) to bring that spin axis world-vertical). The baked livox prim only
        # carries the mount tilt, so without this the spin axis lands horizontal and the scan
        # covers azimuth grossly non-uniformly (whole sectors under-sampled). Compose the
        # canonical correction after the mount on the render-layer copy so it spins about base
        # +Z => uniform 360deg coverage. base_link_T_sensor below stays the original mount, so
        # the published point frame is unchanged (render node applies it with identity rotation).
        R_canon = Gf.Matrix4d(0, -1, 0, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0, 0, 0, 1)
        UsdGeom.Xformable(ref_prim).MakeMatrixXform().Set(R_canon * mat)

        t = mat.ExtractTranslation()
        q = mat.ExtractRotation().GetQuat()
        imag = q.GetImaginary()
        bl_xyz = [float(t[0]), float(t[1]), float(t[2])]
        bl_wxyz = [float(q.GetReal()), float(imag[0]), float(imag[1]), float(imag[2])]

        lidar_manifest_entries.append(
            {
                "render_product_path": rp_path,
                "topic": lidar_cfg.get("topic", ""),
                "frame_id": lidar_cfg.get("frame_id", body),
                "parent_body": body,
                "base_link_T_sensor": {"xyz": bl_xyz, "wxyz": bl_wxyz},
            }
        )
        print(f"[assemble_scene] lidar[{i}]: {lidar_dst_path}  -> {lidar_cfg.get('topic', '')}")

    stage.GetRootLayer().Save()
    # Render layer references the source robot stage (camera/lidar parents)
    # with absolute paths via ``AddReference(robot_usd_path, ...)`` above.
    # Rewrite to anchored-relative so moving the assets tree doesn't
    # break the references.
    _relativize_layer_paths(out_path)
    print(f"[assemble_scene] render layer written: {out_path}")

    return out_path, authored_cam_indices, lidar_manifest_entries


def assemble_scene(
    scene: str,
    output_dir: str = "/tmp/isaacsim_stage",
    base_path: str = "",
    write_manifest: bool = True,
) -> dict:
    from pxr import Gf, Sdf, Usd, UsdGeom, Vt

    cfg_path = os.path.abspath(scene)
    if not base_path:
        base_path = os.getcwd()

    if not cfg_path.endswith((".yaml", ".yml")):
        print(
            f"[assemble_scene] ERROR: --scene must be a YAML file (.yaml/.yml), got: {cfg_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    with open(cfg_path) as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        print(
            f"[assemble_scene] ERROR: scene YAML did not parse to a mapping: {cfg_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    robot_section = config.get("robot")
    if not isinstance(robot_section, dict):
        print(
            f"[assemble_scene] ERROR: scene YAML missing 'robot' mapping: {cfg_path}",
            file=sys.stderr,
        )
        sys.exit(1)
    try:
        robot_name = robot_section["robot_name"]
    except KeyError:
        print(
            f"[assemble_scene] ERROR: scene YAML missing 'robot.robot_name': {cfg_path}",
            file=sys.stderr,
        )
        sys.exit(1)
    robot_prefix = robot_section.get("robot_prefix", robot_name)
    cameras = config.get("cameras") or []
    # Single source for the camera tier filter: the scene YAML's top-level
    # ``camera_mode`` field (off/cam3/cam4/cam6/full, default cam3).
    camera_mode = config.get("camera_mode", "cam3")
    # Strip the depth sub-stream from any camera entry whose depth_group
    # tier exceeds the active camera_mode — in place, before
    # ``author_render_layer``'s USD-authoring loop and this function's own
    # manifest-emission loop both read the same ``cameras`` list, so one
    # pass here keeps them consistent. The overall per-entry camera_group
    # tier / enabled gate is applied separately inside author_render_layer.
    _depth_mode_rank = camera_mode_rank(camera_mode)
    for _cam_cfg in cameras:
        if not isinstance(_cam_cfg, dict):
            continue
        _depth_tier = _cam_cfg.get("depth_group") or _cam_cfg.get("camera_group", "cam3")
        if _depth_mode_rank < camera_mode_rank(_depth_tier):
            _topic = _cam_cfg.get("topic")
            if isinstance(_topic, dict) and _topic.get("depth"):
                _cam_cfg["topic"] = {**_topic, "depth": ""}
            _cam_cfg["dds_depth_topic"] = ""
    lidars = config.get("lidars") or []
    # FreeCam (interactive RViz-driven camera) resolution. Optional scene-yaml
    # override; defaults to 1280x720. Drives both the render-product resolution
    # (actual RTX image size) and the manifest CameraInfo below.
    free_cam_w = int(config.get("free_cam_width", 1280))
    free_cam_h = int(config.get("free_cam_height", 720))
    # ``scene:`` is optional. Common newton-standalone setups (e.g. a flat
    # ground + robot, no decorative props) leave the key out or set it to
    # null. Synthesize a minimal default world in that case — empty USDA +
    # a visible ground plane prim + a DomeLight — so OVRTX has something to
    # render. Newton's ``builder.add_ground_plane()`` always runs anyway, so
    # physics is unaffected; the prim is purely for the visualizer.
    scene_usd_rel = config.get("scene")
    if scene_usd_rel is None or (isinstance(scene_usd_rel, str) and not scene_usd_rel.strip()):
        from pxr import UsdLux

        empty_path = os.path.join(output_dir, "scene_empty.usda")
        os.makedirs(output_dir, exist_ok=True)
        # Always re-author so edits to this default land on every rebuild.
        if os.path.isfile(empty_path):
            os.remove(empty_path)
        empty_stage = Usd.Stage.CreateNew(empty_path)
        empty_stage.SetMetadata("upAxis", "Z")
        empty_stage.SetMetadata("metersPerUnit", 1.0)

        # Root /World Xform so referencers have somewhere to nest under.
        world = UsdGeom.Xform.Define(empty_stage, "/World")
        empty_stage.SetDefaultPrim(world.GetPrim())

        # Visual ground plane: 50×50 m, slight grey, sits at z=0 to match
        # Newton's add_ground_plane (which is at z=0 by default).
        ground = UsdGeom.Mesh.Define(empty_stage, "/World/GroundPlane")
        size = 25.0
        ground.CreatePointsAttr(
            [
                Gf.Vec3f(-size, -size, 0.0),
                Gf.Vec3f(size, -size, 0.0),
                Gf.Vec3f(size, size, 0.0),
                Gf.Vec3f(-size, size, 0.0),
            ]
        )
        ground.CreateFaceVertexCountsAttr([4])
        ground.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        ground.CreateExtentAttr([Gf.Vec3f(-size, -size, 0.0), Gf.Vec3f(size, size, 0.0)])
        ground.CreateDisplayColorAttr([Gf.Vec3f(0.4, 0.4, 0.4)])

        # DomeLight for ambient illumination so the renderer doesn't show a
        # near-black scene. ``intensity=1000`` is a typical Isaac Sim default
        # for an unstyled studio dome.
        dome = UsdLux.DomeLight.Define(empty_stage, "/World/DefaultDomeLight")
        dome.CreateIntensityAttr(1000.0)
        dome.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))

        empty_stage.GetRootLayer().Save()
        scene_usd_resolved = empty_path
        print(f"[assemble_scene] empty 'scene:' → synthesised {empty_path} (ground + dome light)")
    else:
        if not isinstance(scene_usd_rel, str):
            print(
                f"[assemble_scene] ERROR: scene YAML 'scene' must be a string path "
                f"or empty (got {type(scene_usd_rel).__name__}): {cfg_path}",
                file=sys.stderr,
            )
            sys.exit(1)
        scene_usd_abs = scene_usd_rel if os.path.isabs(scene_usd_rel) else os.path.join(base_path, scene_usd_rel)
        if os.path.isfile(scene_usd_abs):
            scene_usd_resolved = scene_usd_abs
        else:
            print(
                f"[assemble_scene] ERROR: scene USD not found: {scene_usd_abs}\n"
                f"  (scene YAML 'scene: {scene_usd_rel}', base_path={base_path})",
                file=sys.stderr,
            )
            sys.exit(1)
    # Gate 2: robot USDA source resolution. Precedence:
    #   1. ``robot.robot_source.prebaked.usda`` (explicit, wins over
    #      everything else) — path relative to ``base_path`` unless
    #      absolute. Set by scenes that ship a hand-authored robot USDA
    #      and want to skip the URDF→USD assemble_robot stage entirely
    #      (e.g. the mujoco engine path, no Isaac Sim 6.x requirement).
    #      If ``prebaked.visual_only: true``, the prebaked USD is used
    #      ONLY for visual rendering (robot_visual.usda); physics still
    #      uses the staged URDF. Needed when the prebaked has different
    #      joint topology from the URDF-based physics (e.g. IPC cloth tasks
    #      where visual = /genie/... but physics = /robot/...).
    #   2. ``robot.robot_source.urdf`` present → ``staged_robot_usd``
    #      (the assemble_robot output, must exist).
    #   3. Neither set → pre-baked assets/robot/<robot_name>/robot.usda
    #      (legacy convention).
    #
    # A missing staged file in case (2) is NOT silently fallback'd: that
    # almost always means assemble_robot crashed or the cache was
    # partially deleted, and silent fallback would load the wrong robot.
    _robot_source = robot_section.get("robot_source") or {}
    _urdf_block = _robot_source.get("urdf") if isinstance(_robot_source, dict) else None
    _prebaked_block = _robot_source.get("prebaked") if isinstance(_robot_source, dict) else None
    _prebaked_usda_rel = ""
    _prebaked_visual_only = False
    if isinstance(_prebaked_block, dict):
        _prebaked_usda_rel = str(_prebaked_block.get("usda") or "").strip()
        _prebaked_visual_only = bool(_prebaked_block.get("visual_only", False))
    robot_from_urdf = isinstance(_robot_source, dict) and "urdf" in _robot_source and not (_prebaked_usda_rel and not _prebaked_visual_only)

    staged_robot_usd = os.path.join(output_dir, "robot.usda")
    prebaked_robot_usd = os.path.join(base_path, "robot", robot_name, "robot.usda")

    # Resolve prebaked path once (used by both physics and visual paths below)
    prebaked_resolved = None
    if _prebaked_usda_rel:
        prebaked_resolved = (
            _prebaked_usda_rel if os.path.isabs(_prebaked_usda_rel) else os.path.join(base_path, _prebaked_usda_rel)
        )

    # Physics robot USD
    if _prebaked_usda_rel and not _prebaked_visual_only:
        robot_usd_path = prebaked_resolved
        robot_usd_origin = "prebaked (scene.robot.robot_source.prebaked.usda)"
    elif robot_from_urdf:
        robot_usd_path = staged_robot_usd
        robot_usd_origin = "staged (from the urdf_to_usd stage)"
    else:
        robot_usd_path = prebaked_robot_usd
        robot_usd_origin = "pre-baked assets/robot/<robot_name>/robot.usda"

    print(f"[assemble_scene] config:     {cfg_path}")
    print(f"[assemble_scene] robot:      {robot_name} (prefix={robot_prefix})")
    print(f"[assemble_scene] robot usd:  {robot_usd_path} [{robot_usd_origin}]")
    print(f"[assemble_scene] from_urdf:  {robot_from_urdf}")
    print(f"[assemble_scene] cameras:    {len(cameras)}")
    print(f"[assemble_scene] output dir: {output_dir}")
    print()

    os.makedirs(output_dir, exist_ok=True)

    if not os.path.isfile(robot_usd_path):
        print(
            f"[assemble_scene] ERROR: robot USD not found: {robot_usd_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Open with LoadAll so any payloads (e.g. the Sensors variant payload that
    # carries cameras like ``Head_Camera``) are realized — the in-source camera
    # lookup below depends on these prims being resolvable on the open stage.
    robot_stage = Usd.Stage.Open(robot_usd_path, load=Usd.Stage.LoadAll)
    if not robot_stage:
        print(
            f"[assemble_scene] ERROR: cannot open robot USD: {robot_usd_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    robot_visual_path = os.path.join(output_dir, "robot_visual.usda")
    if _prebaked_usda_rel and _prebaked_visual_only:
        # Physics stage (robot_stage above) is the staged URDF USD; its
        # topology drives Newton/MuJoCo. For rendering we want the authored
        # prebaked asset instead, which carries materials/meshes but a
        # different joint namespace. Open it as a separate stage and export
        # robot_visual.usda as a thin reference to IT, not to the physics USD.
        _visual_src_stage = Usd.Stage.Open(prebaked_resolved, load=Usd.Stage.LoadAll)
        if not _visual_src_stage:
            print(
                f"[assemble_scene] ERROR: cannot open prebaked visual USD: {prebaked_resolved}",
                file=sys.stderr,
            )
            sys.exit(1)
        _export_visual_robot(_visual_src_stage, robot_prefix, robot_visual_path)
        print(f"[assemble_scene] visual usd: {prebaked_resolved} [prebaked (visual_only)]")
    else:
        _export_visual_robot(robot_stage, robot_prefix, robot_visual_path)

    render_layer_path, authored_cam_indices, lidar_manifest_entries = author_render_layer(
        output_dir,
        robot_usd_path,
        robot_stage,
        robot_prefix,
        cameras,
        lidars,
        free_cam_w,
        free_cam_h,
        camera_mode,
    )
    out_path = render_layer_path

    # Newton scene layer — cloth/softbody USD references for the physics stage.
    # Separate from render_layer.usda (which is mounted under /RenderOVRTX on
    # the physics stage and under /RenderOVRTX on the render stage). Cloth
    # USDs must be at /World/<name> directly on the root stage so Newton's
    # builder.add_usd sees the mesh and our inject hook can read it.
    # Written as newton_scene.usda next to render_layer.usda; _open_scene_with_
    # references adds it as a sublayer when present.
    newton_cfg = config.get("newton")
    if isinstance(newton_cfg, dict) and newton_cfg.get("entries"):
        newton_scene_path = os.path.join(output_dir, "newton_scene.usda")
        newton_stage = Usd.Stage.CreateNew(newton_scene_path)
        wrote = 0
        for entry in newton_cfg["entries"]:
            name = entry.get("name", "")
            kind = entry.get("kind", "")
            mesh_ref = entry.get("mesh_usd", "")
            pose = entry.get("pose") or [0, 0, 0, 0, 0, 0, 1]
            if not name:
                continue
            prim_path = f"/World/{name}"

            # -------------------------------------------------------------
            # Primitive: static box collider.
            # -------------------------------------------------------------
            # Authored as ``UsdGeom.Cube`` (size=2 → ±1 in each axis by
            # default) with translate + scale xformOps and a quaternion
            # rotation. ``UsdPhysics.CollisionAPI`` makes Newton's
            # ``add_usd`` register it as a world-static shape so cloth
            # and articulations collide against it.
            #
            # Unlike cloth — where particle_q is in world frame and the
            # prim MUST be at identity to avoid a double-transform — a
            # static box has no per-frame writeback. The xform is
            # authored once and Hydra composes it normally; Newton reads
            # the same transform during USD parsing.
            # -------------------------------------------------------------
            if kind == "box":
                params = entry.get("params") or {}
                he = params.get("half_extents") or [0.5, 0.5, 0.5]
                try:
                    hx, hy, hz = (float(he[0]), float(he[1]), float(he[2]))
                except Exception:
                    print(
                        f"[assemble_scene] WARN: box {name!r} half_extents " f"{he!r} not 3 floats — skipping",
                        file=sys.stderr,
                    )
                    continue
                cube = UsdGeom.Cube.Define(newton_stage, prim_path)
                cube.GetSizeAttr().Set(2.0)  # half-size = 1 in each axis
                t_op = cube.AddTranslateOp()
                t_op.Set(Gf.Vec3d(float(pose[0]), float(pose[1]), float(pose[2])))
                r_op = cube.AddOrientOp()
                r_op.Set(
                    Gf.Quatf(
                        float(pose[6]),
                        float(pose[3]),
                        float(pose[4]),
                        float(pose[5]),
                    )
                )
                s_op = cube.AddScaleOp()
                s_op.Set(Gf.Vec3f(hx, hy, hz))
                # Mark as a static collider so Newton's add_usd picks it up.
                try:
                    from pxr import UsdPhysics

                    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
                except Exception as exc:
                    print(
                        f"[assemble_scene] WARN: could not apply " f"UsdPhysics.CollisionAPI to box {name!r}: {exc}",
                        file=sys.stderr,
                    )
                print(
                    f"[assemble_scene] newton box: {name} @ {prim_path} "
                    f"pos={pose[:3]} half_extents=({hx}, {hy}, {hz})"
                )
                wrote += 1
                continue

            # -------------------------------------------------------------
            # Geometry from external mesh USD (cloth and friends).
            # -------------------------------------------------------------
            if not mesh_ref:
                continue
            mesh_abs = None
            for candidate in [
                mesh_ref if os.path.isabs(mesh_ref) else None,
                os.path.join(base_path, mesh_ref) if base_path else None,
                os.path.join(os.path.dirname(cfg_path), mesh_ref),
            ]:
                if candidate and os.path.isfile(candidate):
                    mesh_abs = os.path.abspath(candidate)
                    break
            if mesh_abs is None:
                print(
                    f"[assemble_scene] WARN: newton cloth mesh not found: {mesh_ref}",
                    file=sys.stderr,
                )
                continue
            xform_prim = UsdGeom.Xform.Define(newton_stage, prim_path)
            xform_prim.GetPrim().GetReferences().AddReference(mesh_abs)
            # IMPORTANT: keep this prim at identity. The spawn ``pose`` is
            # consumed at runtime by ``add_cloth_mesh(pos=…, rot=…)`` and
            # baked into Newton's ``particle_q`` (world frame). Our cloth
            # writeback then copies ``particle_q`` straight into the
            # mesh's ``points`` attribute — Hydra interprets ``points``
            # as the prim's LOCAL frame and composes with this prim's
            # xform. Authoring a non-identity translate/orient here
            # therefore applies the spawn pose TWICE, and the visual
            # cloth ends up offset from where Newton simulated it (the
            # observed symptom: shirt drapes onto empty space next to
            # the robot rather than where the robot actually is).
            print(
                f"[assemble_scene] newton cloth: {name} @ {prim_path} "
                f"← {os.path.basename(mesh_abs)} (xform=identity; "
                f"spawn pose {pose[:3]} applied at runtime via add_cloth_mesh)"
            )
            wrote += 1
        if wrote:
            newton_stage.GetRootLayer().Save()
            print(f"[assemble_scene] newton scene written: {newton_scene_path} ({wrote} entr(ies))")
            # Make cloth/box asset references inside the newton scene
            # portable.  Cloth USDs commonly live in
            # ``/scenes/blank/<thing>.usd`` (shared assets across all
            # cloth scenes); authoring them as absolute paths means
            # the whole scene tree can't be relocated.  Rewrite to
            # ``../blank/<thing>.usd`` so the entire scenes/ tree
            # is movable as a unit — same convention the
            # ``robot_runtime.usda`` dump uses.
            _relativize_layer_paths(newton_scene_path)
        else:
            del newton_stage
            if os.path.exists(newton_scene_path):
                os.remove(newton_scene_path)

    # Manifest stores paths RELATIVE to ``base_path`` (typically the workspace
    # root). The consumer (genie_sim_engine.py)
    # joins them back with ``base_path`` before handing absolute paths to
    # Omni APIs. This keeps the on-disk manifest portable across machines
    # and makes ``./assets/scenes/scene/manifest.json`` readable.
    def _rel(p: str) -> str:
        if not p:
            return p
        ap = os.path.abspath(p)
        base_abs = os.path.abspath(base_path)
        try:
            if os.path.commonpath([ap, base_abs]) != base_abs:
                return ap
            return os.path.relpath(ap, start=base_abs)
        except ValueError:
            return ap

    # The manifest is intentionally limited to **static** fields — paths
    # to assets, prim namespaces, derived booleans like ``robot_from_urdf``.
    # Anything that's a runtime *behavior* (mimic, fix_base, init_joint_pos,
    # overrides) is read live from the scene yaml at engine startup
    # via ``scene_yaml`` below, so the operator can edit the yaml and
    # relaunch with a cache HIT to pick up changes — the bake cache only
    # caches the slow URDF→USD conversion, not behavior. A staged debug
    # snapshot is also written to ``<stage_dir>/scene.yaml`` by
    # ``stage_yaml_snapshot`` for post-mortem ``diff`` purposes; the engine
    # reads the original, not the snapshot.
    # Emit manifest through the shared typed :class:`Manifest`. Every
    # camera / lidar / path field is normalized to the flat schema that
    # ``render_node.cpp`` and downstream Python readers consume. Details
    # unique to the Isaac path (per-camera authoring gate,
    # ``authored_cam_indices``; per-scene FreeCam intrinsics) are
    # translated to the typed ``CameraManifest`` shape below.
    from genie_sim_engine_core.schema import (
        CameraManifest,
        LidarBaseLinkT,
        LidarManifest,
        Manifest,
    )

    from pathlib import Path as _Path

    _base_path = _Path(base_path) if base_path else _Path.cwd()

    def _opt_path(rel: str):
        return _Path(rel) if rel else None

    typed_cameras: list[CameraManifest] = []
    for i, cam_cfg in enumerate(cameras):
        if not isinstance(cam_cfg, dict):
            continue
        if i not in authored_cam_indices:
            print(
                f"[assemble_scene] manifest: skipping cameras[{i}] — its render product was not "
                f"authored on the render layer (see WARN above)",
                file=sys.stderr,
            )
            continue
        cam_rel_path = cam_cfg.get("prim_path")
        if not cam_rel_path:
            continue
        sensor = cam_cfg.get("sensor") or {}
        topic = cam_cfg.get("topic") or {}
        intrinsic = cam_cfg.get("intrinsic") or {}
        rgb_topic = topic.get("rgb", "") if isinstance(topic, dict) else str(topic or "")
        depth_topic = topic.get("depth", "") if isinstance(topic, dict) else ""
        typed_cameras.append(
            CameraManifest(
                path=cam_rel_path,
                topic=rgb_topic,
                depth_topic=depth_topic,
                dds_topic=str(cam_cfg.get("dds_topic") or ""),
                dds_depth_topic=str(cam_cfg.get("dds_depth_topic") or ""),
                render_product_path=f"/RenderOVRTX/Cam_{i}",
                frame_id=str(cam_cfg.get("frame_id") or ""),
                width=int(sensor.get("width", 1280)),
                height=int(sensor.get("height", 800)),
                fx=float(intrinsic.get("fx", 610.0)),
                fy=float(intrinsic.get("fy", 610.0)),
                cx=float(intrinsic.get("cx", 640.0)),
                cy=float(intrinsic.get("cy", 400.0)),
                k1=float(intrinsic.get("k1", 0.0)),
                k2=float(intrinsic.get("k2", 0.0)),
                p1=float(intrinsic.get("p1", 0.0)),
                p2=float(intrinsic.get("p2", 0.0)),
                k3=float(intrinsic.get("k3", 0.0)),
                k4=float(intrinsic.get("k4", 0.0)),
                model=(cam_cfg.get("model") or "").strip(),
                min_range=float(sensor.get("min_range", 0.01)),
                max_range=float(sensor.get("max_range", 10000.0)),
                is_free_cam=False,
                depth_scale=(float(cam_cfg["depth_scale"]) if cam_cfg.get("depth_scale") is not None else None),
            )
        )
    typed_cameras.append(
        CameraManifest(
            path="FreeCam",
            topic="/genie_sim/free_camera_rgb",
            render_product_path="/RenderOVRTX/FreeCam",
            width=free_cam_w,
            height=free_cam_h,
            fx=free_cam_w / 2.0,
            fy=free_cam_w / 2.0,
            cx=free_cam_w / 2.0,
            cy=free_cam_h / 2.0,
            is_free_cam=True,
        )
    )

    typed_lidars: list[LidarManifest] = []
    for lid in lidar_manifest_entries:
        typed_lidars.append(
            LidarManifest(
                render_product_path=str(lid.get("render_product_path", "")),
                topic=str(lid.get("topic", "")),
                frame_id=str(lid.get("frame_id", "base_link")),
                parent_body=str(lid.get("parent_body", "")),
                base_link_T_sensor=LidarBaseLinkT.from_dict(lid.get("base_link_T_sensor")),
            )
        )

    typed_manifest = Manifest(
        physics_engine="",  # Isaac subprocess doesn't know the launcher's engine label.
        base_path=_base_path,
        stage_dir=_Path(output_dir),
        scene_yaml=_Path(cfg_path) if cfg_path else None,
        robot_prefix=robot_prefix or "genie",
        robot_from_urdf=bool(robot_from_urdf),
        scene_usda=_opt_path(_rel(scene_usd_resolved)),
        robot_usda=_opt_path(_rel(robot_usd_path)),
        robot_visual_usda=_opt_path(_rel(robot_visual_path)),
        render_layer_usda=_opt_path(_rel(out_path)),
        free_cam_prim_path="/RenderOVRTX/Cameras/FreeCam",
        cameras=typed_cameras,
        lidars=typed_lidars,
    )
    manifest = typed_manifest.to_dict()
    manifest_path = os.path.join(output_dir, "manifest.json")
    if write_manifest:
        typed_manifest.write(_Path(manifest_path))
        print(f"[assemble_scene] manifest written: {manifest_path}")
    else:
        print(f"[assemble_scene] manifest built (write deferred to caller): {manifest_path}")
    print()
    print("[assemble_scene] done!")
    return manifest


def main(argv: list[str] | None = None) -> None:
    """Subprocess entry — argv matches the pre-migration CLI shape."""
    import argparse

    parser = argparse.ArgumentParser(prog="assemble_scene")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--output-dir", default="/tmp/isaacsim_stage")
    parser.add_argument("--base-path", default="")
    parser.add_argument(
        "--no-write-manifest",
        action="store_true",
        help="Compute the manifest dict but DO NOT write manifest.json; "
        "used by the pipeline runner when core owns the write.",
    )
    args = parser.parse_args(argv)
    assemble_scene(
        scene=args.scene,
        output_dir=args.output_dir,
        base_path=args.base_path,
        write_manifest=not args.no_write_manifest,
    )


if __name__ == "__main__":
    main()

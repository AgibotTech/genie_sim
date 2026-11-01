# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Assemble a multi-environment USD stage and manifest with pxr.

Reference the single-environment assets into spaced environment scopes and
author per-env camera anchors and render products. The manifest supplies
the same layout to physics and rendering. Runtime body transforms and
camera anchors are published by the render bridge.

The layout is inspired by Isaac Sim's ``GridCloner.clone`` and also works
with the Kit-free OVRTX renderer.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Inputs / outputs
# ---------------------------------------------------------------------------

# IsaacSim's default_viz_camera defaults — debug-friendly third-person
# view that frames "robot + workbench + manipulated object" in one shot.
# Pose is expressed in the env's local frame (so it auto-translates with
# /World/envs/env_<i>'s xformOp:translate).
VIZ_CAM_EYE = (-0.2, -1.8, 1.8)
VIZ_CAM_TARGET = (-0.4, 0.0, 0.9)
VIZ_CAM_FOCAL_LENGTH_MM = 18.0
VIZ_CAM_H_APERTURE_MM = 20.955
VIZ_CAM_V_APERTURE_MM = 15.2908
VIZ_CAM_CLIP_NEAR = 0.01
VIZ_CAM_CLIP_FAR = 100.0


def _lookat_quat_wxyz(eye, target, world_up=(0.0, 0.0, 1.0)):
    """Return ``(qw, qx, qy, qz)`` for a USD camera at ``eye`` looking at
    ``target``. Mirrors ``rl_renderer._lookat_quaternion`` exactly so the
    OVRTX viz camera matches the legacy IsaacSim debug view byte-for-byte.
    """
    import math as _math

    def _norm(v):
        n = _math.sqrt(sum(c * c for c in v))
        return tuple(c / n for c in v) if n > 1e-9 else v

    def _cross(a, b):
        return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])

    fwd = _norm(tuple(t - e for t, e in zip(target, eye)))  # local -Z
    right = _norm(_cross(fwd, world_up))  # local +X
    up = _cross(right, fwd)  # local +Y
    lz = tuple(-f for f in fwd)  # local +Z

    m00, m01, m02 = right[0], up[0], lz[0]
    m10, m11, m12 = right[1], up[1], lz[1]
    m20, m21, m22 = right[2], up[2], lz[2]

    trace = m00 + m11 + m22
    if trace > 0:
        s = 0.5 / _math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m21 - m12) * s
        y = (m02 - m20) * s
        z = (m10 - m01) * s
    elif m00 > m11 and m00 > m22:
        s = 2.0 * _math.sqrt(max(0.0, 1.0 + m00 - m11 - m22))
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * _math.sqrt(max(0.0, 1.0 + m11 - m00 - m22))
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = 2.0 * _math.sqrt(max(0.0, 1.0 + m22 - m00 - m11))
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    return w, x, y, z


def _load_manifest(path: str) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _save_manifest(manifest: dict[str, Any], path: str) -> None:
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)


# ---------------------------------------------------------------------------
# USD construction
# ---------------------------------------------------------------------------


def _write_scene_multi_usda(
    output_path: str,
    base_scene_usda: str,
    robot_usda: str,
    robot_root_name: str,
    foreground_usda: str,
    foreground_root_name: str,
    num_envs: int,
    env_spacing: float,
) -> None:
    """Generate the multi-env scene USD using a hand-written .usda string.

    We don't go through pxr.Usd here for the file layout because plain
    text is more diffable and easier to inspect. ``robot.usda`` and the raw
    ``foreground.usda`` (table + manipulable objects) are *referenced* — their
    mesh bits are NOT copied N times.

    Each env (including env_0) gets BOTH a robot AND its own foreground under
    ``/World/envs/env_<i>``, so every env is a complete scene (robot + table +
    objects). The floor + global lights live in ``base_scene_usda`` (the
    default-env wrapper) and stay SHARED — they are not per-env. The base
    scene's own foreground copy at the origin (``/World/Objects``) is
    DEACTIVATED here, since in multi-env it is undriven and would render as a
    stray static object set; each env's objects are driven into its own
    ``/World/envs/env_<i>/World/Objects/...`` prims instead.
    """
    base_scene_abs = str(Path(base_scene_usda).resolve())
    robot_usda_abs = str(Path(robot_usda).resolve())
    foreground_abs = str(Path(foreground_usda).resolve())

    envs_block = []
    # All envs live under /World/envs/env_<i>. Layout = +X * env_spacing.
    # The viz cameras themselves are NOT nested here — instead they're
    # authored directly in render_layer.usda with pre-computed world
    # transforms (see _add_viz_cameras_and_render_products). Reason:
    # OVRTX loads render_layer.usda standalone for RenderProduct lookup
    # and can't resolve cross-layer Camera targets back to scene_multi.
    for i in range(num_envs):
        x = i * env_spacing
        # The robot hangs under a child named after its root prim and the
        # foreground under one named "World", so env_<i> composes robot links
        # at /World/envs/env_<i>/genie/<link> and objects at
        # /World/envs/env_<i>/World/Objects/<name>/entity — matching the
        # labels the engine publishes, so update_poses can address them by
        # prefixing `/World/envs/env_<i>/`.
        # Both carry omni:resetXformStack, so the renderer writes them as
        # world transforms and this env translation does not apply to them;
        # update_poses bakes the same +X offset into the poses instead.
        envs_block.append(
            f'        def Xform "env_{i}"\n'
            f"        {{\n"
            f"            double3 xformOp:translate = ({x}, 0, 0)\n"
            f'            uniform token[] xformOpOrder = ["xformOp:translate"]\n'
            f'            def "{robot_root_name}" (\n'
            f"                prepend references = @{robot_usda_abs}@\n"
            f"            )\n"
            f"            {{\n"
            f"            }}\n"
            f'            def "{foreground_root_name}" (\n'
            f"                prepend references = @{foreground_abs}@\n"
            f"            )\n"
            f"            {{\n"
            f"            }}\n"
            f"        }}\n"
        )

    content = (
        "#usda 1.0\n"
        "(\n"
        '    defaultPrim = "World"\n'
        "    metersPerUnit = 1\n"
        '    upAxis = "Z"\n'
        "    subLayers = [\n"
        f"        @{base_scene_abs}@\n"
        "    ]\n"
        ")\n"
        "\n"
        'over "World"\n'
        "{\n"
        # Deactivate the shared origin foreground that comes in via the base
        # scene subLayer — in multi-env it is undriven and would render as a
        # stray static object set; each env carries its own below.
        '    over "Objects" (\n'
        "        active = false\n"
        "    )\n"
        "    {\n"
        "    }\n"
        '    def Xform "envs"\n'
        "    {\n" + "".join(envs_block) + "    }\n"
        "}\n"
    )

    with open(output_path, "w") as f:
        f.write(content)


# ---------------------------------------------------------------------------
# Render layer USDA modification
# ---------------------------------------------------------------------------


def _env0_head_camera_path(stage) -> Optional[str]:
    """Resolve the env-0 head Camera prim path from the head RenderProduct.

    The single-env render layer names its head Camera from the task's camera
    config (e.g. ``head_front_Camera`` from ``prim_path: head_link3/head_front_Camera``),
    NOT a fixed ``Head_Front_Camera`` — so we must not hardcode the leaf name.
    ``/RenderOVRTX/Cam_0`` (the env-0 head RenderProduct authored by the
    single-env ``author_render_layer``) binds its Camera via a ``camera``
    relationship; read the camera path from there. Returns ``None`` when Cam_0
    or its ``camera`` target is absent (callers decide whether that's fatal).
    """
    rp = stage.GetPrimAtPath("/RenderOVRTX/Cam_0")
    if not rp.IsValid():
        return None
    rel = rp.GetRelationship("camera")
    targets = rel.GetTargets() if rel else []
    if not targets:
        return None
    return str(targets[0])


def _add_env_cameras_to_render_layer(
    render_layer_path: str,
    num_envs: int,
    env_spacing: float,
    template_camera: dict[str, Any],
    head_height_m: float = 1.5,
) -> None:
    """Append N-1 new Camera + RenderProduct prims to the render layer.

    The source ``head_link3`` Xform has **no xformOp** in the USD file
    on purpose — its transform is populated at runtime by OVRTX from
    ``/tf_render`` (child_frame_id=head_link3). We mirror that here:
    each ``env_<i>_head_link3`` Xform is also empty, and the sim_server
    side broadcasts a corresponding ``env_<i>_head_link3`` tf each tick
    (= env_0's head transform with X shifted by ``i * env_spacing``).
    See ``render/bridges/_upstream_base.update_poses`` for the
    publish-side expansion.

    Cameras are identical to env_0's — same intrinsics, same Camera
    local transform (identity translate + identity orient under the
    head_link3 anchor). Only the anchor's tf differs per env.

    ``head_height_m`` is kept in the signature for backwards
    compatibility but is no longer consulted.
    """
    from pxr import Usd, UsdGeom, Sdf

    stage = Usd.Stage.Open(render_layer_path)
    if not stage:
        raise RuntimeError(f"failed to open {render_layer_path}")

    # Cam_0 (env-0 head RenderProduct) supplies BOTH the resolution /
    # rendermode we clone AND the env-0 head Camera path (via its ``camera``
    # relationship). Resolve the camera from the relationship rather than
    # hardcoding a ``Head_Front_Camera`` leaf — the single-env render layer
    # names it from the task's camera config (e.g. ``head_front_Camera``).
    src_rp = stage.GetPrimAtPath("/RenderOVRTX/Cam_0")
    if not src_rp.IsValid():
        raise RuntimeError(
            f"{render_layer_path}: expected /RenderOVRTX/Cam_0 (the env-0 " f"camera RenderProduct) but it's missing"
        )
    src_cam_path = _env0_head_camera_path(stage)
    if not src_cam_path:
        raise RuntimeError(
            f"{render_layer_path}: /RenderOVRTX/Cam_0 has no 'camera' "
            f"relationship target — cannot resolve the env-0 head camera"
        )
    src_cam_prim = stage.GetPrimAtPath(src_cam_path)
    if not src_cam_prim.IsValid():
        raise RuntimeError(
            f"{render_layer_path}: env-0 head camera {src_cam_path} " f"(from /RenderOVRTX/Cam_0) is missing"
        )
    src_cam_xf = UsdGeom.Xformable(src_cam_prim)
    src_cam_ops = list(src_cam_xf.GetOrderedXformOps())

    def _clone_cam_local_ops(src_ops, dst_xf):
        """Replicate a Camera's local xform ops verbatim onto dst_xf."""
        dst_xf.ClearXformOpOrder()
        for op in src_ops:
            new_op = dst_xf.AddXformOp(op.GetOpType(), op.GetPrecision())
            new_op.Set(op.Get())

    for i in range(1, num_envs):
        # ── Empty Xform — runtime tf_render fills it (matches env_0). ──
        parent_path = f"/RenderOVRTX/Cameras/env_{i}_head_link3"
        parent = stage.DefinePrim(parent_path, "Xform")
        UsdGeom.Xformable(parent).ClearXformOpOrder()

        # ── Clone Camera prim verbatim under that parent ──
        cam_path = parent_path + f"/Head_Front_Camera_env_{i}"
        cam_prim = UsdGeom.Camera.Define(stage, cam_path)
        cam_prim.CreateClippingRangeAttr((0.01, 10000))
        cam_prim.CreateFocalLengthAttr(template_camera.get("focal_length", 10.007404))
        cam_prim.CreateHorizontalApertureAttr(template_camera.get("h_aperture", 20.955))
        cam_prim.CreateVerticalApertureAttr(template_camera.get("v_aperture", 13.098667))
        cam_xf = UsdGeom.Xformable(cam_prim)
        _clone_cam_local_ops(src_cam_ops, cam_xf)

        # ── Matching RenderProduct ──
        rp_path = f"/RenderOVRTX/Cam_env_{i}"
        rp_prim = stage.DefinePrim(rp_path, "RenderProduct")
        rp_prim.CreateRelationship("camera").AddTarget(Sdf.Path(cam_path))
        for attr in src_rp.GetAttributes():
            if attr.GetName() in ("omni:rtx:rendermode", "resolution"):
                new_attr = rp_prim.CreateAttribute(attr.GetName(), attr.GetTypeName())
                new_attr.Set(attr.Get())

        ldr = stage.DefinePrim(rp_path + "/LdrColor", "RenderVar")
        ldr.CreateAttribute("sourceName", Sdf.ValueTypeNames.String).Set("LdrColor")
        rp_prim.CreateRelationship("orderedVars").AddTarget(Sdf.Path(rp_path + "/LdrColor"))

    stage.GetRootLayer().Save()


def _add_viz_cameras_and_render_products(
    render_layer_path: str,
    num_envs: int,
    env_spacing: float,
) -> None:
    """Author per-env default_viz_camera Camera + RenderProduct prims.

    Both prims live in render_layer.usda (NOT scene_multi.usda) because
    OVRTX loads render_layer.usda standalone to discover RenderProducts
    and cannot resolve cross-layer Camera relationship targets.

    Each env's viz Camera world transform is precomputed:
      eye_world = (i*env_spacing + VIZ_CAM_EYE[0], -1.8, 1.8)
      target_world = (i*env_spacing + VIZ_CAM_TARGET[0], 0, 0.9)
    so the camera ends up at the same relative pose to env_<i>'s
    contents that IsaacSim's GridCloner-cloned default_viz_camera
    achieved (just baked into world coords here, since OVRTX's render
    layer doesn't see scene_multi's env_<i> Xform).
    """
    from pxr import Usd, UsdGeom, Sdf, Gf

    stage = Usd.Stage.Open(render_layer_path)
    if not stage:
        raise RuntimeError(f"failed to open {render_layer_path}")

    src_rp = stage.GetPrimAtPath("/RenderOVRTX/Cam_0")
    if not src_rp.IsValid():
        raise RuntimeError(f"{render_layer_path}: expected /RenderOVRTX/Cam_0 RenderProduct")

    # The eye→target direction is identical for every env (both endpoints
    # shift by the same +X offset), so we can compute the orientation
    # once and reuse it.
    qw, qx, qy, qz = _lookat_quat_wxyz(VIZ_CAM_EYE, VIZ_CAM_TARGET)

    for i in range(num_envs):
        x_offset = i * env_spacing
        eye_world = (
            VIZ_CAM_EYE[0] + x_offset,
            VIZ_CAM_EYE[1],
            VIZ_CAM_EYE[2],
        )

        # ── Camera prim ──────────────────────────────────────────────
        cam_path = f"/RenderOVRTX/Cameras/viz_env_{i}"
        cam_prim = UsdGeom.Camera.Define(stage, cam_path)
        cam_prim.CreateClippingRangeAttr(Gf.Vec2f(VIZ_CAM_CLIP_NEAR, VIZ_CAM_CLIP_FAR))
        cam_prim.CreateFocalLengthAttr(VIZ_CAM_FOCAL_LENGTH_MM)
        cam_prim.CreateHorizontalApertureAttr(VIZ_CAM_H_APERTURE_MM)
        cam_prim.CreateVerticalApertureAttr(VIZ_CAM_V_APERTURE_MM)

        cam_xf = UsdGeom.Xformable(cam_prim)
        cam_xf.ClearXformOpOrder()
        cam_xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*eye_world))
        cam_xf.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Quatf(qw, qx, qy, qz))

        # ── RenderProduct prim ───────────────────────────────────────
        rp_path = f"/RenderOVRTX/Cam_viz_{i}"
        rp_prim = stage.DefinePrim(rp_path, "RenderProduct")
        rp_prim.CreateRelationship("camera").AddTarget(Sdf.Path(cam_path))

        for attr in src_rp.GetAttributes():
            if attr.GetName() in ("omni:rtx:rendermode", "resolution"):
                new_attr = rp_prim.CreateAttribute(attr.GetName(), attr.GetTypeName())
                new_attr.Set(attr.Get())

        ldr = stage.DefinePrim(rp_path + "/LdrColor", "RenderVar")
        ldr.CreateAttribute("sourceName", Sdf.ValueTypeNames.String).Set("LdrColor")
        rp_prim.CreateRelationship("orderedVars").AddTarget(Sdf.Path(rp_path + "/LdrColor"))

    stage.GetRootLayer().Save()


# ---------------------------------------------------------------------------
# Manifest extension
# ---------------------------------------------------------------------------


def _extend_manifest_cameras(
    manifest: dict[str, Any],
    num_envs: int,
) -> dict[str, Any]:
    """Add env_1..N-1 camera entries to the manifest.

    Convention: env_0 keeps the original topic name; env_i (i>=1) uses
    the ``env_{i}_`` prefix. Free-cam-style cameras are left untouched
    (single instance shared across all envs).
    """
    out = copy.deepcopy(manifest)

    # Find the env-0 head camera template (first non-free_cam entry).
    template_idx = None
    template_cam = None
    for idx, cam in enumerate(out["cameras"]):
        if cam.get("is_free_cam"):
            continue
        template_idx = idx
        template_cam = cam
        break

    if template_cam is None:
        raise RuntimeError("manifest.cameras has no non-free_cam entry to use as template")

    new_cams: list[dict[str, Any]] = []
    for i in range(1, num_envs):
        c = copy.deepcopy(template_cam)
        base_topic = c["topic"]
        c["topic"] = _env_inject(base_topic, i)
        if "depth_topic" in c:
            c["depth_topic"] = _env_inject(c["depth_topic"], i)
        c["render_product_path"] = f"/RenderOVRTX/Cam_env_{i}"
        c["path"] = f"env_{i}_head_link3/Head_Front_Camera_env_{i}"
        c["frame_id"] = f"env_{i}_head_link3"
        new_cams.append(c)

    # Insert env_1..N-1 right after env_0's head camera entry.
    out["cameras"] = out["cameras"][: template_idx + 1] + new_cams + out["cameras"][template_idx + 1 :]
    return out


def _add_viz_cameras_to_manifest(
    manifest: dict[str, Any],
    num_envs: int,
) -> dict[str, Any]:
    """Append per-env default_viz_camera entries to manifest.cameras.

    These are marked ``is_free_cam: True`` so OVRTX skips its anchor-map
    registration (we don't want runtime tf_render to overwrite the
    USD-baked viz pose). Resolution / fx / fy are computed from the
    IsaacSim-parity intrinsics (focal=18mm, hAp=20.955mm, vAp=15.2908mm)
    using the head camera's image dimensions.
    """
    out = copy.deepcopy(manifest)

    head_template: dict[str, Any] | None = None
    for cam in out.get("cameras", []) or []:
        if cam.get("is_free_cam"):
            continue
        head_template = cam
        break

    width = int((head_template or {}).get("width", 640))
    height = int((head_template or {}).get("height", 400))

    fx = VIZ_CAM_FOCAL_LENGTH_MM * width / VIZ_CAM_H_APERTURE_MM
    fy = VIZ_CAM_FOCAL_LENGTH_MM * height / VIZ_CAM_V_APERTURE_MM
    cx = width / 2.0
    cy = height / 2.0

    new_cams: list[dict[str, Any]] = []
    for i in range(num_envs):
        new_cams.append(
            {
                "render_product_path": f"/RenderOVRTX/Cam_viz_{i}",
                "topic": f"/genie_sim/env_{i}_viz_camera_rgb",
                "depth_topic": "",
                # Short relative ``path`` (no leading slash) — keeps OVRTX's
                # anchor-map parser from splitting on a leading "/" and
                # crashing when ``cam.path.substr(0, 0)`` becomes "".
                # The actual Camera prim is wired up via the RenderProduct's
                # ``camera`` relationship target authored elsewhere.
                "path": f"viz_env_{i}/default_viz_camera",
                "frame_id": f"env_{i}_viz_camera",
                "width": width,
                "height": height,
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
                "k1": 0.0,
                "k2": 0.0,
                "p1": 0.0,
                "p2": 0.0,
                "k3": 0.0,
                "k4": 0.0,
                "model": "",
                "is_free_cam": True,
            }
        )

    out["cameras"] = list(out.get("cameras", []) or []) + new_cams
    return out


def _write_reset_xform_wrapper(
    out_path: Path,
    src_abs: str,
    default_name: str,
    rel_paths: list[str],
    root_kw: str = "Xform",
) -> None:
    """Write a wrapper USD ``def <root_kw> "<default_name>" (references=@src@)``
    that flags each prim in ``rel_paths`` (paths relative to the source's
    defaultPrim) with ``omni:resetXformStack=1`` via nested ``over`` blocks.

    Shared by the robot- and foreground-wrapper authors. USD doesn't accept an
    ``over`` with embedded slashes at parse time, so we build a name tree and
    emit nested overs.
    """
    lines: list[str] = [
        "#usda 1.0",
        "(",
        f'    defaultPrim = "{default_name}"',
        "    customLayerData = {",
        '        string creator = "rlinf assemble_multi_env"',
        "    }",
        ")",
        "",
        f'def {root_kw} "{default_name}" (',
        f"    prepend references = @{src_abs}@",
        ")",
        "{",
    ]
    tree: dict = {}
    for rel in rel_paths:
        cursor = tree
        for seg in rel.split("/"):
            cursor = cursor.setdefault(seg, {})

    def _emit_overs(node: dict, indent_level: int) -> None:
        ind = "    " * indent_level
        for name in sorted(node.keys()):
            children = node[name]
            lines.append(f'{ind}over "{name}"')
            lines.append(f"{ind}{{")
            lines.append(f"{ind}    custom bool omni:resetXformStack = 1")
            if children:
                _emit_overs(children, indent_level + 1)
            lines.append(f"{ind}}}")

    _emit_overs(tree, indent_level=1)
    lines.append("}")
    lines.append("")
    Path(out_path).write_text("\n".join(lines))


def _author_foreground_reset_xform_wrapper(
    src_foreground_usda: Path,
    out_dir: Path,
) -> tuple[Path, str]:
    """Wrapper over the RAW foreground that flags each driven OBJECT frame.

    The engine drives each object's ``.../entity`` prim (one rigid frame per
    object — these are exactly the ``World/Objects/<name>/entity`` labels it
    publishes on /tf_render). Flag those with ``omni:resetXformStack`` so OVRTX
    writes each object's WORLD pose straight to its per-env prim
    (``/World/envs/env_<i>/World/Objects/<name>/entity``); the env's +X offset
    is ignored by the reset-stack prim and update_poses bakes it into the pose,
    exactly like the robot links. The object geometry under ``entity`` (its
    payload) is rigid relative to entity, so it does NOT need the flag.

    Returns ``(wrapper_path, root_name)`` (root_name is the foreground's
    defaultPrim, typically ``World``).
    """
    from pxr import Usd

    src_stage = Usd.Stage.Open(str(src_foreground_usda), load=Usd.Stage.LoadAll)
    if not src_stage:
        raise RuntimeError(f"failed to open foreground usda: {src_foreground_usda}")
    default_prim = src_stage.GetDefaultPrim()
    if not default_prim or not default_prim.IsValid():
        raise RuntimeError(
            f"{src_foreground_usda}: no defaultPrim — foreground wrapper "
            f"requires one so a 'prepend references' picks the right entry"
        )
    default_name = default_prim.GetName()  # "World"
    default_path_str = str(default_prim.GetPath())  # "/World"

    rel_paths: list[str] = []
    for prim in src_stage.Traverse():
        path_str = str(prim.GetPath())
        if path_str.endswith("/entity") and path_str.startswith(default_path_str + "/"):
            rel_paths.append(path_str[len(default_path_str) + 1 :])

    out_path = out_dir / "foreground_reset_xform.usda"
    _write_reset_xform_wrapper(out_path, str(Path(src_foreground_usda).resolve()), default_name, rel_paths)
    print(
        f"[_author_foreground_reset_xform_wrapper] flagged {len(rel_paths)} "
        f"object entity prim(s) under /{default_name}",
        flush=True,
    )
    return out_path, default_name


def _author_robot_reset_xform_wrapper(
    src_robot_usda: Path,
    out_dir: Path,
) -> tuple[Path, str]:
    """Reference the source robot and mark links with ``omni:resetXformStack``.

    The renderer receives world-space link transforms. Resetting each link's
    transform stack prevents parent transforms from being applied again.
    Preserve the source default prim and leave managed assets unchanged.
    """
    from pxr import Usd, UsdGeom

    # Walk the source robot stage to enumerate every link Xform path.
    # We only flag Xform-typed prims (Scope and other types don't have
    # transform composition, and Cylinder/Cube etc. are leaf collision
    # primitives that don't get tf'd).
    src_stage = Usd.Stage.Open(str(src_robot_usda), load=Usd.Stage.LoadAll)
    if not src_stage:
        raise RuntimeError(f"failed to open source robot.usda: {src_robot_usda}")
    default_prim = src_stage.GetDefaultPrim()
    if not default_prim or not default_prim.IsValid():
        raise RuntimeError(
            f"{src_robot_usda}: no defaultPrim — wrapper requires one "
            f"so a 'prepend references' picks the right entry"
        )
    default_name = default_prim.GetName()  # typically "robot"

    # Collect relative paths under defaultPrim, stripping the
    # /<defaultPrim> prefix so the wrapper's `over` blocks compose
    # correctly when the wrapper itself is referenced.
    link_rel_paths: list[str] = []
    default_path_str = str(default_prim.GetPath())  # e.g. "/robot"
    for prim in src_stage.Traverse():
        if prim.GetTypeName() != "Xform":
            continue
        path_str = str(prim.GetPath())
        if not path_str.startswith(default_path_str + "/"):
            continue
        rel = path_str[len(default_path_str) + 1 :]
        link_rel_paths.append(rel)

    out_path = out_dir / "robot_reset_xform.usda"
    _write_reset_xform_wrapper(out_path, str(src_robot_usda.resolve()), default_name, link_rel_paths)
    print(
        f"[_author_robot_reset_xform_wrapper] flagged {len(link_rel_paths)} " f"link Xform(s) under /{default_name}",
        flush=True,
    )
    return out_path, default_name


def _rewrite_external_relative_refs(usda_path: Path, source_dir: Path) -> int:
    """Replace ``@../...@`` asset refs in a copied ``.usda`` with absolute paths.

    Upstream USD assets often reference sibling directories via
    ``@../blank/foo.usd@``-style paths, relative to the .usda file's
    *original* location. When we copy the .usda into a new directory
    (here, ``out_dir``), those relative refs would resolve against the
    new directory and silently break (e.g. /tmp/rlinf_multi_4env/../blank
    → /tmp/blank, which usually doesn't exist).

    We fix this by rewriting any reference that starts with ``..`` to an
    absolute path resolved against ``source_dir`` — where the .usda
    originally lived. Absolute and same-directory refs are untouched.

    Returns the number of references rewritten.
    """
    import re

    text = usda_path.read_text(errors="replace")
    pattern = re.compile(r"@([^@\n]+)@")
    rewrites: list[int] = []

    def _sub(m: "re.Match[str]") -> str:
        ref = m.group(1).strip()
        if not ref or ref.startswith("/") or not ref.startswith(".."):
            return m.group(0)
        absolute = (source_dir / ref).resolve(strict=False)
        rewrites.append(1)
        return f"@{absolute}@"

    new_text, _ = pattern.subn(_sub, text)
    n = len(rewrites)
    if n > 0:
        usda_path.write_text(new_text)
    return n


def _env_inject(topic: str, env_id: int) -> str:
    """Mirror of ``_multi_env_helpers.env_topic_name`` (kept local to
    avoid importing from a sibling package at script-tool time)."""
    if env_id == 0:
        return topic
    parts = topic.lstrip("/").split("/", 1)
    if len(parts) == 1:
        return f"/env_{env_id}_{parts[0]}"
    ns, rest = parts
    return f"/{ns}/env_{env_id}_{rest}"


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


def assemble_multi_env(
    single_env_manifest: str,
    num_envs: int,
    output_dir: str,
    env_spacing: float = 15.0,
    head_height_m: float = 1.5,
) -> str:
    """Build a multi-env stage and return its absolute manifest path.

    Args:
        single_env_manifest: Existing single-environment manifest.
        num_envs: Total environment count, including env 0; at least one.
        output_dir: Destination for authored layers and the new manifest.
            Existing assets remain referenced at their source paths.
        env_spacing: Separation in meters along the X axis.
        head_height_m: Compatibility parameter passed to camera assembly.

    Feed the resulting manifest to both physics and rendering backends.
    """
    if num_envs < 1:
        raise ValueError(f"num_envs must be >= 1 (got {num_envs})")

    src_manifest = _load_manifest(single_env_manifest)
    src_dir = Path(single_env_manifest).resolve().parent

    # Manifest paths are mostly relative to ``base_path`` (set by the
    # upstream assemble_scene). Fall back to the manifest's own
    # directory if base_path isn't set.
    src_base_path = Path(src_manifest.get("base_path", "") or src_dir).resolve()

    def _resolve_against_base(path_str: str) -> Path:
        """Resolve a manifest path field against ``base_path``; if it's
        already absolute, return as-is."""
        p = Path(path_str)
        return p if p.is_absolute() else (src_base_path / p)

    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # render_layer.usda is the only source file we must modify (to add
    # viz cameras and the per-env head cameras). Copy it once into
    # out_dir and operate on the copy; everything else stays at the
    # source path. We rewrite the copy's ``..``-relative refs (e.g.
    # ``@../blank/T_Shirt_fold.usd@``) so they resolve against the
    # *source* directory rather than against the new out_dir location.
    src_render_layer = _resolve_against_base(src_manifest["render_layer_usda"])
    if not src_render_layer.is_file():
        raise FileNotFoundError(f"render_layer.usda not found: {src_render_layer}")
    render_layer_dst = out_dir / src_render_layer.name
    shutil.copy2(src_render_layer, render_layer_dst)
    rewrites = _rewrite_external_relative_refs(render_layer_dst, src_dir)
    if rewrites > 0:
        print(
            f"[assemble_multi_env] rewrote {rewrites} '..'-relative ref(s) " f"in {render_layer_dst.name}",
            flush=True,
        )

    # Source-side absolute paths we'll just point the manifest at:
    src_robot_usda = _resolve_against_base(src_manifest["robot_usda"])
    src_scene_usda = _resolve_against_base(src_manifest["scene_usda"])
    if not src_robot_usda.is_file():
        raise FileNotFoundError(f"robot.usda not found: {src_robot_usda}")
    if not src_scene_usda.is_file():
        raise FileNotFoundError(f"scene_usda not found: {src_scene_usda}")

    # Build scene_multi.usda referencing the original scene + N robot
    # clones — only when num_envs > 1. For N=1 we use the source base
    # scene directly (it already has its own robot at /genie/...).
    if num_envs > 1:
        scene_multi_path = out_dir / "scene_multi.usda"
        # Author a robot WRAPPER first: (a) it carries a defaultPrim so the
        # per-env reference resolves — the raw robot's root prim name is
        # robot-specific (e.g. /genie), NOT the literal /robot the old code
        # assumed, which silently composed EMPTY env prims; and (b) it flags
        # every link Xform with omni:resetXformStack so OVRTX writes each
        # link's world pose directly instead of compounding parent transforms
        # (else the mesh tree scatters). The env reference reuses the
        # wrapper's root name so a body label `genie/<link>` maps to the prim
        # `/World/envs/env_<i>/genie/<link>`.
        robot_wrapper, robot_root = _author_robot_reset_xform_wrapper(Path(src_robot_usda), out_dir)
        # RAW foreground (table + manipulable objects, no floor/lights),
        # replicated under each env so every env has its own complete scene.
        # Falls back to scene_usda for older manifests lacking the field.
        fg_src = src_manifest.get("foreground_usda") or src_manifest["scene_usda"]
        src_foreground_usda = _resolve_against_base(fg_src)
        if not src_foreground_usda.is_file():
            raise FileNotFoundError(f"foreground_usda not found: {src_foreground_usda}")
        fg_wrapper, fg_root = _author_foreground_reset_xform_wrapper(Path(src_foreground_usda), out_dir)
        _write_scene_multi_usda(
            output_path=str(scene_multi_path),
            base_scene_usda=str(src_scene_usda),
            robot_usda=str(robot_wrapper),
            robot_root_name=robot_root,
            foreground_usda=str(fg_wrapper),
            foreground_root_name=fg_root,
            num_envs=num_envs,
            env_spacing=env_spacing,
        )
        scene_usda_for_manifest = str(scene_multi_path)

        template_intrinsics = _read_camera_intrinsics_template(str(render_layer_dst))
        _add_env_cameras_to_render_layer(
            render_layer_path=str(render_layer_dst),
            num_envs=num_envs,
            env_spacing=env_spacing,
            template_camera=template_intrinsics,
            head_height_m=head_height_m,
        )
    else:
        # N=1: keep base scene as-is; the head camera config in the
        # original render_layer is already correct.
        scene_usda_for_manifest = str(src_scene_usda)

    # Author per-env default_viz_camera RenderProducts (always — even
    # for num_envs == 1 — so debug viewers always have a stable
    # external view to subscribe to).
    _add_viz_cameras_and_render_products(
        render_layer_path=str(render_layer_dst),
        num_envs=num_envs,
        env_spacing=env_spacing,
    )

    # Build new manifest pointing at the source-side robot/scene + the
    # locally-modified render_layer + extended cameras.
    out_manifest = copy.deepcopy(src_manifest)
    out_manifest["scene_usda"] = scene_usda_for_manifest
    out_manifest["render_layer_usda"] = str(render_layer_dst)
    # For multi-env, scene_multi.usda ALREADY contains the N per-env robots
    # (referenced under /World/envs/env_<i>/<root>). Leaving robot_usda set would
    # make the render node (tiled init_stage / OVRTX render_node) ALSO add a
    # PRIMARY robot at the origin under /<robot_prefix> — a stray Nth robot with
    # no table. So drop it for multi-env; single-env still needs it (that path
    # renders the primary /genie robot directly, no /World/envs clones).
    out_manifest["robot_usda"] = "" if num_envs > 1 else str(src_robot_usda)
    robot_visual = src_manifest.get("robot_visual_usda", "")
    if robot_visual:
        out_manifest["robot_visual_usda"] = str(_resolve_against_base(robot_visual))
    out_manifest["base_path"] = str(out_dir)
    out_manifest["num_envs"] = num_envs
    out_manifest["env_spacing"] = env_spacing

    if num_envs > 1:
        out_manifest = _extend_manifest_cameras(out_manifest, num_envs)
    out_manifest = _add_viz_cameras_to_manifest(out_manifest, num_envs)

    out_manifest_path = out_dir / "manifest.json"
    _save_manifest(out_manifest, str(out_manifest_path))

    return str(out_manifest_path)


def _read_camera_intrinsics_template(render_layer_path: str) -> dict[str, Any]:
    """Read focal_length / aperture from the existing env-0 head camera."""
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(render_layer_path)
    cam_path = _env0_head_camera_path(stage)
    cam_prim = stage.GetPrimAtPath(cam_path) if cam_path else None
    if cam_prim is None or not cam_prim.IsValid():
        # Fall back to safe defaults from the original scene_flat_g2_sp_fsvbd.
        return {"focal_length": 10.007404, "h_aperture": 20.955, "v_aperture": 13.098667}

    cam = UsdGeom.Camera(cam_prim)
    return {
        "focal_length": float(cam.GetFocalLengthAttr().Get() or 10.007404),
        "h_aperture": float(cam.GetHorizontalApertureAttr().Get() or 20.955),
        "v_aperture": float(cam.GetVerticalApertureAttr().Get() or 13.098667),
    }


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a multi-env OVRTX stage from a single-env manifest")
    parser.add_argument(
        "--input-manifest",
        required=True,
        help="path to an existing single-env manifest.json " "(produced by upstream assemble_scene)",
    )
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument(
        "--output-dir",
        required=True,
        help="where to write the multi-env scene + manifest",
    )
    parser.add_argument(
        "--env-spacing",
        type=float,
        default=1.5,
        help="distance (m) between consecutive envs along +X",
    )
    parser.add_argument(
        "--head-height",
        type=float,
        default=1.5,
        help="approximate head height (m) for static env_i camera placement " "until upstream multi-world FK lands",
    )
    args = parser.parse_args(argv)

    out = assemble_multi_env(
        single_env_manifest=args.input_manifest,
        num_envs=args.num_envs,
        output_dir=args.output_dir,
        env_spacing=args.env_spacing,
        head_height_m=args.head_height,
    )
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

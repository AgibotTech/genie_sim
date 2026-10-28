# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Shared Kit / stage / pose plumbing for the Isaac render nodes.

Factored out of ``isaacsim_render.py`` so BOTH the per-camera ROS node and the
new ``isaacsim_tiled_render.py`` (TiledCamera + SHM) reuse the same manifest
parsing, stage loading, camera intrinsics, and /tf_render prim driving.

Contract: the pxr / isaacsim / omni imports are done LAZILY inside the functions
so this module is importable on a plain host (for ``Manifest`` parsing / unit
tests) while the Kit-dependent functions run only inside a live SimulationApp.
The entry script MUST create ``SimulationApp`` before calling any of the
Kit-dependent functions below.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

# --------------------------------------------------------------------------
# Adaptive RTX subframes: how many subframes one RTX pass renders, scaled by
# how many ticks were skipped since the last rendered frame.
# --------------------------------------------------------------------------
# Where this came from: the render-only node renders on demand (once per
# /tf_render tick change), and its bridge publishes no poses at all when
# need_images is false (render/bridges/isaac_tiled.py's
# _skip_poses_when_uncollected). So with images_at_chunk_end_only=true the
# first K-1 steps of a chunk never render, and step K's poses are one large
# jump away from the last rendered frame. A single subframe then has neither
# reusable temporal history (disocclusion) nor enough samples, and the
# denoiser can only emit an unconverged image. Rendering every step, the
# accumulation carries across steps and the last frame is clean.
#
# The fused node reached the same situation from the other direction: it
# rendered every step until a step could decline its image, and on-demand
# rendering is what gave it gaps. Hence one definition here rather than a copy
# in each node -- two copies drift, and the drift is invisible: one backend's
# pictures quietly softer than the other's.
#
# Why adapt on the gap instead of configuring a constant: only the node knows
# how many frames were skipped. The client's policy (image every step, only at
# chunk end, something else later) is opaque to it. The gap IS "how much
# accumulation this frame is missing", so using it as the subframe count is
# self-explanatory:
#   gap <= 1   rendered every tick -> 1 subframe, byte-identical to before
#              this existed, zero added cost
#   gap == K   K-1 were skipped -> K subframes (capped), restoring what was
#              missed
#
# Env knobs keep their original ``RLINF_TILED_*`` spelling: they are the names
# already in deployment scripts and runbooks, and renaming them would break
# those silently for a cosmetic gain.
#: Cap, against the pathological case (a long pause makes gap run to
#: thousands). Measured on the render-only node (4090, 3 tiles, 640x400):
#: 1 subframe = 65.4 ms, 8 subframes = 122.0 ms -- 1.87x, not 8x, because the
#: ~57 ms of fixed cost (scene traversal, tile assembly) is paid either way and
#: only shading scales. Which is the whole reason adapting beats rendering
#: every tick: a few expensive frames still cost far less than many cheap ones.
RT_SUBFRAMES_MAX = max(1, int(os.environ.get("RLINF_TILED_RT_SUBFRAMES_MAX", "8") or 8))
#: Hard override; empty (the default) means "adapt".
RT_SUBFRAMES_FIXED = os.environ.get("RLINF_TILED_RT_SUBFRAMES", "").strip()
RT_SUBFRAMES_RESET = max(1, int(os.environ.get("RLINF_TILED_RT_SUBFRAMES_RESET", "16") or 16))


def subframes_for_gap(gap, after_reset: bool = False) -> int:
    """Subframes to render, given how many ticks were skipped since the last
    rendered frame.

    ``after_reset``: render at the cap, ignoring the gap. A reset teleports the
    robot AND every object inside a single tick, so the gap reads as 1 while
    the temporal history is in fact entirely stale -- nothing in the previous
    frame is still where it was. Measured on the render-only node: the first
    frame after a reset had laplacian variance 96.3 against 129.2 for the same
    moment rendered every tick, while mid-episode (gap=10 -> 8 subframes) came
    out at 123.6 against 113.4. Only the reset frame was actually short.

    ``gap is None``: cold start or hot reload, where nothing is accumulated at
    all. That is the long-gap case, not the gap-of-1 case.
    """
    if RT_SUBFRAMES_FIXED:
        return max(1, int(RT_SUBFRAMES_FIXED))
    if after_reset:
        return RT_SUBFRAMES_RESET
    if gap is None:
        return RT_SUBFRAMES_MAX
    return max(1, min(int(gap), RT_SUBFRAMES_MAX))


# ---------------------------------------------------------------------------
# Manifest parsing (pure Python — host importable)
# ---------------------------------------------------------------------------


def load_scene_config(manifest_path) -> dict:
    """Read the scene named by the manifest, including legacy fixed filenames.

    An explicit missing/broken scene must fail: falling back to a neighbouring
    file could silently reset this task using another task's instance table.
    Path anchoring matches Manifest.load and the core Manifest contract.
    """
    import yaml

    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    scene_path = manifest.get("scene_yaml")
    if scene_path:
        cfg_path = Path(scene_path)
        if not cfg_path.is_absolute():
            base = Path(manifest["base_path"]) if manifest.get("base_path") else manifest_path.parent
            cfg_path = base / cfg_path
    else:
        cfg_path = manifest_path.parent / "rl_scene_cfg.json"
        if not cfg_path.is_file():
            return {}
    cfg = yaml.safe_load(cfg_path.read_text())
    if not isinstance(cfg, dict):
        raise ValueError(f"scene config must be a mapping: {cfg_path}")
    return cfg


@dataclass
class CameraConfig:
    path: str
    topic: str
    render_product_path: str
    width: int = 1280
    height: int = 800
    # This camera's NATIVE (benchmark) resolution — the render tile is uniform
    # (width x height) for all cameras, but the vertical FOV is derived from the
    # native aspect so a non-1.6 camera stretches (not crops) into the tile.
    # 0 => fall back to width/height (square-with-tile, old behaviour).
    native_width: int = 0
    native_height: int = 0
    fx: float = 610.0
    fy: float = 610.0
    cx: float = 640.0
    cy: float = 400.0
    min_range: float = 0.01
    max_range: float = 10000.0
    is_free_cam: bool = False
    # Debug/viz "freecam": if set, build_gridcloner_stage AUTHORS a world-anchored
    # look-at camera (child of each env) at these env-LOCAL offsets instead of
    # resolving a robot camera. [x,y,z] eye + [x,y,z] target, relative to env origin.
    freecam_eye: Optional[List[float]] = None
    freecam_target: Optional[List[float]] = None


@dataclass
class Manifest:
    scene_usda: str
    #: Optional. Stage the render node OPENS: global env only, no foreground.
    #: Empty => open ``scene_usda`` and deactivate the strays (no env shell).
    #: Added WITHOUT a schema-version bump, per the core manifest schema's
    #: carve-out for optional fields the C++ mirror does not read.
    env_shell_usda: str = ""
    render_layer_usda: str = ""
    robot_usda: str = ""
    foreground_usda: str = ""
    background_usda: str = ""
    # Per-env background override (per-env room generalization).
    # Empty (the default) => every env clones ``background_usda`` verbatim
    # (backward compatible). When non-empty, entry ``i`` (aligned to env
    # index, NOT to any other list in this class) is the absolute
    # ``background.usda`` path env ``i`` should get instead; a shorter list
    # or an empty string at some index leaves THAT env on the
    # ``background_usda`` default (no override, not "no background"). This
    # is a build-time assignment, not a per-reset one: whoever populates
    # this (task_loader / whatever calls build_stage_manifest) decides the
    # room ONCE per stage build, using
    # ``randomize.background``/``randomize.sampler`` — see
    # docs/GENERALIZATION_DESIGN.md Section 9. Meaningless unless a
    # background is being loaded at all (``task_background.
    # resolve_use_background`` decided that already; this field only
    # decides WHICH room once that decision is yes).
    background_usda_per_env: List[str] = field(default_factory=list)
    robot_prefix: str = ""
    free_cam_prim_path: str = ""
    cameras: List[CameraConfig] = field(default_factory=list)
    #: ``robot.body_paths`` — short link name -> absolute USD prim path. Used to
    #: drive the robot from live tf: the link tree is authoritative, and the
    #: short names are what /tf frames are called.
    robot_body_paths: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, manifest_path) -> "Manifest":
        """Load a ``manifest.json`` (same schema as the C++/OVRTX RenderNode +
        isaacsim_render.py). Relative paths resolve against ``base_path`` (or the
        manifest's own dir)."""
        manifest_path = Path(manifest_path).resolve()
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        data = json.loads(manifest_path.read_text())
        base_dir = Path(data["base_path"]) if data.get("base_path") else manifest_path.parent

        def _resolve(p: str) -> str:
            if not p:
                return p
            pp = Path(p)
            return str(pp if pp.is_absolute() else (base_dir / pp).resolve())

        scene_usda = _resolve(data.get("scene_usda") or data.get("usd_path", ""))
        if not scene_usda:
            raise KeyError("Manifest missing both 'scene_usda' and 'usd_path'")
        cams: List[CameraConfig] = []
        for i, cj in enumerate(data.get("cameras", [])):
            w = int(cj.get("width", 1280))
            h = int(cj.get("height", 800))
            cams.append(
                CameraConfig(
                    path=cj["path"],
                    topic=cj["topic"],
                    render_product_path=cj.get("render_product_path") or f"/RenderOVRTX/Cam_{i}",
                    width=w,
                    height=h,
                    native_width=int(cj.get("native_width", 0)) or w,
                    native_height=int(cj.get("native_height", 0)) or h,
                    fx=float(cj.get("fx", 610.0)),
                    fy=float(cj.get("fy", 610.0)),
                    cx=float(cj.get("cx", w / 2.0)),
                    cy=float(cj.get("cy", h / 2.0)),
                    min_range=float(cj.get("min_range", 0.01)),
                    max_range=float(cj.get("max_range", 10000.0)),
                    is_free_cam=bool(cj.get("is_free_cam", False)),
                    freecam_eye=cj.get("freecam_eye"),
                    freecam_target=cj.get("freecam_target"),
                )
            )
        return cls(
            scene_usda=scene_usda,
            env_shell_usda=_resolve(data.get("env_shell_usda", "")),
            render_layer_usda=_resolve(data.get("render_layer_usda", "")),
            robot_usda=_resolve(data.get("robot_usda", "")),
            foreground_usda=_resolve(data.get("foreground_usda", "")),
            background_usda=_resolve(data.get("background_usda", "")),
            # Empty entries stay empty ("no override for this env") rather
            # than resolving to base_dir itself -- ``_resolve("")`` already
            # returns "" unchanged, matching ``background_usda``'s own
            # empty-string handling above.
            background_usda_per_env=[_resolve(p) for p in (data.get("background_usda_per_env") or [])],
            robot_prefix=data.get("robot_prefix", ""),
            free_cam_prim_path=data.get("free_cam_prim_path", ""),
            cameras=cams,
            robot_body_paths=dict((data.get("robot") or {}).get("body_paths") or {}),
        )

    def obs_cameras(self) -> List[CameraConfig]:
        """The observation cameras = non-free-cam entries (per-env head cams).
        Free/viz cameras (``is_free_cam``) are debug-only, excluded from obs."""
        return [c for c in self.cameras if not c.is_free_cam]


# ---------------------------------------------------------------------------
# Kit-dependent helpers (call only after SimulationApp is created)
# ---------------------------------------------------------------------------


def set_camera_intrinsics(prim, cam: CameraConfig, focal_length: float = 1.0) -> None:
    """Reproduce the robot-USDA camera's CALIBRATED model in the uniform obs tile.

    Each obs camera is a real robot-USDA ``Camera`` prim carrying a fully
    calibrated OpenCV pinhole model (``omni:lensdistortion:opencvPinhole`` fx/fy/
    cx/cy + k/p distortion, at ``imageSize``) with an
    ``OmniLensDistortionOpenCvPinholeAPI`` schema applied. That IS the camera the
    benchmark renders, and its true FOV / principal point / DISTORTION come from
    that model — NOT from the prim's nominal ``focalLength``/``horizontalAperture``,
    which can and do disagree (e.g. the wrist's baked aperture says hFOV 88 deg
    while its calibrated fx says 105 deg). The RTX pass (shared by the tiled
    sensor) applies the schema's distortion — VERIFIED by forcing a large k1 and
    seeing straight edges bow — so we must feed it a model consistent with the
    tile resolution:

      * rescale the OpenCV model + ``imageSize`` to the tile so the RTX
        distortion pass reproduces the benchmark's warp. The rescale is a change
        of pixel units only: normalized coords ``(x-cx)/fx`` are invariant under
        ``fx,cx *= sx``, so a NON-UNIFORM rescale (wrist 1280x1056 -> 640x400) is
        exactly "benchmark render at native + non-uniform resize into the tile"
        (distortion coeffs are normalized => carry over unchanged);
      * also derive the USD aperture (+ offset) from the same fx/fy/cx/cy as a
        self-consistent fallback for any path that reads the frustum instead.

    So the calibrated FOV + distortion RESIZE (stretch) into the uniform tile
    rather than being cropped. Synthesize mode (a camera with NO baked OpenCV
    model) falls back to a plain pinhole from ``cam.fx`` with the native-aspect
    vertical FOV.
    """
    ox = "omni:lensdistortion:opencvPinhole"
    fx_a = prim.GetAttribute(f"{ox}:fx")
    fy_a = prim.GetAttribute(f"{ox}:fy")
    cx_a = prim.GetAttribute(f"{ox}:cx")
    cy_a = prim.GetAttribute(f"{ox}:cy")
    img_a = prim.GetAttribute(f"{ox}:imageSize")
    have_opencv = fx_a and fx_a.IsValid() and fx_a.Get() and fy_a and fy_a.IsValid() and fy_a.Get()
    if have_opencv:  # mirror mode: calibrated OpenCV model
        focal = float(prim.GetAttribute("focalLength").Get() or 1.0)
        fx, fy = float(fx_a.Get()), float(fy_a.Get())
        if img_a and img_a.IsValid() and img_a.Get() is not None:
            nat_w, nat_h = (int(v) for v in img_a.Get())
        else:
            nat_w, nat_h = cam.native_width or cam.width, cam.native_height or cam.height
        cx = float(cx_a.Get()) if (cx_a and cx_a.IsValid()) else nat_w / 2.0
        cy = float(cy_a.Get()) if (cy_a and cy_a.IsValid()) else nat_h / 2.0
        # Aperture reproducing the calibrated FOV (resolution-independent), and a
        # principal-point aperture offset for the off-centre cx/cy.
        h_ap = focal * nat_w / fx
        v_ap = focal * nat_h / fy
        h_off = (cx - nat_w / 2.0) * h_ap / nat_w
        v_off = (cy - nat_h / 2.0) * v_ap / nat_h
        prim.GetAttribute("horizontalAperture").Set(h_ap)
        prim.GetAttribute("verticalAperture").Set(v_ap)
        for an, av in (("horizontalApertureOffset", h_off), ("verticalApertureOffset", v_off)):
            a = prim.GetAttribute(an)
            if a and a.IsValid():
                a.Set(float(av))
        # Rescale the OpenCV model to the tile (non-uniform => stretch), keeping
        # distortion. imageSize -> tile so the model stays self-consistent.
        sx, sy = cam.width / nat_w, cam.height / nat_h
        for a, s in ((fx_a, sx), (fy_a, sy), (cx_a, sx), (cy_a, sy)):
            if a and a.IsValid() and a.Get() is not None:
                a.Set(float(a.Get()) * s)
        if img_a and img_a.IsValid():
            from pxr import Gf  # noqa: PLC0415

            img_a.Set(Gf.Vec2i(int(cam.width), int(cam.height)))
        print(
            f"[cam-intrinsics] {cam.render_product_path}: USDA opencv "
            f"{nat_w}x{nat_h} fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f} "
            f"-> aperture h={h_ap:.3f} v={v_ap:.3f} off=({h_off:.3f},{v_off:.3f}) "
            f"-> tile {cam.width}x{cam.height}",
            flush=True,
        )
        return
    # synthesize mode: plain pinhole from cam.fx, native-aspect vertical FOV.
    focal = focal_length
    h_ap = cam.width * focal / cam.fx
    nat_w = cam.native_width or cam.width
    nat_h = cam.native_height or cam.height
    v_ap = h_ap * (nat_h / nat_w)
    prim.GetAttribute("focalLength").Set(focal)
    prim.GetAttribute("horizontalAperture").Set(h_ap)
    prim.GetAttribute("verticalAperture").Set(v_ap)
    fx = cam.width * focal / h_ap
    fy = cam.height * focal / v_ap
    print(
        f"[cam-intrinsics] {cam.render_product_path}: SYNTH fx={fx:.1f} "
        f"fy={fy:.1f} native={nat_w}x{nat_h} tile={cam.width}x{cam.height}",
        flush=True,
    )
    for attr, val in (("cx", cam.width / 2.0), ("cy", cam.height / 2.0), ("fx", fx), ("fy", fy)):
        a = prim.GetAttribute(f"{ox}:{attr}")
        if a and a.IsValid():
            a.Set(float(val))


def author_lookat_camera(stage, path, eye, target, up=(0.0, 0.0, 1.0), h_aperture=2.0, aspect=1.6, focal=1.0):
    """Author a UsdGeom.Camera at ``eye`` looking at ``target`` (USD camera looks
    down local -Z). Wide FOV + huge far clip so it frames a whole env. ``eye``/
    ``target`` are in the coordinate space of ``path``'s parent (env-local when
    ``path`` is a child of an env prim). Returns the camera prim."""
    from pxr import Gf, UsdGeom

    cam = UsdGeom.Camera.Define(stage, path)
    prim = cam.GetPrim()
    eye_v, tgt_v, up_v = Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(*up)
    fwd = (tgt_v - eye_v).GetNormalized()
    right = Gf.Cross(fwd, up_v).GetNormalized()
    trueup = Gf.Cross(right, fwd).GetNormalized()
    # Rotation-only matrix (rows = camera X/Y/Z basis in parent space) -> quat.
    # Author as translate + orient (NOT a single xformOp:transform matrix): the
    # RTX camera / TiledCameraSensor renders black off a matrix xformOp on a
    # freshly-authored camera (same gotcha objects hit), fine off translate+orient.
    m = Gf.Matrix4d()
    m.SetRow(0, Gf.Vec4d(right[0], right[1], right[2], 0.0))
    m.SetRow(1, Gf.Vec4d(trueup[0], trueup[1], trueup[2], 0.0))
    m.SetRow(2, Gf.Vec4d(-fwd[0], -fwd[1], -fwd[2], 0.0))
    m.SetRow(3, Gf.Vec4d(0.0, 0.0, 0.0, 1.0))
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(eye_v)
    xf.AddOrientOp().Set(Gf.Quatf(m.ExtractRotationQuat()))
    cam.CreateFocalLengthAttr(float(focal))
    cam.CreateHorizontalApertureAttr(float(h_aperture))
    cam.CreateVerticalApertureAttr(float(h_aperture / max(aspect, 1e-3)))
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 1000.0))
    return prim


def apply_xform(prim, translation, rotation) -> None:
    """Write translate + orient onto an existing xform prim (Gf.Vec3d / Gf.Quatd)."""
    from pxr import Gf, Sdf, UsdGeom

    if not prim or not prim.IsValid():
        return
    t_attr = prim.GetAttribute("xformOp:translate")
    order_attr = prim.GetAttribute("xformOpOrder")
    order = list(order_attr.Get()) if order_attr and order_attr.Get() else []
    # Rebuild the op stack when there's no translate op OR the translate op is not
    # in xformOpOrder. Foreground object /entity prims (from the payload) carry a
    # xformOp:transform MATRIX, not translate — so a bare t_attr.Set() below would
    # be silently ignored (the object stays at its authored pose while the driver
    # thinks it drove it). One rebuild per prim makes the order [translate, orient].
    if not t_attr or not t_attr.IsValid() or "xformOp:translate" not in order:
        xf = UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(translation)
        # ClearXformOpOrder drops the ORDER but not the pre-existing op ATTRIBUTES:
        # some payload objects author a xformOp:orient at DOUBLE precision (quatd).
        # AddOrientOp() defaults to float and RAISES a Tf error on that mismatch,
        # which crashes the render node. Match the existing precision when it's quatd.
        oo = prim.GetAttribute("xformOp:orient")
        if oo and oo.IsValid() and oo.GetTypeName() == Sdf.ValueTypeNames.Quatd:
            xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(rotation))
        else:
            xf.AddOrientOp().Set(Gf.Quatf(rotation))
        return
    t_attr.Set(translation)
    o_attr = prim.GetAttribute("xformOp:orient")
    if o_attr and o_attr.IsValid():
        o_attr.Set(Gf.Quatf(rotation) if o_attr.GetTypeName() == Sdf.ValueTypeNames.Quatf else rotation)


def reference_scene_stage(scene_usda: str) -> None:
    """Load a scene by referencing it into an empty stage's ``/World``.

    Deliberately not ``open_stage``. Under Isaac Sim 5.1 an ``open_stage`` of a
    real multi-sublayer scene followed by ``simulation_app.update()`` segfaults
    whenever ``isaacsim.ros2.bridge`` is enabled (crash frames in
    ``libomni.graph.core.plugin.so`` / ``libomni.graph.image.core.plugin.so``).
    Bisected: the bridge alone is fine, a trivial generated stage is fine, and
    the scene alone is fine -- only the combination crashes. Referencing the
    scene in instead of opening it survives repeated updates, which is the
    pattern the verified data-collection replay pipeline uses.

    Referencing at ``/World`` (after dropping the default stage's own
    ``/World``) keeps the scene's prim paths identical to what ``open_stage``
    produced, so manifest / deformable-channel ``prim_path`` values still
    resolve.
    """
    from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage

    stage = get_current_stage()
    default_world = stage.GetPrimAtPath("/World")
    if default_world and default_world.IsValid():
        stage.RemovePrim("/World")
    add_reference_to_stage(scene_usda, "/World")


def init_stage(manifest: Manifest, simulation_app, *, warmup: int = 10, create_world: bool = True):
    """Load scene_usda, attach robot + render-layer sublayers. Returns
    ``(world, stage)`` (``world`` is ``None`` when ``create_world=False`` — the
    tiled render path needs no physics World, it renders via the Replicator
    orchestrator). Mirrors isaacsim_render._init_stage."""
    from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage

    reference_scene_stage(manifest.scene_usda)
    for _ in range(warmup):
        simulation_app.update()
    world = None
    if create_world:
        from isaacsim.core.api import World

        world = World(stage_units_in_meters=1.0)
    stage = get_current_stage()
    robot_root = ""
    if manifest.robot_usda:
        robot_root = "/" + manifest.robot_prefix if manifest.robot_prefix else "/Robot"
        add_reference_to_stage(manifest.robot_usda, robot_root)
        select_physics_variant(stage, robot_root, "none")
    if manifest.render_layer_usda:
        add_reference_to_stage(manifest.render_layer_usda, "/RenderOVRTX")
    if robot_root:
        disable_physics_subtree(stage, robot_root, tag="init_stage")
    if world is not None:
        world.reset()
    for _ in range(warmup):
        simulation_app.update()  # let the added references + prims settle
    return world, stage


def _find_prim_by_leaf(stage, root_path: str, leaf_name: str) -> Optional[str]:
    """First prim under ``root_path`` whose leaf name == ``leaf_name`` (depth
    traversal). Used to locate a robot link (e.g. head_link3) whose full path is
    nested + robot-specific."""
    from pxr import Usd

    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return None
    for prim in Usd.PrimRange(root):
        if prim.GetName() == leaf_name:
            return str(prim.GetPath())
    return None


def select_physics_variant(stage, root_path: str, selection: str = "none") -> bool:
    """Point the robot prim's ``Physics`` variantSet at ``selection``.

    Robot assets carry a ``Physics`` variantSet (``mujoco`` / ``none`` /
    ``physics`` / ``physx``) whose default selection is ``physx``. That payload
    only layers ``PhysxJointAPI`` + ``maxJointVelocity`` **overs** onto the joint
    prims -- the ``physics:body0`` / ``body1`` relationships live in the
    ``physics`` variant instead -- so PhysX receives joints with no bodies at
    either end and logs ``CreateJoint - no bodies defined`` once per joint.

    The render path drives every link from tf and needs no solver at all, so
    selecting ``none`` keeps the physics payload from ever loading. Returns
    whether a selection was made."""
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return False
    vsets = root.GetVariantSets()
    if not vsets.HasVariantSet("Physics"):
        return False
    vset = vsets.GetVariantSet("Physics")
    if selection not in vset.GetVariantNames():
        return False
    return bool(vset.SetVariantSelection(selection))


def fused_body_tables(body_rel_paths, object_rel_paths):
    """The fused ``SceneSchema`` body triple: ``(paths, kinds, link_names)``.

    Objects must sit in the body array together with the robot links, and
    their labels must keep the **full env-relative path**, not be trimmed to
    the leaf name. Neither point is a style choice -- both are hard
    requirements of the eval pipeline:

    * ``sim_handler._schema_has_objects()`` uses
      ``any("/Objects/" in n for n in body_link_names)`` to decide whether to
      build the success-check tree at all. If objects aren't in this table,
      that gate is always false, the checker tree is **never built**, and
      every step just writes reward 0 / success False with no log line at
      all.
    * ``ApiCoreShim.refresh()`` only builds the object index for labels
      containing ``/Objects/``; robot links are looked up by their full
      ``genie/<link>`` path instead. Trim it down to ``<link>`` and the
      gripper can no longer be found either.

    This is also the exact shape of the newton path (``World/Objects/<id>/
    entity`` and ``genie/<link>`` both in the body array, with
    ``object_names`` empty instead), so the two engines present the checker
    with the same table.

    The separate ``object_names`` / ``get_object_poses()`` channel is left
    unchanged -- it has other consumers and doesn't conflict with this, at
    the cost of storing an extra M x 7 floats per env.

    Note: once a caller adds objects to the body table, ``get_body_poses()``
    must be updated to fill in their poses too. Having an entry in the table
    but not in the poses is worse than the original bug -- the shim would
    get an out-of-range index, or one pointing at a different body.
    """
    paths = list(body_rel_paths) + list(object_rel_paths)
    kinds = ["robot_link"] * len(body_rel_paths) + ["rigid_object"] * len(object_rel_paths)
    return paths, kinds, list(paths)


def local_aabb_center_half(stage, prim_path):
    """``[cx, cy, cz, hx, hy, hz]`` -- the prim's AABB in its **own local
    frame** (center + half-extents).

    Same semantics as newton's ``/rlinf/body_aabb_local`` table: static,
    local-frame, with the client composing it against the live pose into a
    world-frame AABB (``_world_aabb_from_local``). The geometry doesn't
    change at runtime, so this table is computed once at startup and sent
    once with the schema.

    Uses ``ComputeUntransformedBound``, not ``ComputeLocalBound``: the latter
    **includes** the prim's own transform (verified identical to
    ``ComputeWorldBound``), so using it as the static table would bake the
    object's world position in, and composing it with the live pose a
    second time would double it.

    Returns all zeros when the prim doesn't exist, or its bounding box is
    empty/infinite (no renderable geometry) -- matching newton's own
    degraded behaviour of "send a zero box when the table hasn't arrived",
    which the caller (``api_core_shim.get_obj_aabb``) already treats as
    best-effort.
    """
    from pxr import Usd, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return [0.0] * 6
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    try:
        rng = cache.ComputeUntransformedBound(prim).ComputeAlignedRange()
    except Exception:  # noqa: BLE001 -- missing/unevaluable geometry both degrade to a zero box
        return [0.0] * 6
    if rng.IsEmpty():
        return [0.0] * 6
    lo, hi = rng.GetMin(), rng.GetMax()
    vals = [float(lo[0]), float(lo[1]), float(lo[2]), float(hi[0]), float(hi[1]), float(hi[2])]
    if not all(v == v and abs(v) != float("inf") for v in vals):
        return [0.0] * 6
    return [
        (vals[0] + vals[3]) * 0.5,
        (vals[1] + vals[4]) * 0.5,
        (vals[2] + vals[5]) * 0.5,
        (vals[3] - vals[0]) * 0.5,
        (vals[4] - vals[1]) * 0.5,
        (vals[5] - vals[2]) * 0.5,
    ]


def _static_prop_warn(msg: str) -> None:
    print(f"[static-props] {msg}", flush=True)


def _resolve_prop_in_env(stage, env_root: str, want: str, tail_index: dict, cache):
    """Resolve ``want`` inside **one specific env**'s subtree and compute its
    world-frame box; returns ``None`` on failure (a warning has already been
    logged). The inner unit that ``static_prop_world_aabbs`` calls once per
    env; split out on its own because the outer function needs to evaluate
    the same ``want`` separately across multiple envs.
    """
    prim = stage.GetPrimAtPath(want)
    if not (prim and prim.IsValid() and str(prim.GetPath()).startswith(env_root + "/")):
        # The exact full-path match only counts if the matched prim actually
        # sits under **this** env's subtree: in a multi-env scene, the same
        # literal ``want`` string matches exactly one absolute path, which
        # can only ever be one particular env's own copy -- it must not be
        # mistaken for "every env matched this same prim exactly" (which
        # would give every env the same box, falling back into the old
        # single-env bug).
        tail = want.rstrip("/").rsplit("/", 1)[-1]
        candidates = tail_index.get(tail, [])
        if not candidates:
            _static_prop_warn(
                f"WARN: static prop {want!r} not found under {env_root!r} "
                f"(no prim named {tail!r}) -- checkers referencing it will "
                f"get a zero box"
            )
            return None
        # Shallowest wins: a same-named child mesh node under the prop
        # (prop/mesh/... sharing the name) must not outrank the prop itself.
        depth = min(p.count("/") for p in candidates)
        shallowest = sorted(p for p in candidates if p.count("/") == depth)
        if len(shallowest) > 1:
            _static_prop_warn(
                f"WARN: static prop {want!r} is ambiguous under {env_root!r} "
                f"-- {len(shallowest)} prims share the name {tail!r} at the "
                f"same depth ({', '.join(shallowest[:4])}); refusing to "
                f"guess -- checkers referencing it will get a zero box"
            )
            return None
        prim = stage.GetPrimAtPath(shallowest[0])
    rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    if rng.IsEmpty():
        _static_prop_warn(
            f"WARN: static prop {want!r} has an empty bound under " f"{env_root!r} (no renderable geometry?) -- skipped"
        )
        return None
    lo, hi = rng.GetMin(), rng.GetMax()
    return [float(lo[0]), float(lo[1]), float(lo[2]), float(hi[0]), float(hi[1]), float(hi[2])]


def static_prop_world_aabbs(stage, env_roots, requested):
    """**Per-env, world-frame** bounding boxes for static scene props (no body
    row, not part of physics).

    Background: ``Inside``/``InBBox``-style checkers need the bounding box of
    a passive object (e.g. a storage bin), but the fused backend's
    ``get_body_aabbs()`` only indexes by rigid body -- the storage bin comes
    from the background usda, is static set-dressing geometry, and
    ``discover()`` already excludes it from the object table (see that
    method's ``rigid_body_is_enabled`` check). If it can't be found there, it
    degrades to a zero box. A zero box contains no points, so the checker
    always fails, and **silently** -- the symptom is indistinguishable from
    "the policy is just bad".

    Key difference from ``body_local_aabbs``: that table stores **local-frame**
    boxes, which the client composes with each env's live pose into a world
    box (because bodies move, and each env has its own pose row). Props don't
    move and have no pose row -- there's nothing to compose with -- so here
    the world-frame box (``ComputeWorldBound``, not
    ``ComputeUntransformedBound``) is computed and published directly.

    **Why this has to be per-env rather than just computed on env_0.** The
    fused backend uses GridCloner to clone a separate ``Background`` per
    env, and the env spacing (``env_spacing``, defaulting to 15 meters in both
    ``isaacsim_fused_sim.py`` and ``sim_server.py``) is a **real world-space
    offset** -- completely unlike the newton path, where ``spacing=(0,0,0)``
    overlaps every env in place and shares one global copy of the props. The
    object world poses the checker reads (``view.get_world_poses()``, used
    by ``_poses_into``) carry this same grid offset, so env_1's bottle is
    actually sitting 15 meters away from env_0's storage-bin box -- publishing
    only env_0's box means env_1/2/3...'s ``Inside`` checker is forever
    judging against a box 15 meters from itself, always fails, and "fixing"
    it this way would be the same as not fixing it at all. So this
    resolves **each** env in ``env_roots`` independently and computes its own
    ``ComputeWorldBound``; if one env fails to resolve a prop, that prop is
    skipped for **all** envs (never publish a partial table -- a partial
    table is more dangerous than publishing nothing, since consumers can
    easily forget to check whether a given env even has this row).

    Path resolution follows the same rules as the newton reference
    implementation (``engine/newton/static_props.py``), run independently per
    env: the checker sends ``/World/<name>``, while in the composed stage it
    is ``{env_root}/Background/<name>``, so it first tries an exact
    **full-path** match (requiring the matched prim to sit under this env's
    own subtree -- see ``_resolve_prop_in_env``); if that fails, it falls
    back to searching this env's subtree by **leaf name**, taking the
    shallowest match; if flattening still leaves multiple candidates, it
    refuses to guess -- guessing wrong about a prop's box produces a score
    that "looks plausible but is wrong", which is harder to catch than having
    no box at all. The search is scoped to each env's own subtree, never the
    whole stage -- otherwise a leaf-name search would treat every env's own
    storage bin as a "same-name candidate" and immediately trigger the
    duplicate-name rejection.

    Returns ``(names, boxes)``:
      * ``names`` -- props that resolved successfully on **every** env; the
        elements are the **requested** paths (not the resolved stage paths),
        since the consumer (``api_core_shim``) looks the table up verbatim by
        the string the checker sent.
      * ``boxes`` -- ``{name: [[min_x,min_y,min_z,max_x,max_y,max_z], ...]}``,
        where the outer list's length equals ``len(env_roots)`` and index i
        corresponds to env i, world-frame, each row with ``min <= max``,
        following the same 6-element convention as ``get_body_aabbs()`` with
        just an extra env axis on top -- this is the **exact shape** of
        ``SceneSchema.static_prop_world_aabbs`` once it's on the wire, which
        both ``isaac_fused.py`` and ``api_core_shim.py`` read assuming this
        shape.

    Any failure to resolve / an empty bounding box / any exception **skips
    and logs a warning** rather than publishing a zero box -- a zero box
    looks like a real answer and masks the problem, while skipping lets
    ``api_core_shim``'s own "nobody recognizes this path" warning fire
    normally. No exception ever propagates: prop boxes are a nice-to-have for
    evaluation, not worth failing engine startup over.
    """
    from pxr import Usd, UsdGeom

    names: list = []
    boxes: dict = {}
    wanted = [str(w) for w in (requested or ()) if str(w).strip()]
    env_roots = [str(r) for r in (env_roots or ())]
    if not wanted or not env_roots:
        return names, boxes

    try:
        # purpose uses [default_, render]: props are written against render
        # geometry; guide/proxy would pull in auxiliary geometry, not the
        # prop itself (same tradeoff as the newton reference implementation).
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
        # Build one tail-name index per env, built once (not once per prop)
        # -- the env subtree doesn't change during this call, so N props x M
        # envs amortizes to M traversals instead of N x M. A position with a
        # missing env subtree is left as None, as a placeholder to keep
        # alignment with env_roots' indices.
        env_indexes: list = []
        for env_root in env_roots:
            root_prim = stage.GetPrimAtPath(env_root)
            if not root_prim or not root_prim.IsValid():
                _static_prop_warn(
                    f"WARN: env root {env_root!r} missing on "
                    f"the stage; any static prop requiring it "
                    f"will be skipped for ALL envs"
                )
                env_indexes.append(None)
                continue
            idx: dict = {}
            for prim in Usd.PrimRange(root_prim):
                idx.setdefault(prim.GetName(), []).append(str(prim.GetPath()))
            env_indexes.append(idx)
    except Exception as exc:  # noqa: BLE001 -- prop boxes are a nice-to-have, must not take down startup
        _static_prop_warn(f"WARN: stage traversal failed ({exc!r}); no " f"static prop boxes published")
        return names, boxes

    for want in wanted:
        per_env_boxes: list = []
        resolved = True
        for env_root, tail_index in zip(env_roots, env_indexes):
            if tail_index is None:
                resolved = False
                break
            try:
                box = _resolve_prop_in_env(stage, env_root, want, tail_index, cache)
            except Exception as exc:  # noqa: BLE001
                _static_prop_warn(
                    f"WARN: static prop {want!r} failed under " f"{env_root!r} ({exc!r}) -- skipped for " f"ALL envs"
                )
                box = None
            if box is None:
                resolved = False
                break
            per_env_boxes.append(box)
        if resolved:
            names.append(want)
            boxes[want] = per_env_boxes

    return names, boxes


def rigid_body_is_enabled(prim) -> bool:
    """True if and only if ``prim`` is a rigid body that **the physics engine
    will actually build**.

    ``HasAPI(RigidBodyAPI)`` only tells you the prim carries the rigid-body
    schema, not that it's turned on. Assets often use
    ``rigidBodyEnabled = false`` to mean "this is set dressing": measured on
    the ``laboratory_2`` room's ``Background`` subtree once mounted into the
    world, 246 prims carry ``RigidBodyAPI``, of which 176 (cardboard boxes,
    notebooks, etc.) are overridden to false by the room layer -- overriding
    a true the asset itself authored -- so the check must read the
    **composed** value, not just one layer.

    PhysX honours this flag and won't create rigid bodies for them. Feeding
    these paths into a ``RigidPrim`` view gets "did not match any rigid
    bodies", which then triggers a GPU assertion on
    ``get_angular_velocities()`` (``GpuRigidBodyView.cpp: 371``) that takes
    the whole process down with it.

    This check is aligned with the newton path's
    ``_strip_disabled_physics_schemas``
    (``genie_sim_engine_newton/.../setup/model.py``): that side de-applies
    ``PhysicsRigidBodyAPI`` outright on any prim whose composed value is
    false, before ``builder.add_usd()`` runs, so Newton never sees them in
    the first place.

    Returns True when the attribute isn't authored at all -- USD's default
    is true, and "not written" doesn't mean "turned off". (Once a schema is
    applied, ``GetRigidBodyEnabledAttr()`` is always valid, and ``Get()``
    returns the schema fallback ``True`` when unauthored; so there's no need
    to additionally check whether the attr is empty here -- that branch is
    unreachable given ``HasAPI`` is already true.)
    """
    from pxr import UsdPhysics

    if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
        return False
    return UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get() is not False


def disable_physics_subtree(stage, root_path: str, *, tag: str = "render") -> int:
    """Remove physics / articulation / collision APIs from every prim under
    ``root_path`` so the render's kinematic per-prim world-pose writes (from tf)
    aren't overridden or "pushed back" by the articulation/rigid-body solver.
    Mirrors data-collection kinematic_replay.disable_physics_on_stage in
    link-replay mode (include_robot=True). Returns the number of APIs removed."""
    from pxr import Usd, UsdPhysics, PhysxSchema

    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return 0
    n = 0
    apis = [
        UsdPhysics.ArticulationRootAPI,
        UsdPhysics.RigidBodyAPI,
        UsdPhysics.CollisionAPI,
        PhysxSchema.PhysxArticulationAPI,
        PhysxSchema.PhysxRigidBodyAPI,
        PhysxSchema.PhysxCollisionAPI,
    ]
    for prim in Usd.PrimRange(root):
        for api in apis:
            try:
                if prim.HasAPI(api):
                    prim.RemoveAPI(api)
                    n += 1
            except Exception:  # noqa: BLE001
                pass
    print(f"[{tag}] disabled {n} physics APIs under {root_path}", flush=True)
    return n


def make_render_only(stage, root_path: str) -> None:
    """Make everything under ``root_path`` scenery: no colliders, rigid bodies,
    articulations or joints.

    For per-env background rooms swapped into a physics-alive (fused) stage.
    Background randomization is only allowed for tasks whose room is pure
    decor, and on the render-only path the room never meets physics at all; on
    the fused path it would be simulated, and a room's furniture can overlap
    the robot or the table. Instanced prims are de-instanced first, because
    edits cannot be authored on instance proxies. Joints are disabled rather
    than left pointing at bodies that no longer exist.
    """
    from pxr import Usd, UsdPhysics

    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return
    apis = [UsdPhysics.ArticulationRootAPI, UsdPhysics.RigidBodyAPI, UsdPhysics.CollisionAPI]
    try:
        from pxr import PhysxSchema

        apis += [PhysxSchema.PhysxArticulationAPI, PhysxSchema.PhysxRigidBodyAPI, PhysxSchema.PhysxCollisionAPI]
    except ImportError:  # a plain pxr build without the PhysX schemas
        pass
    while True:
        instances = [p for p in Usd.PrimRange(root) if p.IsInstance()]
        if not instances:
            break
        for prim in instances:
            prim.SetInstanceable(False)
    n_apis = n_joints = 0
    for prim in Usd.PrimRange(root):
        for api in apis:
            if prim.HasAPI(api):
                prim.RemoveAPI(api)
                n_apis += 1
        if prim.IsA(UsdPhysics.Joint):
            UsdPhysics.Joint(prim).CreateJointEnabledAttr().Set(False)
            n_joints += 1
    print(
        f"[gridcloner] {root_path} is scenery only: removed {n_apis} physics API(s), " f"disabled {n_joints} joint(s)",
        flush=True,
    )


def _ensure_physics_scene(stage) -> None:
    """Create (or reuse) the stage's ``UsdPhysics.Scene`` for the FUSED
    physics+render backend and turn on GPU dynamics + GPU broadphase.

    Only called from ``build_gridcloner_stage(..., physics=True)`` -- the
    render-only path (the default) never calls this.

    "Check first": mirrors ``_locate_physics_scene`` in
    genie_sim_engine_isaacsim's ``kit/stage.py`` -- try the conventional
    ``/physicsScene`` path, else fall back to scanning the whole stage for
    an already-authored ``UsdPhysics.Scene`` prim, because the base
    ``scene_usda`` MAY already carry one (authored for some other consumer
    of that same file) at a non-default path. ``UsdPhysics.Scene.Define``
    on an EXISTING scene's path just wraps it (no duplicate scene prim is
    created); the gravity/GPU attrs below are then set on whichever scene
    was found, same as a freshly created one.

    Gravity magnitude follows Isaac's own convention for this, reproduced
    by hand because this path authors raw USD/PhysX schemas instead of
    going through a ``PhysicsContext``: ``isaacsim.core.api``'s
    ``PhysicsContext.__init__`` sets its auto-created default scene's
    gravity to ``-9.81 / meters_per_unit`` (see
    ``physics_context.py``: ``self.set_gravity(value=-9.81 /
    meters_per_unit)``). Gravity has to be expressed in stage-units/s^2,
    not meters/s^2, so the division is what makes this correct for ANY
    stage rather than for one assumed unit.

    Do NOT hardcode a unit here on the strength of the "robot is meters,
    stage is cm" comments elsewhere in this file: those describe Isaac's
    ``add_reference_to_stage`` x100 reconcile, which is a different thing
    from what the stage DECLARES via ``metersPerUnit``. Measured on a real
    fused run, the composed stage reported ``metersPerUnit=1`` (metres) --
    so an assumed /0.01 would have been 100x too STRONG. The log line below
    prints both the raw magnitude and its m/s^2 equivalent precisely so this
    never has to be assumed again.

    GPU dynamics + GPU broadphase attribute names/values (``physxScene:
    enableGPUDynamics``, ``physxScene:broadphaseType`` = "GPU") and the
    ``UsdPhysics.Scene.Define`` + gravity direction/magnitude pattern are
    the same ones already used in this repo's geniesim_benchmark
    (``api_core.py`` / ``usd_utils.py``) -- reused verbatim rather than
    inventing new values.
    """
    from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics

    scene_path = "/physicsScene"
    existing = stage.GetPrimAtPath(scene_path)
    if not existing or not existing.IsValid():
        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.Scene):
                existing = prim
                scene_path = prim.GetPath().pathString
                break
    reused = bool(existing and existing.IsValid())
    scene = UsdPhysics.Scene.Define(stage, Sdf.Path(scene_path))
    # metersPerUnit falls back to 1.0 (meters) if somehow unauthored --
    # matches UsdGeom's own documented default so this never divides by 0.
    mpu = UsdGeom.GetStageMetersPerUnit(stage) or 1.0
    gravity_mag = 9.81 / mpu
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr().Set(gravity_mag)
    physx_scene = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
    physx_scene.CreateEnableGPUDynamicsAttr().Set(True)
    physx_scene.CreateBroadphaseTypeAttr().Set("GPU")
    print(
        f"[gridcloner] physics scene {'reused' if reused else 'created'} at "
        f"{scene_path}: gravity=-9.81 m/s^2 ({gravity_mag:.3f} stage-units/s^2 "
        f"@ metersPerUnit={mpu:g}), GPU dynamics=ON, broadphase=GPU",
        flush=True,
    )


def build_gridcloner_stage(
    manifest: Manifest, simulation_app, num_envs: int, env_spacing: float, *, warmup: int = 10, physics: bool = False
):
    """Multi-env render stage via Isaac **GridCloner** (the Isaac tiled backend's
    way of organizing multi-env — replaces the hand-authored scene_multi.usda).

    Authors ONE env's content (robot + foreground table/blocks + head camera)
    under ``/World/envs/env_0``, then clones it into N envs at a grid of origins,
    so every env is a faithful copy at its own origin — the robot/table/blocks
    relative layout is preserved and there are NO resetXformStack hacks. Floor +
    lights stay global (not cloned).

    Returns ``(stage, cam_paths, env_origins)``:
      * ``cam_paths``   — per-env head Camera prim path (for TiledCameraSensor).
      * ``env_origins`` — per-env (x,y,z) grid origins; the caller ADDS these when
        driving each env's robot from /tf_render (the renderer owns the layout).

    ``manifest.background_usda_per_env`` (optional): when non-empty,
    re-points env ``i``'s ``Background`` prim at a DIFFERENT room after the
    clone, instead of every env sharing the one room referenced under
    ``env_0`` before cloning. See the field's docstring on ``Manifest`` and
    step "5b" below for why this is safe with ``copy_from_source=True``.

    ``physics`` (keyword-only, default ``False``): selects which of the two
    consumers of this stage we're building for.

      * ``False`` (render-only, the default behaviour, UNCHANGED byte-for-byte):
        physics is stripped (step 3b) so the renderer's own kinematic
        per-prim world-pose writes from ``/tf_render`` aren't fought by a
        live articulation/rigid-body solver -- a separate physics process
        owns the sim and streams poses over ROS.
      * ``True`` (the FUSED backend: PhysX + tiled rendering in the SAME
        process): step 3b is skipped -- the ``ArticulationRootAPI``/
        ``RigidBodyAPI``/``CollisionAPI`` schemas on the robot/foreground
        assets are exactly what PhysX needs, there is no external pose
        stream to fight because this process IS the one moving the prims.
        The clone step also switches to ``replicate_physics=True`` (PhysX's
        own batched-env replicator instead of per-env USD physics parsing),
        and a ``UsdPhysics.Scene`` is created/configured for GPU dynamics.
    """
    from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage, open_stage
    from isaacsim.core.cloner import GridCloner
    from pxr import Gf, Sdf, UsdGeom

    # 1. base scene: /World + /RLinfEnv (floor/lights). NOTE the base scene
    #    SUBLAYERS the single-env foreground into /World/Objects (the original
    #    table + blocks, at the origin). We author our OWN per-env foreground
    #    under each /World/envs/env_i, so DEACTIVATE that sublayered original —
    #    otherwise it renders as a stray 5th table/blocks sitting undriven at the
    #    world origin (num_envs clones + 1 original = N+1 tables).
    # ``env_shell_usda`` (optional, same schema version) is a stage carrying
    # ONLY the global env -- floor + lights -- and no foreground at all. Given
    # one, everything below about deactivating strays is moot: there is nothing
    # to deactivate, because the foreground only ever appears under env_0.
    #
    # That matters beyond tidiness. Opening the full scene and then switching
    # off /World/Objects RE-HOMES any OmniGraph baked into an asset (registered
    # at the base path, torn down, re-created under env_0), and the earlier Isaac SDK
    # segfaults on a re-homed graph -- it is why clean_the_desktop could not
    # load at all.
    #
    # open_stage, not reference_scene_stage: a reference only brings in the
    # defaultPrim (/World), and the floor + lights live at the root in
    # /RLinfEnv. The 5.1 + ros2.bridge crash reference_scene_stage avoids does
    # not apply here -- neither node that builds this stage enables the bridge.
    _shell = getattr(manifest, "env_shell_usda", "") or ""
    open_stage(_shell or manifest.scene_usda)
    for _ in range(warmup):
        simulation_app.update()
    stage = get_current_stage()
    if _shell:
        # os.path, not _Path: that name is a function-local import further down.
        print(
            f"[gridcloner] opened env shell {os.path.basename(_shell)} "
            f"(no foreground on the base stage; nothing to deactivate)",
            flush=True,
        )

    # ``/Background`` is the same problem as ``/World/Objects``, one layer up:
    # the wrapper authors the room into ``scene_usda`` because the physics
    # engine opens that very file and needs the room in it. Here that copy is
    # a second room -- env_0 references its own below -- sitting undriven at
    # the world origin. Two rooms in one place look like one room until
    # something moves: turning the doorknob separates the driven handle from
    # its undriven twin and the render shows two handles.
    #
    # ``/World/Background`` is a THIRD copy: compose_background_into_scene
    # mounts the room twice (root ``/Background`` for the engine, and one under
    # the defaultPrim because a reference only takes the defaultPrim subtree).
    # The old path never switched that one off, so a global room stayed live
    # alongside each env's -- the env-shell path simply never has it.
    if not _shell:
        for stray, what in (
            ("/World/Objects", "the stray 5th table/blocks"),
            ("/Background", "a second, undriven copy of the room"),
        ):
            prim = stage.GetPrimAtPath(stray)
            if prim and prim.IsValid():
                prim.SetActive(False)
                print(f"[gridcloner] deactivated base-scene original {stray} " f"({what})", flush=True)

    envs_scope = "/World/envs"
    env0 = f"{envs_scope}/env_0"
    UsdGeom.Xform.Define(stage, envs_scope)
    UsdGeom.Xform.Define(stage, env0)

    # 2. robot under env_0 via a RAW USD reference (skips Isaac's x100 mpu
    #    reconcile — robot is meters, stage is cm). NO resetXformStack: the driver
    #    places each link with a PARENT-AWARE world-pose write (local = world *
    #    parent^-1), so the env prim's grid offset is applied exactly once by the
    #    hierarchy (GridCloner already separates envs — see set_world_pose in
    #    data-collection's kinematic_replay). Physics is disabled on the whole
    #    stage below so the articulation doesn't fight the per-link writes.
    from pathlib import Path as _Path
    from pxr import Usd as _Usd

    robot_root = manifest.robot_prefix or "genie"
    robot_env0 = f"{env0}/{robot_root}"
    robot_prim = stage.DefinePrim(robot_env0, "Xform")
    robot_prim.GetReferences().AddReference(str(_Path(manifest.robot_usda).resolve()))
    stage.Load(robot_env0)

    # 3. foreground (table + blocks) as DRIVEABLE objects. The physics engine
    #    SIMULATES them and publishes their poses in get_body_poses() under frame
    #    names "World/Objects/<name>/entity" (the objects are just not split out
    #    into get_object_poses yet — Phase 1.5). So the renderer DRIVES them like
    #    robot links, NOT static, so the render reflects the settled/moving physics
    #    state (blocks that fell onto the table, blocks being picked).
    #      * RAW USD reference (skips Isaac's x100 mpu-reconcile: foreground is
    #        meters, stage is cm) so geometry keeps the robot's scale.
    #      * reference at env_0/World so entity paths MATCH the engine frames:
    #        "World/Objects/<name>/entity" -> /World/envs/env_i/World/Objects/
    #        <name>/entity (the bridge prepends /World/envs/env_i/).
    #      * flag each /entity resetXformStack so the driven world poses place them
    #        independently (ignoring the authored object-Xform offset).
    if manifest.foreground_usda:
        world_prim = stage.DefinePrim(f"{env0}/World", "Xform")
        world_prim.GetReferences().AddReference(str(_Path(manifest.foreground_usda).resolve()))
        stage.Load(f"{env0}/World")  # load payloads so /entity exists
    # Optional BACKGROUND room (shelf/counter/door). Referenced RAW (its USD is
    # mpu=1 like the robot/foreground; a RAW ref skips add_reference's x100 cm
    # reconcile) under its OWN scope so it clones per env. Robot (init_base_pose)
    # + objects (authored) live in this room's world frame, so the real workbench
    # is under them — needs a LARGE env_spacing so full rooms don't overlap.
    if manifest.background_usda:
        bg_prim = stage.DefinePrim(f"{env0}/Background", "Xform")
        bg_prim.GetReferences().AddReference(str(_Path(manifest.background_usda).resolve()))
        stage.Load(f"{env0}/Background")
        print(f"[gridcloner] background loaded: " f"{_Path(manifest.background_usda).name}", flush=True)

    # BEFORE the first update: opt env_0's graphs out of auto-instancing, or
    # the clones below all merge into one and only a single env's graph output
    # follows its own input. The clones inherit this override.
    _n_ai = disable_graph_auto_instancing(stage, env0)
    if _n_ai:
        print(
            f"[gridcloner] auto-instancing disabled on {_n_ai} OmniGraph(s) "
            f"under {env0} (merged graphs run in only one env)",
            flush=True,
        )

    # Ticking HERE is what breaks graphs on the tiled path. The graphs run for
    # the first time against a half-assembled stage -- before cloning and
    # before the per-env background swap -- and the Write node latches onto a
    # target attribute that a later Fabric update then drops. The node stays
    # initialised and keeps computing (its compute count climbs, its inputs
    # are right) but has nowhere to write, so env_0's output silently stops
    # following its input while every clone's works.
    #
    # Deferring to the warmup that already runs after assembly fixes it. The
    # fused path still needs the ticks here (PhysX bring-up depends on them),
    # and it does not show the fault because SimulationContext.reset() starts
    # the timeline and re-resolves the bindings.
    if physics:
        for _ in range(warmup):
            simulation_app.update()

    # 3b. Disable physics on the whole env so the articulation / rigid bodies
    #     don't override the kinematic per-prim world-pose writes from /tf_render
    #     (mirrors data-collection's disable_physics_on_stage in link-replay mode).
    #     SKIPPED when physics=True: the FUSED backend runs PhysX and tiled
    #     rendering in the SAME process, so there is no external /tf_render
    #     pose stream to fight -- PhysX itself is what's supposed to move
    #     these prims, and the ArticulationRootAPI/RigidBodyAPI/CollisionAPI
    #     schemas this step would otherwise strip are exactly what PhysX
    #     needs on the robot/foreground assets. A UsdPhysics.Scene (gravity +
    #     GPU dynamics/broadphase) is created/configured here instead.
    if physics:
        print(
            f"[gridcloner] physics=True: keeping physics alive under {env0} "
            "(FUSED backend -- PhysX and tiled rendering share this process) "
            "and enabling GPU dynamics",
            flush=True,
        )
        _ensure_physics_scene(stage)
    else:
        disable_physics_subtree(stage, env0, tag="gridcloner")

    # 4. Resolve EVERY obs camera (the benchmark task's camera_list — head +
    #    wrists) from the robot USD, where they already exist as full Camera prims
    #    under their links (head_link3/head_front_Camera, gripper_l_base_link/
    #    Left_Camera, ...). They clone with the robot + follow it, so we only need
    #    to locate them (by leaf under the named link) and set the render
    #    resolution; their pose + OpenCV-pinhole intrinsics come baked in.
    cam_rels: List[str] = []  # each camera's path relative to env0
    for oc in manifest.obs_cameras():
        if oc.freecam_eye is not None:
            # Debug/viz freecam: AUTHOR a world-anchored look-at camera as a child
            # of env_0 (clones per env; env-LOCAL eye/target so each env's freecam
            # frames its own robot+table). Not on the robot, not driven by
            # /tf_render — stays put while the scene animates.
            fpath = f"{env0}/DebugFreecam"
            author_lookat_camera(stage, fpath, oc.freecam_eye, oc.freecam_target or [0.0, 0.0, 0.0])
            cam_rels.append(fpath[len(env0) + 1 :])
            continue
        link = oc.path.split("/", 1)[0]  # head_link3 / gripper_l_base_link
        leaf = oc.path.rsplit("/", 1)[-1]  # head_front_Camera / Left_Camera
        link_prim = _find_prim_by_leaf(stage, robot_env0, link)
        if not link_prim:
            raise RuntimeError(f"build_gridcloner_stage: camera link '{link}' not found under " f"{robot_env0}")
        cpath = f"{link_prim}/{leaf}"
        cprim = stage.GetPrimAtPath(cpath)
        if not cprim or not cprim.IsValid():
            raise RuntimeError(f"build_gridcloner_stage: camera '{leaf}' not found under {link_prim}")
        set_camera_intrinsics(cprim, oc)
        cam_rels.append(cpath[len(env0) + 1 :])
    print(
        f"[gridcloner] resolved {len(cam_rels)} obs camera(s) from robot USD: "
        f"{[r.rsplit('/', 1)[-1] for r in cam_rels]}",
        flush=True,
    )

    # 5. clone env_0 -> N envs on a single-row grid.
    cloner = GridCloner(spacing=float(env_spacing), num_per_row=int(num_envs), stage=stage)
    cloner.define_base_env(envs_scope)
    env_paths = cloner.generate_paths(f"{envs_scope}/env", int(num_envs))
    # replicate_physics=True (physics=True only): use PhysX's own batched-env
    # replicator instead of per-env USD physics parsing -- verified against
    # the REAL isaacsim.core.cloner source
    # (isaacsim/core/cloner/impl/cloner.py Cloner.clone/.replicate_physics +
    # grid_cloner.py GridCloner.clone, which just forwards to the base): the
    # base_env_path/root_path args replicate_physics() needs are ALREADY
    # cached on this cloner instance by the define_base_env()/
    # generate_paths() calls just above (clone() falls back to
    # self._base_env_path/self._root_path when the kwargs are None) -- but
    # Isaac's OWN test for this exact combination
    # (isaacsim/core/cloner/tests/test_cloner.py
    # ::test_grid_cloner_copy_addition) passes them explicitly alongside
    # copy_from_source=True rather than leaning on that fallback, so we match
    # that rather than the terser (also-correct) form. GridCloner.clone()
    # computes its returned per-env origins via get_clone_transforms() BEFORE
    # branching on replicate_physics, so env_origins below has the same
    # shape/meaning regardless of physics.
    clone_kwargs = dict(source_prim_path=env0, prim_paths=env_paths, replicate_physics=physics, copy_from_source=True)
    if physics:
        clone_kwargs["base_env_path"] = envs_scope
        clone_kwargs["root_path"] = f"{envs_scope}/env_"
    env_origins = cloner.clone(**clone_kwargs)

    # Again, over every env INCLUDING env_0 -- belt and braces, so the clones
    # carry the flag explicitly rather than only by inheriting env_0's. This
    # costs nothing and removes a dependency on inheritance.
    _n_post = sum(disable_graph_auto_instancing(stage, _ep) for _ep in env_paths)
    if _n_post:
        print(
            f"[gridcloner] auto-instancing re-applied to {_n_post} "
            f"OmniGraph(s) across {len(env_paths)} env(s) after cloning",
            flush=True,
        )

    # 5b. Per-env BACKGROUND overrides (per-env room generalization --
    # manifest.background_usda_per_env, see the Manifest field docstring).
    # Empty (the default) => nothing runs here, behaviour is unchanged.
    #
    # Why this is safe with ``copy_from_source=True``: verified against the
    # REAL isaacsim.core.cloner source (Cloner.clone, copy_from_source
    # branch) -- for every clone target it does
    # ``Sdf.CopySpec(layer, source_prim_path, layer, prim_path)``, a full
    # deep copy of env_0's PRIM SPEC TREE into env_i's OWN path in the SAME
    # layer. That is NOT an instance and NOT an inherit (the
    # ``copy_from_source=False`` branch uses ``inheritPathList`` instead,
    # which WOULD make a later per-env edit here fan out to every env) --
    # each env_i's ``Background`` prim after cloning is an independent spec
    # with its OWN copy of the reference list, so clearing and re-adding a
    # reference on env_i touches nothing else. Reproduced standalone with
    # plain pxr (no Kit needed for this mechanism) in
    # tests/test_background_per_env_mount.py, which asserts the exact
    # opposite would happen with ``copy_from_source=False``.
    _bg_requested = 0
    for _i, _ep in enumerate(env_paths):
        if _i >= len(manifest.background_usda_per_env):
            break
        _room = manifest.background_usda_per_env[_i]
        if not _room:
            continue  # no override for this env -- keep the cloned default
        _bg_requested += 1
        _bg_prim = stage.DefinePrim(f"{_ep}/Background", "Xform")
        _bg_prim.GetReferences().ClearReferences()
        _bg_prim.GetReferences().AddReference(str(_Path(_room).resolve()))
        # A swapped-in room brings its own graphs (3 of the 100 rooms carry a
        # faucet ActionGraph), and they have not been through the opt-out.
        _n_swap = disable_graph_auto_instancing(stage, f"{_ep}/Background")
        print(
            f"[gridcloner] per-env background override: {_ep}/Background <- "
            f"{_Path(_room).name}"
            f"{f' (+{_n_swap} graph(s) opted out of auto-instancing)' if _n_swap else ''}",
            flush=True,
        )

    # The foreground table/blocks are PAYLOADS (lazy). Load all payloads so the
    # clones show geometry, not just the (unloaded) payload arc.
    stage.Load("/")
    # On a physics-alive (fused) stage a swapped-in room would be simulated;
    # keep it scenery, as it is on the render-only path. After Load, so the
    # room's own payloads are composed and covered.
    if physics and _bg_requested:
        for _i, _ep in enumerate(env_paths):
            if _i < len(manifest.background_usda_per_env) and manifest.background_usda_per_env[_i]:
                make_render_only(stage, f"{_ep}/Background")
    for _ in range(warmup):
        simulation_app.update()

    # 5c. Independent readback of the per-env background overrides -- same
    # three-tier discipline as lights/material (see
    # docs/GENERALIZATION_DESIGN.md): count via Traverse-equivalent
    # (GetChildren on the composed prim, NOT "AddReference didn't raise")
    # how many of the REQUESTED overrides actually resolved real geometry.
    if _bg_requested:
        _bg_landed = 0
        _bg_failed: List[str] = []
        for _i, _ep in enumerate(env_paths):
            if _i >= len(manifest.background_usda_per_env):
                continue
            _room = manifest.background_usda_per_env[_i]
            if not _room:
                continue
            _bg_prim = stage.GetPrimAtPath(f"{_ep}/Background")
            if _bg_prim and _bg_prim.IsValid() and list(_bg_prim.GetChildren()):
                _bg_landed += 1
            else:
                _bg_failed.append(f"{_ep}/Background <- {_room!r}")
        if _bg_landed == 0:
            raise RuntimeError(
                f"build_gridcloner_stage: requested {_bg_requested} per-env "
                f"background override(s) but NONE resolved any geometry "
                f"after Load() -- {'; '.join(_bg_failed)}"
            )
        if _bg_landed < _bg_requested:
            print(
                f"[gridcloner] WARN: per-env background -- only "
                f"{_bg_landed}/{_bg_requested} requested override(s) "
                f"landed geometry; failed: {'; '.join(_bg_failed)}",
                flush=True,
            )
        else:
            print(f"[gridcloner] per-env background: {_bg_landed}/" f"{_bg_requested} override(s) landed", flush=True)
    # _bg_requested == 0 -> no per-env override was asked for; stay quiet
    # (this is the default -- ``manifest.background_usda_per_env`` empty).

    # Diagnostic: actual WORLD positions of each env's table + robot base, so we
    # can see whether every env's content really sits at its own origin.
    xc = UsdGeom.XformCache()

    def _wpos(path):
        pr = stage.GetPrimAtPath(path) if path else None
        if not pr or not pr.IsValid():
            return None
        t = xc.GetLocalToWorldTransform(pr).ExtractTranslation()
        return tuple(round(float(v), 2) for v in t)

    # Which prims to sample, discovered from env_0 rather than hardcoded. A
    # hardcoded asset name only matches one task's assets, which would make
    # the whole diagnostic silently useless on every other task: the paths
    # would resolve to nothing, so each env would print
    # ``table_w=None block_w=None`` and the count below would read ``ACTIVE
    # tables=0 (expect N)`` -- indistinguishable from "this task has no
    # table" on a task whose table is present and rendering.
    #
    # A support surface is whatever prim is named ``table*``; the sampled
    # object is any other child of ``World/Objects``. Names, not types, because
    # these are USD references whose schema varies by asset.
    #
    # ``s<N>_`` is stripped before matching: a SUPERSET stage names every object
    # after the slot it occupies, so the table arrives as ``s0_table_000`` and a
    # bare ``startswith("table")`` finds nothing on exactly the multi-instance
    # stages where this diagnostic is most worth reading.
    #
    # Returned as (relative path, leaf name) -- the path is what ``_wpos``
    # needs, the leaf is what the ACTIVE count matches on.
    import re as _re  # noqa: PLC0415 -- local, same style as the pxr imports

    def _is_table(name):
        return _re.sub(r"^s\d+_", "", name).lower().startswith("table")

    def _pick_sample_prims():
        root = env_paths[0]
        container = stage.GetPrimAtPath(f"{root}/World/Objects")
        obj_names = [c.GetName() for c in container.GetChildren()] if container and container.IsValid() else []
        table_rel = table_leaf = None
        for n in obj_names:
            if _is_table(n):
                table_rel, table_leaf = f"World/Objects/{n}/entity", n
                break
        if table_leaf is None:
            # Not under World/Objects on every task -- widen before giving up.
            root_prim = stage.GetPrimAtPath(root)
            if root_prim and root_prim.IsValid():
                for p in _Usd.PrimRange(root_prim):
                    if _is_table(p.GetName()):
                        table_leaf = p.GetName()
                        table_rel = str(p.GetPath())[len(root) + 1 :]
                        break
        other = next((n for n in obj_names if n != table_leaf), None)
        return (table_rel, table_leaf, f"World/Objects/{other}/entity" if other else None, other)

    table_rel, table_leaf, object_rel, object_name = _pick_sample_prims()

    for ei, ep in enumerate(env_paths):
        base = _find_prim_by_leaf(stage, f"{ep}/{robot_root}", "base_link")
        rx = stage.GetPrimAtPath(base).GetAttribute("omni:resetXformStack").Get() if base else None
        print(
            f"[gridcloner] {ep}: origin={tuple(round(float(v),2) for v in env_origins[ei])} "
            f"{table_leaf or 'table'}_w="
            f"{_wpos(f'{ep}/{table_rel}') if table_rel else None} "
            f"{object_name or 'object'}_w="
            f"{_wpos(f'{ep}/{object_rel}') if object_rel else None} "
            f"base_w={_wpos(base)} base_resetXform={rx}",
            flush=True,
        )

    # Freecam diagnostic: env_0's freecam world position + look direction (-Z),
    # so we can see whether it points at the robot+table.
    fc = stage.GetPrimAtPath(f"{env_paths[0]}/DebugFreecam")
    if fc and fc.IsValid():
        w = xc.GetLocalToWorldTransform(fc)
        pos = w.ExtractTranslation()
        zrow = w.GetRow(2)
        print(
            f"[gridcloner] freecam_w pos="
            f"{tuple(round(float(v),2) for v in pos)} "
            f"fwd(-Z)={tuple(round(-float(zrow[i]),2) for i in range(3))}",
            flush=True,
        )

    # 6. per-env, per-camera prim paths for TiledCameraSensor, ordered
    #    [env_i][cam_j] -> flat index i*num_cams + j (the bridge reshapes the
    #    tiled buffer to (num_envs, num_cams, H, W, 3) in this order).
    cam_paths = [f"{p}/{rel}" for p in env_paths for rel in cam_rels]
    origins = [tuple(float(v) for v in o) for o in list(env_origins)]
    # Traverse() visits only ACTIVE prims, so this counts real (rendered) tables —
    # must equal num_envs (the deactivated base-scene original must NOT appear).
    # ``table_leaf`` is discovered from the stage (see _pick_sample_prims)
    # rather than hardcoded, so this check is meaningful on every task.
    if table_leaf:
        n_tables = sum(1 for p in stage.Traverse() if p.GetName() == table_leaf)
        tables_note = f"ACTIVE {table_leaf}={n_tables} (expect {num_envs})"
    else:
        tables_note = "ACTIVE tables=n/a (no table* prim in env_0)"
    print(
        f"[gridcloner] {num_envs} envs x {len(cam_rels)} cam(s) @ "
        f"spacing={env_spacing}; origins[:2]={origins[:2]}; "
        f"cam0={cam_paths[0]}; {tables_note}",
        flush=True,
    )
    return stage, cam_paths, origins


def disable_graph_auto_instancing(stage, root: str) -> int:
    """Opt every OmniGraph under ``root`` out of auto-instancing.

    OmniGraph merges structurally identical graphs as an optimisation. Every
    env is a clone of env_0, so the graphs they carry are structurally
    identical by construction and get merged -- and once merged only ONE env's
    output follows its own input. Measured with an observable graph: 1 of 4
    envs tracked its input, the other three sat at the value authored in
    the asset; clearing the flag gives 4 of 4, through the real builder in
    both physics modes and across reset.

    This is an SDK behaviour, not something the scene does wrong, and which
    piece of internal state ends up mis-bound is not established -- only that
    the merge is the path it happens on.

    MUST run before the env's first ``simulation_app.update()``: the merge
    decision is taken when the graphs are first seen. Call it again for any
    graph introduced later (the per-env background swap brings in whatever
    room it mounts).

    Returns how many graphs were opted out, so a caller can log it.
    """
    from pxr import Sdf, Usd  # noqa: PLC0415

    prim = stage.GetPrimAtPath(root)
    if not prim or not prim.IsValid():
        return 0

    # Collect FIRST, mutate afterwards. De-instancing while the PrimRange is
    # live invalidates it ("Iterator points to expired ... instance proxy
    # prim"), so the traversal only records paths.
    paths = [
        p.GetPath()
        for p in Usd.PrimRange(prim, Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate))
        if p.GetTypeName() in ("OmniGraph", "ComputeGraph")
    ]

    n = 0
    for path in paths:
        p = stage.GetPrimAtPath(path)
        # Nested instances need more than one pass: de-instancing the
        # innermost enclosing instance can leave the prim inside another one.
        for _ in range(16):
            if not p or not p.IsValid() or not p.IsInstanceProxy():
                break
            # USD forbids authoring on an instance proxy ("authoring to an
            # instance proxy is not allowed") -- it is a view onto a shared
            # prototype. A graph in that position is shared by every instance
            # anyway, so per-env execution is impossible until it is real.
            inst = p
            while inst and inst.IsInstanceProxy():
                inst = inst.GetParent()
            if not inst or not inst.IsValid():
                break
            inst.SetInstanceable(False)
            print(
                f"[gridcloner] de-instanced {inst.GetPath()} so the " f"OmniGraph inside it can be made per-env",
                flush=True,
            )
            p = stage.GetPrimAtPath(path)
        if not p or not p.IsValid() or p.IsInstanceProxy():
            print(
                f"[gridcloner] WARNING: OmniGraph at {path} stays inside an " f"instance; it will run in only one env",
                flush=True,
            )
            continue
        p.CreateAttribute("allowAutoInstancing", Sdf.ValueTypeNames.Bool).Set(False)
        n += 1
    return n


def resolve_camera_prims(stage, cameras: List[CameraConfig]) -> List[str]:
    """Resolve Camera prim paths for a TiledCamera, from each camera's
    RenderProduct ``camera`` relationship (robust — same binding the
    OVRTX/render_layer authoring uses). Skips unresolved entries."""
    from pxr import UsdRender

    paths: List[str] = []
    for cam in cameras:
        rp = stage.GetPrimAtPath(cam.render_product_path)
        cam_path: Optional[str] = None
        if rp and rp.IsValid():
            rel = UsdRender.Product(rp).GetCameraRel()
            targets = rel.GetTargets() if rel else []
            if targets:
                cam_path = str(targets[0])
        if cam_path and stage.GetPrimAtPath(cam_path).IsValid():
            paths.append(cam_path)
    return paths


def resolve_obs_camera_prims(stage, manifest: Manifest) -> List[str]:
    """Per-env head (obs) Camera prim paths — the default for training obs."""
    return resolve_camera_prims(stage, manifest.obs_cameras())


class TfRenderDriver:
    """Applies /tf_render transforms onto stage prims (mirrors isaacsim_render
    ._on_tf_render). Absolute ``/World/...`` frames (our multi-env scene_multi
    frames) are used verbatim; bare frames get the robot prefix."""

    def __init__(self, stage, robot_prefix: str = "", env_origins=None) -> None:
        self._stage = stage
        self._prefix = robot_prefix
        self._cache: dict = {}
        self._last_tick = 0
        # See _bump_tick(): once-per-generation warning latch for a tick that
        # falls far below the watermark (the signature of a poisoned tick —
        # see set_flat_paths()).
        self._tick_warned = False
        # One-shot warning latch for apply_flat(), used when a flat frame
        # arrives before the path table. Reset once per generation (see
        # set_flat_paths), otherwise it fires for generation one and never
        # again.
        self._flat_unusable_warned = False
        # The other one-shot warning latch for apply_flat(), used when the
        # table IS installed but the sizes disagree (n != len(entries), or the
        # payload is too short). Same shape as the latch above — silently return
        # 0, tick does not advance — except that has_flat_paths() is already
        # True here, so the /tf_render fallback has long since been
        # unsubscribed, which makes this branch the more dangerous one. Reset
        # once per generation, for the same reason.
        self._flat_size_warned = False
        # GridCloner multi-env: each /World/envs/env_<i>/... body's LOCAL physics
        # pose is composed into env_i's frame by MATRIX multiply against env_i's
        # world transform (T_world = T_local * T_env) — not an additive
        # translation — so orientation and any non-translation env transform are
        # handled correctly, and the offset lives in ONE place (the env prim's
        # world matrix). The bridge publishes RAW local poses (no baked offset).
        self._env_origins = list(env_origins) if env_origins else None
        self._env_xf: dict = {}  # env index -> Gf.Matrix4d (cached)
        # path -> (t_attr, o_attr, envM*parent_inv|None, orient_is_double) | None
        # (fast-path cache for apply(); misses cached too — see _build_drive_entry)
        self._drive_cache: dict = {}
        # Fabric fast path, on by default; ``RLINF_TILED_FABRIC=0`` falls back
        # to the USD path. The FSD renderer reads from Fabric anyway, so writing
        # world poses straight into the Fabric scene saves the USD->Fabric sync
        # Hydra does every frame for our ~500 prims.
        #
        # Why it is the default: the USD path once rendered the entire image as
        # uniform grey with nothing reporting an error anywhere — the server was
        # serving, the renderer was producing frames, and the client received
        # correctly shaped but empty images. A silent wrong answer like that
        # does not belong on the default path.
        #
        # It still falls back to the USD path automatically when usdrt cannot be
        # attached; see the warning below.
        self._fabric = os.environ.get("RLINF_TILED_FABRIC", "1") not in ("0", "false", "False")
        self._rt_stage = None
        self._rt_cache: dict = {}  # path -> (wPosAttr, wOriAttr, envM) | None
        self._rt_flat_entries = None
        if self._fabric:
            try:
                import omni.usd  # noqa: PLC0415
                from usdrt import Usd as _RtUsd  # noqa: PLC0415

                self._rt_stage = _RtUsd.Stage.Attach(omni.usd.get_context().get_stage_id())
                print("[tf-driver] Fabric write path ON " "(usdrt world-pose; skips USD->Fabric sync)", flush=True)
            except Exception as _e:  # noqa: BLE001
                print(
                    f"[tf-driver] WARNING: Fabric requested but usdrt attach "
                    f"failed ({_e!r}); falling back to USD xformOps. On "
                    f"2026-07-28 this path was observed to render an EMPTY "
                    f"scene (uniform gray, no error anywhere) — if your "
                    f"frames look blank, that is the first thing to check.",
                    flush=True,
                )
                self._fabric = False
        else:
            print(
                "[tf-driver] WARNING: Fabric write path DISABLED via "
                "RLINF_TILED_FABRIC — using USD xformOps. This path was "
                "observed to render an EMPTY scene (uniform gray, no error "
                "anywhere) on 2026-07-28; verify your frames are not blank.",
                flush=True,
            )

    def _env_world_matrix(self, ei: int):
        """World transform of /World/envs/env_<ei> (authored by GridCloner),
        cached. Falls back to a translate matrix from env_origins if the prim
        can't be read."""
        m = self._env_xf.get(ei)
        if m is None:
            from pxr import Gf, UsdGeom

            prim = self._stage.GetPrimAtPath(f"/World/envs/env_{ei}")
            if prim and prim.IsValid():
                m = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
            elif self._env_origins and ei < len(self._env_origins):
                m = Gf.Matrix4d(1.0)
                m.SetTranslate(Gf.Vec3d(*self._env_origins[ei]))
            else:
                m = Gf.Matrix4d(1.0)
            self._env_xf[ei] = m
        return m

    def _get_prim(self, prim_path: str):
        prim = self._cache.get(prim_path)
        if prim is None:
            prim = self._stage.GetPrimAtPath(prim_path)
            if prim and prim.IsValid():
                self._cache[prim_path] = prim
            else:
                return None
        return prim

    def _bump_tick(self, tick: int) -> None:
        """Advance the monotonic tick watermark (``max()``), warning ONCE per
        generation if ``tick`` falls far below it — that shape (a huge
        watermark, then small ticks that can never clear it) is exactly what
        a poisoned watermark from a prior (killed) generation's engine looks
        like from the inside (see the reload-generation-boundary reset in
        ``set_flat_paths`` below). The 1000 threshold is arbitrary-but-generous:
        real per-step tick deltas are 1, so this only fires on a genuine
        generation-scale jump backward, never on ordinary jitter/reordering."""
        if tick < self._last_tick - 1000 and not self._tick_warned:
            print(
                f"[tf-driver] WARNING: tick {tick} is far below watermark "
                f"{self._last_tick} — looks like a stale/poisoned tick "
                f"watermark surviving from a prior generation",
                flush=True,
            )
            self._tick_warned = True
        self._last_tick = max(self._last_tick, tick)

    def apply(self, transforms) -> int:
        """Apply a list of geometry_msgs TransformStamped-like objects. Returns
        the number of prims actually driven."""
        from pxr import Gf

        # Record the largest global tick carried by this batch of transforms
        # BEFORE the pose-application loop — that loop continues past missing
        # prims, so doing it afterwards would drop ticks.
        #
        # Both the engine and the bridge publish poses, and their naming
        # dialects differ: ``env_<w>/genie/...`` at N>1, absolute prim paths at
        # N==1. Deciding on a leading "/" alone misreads the engine's N==1
        # dialect entirely.
        # poisons the tick with a huge garbage value (last_tick is monotonic via
        # max(), so ONE poisoned decode permanently freezes on-demand rendering
        # into heartbeat-only mode and stalls the frame-sync barrier). Decode
        # ONLY from the bridge's two dialects.
        from rlinf_geniesim.sim_clock import decode_stamp_to_tick  # lazy import

        for _tf in transforms:
            if not _tf.child_frame_id.startswith(("/World/envs/", "/RenderOVRTX/")):
                continue
            _st = _tf.header.stamp
            self._bump_tick(decode_stamp_to_tick(_st.sec, _st.nanosec))
        # Fast path (performance-critical: ~500 prims per control step; the
        # naive version cost 270-300 ms per frame, 64% of the whole RL step).
        # Two mechanisms:
        #   1. a one-time per-prim cache holding the resolved prim, its env world
        #      matrix, the static parent world inverse, the translate/orient
        #      attribute handles and the orient precision — in steady state
        #      there are no more regexes, no XformCache lookups and no
        #      GetAttribute calls;
        #   2. every attribute write is wrapped in an ``Sdf.ChangeBlock``,
        #      collapsing ~1000 USD change notifications into one.
        from pxr import Sdf

        cache = self._drive_cache
        pending = []
        for tf in transforms:
            body = tf.child_frame_id
            if body.startswith("/"):
                prim_path = body
            elif self._prefix:
                prim_path = f"/{self._prefix}/{body}"
            else:
                prim_path = f"/{body}"
            try:
                e = cache[prim_path]
            except KeyError:
                e = self._build_drive_entry(prim_path)
                cache[prim_path] = e
            if e is None:
                continue
            t, r = tf.transform.translation, tf.transform.rotation
            pos = Gf.Vec3d(float(t.x), float(t.y), float(t.z))
            quat = Gf.Quatd(float(r.w), float(r.x), float(r.y), float(r.z))
            env_parent_inv = e[2]
            if env_parent_inv is not None:
                # local = (physics pose composed into env_i world) * parent^-1
                # (PARENT-AWARE write == set_world_pose; env offset applied once).
                m = Gf.Matrix4d()
                m.SetTransform(Gf.Rotation(quat), pos)
                local = m * env_parent_inv
                pos = local.ExtractTranslation()
                # parent^-1 carries the block-root's uniform scale (0.5), so
                # ``local`` is NOT a pure rotation. Gf.ExtractRotationQuat on a
                # scaled matrix returns a DISTORTED rotation (up to ~20deg off);
                # strip scale/shear first so only the true rotation is written.
                quat = local.RemoveScaleShear().ExtractRotationQuat()
            pending.append((e, pos, quat))
        with Sdf.ChangeBlock():
            for e, pos, quat in pending:
                e[0].Set(pos)
                o_attr = e[1]
                if o_attr is not None:
                    o_attr.Set(quat if e[3] else Gf.Quatf(quat))
        return len(pending)

    def _build_drive_entry(self, prim_path: str):
        """One-time resolution for a driven prim: returns
        ``(t_attr, o_attr, envM_x_parent_inv | None, orient_is_double)`` or
        ``None`` for paths that don't resolve (e.g. the engine's bare frames on
        a GridCloner stage — cached so each miss costs one dict hit)."""
        import re
        from pxr import Gf, Sdf, UsdGeom

        prim = self._get_prim(prim_path)
        if prim is None:
            return None
        # Ensure the op stack is [translate, orient] (payload /entity prims carry
        # a matrix op instead — see apply_xform). One-time identity write; the
        # real pose lands in the same apply() call.
        t_attr = prim.GetAttribute("xformOp:translate")
        order_attr = prim.GetAttribute("xformOpOrder")
        order = list(order_attr.Get()) if order_attr and order_attr.Get() else []
        if not t_attr or not t_attr.IsValid() or "xformOp:translate" not in order:
            apply_xform(prim, Gf.Vec3d(0, 0, 0), Gf.Quatd(1, 0, 0, 0))
        t_attr = prim.GetAttribute("xformOp:translate")
        o_attr = prim.GetAttribute("xformOp:orient")
        if not (t_attr and t_attr.IsValid()):
            return None
        if not (o_attr and o_attr.IsValid()):
            o_attr = None
        o_double = bool(o_attr is not None and o_attr.GetTypeName() == Sdf.ValueTypeNames.Quatd)
        env_parent_inv = None
        m = re.search(r"/World/envs/env_(\d+)/", prim_path) if self._env_origins is not None else None
        if m:
            # Precompose envM * parent_world^-1 (both STATIC): steady state is a
            # single matrix multiply per frame.
            parent = prim.GetParent()
            xc = UsdGeom.XformCache()
            pw = xc.GetLocalToWorldTransform(parent) if parent and parent.IsValid() else Gf.Matrix4d(1.0)
            env_parent_inv = self._env_world_matrix(int(m.group(1))) * pw.GetInverse()
        return (t_attr, o_attr, env_parent_inv, o_double)

    def _rt_entry(self, prim_path: str):
        """Fabric counterpart of ``_build_drive_entry``: resolve the prim's
        Fabric WORLD-pose attrs + its env world matrix once, cache misses too.
        Returns ``(worldPosAttr, worldOriAttr, envM)`` or ``None``.

        We write the WORLD pose (``m * envM``) rather than a parent-relative
        local: setting ``_worldPosition``/``_worldOrientation`` auto-authors
        ``omni:fabric:resetXformStack`` so the FSD ignores the USD parent
        hierarchy and honours the world pose directly (one matrix multiply, no
        parent-inverse — and no USD write at all).

        NOTE resetXformStack ignores the ENTIRE ancestor chain, not just pose
        offsets. The benchmark's object slot roots carry
        ``xformOp:scale = (2, 2, 2)`` (see llm_task/*/scene.usda), and that
        scale used to arrive purely through the hierarchy — so once this Fabric
        path writes world pose attributes, it is dropped along with everything
        else and objects render at half their linear size (measured 2026-07-30:
        coloured-pixel area ratio 4.06, linear ratio 2.02, exactly consistent
        with losing scale=2). The consequence is not merely "small": the mesh is
        drawn at half size while its centre stays at the physics centre, so the
        bottom face floats half a block above the table, the shadow detaches
        from the object, and it reads as "sitting on the table askew, violating
        physics".

        So this computes the prim's WORLD scale from the USD hierarchy and
        writes it into Fabric explicitly. Scale does not change per frame, so
        writing it once while building the cache entry is enough
        (``_worldScale`` persists once authored). Identity scales are skipped,
        sparing 56 pointless attributes on the robot's links."""
        try:
            return self._rt_cache[prim_path]
        except KeyError:
            pass
        import re
        from pxr import Gf
        from usdrt import Rt  # noqa: PLC0415

        e = None
        rprim = self._rt_stage.GetPrimAtPath(prim_path)
        if rprim is not None and rprim.IsValid():
            xf = Rt.Xformable(rprim)
            mm = re.search(r"/World/envs/env_(\d+)/", prim_path) if self._env_origins is not None else None
            envM = self._env_world_matrix(int(mm.group(1))) if mm else Gf.Matrix4d(1.0)
            self._author_world_scale(xf, prim_path)
            e = (xf.CreateWorldPositionAttr(), xf.CreateWorldOrientationAttr(), envM)
        self._rt_cache[prim_path] = e
        return e

    def _author_world_scale(self, xf, prim_path: str) -> None:
        """Write ``prim_path``'s world scale, as computed from the USD
        hierarchy, into Fabric explicitly.

        See ``_rt_entry`` for why this is needed: writing a world pose brings
        resetXformStack with it, and scales on the ancestor chain stop applying.
        Identity scales (robot links and the like) are skipped.
        An unreadable prim or a failed write warns once and moves on — wrong
        size beats taking down the whole render chain.
        """
        try:
            from pxr import Gf, UsdGeom  # noqa: PLC0415
            from usdrt import Gf as RtGf  # noqa: PLC0415

            uprim = self._stage.GetPrimAtPath(prim_path)
            if not (uprim and uprim.IsValid()):
                return
            m = UsdGeom.XformCache().GetLocalToWorldTransform(uprim)
            s = [float(Gf.Vec3d(m.GetRow3(i)).GetLength()) for i in range(3)]
            if all(abs(v - 1.0) < 1e-6 for v in s):
                return  # identity, nothing to write
            xf.CreateWorldScaleAttr().Set(RtGf.Vec3f(s[0], s[1], s[2]))
            if not getattr(self, "_scale_authored_log", None):
                self._scale_authored_log = True
                print(
                    f"[tf-driver] Fabric world-scale authored (resetXformStack "
                    f"drops hierarchy scale): {prim_path} -> "
                    f"({s[0]:.3f}, {s[1]:.3f}, {s[2]:.3f})",
                    flush=True,
                )
        except Exception as exc:  # noqa: BLE001
            if not getattr(self, "_scale_warned", False):
                self._scale_warned = True
                print(
                    f"[tf-driver] WARNING: could not author Fabric world "
                    f"scale for {prim_path} ({exc!r}); objects whose slot "
                    f"root carries a scale will render at the WRONG SIZE "
                    f"(and appear to float above their support).",
                    flush=True,
                )

    # ------------------------------------------------------------------
    # FLAT pose channel (perf): the bridge sends one Float64MultiArray per
    # frame ([tick, n, n*7]) + a latched JSON path table, instead of ~500
    # nested TransformStamped (whose rclpy deserialization alone cost
    # ~140ms/frame at 8 envs).
    # ------------------------------------------------------------------

    def set_flat_paths(self, paths) -> None:
        entries = []
        rt_entries = [] if self._fabric else None
        for p in paths:
            try:
                e = self._drive_cache[p]
            except KeyError:
                e = self._build_drive_entry(p)
                self._drive_cache[p] = e
            entries.append(e)
            if self._fabric:
                rt_entries.append(self._rt_entry(p))
        self._flat_entries = entries
        self._rt_flat_entries = rt_entries
        # Reload / generation boundary: installing this latched path table
        # happens exactly once per hot-swap generation, which makes it the
        # authoritative start of the new generation — so the tick watermark is
        # reset here, not only in __init__.
        #
        # Why the reset is required: a hot swap builds a fresh TfRenderDriver
        # (_last_tick back to zero), but the render node's flat-pose
        # subscription is never destroyed and rebuilt, only repointed at the new
        # driver. During the swap window, in-flight messages from the previous
        # generation's engine can land on this new driver and push _last_tick to
        # a large value before the path table is installed — after which the new
        # generation's own small ticks can never exceed that poisoned watermark
        # (max() enforces monotonicity), and on-demand rendering degrades to the
        # heartbeat alone, one step every 2 s for the whole generation.
        #
        # Note: what guarantees this runs every generation is the bridge-side
        # RosRenderBridge.invalidate_flat_path_table(), called by reload_stage()
        # after every successful hot swap — "resend only when the name list
        # changed" would not do, because superset slot naming is identical
        # across instances of the same sub-task.
        self._last_tick = 0
        self._tick_warned = False
        self._flat_unusable_warned = False
        self._flat_size_warned = False
        _drv = sum(1 for e in rt_entries if e is not None) if self._fabric else sum(1 for e in entries if e is not None)
        print(
            f"[tf-driver] flat pose channel: {len(paths)} path(s), "
            f"{_drv} driveable ({'Fabric' if self._fabric else 'USD'})",
            flush=True,
        )
        if _drv == 0 and paths:
            # The table is installed, but not one path resolved to a live
            # prim: apply_flat still advances the tick, the on-demand rendering
            # threshold is still met and acceptance's "flat pose channel:" grep
            # still looks healthy — yet every frame written out is the same
            # still image. This branch affects neither tick nor gating; it is a
            # third distinct shape of "scores fine, picture dead", so it warns
            # separately instead of reusing either tick latch above.
            print(
                "[tf-driver] WARNING: flat pose channel installed with "
                f"{len(paths)} path(s) but 0 resolved to a live prim — "
                "every frame will render the SAME static picture even "
                "though ticks keep advancing and scoring looks normal. "
                "Check the path table against the actual stage.",
                flush=True,
            )

    def has_flat_paths(self) -> bool:
        """Whether this generation's flat path table is installed
        (``set_flat_paths`` has run).

        The render node uses this to decide when it may unsubscribe the
        ``/tf_render`` fallback. The test is "the table is installed", NOT "some
        prim is being driven" — with the table installed but nothing resolved,
        ``apply_flat`` still advances the tick and on-demand rendering keeps
        working. Conversely, a flat frame arriving before the table is UNUSABLE
        (``apply_flat`` returns 0 before advancing the tick), and unsubscribing
        the fallback at that point severs this generation's only pose path.
        """
        return bool(getattr(self, "_flat_entries", None))

    def apply_flat(self, data) -> int:
        """Apply one flat pose frame. ``data`` = array('d') [tick, n, n*7]."""
        import numpy as _np
        from pxr import Gf, Sdf

        entries = getattr(self, "_flat_entries", None)
        if not entries:
            if not getattr(self, "_flat_unusable_warned", False):
                print(
                    "[tf-driver] WARNING: flat pose frame arrived with NO "
                    "path table installed for this generation — no poses "
                    "applied and the tick is NOT advancing, so on-demand "
                    "rendering falls back to the heartbeat and the picture "
                    "goes static. Expect /rlinf/pose_paths to be "
                    "(re)published after every reload.",
                    flush=True,
                )
                self._flat_unusable_warned = True
            return 0
        if len(data) < 2:
            return 0
        A = _np.frombuffer(data, dtype=_np.float64)
        tick, n = int(A[0]), int(A[1])
        if n != len(entries) or A.size < 2 + 7 * n:
            # Same "silently return 0, tick does not advance" shape, but here
            # has_flat_paths() is already True (the table is installed, only the
            # sizes disagree) — the /tf_render fallback was unsubscribed long
            # ago by _flat_cb's has_flat_paths() test above, so this branch is
            # more dangerous than the "table not installed" one: there is no
            # fallback left.
            if not getattr(self, "_flat_size_warned", False):
                print(
                    "[tf-driver] WARNING: flat pose frame size mismatch — "
                    f"expected n={len(entries)} (this generation's path "
                    f"table) but frame carries n={n}, payload size="
                    f"{A.size} (need >= {2 + 7 * n}) — no poses applied and "
                    "the tick is NOT advancing. /tf_render is already "
                    "unsubscribed, so this generation has no fallback "
                    "until /rlinf/pose_paths is republished.",
                    flush=True,
                )
                self._flat_size_warned = True
            return 0
        self._bump_tick(tick)
        P = A[2 : 2 + 7 * n].reshape(n, 7)
        if self._fabric:
            return self._apply_flat_fabric(P)
        pending = []
        for e, row in zip(entries, P):
            if e is None:
                continue
            pos = Gf.Vec3d(float(row[0]), float(row[1]), float(row[2]))
            quat = Gf.Quatd(float(row[3]), float(row[4]), float(row[5]), float(row[6]))
            env_parent_inv = e[2]
            if env_parent_inv is not None:
                m = Gf.Matrix4d()
                m.SetTransform(Gf.Rotation(quat), pos)
                local = m * env_parent_inv
                pos = local.ExtractTranslation()
                # See apply(): parent^-1 injects the block-root uniform scale,
                # so ExtractRotationQuat on the raw ``local`` distorts the
                # rotation. Strip scale/shear before extracting.
                quat = local.RemoveScaleShear().ExtractRotationQuat()
            pending.append((e, pos, quat))
        with Sdf.ChangeBlock():
            for e, pos, quat in pending:
                e[0].Set(pos)
                o_attr = e[1]
                if o_attr is not None:
                    o_attr.Set(quat if e[3] else Gf.Quatf(quat))
        return len(pending)

    def _apply_flat_fabric(self, P) -> int:
        """Fabric variant of the flat-pose apply: write each body's WORLD pose
        (``m * envM``) straight into the Fabric scene — no USD xformOp writes,
        no ``Sdf.ChangeBlock``, no USD->Fabric sync. ``_rt_flat_entries`` is
        aligned index-for-index with the flat rows (built in set_flat_paths)."""
        from pxr import Gf
        from usdrt import Gf as RtGf  # noqa: PLC0415

        rt_entries = self._rt_flat_entries or []
        driven = 0
        for e, row in zip(rt_entries, P):
            if e is None:
                continue
            pos = Gf.Vec3d(float(row[0]), float(row[1]), float(row[2]))
            quat = Gf.Quatd(float(row[3]), float(row[4]), float(row[5]), float(row[6]))
            m = Gf.Matrix4d()
            m.SetTransform(Gf.Rotation(quat), pos)
            world = m * e[2]  # * envM -> WORLD pose
            wp = world.ExtractTranslation()
            # Same reasoning as the USD path (see a8eae62ef5):
            # ExtractRotationQuat returns a distorted rotation for a matrix that
            # carries scale. Today envM is pure translation (identity for a
            # single env, the GridCloner grid offset for multiple), so ``world``
            # is still rigid and this step is currently a no-op — but envM comes
            # from GetLocalToWorldTransform, and the day a scale appears on an
            # env root it would tilt silently. Strip it up front.
            wq = world.RemoveScaleShear().ExtractRotationQuat()
            wi = wq.GetImaginary()
            e[0].Set(RtGf.Vec3d(float(wp[0]), float(wp[1]), float(wp[2])))
            e[1].Set(RtGf.Quatf(float(wq.GetReal()), float(wi[0]), float(wi[1]), float(wi[2])))
            driven += 1
        return driven

    def last_tick(self) -> int:
        return int(self._last_tick)


# ---------------------------------------------------------------------------
# Live deformable-particle channel (used by isaacsim_render.py; available to
# any other Kit node that imports this module)
# ---------------------------------------------------------------------------


def preload_warp_before_kit() -> None:
    """Import every submodule of the real ``warp`` package before Kit starts.

    Must be called before ``import isaacsim`` / ``SimulationApp(...)`` in the
    entry script. Isaac Sim bundles its own ``omni.warp.core`` extension,
    whose ``extension.toml`` declares ``[[python.module]] name = "warp"
    path = "."``. When Kit enables that extension, its python-module loader
    finds ``warp`` already imported (this pip package, needed by the
    Newton/libuipc solver) and appends the extension's own warp/ directory
    onto the existing package's ``__path__`` instead of replacing it — so
    the top-level ``warp`` module stays whichever version was imported
    first, but any submodule not yet imported (e.g. ``warp.fem``) resolves
    against the extension's bundled, differently-versioned copy instead.
    Newton's own eager import chain (``isaacsim.core.experimental.actuators``
    -> ``newton.actuators`` -> ``newton.solvers.implicit_mpm`` ->
    ``import warp.fem``) hits exactly this, and crashes with e.g.
    ``ImportError: cannot import name 'warn' from 'warp._src.utils'`` because
    the two warp versions' internals disagree — which cascades into a native
    crash in omni.graph.core's extension-registration path.

    Importing every submodule up front (before Kit exists to append that
    second path) populates ``sys.modules`` with the correct version for all
    of them, so the later merge is a no-op for anything already cached.
    """
    try:
        import pkgutil
        import warp
    except ImportError:
        return
    for m in pkgutil.walk_packages(warp.__path__, prefix="warp."):
        try:
            __import__(m.name)
        except Exception:
            pass


# Schema names to strip from a cloth/tet mesh prim before writing raw solved
# points to it: they were applied by the scene's own physics setup, and Isaac
# Sim's PhysX would otherwise re-simulate / re-derive the mesh from them and
# fight the points we write. This consumer owns schema removal for
# the supported separate-process Isaac renderer.
_DEFORMABLE_SCHEMAS = (
    "OmniPhysicsDeformableBodyAPI",
    "OmniPhysicsSurfaceDeformableSimAPI",
    "PhysxSurfaceDeformableBodyAPI",
    "PhysxDeformableBodyAPI",
    "PhysxDeformableSurfaceAPI",
    "PhysxAutoAttachmentAPI",
    "PhysicsCollisionAPI",
    "PhysxCollisionAPI",
)


class DeformableChannel:
    """Subscribes to the physics side's deformable-particle channel and writes
    the solved points into the stage's mesh prims.

    Wire format (produced by ``genie_sim_engine_newton``'s
    ``deformable_publisher.py``, consumed identically by the C++ OVRTX render
    node — see ``render_node.cpp``'s ``on_deformable_topology`` /
    ``on_deformable_flat``):

      * ``<topic>_topology`` — latched JSON, sent once: ``{"generation": 1,
        "objects": [{"mesh_id", "name", "prim_path", "n_particles"}, ...]}``.
      * ``<topic>_flat`` — ``Float32MultiArray`` every tick: ``[generation,
        tick, mesh_id_0, n_0, x0,y0,z0, ..., mesh_id_1, n_1, ...]``.

    Points arrive in WORLD space. Isaac Sim's USD path needs mesh-LOCAL, so
    each frame converts via the prim's ``local-to-world`` matrix (inverted
    once per unique prim, cached) — the same technique the verified
    standalone consumer uses. This differs from the OVRTX backend, which
    writes the world points directly and instead authors
    ``omni:resetXformStack=1`` on the prim; USD has no such flag, so the
    conversion happens here.

    The optional Fabric fast path (``usdrt``, as in
    :class:`TfRenderDriver`) can write world-space points without the
    inverse, but it is NOT used here: ``usdrt`` exposes world-position
    attributes for Xforms, not for arbitrary array attributes like a Mesh's
    ``points``, so there is no Fabric equivalent of this write.
    """

    def __init__(self, node, stage, topic: str, logger=None) -> None:
        """Subscribe on ``node`` for ``topic``. ``topic`` empty means the
        channel is disabled and nothing is subscribed."""
        self._node = node
        self._stage = stage
        self._topic = topic
        self._logger = logger
        self._mesh_by_id: dict = {}
        self._n_by_id: dict = {}
        self._inv_cache: dict = {}
        self._last_tick = -1
        self._frames_received = 0
        self._topology_received = False
        self._warned: set = set()

        if node is None or not topic:
            return

        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from std_msgs.msg import Float32MultiArray, String

        node.create_subscription(
            String,
            topic + "_topology",
            self._on_topology,
            QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                history=HistoryPolicy.KEEP_LAST,
            ),
        )
        node.create_subscription(
            Float32MultiArray,
            topic + "_flat",
            self._on_flat,
            QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
            ),
        )
        self._log(f"deformable channel enabled: {topic}_topology / {topic}_flat")

    def _log(self, msg: str) -> None:
        if self._logger is not None:
            self._logger.info(msg)
        else:
            print(f"[deformable] {msg}", flush=True)

    def _warn_once(self, key: str, msg: str) -> None:
        if key in self._warned:
            return
        self._warned.add(key)
        if self._logger is not None:
            self._logger.warn(msg)
        else:
            print(f"[deformable] WARNING: {msg}", flush=True)

    def _on_topology(self, msg) -> None:
        """Parse the latched topology once and bind each declared object to
        its stage prim. A malformed payload is reported and leaves the
        channel inert rather than propagating (an uncaught exception here
        would escape ``rclpy.spin_once`` and kill the render node)."""
        if self._mesh_by_id:
            return  # latched; parsed once
        try:
            payload = json.loads(msg.data)
            objects = payload["objects"]
        except Exception as exc:  # noqa: BLE001
            self._warn_once("topology-parse", f"topology JSON unusable ({exc!r}); deformable channel disabled")
            return

        for obj in objects:
            try:
                mesh_id = int(obj["mesh_id"])
                prim_path = obj["prim_path"]
                n_particles = int(obj["n_particles"])
                name = obj.get("name", prim_path)
            except Exception as exc:  # noqa: BLE001
                self._warn_once("topology-entry", f"topology entry unusable ({exc!r}); skipped")
                continue
            prim = self._stage.GetPrimAtPath(prim_path)
            if not prim or not prim.IsValid():
                self._warn_once(f"missing:{prim_path}", f"deformable prim not in stage: {prim_path}")
                continue
            for schema_name in _DEFORMABLE_SCHEMAS:
                try:
                    if schema_name in prim.GetAppliedSchemas():
                        prim.RemoveAppliedSchema(schema_name)
                except Exception:  # noqa: BLE001
                    pass
            from pxr import UsdGeom as _UsdGeom

            self._mesh_by_id[mesh_id] = _UsdGeom.Mesh(prim)
            self._n_by_id[mesh_id] = n_particles
            self._log(f"  mesh_id={mesh_id} name={name} prim_path={prim_path} n_particles={n_particles}")

        self._topology_received = True
        self._log(f"deformable topology received: {len(self._mesh_by_id)} object(s)")

    def _on_flat(self, msg) -> None:
        if not self._mesh_by_id:
            return  # topology not here yet (or this scene has none)
        data = msg.data
        if len(data) < 2:
            return
        import numpy as np  # local, like every other numpy use here
        from pxr import Gf, Usd, UsdGeom, Vt

        arr = np.asarray(data, dtype=np.float64)
        tick = int(round(arr[1]))
        i = 2

        xf_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        while i + 2 <= len(arr):
            mesh_id = int(round(arr[i]))
            count = int(round(arr[i + 1]))
            i += 2
            if i + count * 3 > len(arr):
                self._warn_once(
                    "truncated",
                    f"flat message truncated: mesh_id={mesh_id} wants {count} "
                    f"points, only {(len(arr) - i) // 3} available",
                )
                break
            mesh = self._mesh_by_id.get(mesh_id)
            if mesh is None:
                i += count * 3
                continue
            declared = self._n_by_id.get(mesh_id)
            if declared is not None and count != declared:
                self._warn_once(
                    f"count:{mesh_id}",
                    f"mesh_id={mesh_id} got {count} points, topology declared {declared}",
                )
            pts_world = arr[i : i + count * 3].reshape(count, 3)
            i += count * 3

            prim_path = str(mesh.GetPrim().GetPath())
            inv = self._inv_cache.get(prim_path)
            if inv is None:
                m = np.array(xf_cache.GetLocalToWorldTransform(mesh.GetPrim()), dtype=np.float64).reshape(4, 4)
                inv = np.linalg.inv(m)
                self._inv_cache[prim_path] = inv
            homo = np.concatenate([pts_world, np.ones((count, 1))], axis=1)
            local = np.ascontiguousarray((homo @ inv)[:, :3], dtype=np.float32)
            try:
                pts = Vt.Vec3fArray.FromNumpy(local)
            except Exception:  # noqa: BLE001
                pts = Vt.Vec3fArray([Gf.Vec3f(float(a), float(b), float(c)) for a, b, c in local])
            mesh.GetPointsAttr().Set(pts)
            # Points moved: the prim's cached bounds are stale, and an
            # out-of-date extent can cull the mesh from the render.
            try:
                mesh.GetExtentAttr().Set(Vt.Vec3fArray([local.min(axis=0), local.max(axis=0)]))
            except Exception:  # noqa: BLE001
                pass

        self._frames_received += 1
        if tick != self._last_tick:
            self._last_tick = tick
            if self._frames_received % 30 == 0:
                self._log(f"deformable channel: tick={tick} ({self._frames_received} message(s) received)")

    def has_topology(self) -> bool:
        return self._topology_received

    def last_tick(self) -> int:
        return int(self._last_tick)


def build_tiled_sensor_group(stage, cam_paths, obs, n_envs: int):
    """Set per-camera intrinsics and build the ``TiledCameraSensor`` group for
    an already-built multi-env stage.

    Returns ``(sensors, n_cams, W, Hmax)`` where ``sensors`` is a list of
    ``(TiledCameraSensor, flat_idx, group_height)``: ``flat_idx[k]`` is the
    global ``env * n_cams + cam`` slot that sensor's ``k``-th tile belongs in,
    which is the index the tiled SHM is addressed by.

    Cameras are grouped by obs HEIGHT because one ``TiledCameraSensor`` needs a
    uniform tile size, while the benchmark's cameras genuinely differ (head
    640x400, wrists 640x528). One sensor per distinct height, all rendered in
    the SAME RTX pass; each group scatters its tiles into the one uniform
    (Hmax-tall) SHM and cameras shorter than Hmax leave the trailing rows zero,
    cropped back to their real height at the obs. A scene whose cameras share a
    height therefore collapses to a single sensor and one bulk copy.

    Lifted out of ``isaacsim_tiled_render.build_scene`` so the FUSED node
    (``isaacsim_fused_sim.py``) builds its sensors through the exact same code
    rather than its own transcription of this grouping. Two copies of an index
    mapping like ``flat_idx`` would be a standing invitation to drift, and a
    drift here does not raise -- it silently puts one camera's picture in
    another camera's SHM slot.

    NOTE: ``build_scene`` still carries its own copy of this logic. Unifying
    it is a one-line change there, deliberately NOT made yet: that function
    is on the production render path and cannot be exercised from the host
    test suite (its module opens a ``SimulationApp`` at import), so it
    should be switched over only when someone can live-run the render node
    and confirm the picture is unchanged.
    """
    from collections import OrderedDict

    # Isaac Sim 6.2 exports the tiled camera runtime from experimental.rtx.
    from isaacsim.sensors.experimental.rtx import TiledCameraSensor

    n_cams = len(obs)
    cam_hw = [(int(c.height), int(c.width)) for c in obs]
    W = cam_hw[0][1]
    Hmax = max(h for h, _w in cam_hw)

    # Intrinsics on env_0's prims; the clones inherit the camera model.
    for cam, path in zip(obs, cam_paths):
        prim = stage.GetPrimAtPath(path)
        if prim and prim.IsValid():
            try:
                set_camera_intrinsics(prim, cam)
            except Exception as e:  # noqa: BLE001
                print(f"[tiled] intrinsics failed for {path}: {e!r}", flush=True)

    groups: "OrderedDict[int, list]" = OrderedDict()
    for j, (hh, _w) in enumerate(cam_hw):
        groups.setdefault(hh, []).append(j)
    sensors = []
    for gh, cam_js in groups.items():
        paths, flat_idx = [], []
        for i in range(int(n_envs)):
            for j in cam_js:
                paths.append(cam_paths[i * n_cams + j])
                flat_idx.append(i * n_cams + j)
        s = TiledCameraSensor(paths, resolution=(gh, W), annotators=["rgb"])
        sensors.append((s, flat_idx, gh))
        print(f"[tiled] sensor @ {W}x{gh}: cams={cam_js} ({len(paths)} tiles)", flush=True)
    return sensors, n_cams, W, Hmax


__all__ = [
    "CameraConfig",
    "Manifest",
    "set_camera_intrinsics",
    "apply_xform",
    "reference_scene_stage",
    "init_stage",
    "resolve_obs_camera_prims",
    "TfRenderDriver",
    "preload_warp_before_kit",
    "DeformableChannel",
    "build_gridcloner_stage",
    "select_physics_variant",
    "disable_physics_subtree",
    "build_tiled_sensor_group",
]

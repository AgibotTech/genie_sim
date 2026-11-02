# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""URDF→USD assembler — impl module.

Exposes :func:`assemble_robot` for :class:`UrdfToUsdStage`. Isaac Sim / pxr / rclpy
imports remain gated inside the individual helper functions so
importing this module does NOT require a running Kit or a ROS node.

**Subprocess entry point:** this module also exposes
``python -m genie_sim_engine_libuipc.assemble._robot_impl --scene … --output-dir …``
so :class:`UrdfToUsdStage` can spawn it in a fresh interpreter. Isaac
Sim's ``SimulationApp`` is a per-process singleton whose ``close()``
call can terminate the interpreter, so running this stage inline
would kill the pipeline runner before the next stage gets to fire.
The subprocess isolation keeps that ``close()`` inside a process the
pipeline runner can outlive.
"""

from __future__ import annotations

import faulthandler
import hashlib
import os
import re
import sys
import traceback

import yaml

faulthandler.enable()  # print C-level stack trace on crash (SIGABRT, SIGSEGV, etc.)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)  # flush CLI output immediately

# Link-name regexes for the selective collision policy in
# ``_apply_post_transformer_collision_policy``. Matched case-insensitively
# against ``prim.GetName()`` for each rigid body. Hits trigger
# `MeshCollisionAPI` authoring (SDF for gripper, convex hull for wheel);
# misses fall through to the "disable colliders on non-contact links"
# branch. See ../../genie_sim_engine_core/docs/pipeline.md § "Selective collision policy" for the
# full decision table — and add new contact patterns there in the same
# diff so docs and code stay in sync.
_COLL_GRIPPER_RE = re.compile(r"gripper|finger|jaw|knuckle|palm", re.IGNORECASE)
_COLL_WHEEL_RE = re.compile(r"wheel|caster|castor", re.IGNORECASE)


def _load_scene_yaml(scene: str) -> tuple[str, dict]:
    cfg_path = os.path.abspath(scene)
    if not cfg_path.endswith((".yaml", ".yml")):
        print(
            f"[assemble_robot] ERROR: --scene must be a YAML file (.yaml/.yml), got: {cfg_path}",
            file=sys.stderr,
        )
        sys.exit(1)
    with open(cfg_path) as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        print(
            f"[assemble_robot] ERROR: scene YAML did not parse to a mapping: {cfg_path}",
            file=sys.stderr,
        )
        sys.exit(1)
    return cfg_path, config


def _fetch_urdf_from_topic(timeout_s: float) -> str:
    """Subscribe to ``/robot_description`` and return the URDF string.

    Uses TRANSIENT_LOCAL durability so the latched URDF published by
    ``robot_state_publisher`` is delivered even when this script
    subscribes after RSP has already published. Blocks up to ``timeout_s``
    waiting for the message; exits with code 1 on timeout.
    """
    try:
        import rclpy
        from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
        from std_msgs.msg import String
    except ImportError as exc:
        print(
            f"[assemble_robot] ERROR: rclpy / std_msgs not importable ({exc}). "
            "Source your ROS 2 workspace before running.",
            file=sys.stderr,
        )
        sys.exit(1)

    rclpy.init(args=None)
    node = rclpy.create_node("assemble_robot_urdf_fetcher")
    qos = QoSProfile(
        depth=1,
        history=QoSHistoryPolicy.KEEP_LAST,
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    )

    holder: dict[str, str] = {}

    def _on_msg(msg):
        if not holder:
            holder["urdf"] = msg.data

    sub = node.create_subscription(String, "/robot_description", _on_msg, qos)

    deadline = node.get_clock().now().nanoseconds + int(timeout_s * 1e9)
    try:
        while rclpy.ok() and not holder:
            if node.get_clock().now().nanoseconds >= deadline:
                break
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_subscription(sub)
        node.destroy_node()
        rclpy.shutdown()

    if not holder:
        print(
            f"[assemble_robot] ERROR: timed out after {timeout_s:.1f}s waiting for "
            "/robot_description. Is robot_state_publisher running?",
            file=sys.stderr,
        )
        sys.exit(1)

    urdf = holder["urdf"]
    if not urdf.strip():
        print(
            "[assemble_robot] ERROR: /robot_description message was empty.",
            file=sys.stderr,
        )
        sys.exit(1)
    return urdf


def _resolve_package_uris(urdf_str: str, urdf_dir: str) -> str:
    """Rewrite every ``package://<pkg>/<rel>`` URI to a minimal RELATIVE path.

    Isaac Sim's URDF importer only walks ``ROS_PACKAGE_PATH`` to resolve
    ``package://`` URIs, and that env var is unreliable under ROS 2 (which
    uses ament_index). To sidestep that without baking absolute paths into
    the URDF, we:

    1. Create a ``pkg/`` directory next to the URDF (``urdf_dir``).
    2. Symlink each referenced package's share directory into
       ``urdf_dir/pkg/<pkg>``.
    3. Rewrite ``package://<pkg>/<rel>`` to ``pkg/<pkg>/<rel>`` -- a path
       relative to the URDF file itself.

    The result is a portable URDF whose mesh references are short, readable,
    and resolved by the importer without any environment configuration.
    """
    try:
        from ament_index_python.packages import (
            PackageNotFoundError,
            get_package_share_directory,
        )
    except ImportError:
        print(
            "[assemble_robot] ERROR: ament_index_python is not importable while "
            "resolving package:// URIs. Source your ROS 2 workspace.",
            file=sys.stderr,
        )
        sys.exit(1)

    pkg_root = os.path.join(urdf_dir, "pkg")
    os.makedirs(pkg_root, exist_ok=True)

    share_cache: dict[str, str] = {}

    def _share(pkg: str) -> str:
        if pkg not in share_cache:
            try:
                share_cache[pkg] = get_package_share_directory(pkg)
            except PackageNotFoundError:
                print(
                    f"[assemble_robot] ERROR: URDF references package://{pkg}/... but "
                    f"package {pkg!r} is not on the ament index. Source the workspace "
                    f"that installs it.",
                    file=sys.stderr,
                )
                sys.exit(1)
        return share_cache[pkg]

    def _ensure_symlink(pkg: str) -> None:
        link_path = os.path.join(pkg_root, pkg)
        target = _share(pkg)
        # If a stale symlink/file exists, replace it so the link tracks the
        # currently-sourced workspace.

        if os.path.islink(link_path) or os.path.exists(link_path):
            try:
                current = os.readlink(link_path)
            except OSError:
                current = None
            if current == target:
                return
            try:
                if os.path.islink(link_path) or os.path.isfile(link_path):
                    os.unlink(link_path)
                else:
                    # A real directory was placed here; refuse to delete it.
                    print(
                        f"[assemble_robot] ERROR: {link_path} exists and is not a symlink; " f"refusing to overwrite.",
                        file=sys.stderr,
                    )
                    sys.exit(1)
            except OSError as exc:
                print(
                    f"[assemble_robot] ERROR: failed to clear stale link {link_path}: {exc}",
                    file=sys.stderr,
                )
                sys.exit(1)
        try:
            os.symlink(target, link_path)
        except OSError as exc:
            print(
                f"[assemble_robot] ERROR: failed to symlink {link_path} -> {target}: {exc}",
                file=sys.stderr,
            )
            sys.exit(1)

    pattern = re.compile(r"package://([A-Za-z_][A-Za-z0-9_]*)/([^\"'<>\s]+)")

    def _sub(match: re.Match) -> str:
        pkg, rel = match.group(1), match.group(2)
        _ensure_symlink(pkg)
        rel_path = os.path.join("pkg", pkg, rel)
        # Sanity-check: the resolved file must actually exist (via the link).
        check_path = os.path.join(urdf_dir, rel_path)
        if not os.path.exists(check_path):
            print(
                f"[assemble_robot] WARNING: resolved package:// path does not exist "
                f"on disk: {check_path} (from package://{pkg}/{rel})",
                file=sys.stderr,
            )
        return rel_path

    return pattern.sub(_sub, urdf_str)


def _stage_urdf_from_string(urdf_str: str, dest_dir: str) -> tuple[str, str, dict[str, str]]:
    """Stage a URDF string to disk in two forms for the URDF→USD converters.

    Returns (urdf_path, raw_urdf_path, ros_packages) where:
      urdf_path    — robot.urdf with package:// URIs rewritten to relative paths
      raw_urdf_path — robot_raw.urdf with original package:// URIs intact
      ros_packages  — {pkg_name: share_dir} for all referenced packages
    """
    os.makedirs(dest_dir, exist_ok=True)

    # Write the raw URDF (package:// URIs intact) for the 6.2 importer.
    # The 6.2 converter auto-discovers packages by walking up from the URDF
    # file's directory looking for <pkg_name>/<rel_path>. Create top-level
    # symlinks <dest_dir>/<pkg_name> -> share_dir so the walker finds them.
    raw_urdf_path = os.path.join(dest_dir, "robot_raw.urdf")
    with open(raw_urdf_path, "w") as f:
        f.write(urdf_str)

    # Collect ros_packages and create top-level symlinks for auto-discovery.
    try:
        from ament_index_python.packages import get_package_share_directory, PackageNotFoundError

        pkg_names = sorted(set(re.findall(r"package://([A-Za-z_][A-Za-z0-9_]*)/", urdf_str)))
        ros_packages: dict[str, str] = {}
        for pkg in pkg_names:
            try:
                share = get_package_share_directory(pkg)
                ros_packages[pkg] = share
                link = os.path.join(dest_dir, pkg)
                if os.path.lexists(link):
                    if os.path.islink(link) and os.readlink(link) == share:
                        continue
                    if os.path.isdir(link) and not os.path.islink(link):
                        continue
                    os.unlink(link)
                os.symlink(share, link)
            except PackageNotFoundError:
                pass
    except ImportError:
        ros_packages = {}

    # Rewrite every ``package://<pkg>/...`` URI to a path relative to the URDF
    # via symlinks under ``<dest_dir>/pkg/<pkg>``. For the 5.x kit-command importer.
    urdf_str = _resolve_package_uris(urdf_str, urdf_dir=dest_dir)

    urdf_path = os.path.join(dest_dir, "robot.urdf")
    with open(urdf_path, "w") as f:
        f.write(urdf_str)
    return urdf_path, raw_urdf_path, ros_packages


def _parse_material_override_blocks(urdf_str: str) -> list[dict]:
    """Read inline ``<material_override>`` blocks nested inside ``<visual>``.

    URDF's standard ``<material>`` element carries diffuse color only —
    no place to author roughness / metallic. Mesh files (DAE / OBJ) DO
    carry per-material PBR, but DAE's roughness/metallic round-trips
    poorly through most DCC exporters, and the 6.2 ``urdf_usd_converter``
    only reads diffuse / specular / emission / transparent from DAE.

    A ``<material_override>`` authored as a child of a ``<visual>`` patches
    the metallic / roughness of **that visual's** mesh-embedded material(s)
    in place — keeping albedo and textures::

        <link name="chassis_link">
          <visual>
            <geometry><mesh filename="${mesh_dir}/tracer_base.dae"/></geometry>
            <material_override>
              <roughness>0.30</roughness>
              <metallic>0.85</metallic>
            </material_override>
          </visual>
        </link>

    Authoring inline (rather than a robot-level block keyed by link/mesh)
    means the override auto-scopes to exactly the geometry it sits next to:
    a link with several visuals just gets one ``<material_override>`` per
    visual, no attributes needed. The block is silently ignored by urdfdom,
    robot_state_publisher, MoveIt and Isaac's URDF→USD converter (it parses
    as an undefined element).

    Returns a list of specs, one per inline block::

        [{"link": "chassis_link", "target": "tracer_base",
          "entry": {"roughness": 0.30, "metallic": 0.85}}, ...]

    ``target`` is the name of the USD prim the visual's geometry becomes —
    the converter names a mesh wrapper after the mesh file stem
    (``tracer_base.dae`` → ``tracer_base``) and a primitive after its shape
    (``<sphere>`` → ``sphere``, ``<cylinder>`` → ``cylinder``, ``<box>`` →
    ``cube``). ``None`` if the geometry can't be identified. Either PBR
    field may be omitted; empty / malformed blocks are dropped.
    """

    import os
    import xml.etree.ElementTree as ET

    try:
        root = ET.fromstring(urdf_str)
    except ET.ParseError:
        return []

    # primitive shape tag -> USD prim name the converter authors
    _PRIM_NAME = {"sphere": "sphere", "cylinder": "cylinder", "box": "cube"}

    def _geometry_target(visual) -> str | None:
        geom = visual.find("geometry")
        if geom is None:
            return None
        mesh = geom.find("mesh")
        if mesh is not None and mesh.attrib.get("filename"):
            return os.path.splitext(os.path.basename(mesh.attrib["filename"].strip()))[0]
        for tag, usd_name in _PRIM_NAME.items():
            if geom.find(tag) is not None:
                return usd_name
        return None

    out: list[dict] = []
    for link in root.iter("link"):
        link_name = link.attrib.get("name")
        if not link_name:
            continue
        for visual in link.findall("visual"):
            block = visual.find("material_override")
            if block is None:
                continue
            entry: dict[str, float] = {}
            for field in ("roughness", "metallic"):
                elem = block.find(field)
                if elem is None or elem.text is None:
                    continue
                try:
                    entry[field] = float(elem.text.strip())
                except ValueError:
                    continue
            if not entry:
                continue
            out.append(
                {
                    "link": link_name,
                    "target": _geometry_target(visual),
                    "entry": entry,
                }
            )
    return out


def _resolve_material_authoring_target(material_prim, logger=None):
    """Resolve an in-stage material prim to its (source-layer, master-path).

    AS3 USD has materials inside instance proxies — visual wrappers
    reference ``payloads/instances.usda`` which in turn references
    ``payloads/materials.usda``. UsdStage refuses authoring on instance
    proxies AND on the anonymous prototype root the proxies resolve to,
    so to write the override we walk back through the composition arcs
    and edit the master material at its definition site
    (``materials.usda</Materials/X>``).

    Returns ``(layer, master_prim_path)`` or ``(None, None)`` when the
    composition can't be traced (e.g. the override target is a directly
    authored material in the root layer, where editing via the in-stage
    prim is fine and the caller falls back to that).
    """
    from pxr import Usd  # noqa: E402

    if not material_prim.IsInstanceProxy():
        # Directly authored — caller authors via the in-stage prim.
        return None, None
    proto = material_prim.GetPrimInPrototype()
    if not proto or not proto.IsValid():
        return None, None
    # Walk arcs to the deepest reference into a non-anonymous layer
    # whose layer hosts an actual spec for this prim. instances.usda
    # references materials.usda — we want materials.usda as the
    # canonical edit site so the override propagates everywhere.
    target_layer = None
    target_path = None
    for arc in Usd.PrimCompositionQuery(proto).GetCompositionArcs():
        if not arc.HasSpecs():
            continue
        layer = arc.GetTargetLayer()
        if layer is None or layer.anonymous:
            continue
        target_layer = layer
        target_path = arc.GetTargetPrimPath()
    if target_layer is None or target_path is None:
        if logger is not None:
            logger.warn(
                f"[material_override] could not trace authoring target for "
                f"{material_prim.GetPath()}; instance-proxy edit will be skipped."
            )
    return target_layer, target_path


def _set_pbr_inputs(material_prim, entry: dict[str, float], logger=None) -> bool:
    """Set roughness / metallic on a ``UsdShade.Material``'s preview surface.

    Sets the Material-level interface input first (present after the
    converter's ``addPreviewMaterialInterface`` pass), falling back to the
    underlying ``UsdShade.Shader`` input. Returns True if at least one
    field was written. Diffuse / albedo / textures are left untouched —
    only the two PBR scalars are patched.
    """
    from pxr import UsdShade

    mat = UsdShade.Material(material_prim)
    if not mat:
        return False

    wrote = False
    for field, value in entry.items():
        target_input = mat.GetInput(field)
        if not target_input:
            surface_output = mat.GetSurfaceOutput()
            shader_prim = None
            if surface_output:
                src = surface_output.GetConnectedSource()
                if src and src[0]:
                    shader_prim = src[0].GetPrim()
            if shader_prim and shader_prim.IsValid():
                shader = UsdShade.Shader(shader_prim)
                target_input = shader.GetInput(field)
        if target_input:
            target_input.Set(float(value))
            wrote = True
        elif logger is not None:
            logger.warn(
                f"[material_override] material {material_prim.GetPath()}: "
                f"no '{field}' input on Material or surface shader; skipped"
            )
    return wrote


def _collect_wrapper_materials(wrapper_prim):
    """Collect material prim paths bound anywhere under a visual-wrapper Xform.

    A visual wrapper's subtree is pure geometry (instanceable ``Mesh`` /
    ``GeomSubset`` refs) — child links are *siblings* of the wrapper, never
    descendants — so no link-boundary pruning is needed. Walks into
    instance proxies so the instanced mesh prims and their bindings are
    visible. Returns a set of ``Sdf.Path`` to bound ``UsdShade.Material``
    prims (which live in ``/<prefix>/Materials``, shared and editable).
    """
    from pxr import Usd, UsdShade

    paths = set()
    for prim in Usd.PrimRange(wrapper_prim, Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdShade.MaterialBindingAPI):
            continue
        rel = UsdShade.MaterialBindingAPI(prim).GetDirectBindingRel()
        if not rel:
            continue
        for t in rel.GetTargets():
            paths.add(t)
    return paths


def _link_visual_wrappers(link_prim):
    """Return a link's own visual geometry children (wrappers + primitives).

    A link Xform's direct children are either child-link Xforms (which
    carry ``UsdPhysics.RigidBodyAPI`` once ``add_rigid_body_schemas`` has
    run) or visual / collision geometry (which do NOT). We keep the
    geometry and drop the child links by that test — robust against the
    mesh-stem-vs-link-name collision (e.g. an ``arm_*_base_link`` whose
    ``base_link.dae`` produces a wrapper named ``base_link``, colliding
    with the root link name). Mesh visuals appear as wrapper ``Xform``\\ s
    (named after the mesh stem); primitive visuals appear as ``Gprim``\\ s
    (``Sphere`` / ``Cube`` / ``Cylinder``, named after the shape). Collision
    geometry (``purpose == guide``) is skipped.

    Returns a list of ``(name, prim)``.
    """
    from pxr import UsdGeom, UsdPhysics

    out = []
    for child in link_prim.GetChildren():
        if not (child.IsA(UsdGeom.Xform) or child.IsA(UsdGeom.Gprim)):
            continue
        if child.HasAPI(UsdPhysics.RigidBodyAPI):
            continue  # child link, not visual geometry
        purpose = UsdGeom.Imageable(child).GetPurposeAttr().Get()
        if purpose == UsdGeom.Tokens.guide:
            continue  # collision geometry
        out.append((child.GetName(), child))
    return out


def _find_link_prim(stage, robot_prefix: str, link_name: str):
    """Return the real link Xform named ``link_name`` under ``/<robot_prefix>``.

    Disambiguates the mesh-wrapper-vs-link name collision: when several
    prims share the name, the real link is the one carrying
    ``UsdPhysics.RigidBodyAPI`` (mesh wrappers never do). Returns ``None``
    if not found, or the first plain match if none carry RigidBodyAPI
    (defensive — shouldn't happen once schemas are applied).
    """
    from pxr import Sdf, Usd, UsdPhysics

    root = stage.GetPrimAtPath(Sdf.Path(f"/{robot_prefix}"))
    if not root or not root.IsValid():
        return None
    fallback = None
    for p in Usd.PrimRange(root):
        if p.GetName() != link_name:
            continue
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            return p
        if fallback is None:
            fallback = p
    return fallback


def _apply_material_overrides(
    stage,
    overrides: list[dict],
    robot_prefix: str,
    logger=None,
) -> int:
    """Patch metallic / roughness on each inline override's target material(s).

    For each spec (one inline ``<material_override>`` from a ``<visual>``):
    find the enclosing link's USD Xform (via ``RigidBodyAPI``, so a
    same-named mesh wrapper can't shadow it), pick the visual geometry whose
    prim name matches ``target`` (the mesh stem or primitive shape name),
    collect the materials it binds, and overwrite their preview-surface
    roughness/metallic — leaving albedo / textures intact.

    When ``target`` is ``None`` (geometry couldn't be identified) the
    override falls back to every visual on the link — correct when the link
    has a single visual, and logged so the operator can tell. A material
    shared by multiple visuals/links is patched once per override that
    reaches it; conflicting overrides resolve last-applied-wins. Returns the
    number of (target, material) patches applied.
    """
    if not overrides:
        return 0

    from pxr import Usd  # noqa: E402

    # Per-source-layer Usd.Stage cache. Materials shared across links
    # / sides resolve to the same (layer, path) tuple, so opening the
    # layer's stage once (and saving once at the end) is both faster
    # and ensures every override on the same master is committed.
    layer_stages: dict[str, Usd.Stage] = {}

    applied = 0
    for spec in overrides:
        link_name = spec["link"]
        target = spec["target"]
        entry = spec["entry"]

        link_prim = _find_link_prim(stage, robot_prefix, link_name)
        if link_prim is None:
            if logger is not None:
                logger.warn(f"[material_override] link '{link_name}' not found under " f"/{robot_prefix}; skipped")
            continue

        wrappers = _link_visual_wrappers(link_prim)
        if target is not None:
            matched = [(n, p) for (n, p) in wrappers if n == target]
            if not matched:
                if logger is not None:
                    logger.warn(
                        f"[material_override] {link_name}: no visual geometry named "
                        f"'{target}' (mesh stem / primitive); falling back to all "
                        f"visuals on the link"
                    )
                matched = wrappers
            wrappers = matched

        if not wrappers and logger is not None:
            logger.warn(f"[material_override] {link_name}: no visual geometry to patch")

        label = f"{link_name}/{target}" if target else link_name
        for _wname, wprim in wrappers:
            for mp in _collect_wrapper_materials(wprim):
                material_prim = stage.GetPrimAtPath(mp)
                if not material_prim or not material_prim.IsValid():
                    continue
                # Two cases:
                #   - Directly-authored material: it sits under
                #     /robot/Materials and is editable in place via the
                #     composed stage.
                #   - AS3-instanced material: lives inside an instance
                #     proxy; the edit must go to the source layer
                #     (payloads/materials.usda) at the master prim
                #     path, otherwise USD blocks the edit
                #     ("authoring to an instance proxy is not allowed").
                edit_prim = material_prim
                source_layer, master_path = _resolve_material_authoring_target(material_prim, logger=logger)
                if source_layer is not None and master_path is not None:
                    edit_stage = layer_stages.get(source_layer.identifier)
                    if edit_stage is None:
                        edit_stage = Usd.Stage.Open(source_layer)
                        if edit_stage is None:
                            if logger is not None:
                                logger.warn(
                                    f"[material_override] could not open source layer "
                                    f"{source_layer.identifier} for {material_prim.GetPath()}"
                                )
                            continue
                        layer_stages[source_layer.identifier] = edit_stage
                    edit_prim = edit_stage.GetPrimAtPath(master_path)
                    if not edit_prim or not edit_prim.IsValid():
                        if logger is not None:
                            logger.warn(
                                f"[material_override] master {master_path} not found in "
                                f"{source_layer.identifier}; skipped."
                            )
                        continue
                if _set_pbr_inputs(edit_prim, entry, logger=logger):
                    applied += 1
                    if logger is not None:
                        fields = ", ".join(f"{k}={v}" for k, v in entry.items())
                        logger.info(f"[material_override] {label}: patched " f"{edit_prim.GetPath().name} ({fields})")

    # Persist any source-layer edits made for instance-proxy
    # materials. The composed-stage caller saves the root layer
    # itself; this loop saves the leaf source layers.
    for path, edit_stage in layer_stages.items():
        try:
            edit_stage.GetRootLayer().Save()
        except Exception as exc:
            if logger is not None:
                logger.warn(f"[material_override] failed to save {path}: {exc}")
    # Explicitly drop the source-layer stage refs so their Sdf.Layer
    # instances can be evicted from the process-global layer cache.
    # See the hygiene comment in
    # ``_apply_post_transformer_material_overrides`` for why a stale
    # cached layer between post-transformer steps silently swallows
    # a later ``AddAppliedSchema`` call.
    layer_stages.clear()
    return applied


# Boolean custom-attribute names the engine reads to identify gripper
# topology WITHOUT the old hard-coded name patterns (``"link4"`` substring,
# ``gripper_[lr]_(inner|outer)_joint[034]$`` regex).  Authored at assemble
# time by ``_derive_gripper_annotations`` + ``_author_gripper_annotations``.
# Derive pad and coupler annotations from the URDF gripper topology.
_ANNOT_GRIPPER_PAD = "newton:isGripperPad"
_ANNOT_LOOP_COUPLER = "newton:isLoopCoupler"

# URDF joint types whose child link rides on a moving degree of freedom.
# A fingertip pad is a LEAF of the movable gripper chain, so we walk only
# these edges when finding chain endpoints (fixed / floating joints don't
# extend the actuated chain).
_URDF_MOVABLE_JOINT_TYPES = frozenset({"revolute", "continuous", "prismatic"})


def _derive_gripper_annotations(urdf_str: str) -> dict:
    """Derive the gripper pad-body + 4-bar coupler-joint sets from URDF
    STRUCTURE (no name literals beyond the shared ``_COLL_GRIPPER_RE``
    family classifier).

    Returns ``{"pads": set[str], "couplers": set[str]}`` — link names and
    joint names respectively.  Empty sets on parse failure or when the
    robot has no gripper (safe: the engine falls back to its legacy
    literals when no annotation is present).

    Physical-truth definitions (see the plan's "reconnaissance correction"
    section):

    * **couplers** — every URDF ``<mimic>``-follower gripper joint whose
      parent link is NOT the gripper base link.  These are the passive
      joints of the 4-bar linkage that the engine must unlock from PD
      control so the closed loop can move.  A follower whose parent IS a
      gripper ``*_base_link`` is the second DRIVEN joint (``outer_joint1``,
      mimicking the master with multiplier −1) — a real actuated joint,
      NOT a passive coupler — so it is EXCLUDED.  On G2 this yields
      exactly 12 (6 per gripper): ``gripper_[lr]_(inner|outer)_joint[034]``,
      excluding the two base-parented driven ``outer_joint1``.

    * **pads** — fingertip pad LINKS: the leaf links of the *movable*
      gripper joint chains.  A link qualifies when it is

        1. the child of a movable (revolute/continuous/prismatic) joint,
        2. itself gripper-classified (``_COLL_GRIPPER_RE``),
        3. NOT a ``<loop_joint>`` endpoint (those ``*_link2`` bodies close
           the 4-bar loop and are not fingertips), and
        4. a chain endpoint — no movable gripper child of its own.

      This structural rule lands on exactly the 4 real pads
      (``gripper_[lr]_(inner|outer)_link4``) on G2 and deliberately
      EXCLUDES ``arm_[lr]_link4`` / ``body_link4`` (not gripper-classified)
      and the ``gripper_[lr]_camera_link`` bodies (attached by a *fixed*
      joint, so never a movable-chain child).  That is the whole point of
      moving off the old ``"link4" in label`` substring test, which wrongly
      hit the arm / body ``link4`` bodies.
    """
    import xml.etree.ElementTree as ET

    out = {"pads": set(), "couplers": set()}
    try:
        root = ET.fromstring(urdf_str)
    except ET.ParseError:
        return out

    def _is_gripper(name: str | None) -> bool:
        return bool(name and _COLL_GRIPPER_RE.search(name))

    # Links that close the 4-bar loop (``<loop_joint>`` endpoints). These
    # are structurally part of the gripper chain but are NOT fingertips.
    loop_links: set[str] = set()
    for lj in root.iter("loop_joint"):
        for tag in ("link1", "link2"):
            e = lj.find(tag)
            if e is not None and e.get("link"):
                loop_links.add(e.get("link"))

    movable_children: dict[str, set[str]] = {}  # parent link -> movable child links
    movable_child_links: set[str] = set()  # every link that is a movable joint's child
    for j in root.iter("joint"):
        jname = j.get("name") or ""
        jtype = (j.get("type") or "").strip().lower()
        parent_el = j.find("parent")
        child_el = j.find("child")
        parent = parent_el.get("link") if parent_el is not None else None
        child = child_el.get("link") if child_el is not None else None

        # Coupler = gripper-classified <mimic>-follower joint whose parent
        # link is NOT the gripper base link.  A follower parented on a
        # gripper ``*_base_link`` is the second DRIVEN joint (outer_joint1,
        # multiplier −1), not a passive 4-bar coupler, so it is excluded.
        if j.find("mimic") is not None and _is_gripper(jname):
            parent_is_gripper_base = _is_gripper(parent) and bool(parent and parent.endswith("_base_link"))
            if not parent_is_gripper_base:
                out["couplers"].add(jname)

        if jtype in _URDF_MOVABLE_JOINT_TYPES and parent and child:
            movable_children.setdefault(parent, set()).add(child)
            movable_child_links.add(child)

    # Candidate pads: gripper links reached by a movable joint, minus the
    # loop-closure bodies.  A candidate is a real pad iff it has no movable
    # gripper child (chain endpoint).
    candidates = {link for link in movable_child_links if _is_gripper(link) and link not in loop_links}
    for link in candidates:
        children = movable_children.get(link, set())
        if not any(child in candidates for child in children):
            out["pads"].add(link)

    return out


def _author_gripper_annotations(stage, urdf_str: str, logger=None) -> dict:
    """Author ``newton:isGripperPad`` / ``newton:isLoopCoupler`` booleans
    onto the derived pad bodies + coupler joints of an OPEN ``robot.usda``
    stage.  Pure-add overlay; idempotent (re-running re-derives the same
    sets and re-sets the same ``True`` values).

    * ``newton:isGripperPad = True`` on each pad LINK/BODY prim — matched
      by name AND ``UsdPhysics.RigidBodyAPI`` so a same-named visual mesh
      wrapper (named after the mesh stem, which never carries RigidBodyAPI)
      can't shadow the real link.
    * ``newton:isLoopCoupler = True`` on each coupler JOINT prim — matched
      by name among ``UsdPhysics.Joint`` prims.

    Returns ``{"pads_authored": int, "couplers_authored": int}``.
    """
    from pxr import Sdf, UsdPhysics

    ann = _derive_gripper_annotations(urdf_str)
    pads = ann["pads"]
    couplers = ann["couplers"]
    if not pads and not couplers:
        return {"pads_authored": 0, "couplers_authored": 0}

    n_pads = 0
    n_couplers = 0
    for prim in stage.Traverse():
        name = prim.GetName()
        if name in couplers and (prim.IsA(UsdPhysics.Joint) or prim.HasAPI(UsdPhysics.Joint)):
            attr = prim.GetAttribute(_ANNOT_LOOP_COUPLER)
            if not attr or not attr.IsValid():
                attr = prim.CreateAttribute(_ANNOT_LOOP_COUPLER, Sdf.ValueTypeNames.Bool)
            attr.Set(True)
            n_couplers += 1
        if name in pads and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            attr = prim.GetAttribute(_ANNOT_GRIPPER_PAD)
            if not attr or not attr.IsValid():
                attr = prim.CreateAttribute(_ANNOT_GRIPPER_PAD, Sdf.ValueTypeNames.Bool)
            attr.Set(True)
            n_pads += 1

    if logger is not None:
        missing_pads = pads - {
            p.GetName() for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI) and p.GetName() in pads
        }
        note = f" (WARN: {sorted(missing_pads)} not found as RigidBodyAPI bodies)" if missing_pads else ""
        logger.info(
            f"[gripper-annot] authored {_ANNOT_GRIPPER_PAD}=True on {n_pads} pad body(ies) "
            f"(derived {len(pads)}: {sorted(pads)}); {_ANNOT_LOOP_COUPLER}=True on "
            f"{n_couplers} coupler joint(s) (derived {len(couplers)}){note}."
        )
    return {"pads_authored": n_pads, "couplers_authored": n_couplers}


def _apply_mimic_joint_overlay(usd_path: str, logger=None, urdf_path: str | None = None) -> dict:
    """Author USD attributes URDF can't express, on the gripper +
    arm joints, into the final ``robot.usda``'s root layer.

    Scope (deliberately narrow)
    ---------------------------
    Per-joint kp / kd authoring is **out of scope**.  The PhysX layer
    handles tuning at runtime via ``kit/stage.py:_configure_drives``
    reading ``config/physics_params.yaml`` (per-class ``usd_drive_api.default_revolute``,
    ``usd_drive_api.chassis_drive_joint``, etc.) — that's the supported
    iteration loop, no rebake required.  Newton-standalone has its own
    runtime fallback in ``MuJoCoWarpAdapter.prepare_model`` (per-joint
    ``kp = effort × 10``).  Both paths leave kp/kd dynamic.

    What this overlay DOES author (USD-only, no runtime equivalent):

      1. ``physxJoint:armature`` — reflected rotor inertia.  URDF has
         no tag; ``physics_params.yaml`` doesn't carry it either.  Authored
         per-class because the reference G2 MJCF authors per-class:

           hand:           0.001
           arm shoulder:   0.15
           arm mid:        0.08
           arm wrist:      0.04

      2. Gripper master drive — ``drive:angular:physics:stiffness``
         and ``damping`` on the inner_joint1 master.  The reference
         MJCF emits ``gainprm="5"``, which can't be expressed in URDF
         (URDF has no joint stiffness tag).  ``physics_params.yaml`` has a
         ``usd_drive_api.gripper.master_stiffness`` field but
         ``_configure_drives`` only applies it on the Kit path; for
         newton-standalone the USD attribute is authored so both
         engines see the same authored value.

      3. Gripper followers — ``drive:angular:physics:stiffness = 0``
         + ``damping = 0``.  Newton's import resolves
         ``JointTargetMode.from_gains(0, 0, has_drive=True)`` → EFFORT
         actuator with idle ctrl (no PD); the equality constraint
         from ``NewtonMimicAPI`` propagates the master's motion to
         followers cleanly.  Without this, the URDF→USD pipeline's
         vestigial 625 default emits a POSITION actuator that
         fights the equality constraint → gripper chatters.  This
         matches what the mjwarp adapter's mimic-mute path achieves
         dynamically; authoring it makes both engines (isaac_newton
         AND newton-standalone) take the same shape.

    Returns counts dict.  Idempotent.
    """
    from pxr import Sdf, Usd, UsdPhysics

    if not os.path.isfile(usd_path):
        return {}

    import math as _math

    _DEG_TO_RAD = _math.pi / 180.0

    # Uniform passive-joint baseline — x2_31dof_hand convention.
    # Every revolute joint (arm + body + head + gripper + chassis)
    # gets the SAME armature + frictionloss, and dof_damping=0.
    # See physics_params.yaml top-of-file rationale and the
    # default class="x2" block in ~/robot_model-t2/x2_31dof_hand.xml
    # for the source pattern:
    #
    #     <joint damping="0.0" armature="0.03" frictionloss="0.3"/>
    #
    # Strategy: no viscous damping → no integrator-implicit
    # velocity-feedback dependence → simpler tuning + matches RL/WBC
    # references.  The small uniform armature gives just enough
    # ``kp/dt²`` stability headroom, and ``frictionloss`` provides
    # a Coulomb-like static-friction floor that dissipates energy
    # at low velocity without introducing speed-dependent dynamics.
    #
    # Passive MJC attributes retain SI units; DriveAPI gains below use
    # the USD per-degree convention independently.
    _UNIFORM_ARMATURE = 0.03
    _UNIFORM_FRICTIONLOSS = 0.3

    # Sub-class armature is unified to a single value matching the x2
    # convention. Names are kept as aliases for downstream code that
    # imports them by name.
    _HAND_ARMATURE = _UNIFORM_ARMATURE
    _ARM_SHOULDER_ARMATURE = _UNIFORM_ARMATURE
    _ARM_MID_ARMATURE = _UNIFORM_ARMATURE
    _ARM_WRIST_ARMATURE = _UNIFORM_ARMATURE

    # Reference G2 MJCF authors gainprm="5" on the gripper master.
    # PhysX/Newton convention: drive stiffness in USD is per-DEGREE,
    # so write 5 × π/180 to land at joint_target_ke = 5 in Newton.
    _HAND_MASTER_FINAL_KP = 5.0
    _HAND_MASTER_FINAL_KD = 0.0
    _HAND_MASTER_USD_STIFFNESS = _HAND_MASTER_FINAL_KP * _DEG_TO_RAD
    _HAND_MASTER_USD_DAMPING = _HAND_MASTER_FINAL_KD * _DEG_TO_RAD

    # Joint-level viscous damping (``dof_damping`` in MuJoCo terms).
    # The reference G2 MJCF authors ``damping=0.05`` on every hand
    # joint via ``<default class="hand_joint">``; the URDF authors
    # ``<dynamics damping="0">`` so Newton's importer leaves
    # ``joint_damping=0`` for the gripper.  That means the ONLY
    # source of energy dissipation in the gripper sub-chain is the
    # PD damping on the master (``kd``), which is force-capped at
    # the URDF effort limit (5 N·m).
    #
    # Under any external excitation — typically the arm moving and
    # propagating a reaction torque through the arm-wrist-gripper
    # kinematic chain — the master can't dissipate energy fast
    # enough, the equality constraint propagates the master's motion
    # to followers (which themselves have ``kp=kd=0``), and the
    # whole gripper rings out at the excitation frequency.  Verified
    # in pure CPU-MuJoCo via ``test_robot_xml_dynamic.py
    # --ctrl-mode sweep --sweep-actuator position:idx22_arm_l_joint2
    # --sweep-amp 0.3``: a 0.3 rad arm sweep at 0.5 Hz drives the
    # gripper followers to ±1.7 rad amplitude (1700× amplification).
    #
    # The fix is to author joint-level viscous damping (``mjc:damping``,
    # which SchemaResolverMjc imports as SI passive joint damping).
    # Reference's 0.05 value was tuned against the
    # G2's hand-joint inertia and is adopted verbatim — independent
    # from the actuator force cap, so it composes cleanly with PD /
    # effort-limit tuning.
    #
    # The selected Newton SDK consumes mjc:damping in SI units.
    _HAND_JOINT_DAMPING_USD = 0.05

    _MASTER_SUFFIX = "inner_joint1"

    def _classify(name: str):
        """Return (class, armature_or_None).  ``None`` armature means
        leave bare (only the drive treatment applies).  ``skip`` class
        means don't author anything — chassis stays in
        ``_configure_drives`` (physics_params.yaml-driven runtime tuning) and
        body / head / non-arm joints don't need armature per the
        reference MJCF."""
        if "gripper" in name:
            if name.endswith(_MASTER_SUFFIX):
                return ("hand_master", _HAND_ARMATURE)
            return ("hand_follower", _HAND_ARMATURE)
        if "arm_" in name and "_joint" in name:
            try:
                k = int(name.rsplit("_joint", 1)[-1])
            except ValueError:
                return ("skip", None)
            if k in (1, 2):
                return ("arm_shoulder", _ARM_SHOULDER_ARMATURE)
            if k in (3, 4, 5):
                return ("arm_mid", _ARM_MID_ARMATURE)
            if k in (6, 7):
                return ("arm_wrist", _ARM_WRIST_ARMATURE)
        return ("skip", None)

    from .physics_variant import select_physics_variant, validate_urdf_joints

    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        if logger is not None:
            logger.warn(f"[joint-overlay] failed to open {usd_path}")
        return {}
    select_physics_variant(stage, logger=logger)
    if urdf_path is None:
        candidate = os.path.join(os.path.dirname(usd_path), "robot_raw.urdf")
        if os.path.isfile(candidate):
            urdf_path = candidate
    if urdf_path and os.path.isfile(urdf_path):
        validate_urdf_joints(stage, urdf_path)

    counts: dict = {}
    n_stale_drive_cleared = 0
    n_mimic_api_applied = 0
    n_uniform_armature_set = 0
    n_uniform_friction_set = 0

    for prim in stage.Traverse():
        if not (prim.IsA(UsdPhysics.Joint) or prim.HasAPI(UsdPhysics.Joint)):
            continue
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        name = prim.GetName()
        klass, armature = _classify(name)

        # CLEANUP — drive:angular:* on non-gripper joints conflicts with
        # the physics_params.yaml + ``_configure_drives`` runtime tuning
        # loop the PhysX layer relies on.  Walk every revolute joint
        # (including ``skip``-classified ones like body / head /
        # chassis) and remove any drive:angular override that isn't a
        # gripper joint, so the runtime layer fully owns those values.
        if klass != "hand_master" and klass != "hand_follower":
            for attr_name in (
                "drive:angular:physics:stiffness",
                "drive:angular:physics:damping",
            ):
                attr = prim.GetAttribute(attr_name)
                if attr and attr.IsValid() and attr.HasAuthoredValue():
                    # ``Clear`` removes the authored opinion in the
                    # current edit target (robot.usda's root layer).
                    # The underlying payload's value (or absence) is
                    # restored, which is what the runtime layer needs
                    # so it can author from yaml.
                    attr.Clear()
                    n_stale_drive_cleared += 1

        # Universal passive-joint baseline — author on EVERY revolute
        # joint regardless of class.  Two attributes:
        #
        #   ``physxJoint:armature``  — reflected rotor inertia
        #                              (Newton + PhysX both read this)
        #   ``mjc:frictionloss``     — stick-slip Coulomb-style friction
        #                              floor (Newton's mjwarp converter
        #                              reads it; PhysX ignores it)
        #
        # x2_31dof_hand convention: every joint carries the SAME
        # passive tuning.  The class-specific block below only owns
        # drive (DriveAPI) and mimic-schema authoring; armature +
        # frictionloss are global to keep the model uniform.
        #
        # The authoring runs BEFORE the ``klass == "skip"`` early-out
        # so body / head / chassis joints (which return ``skip``) ALSO
        # get the universal values — that's the whole point of
        # "uniform across all joints".
        arm_attr = prim.GetAttribute("physxJoint:armature")
        if not arm_attr or not arm_attr.IsValid():
            arm_attr = prim.CreateAttribute("physxJoint:armature", Sdf.ValueTypeNames.Float)
        arm_attr.Set(_UNIFORM_ARMATURE)
        n_uniform_armature_set += 1

        # frictionloss has NO π/180 quirk (verified in
        # ``MuJoCoWarpAdapter._per_class_passive_joint`` line 828).
        # Author the physical value verbatim.
        fl_attr = prim.GetAttribute("mjc:frictionloss")
        if not fl_attr or not fl_attr.IsValid():
            fl_attr = prim.CreateAttribute("mjc:frictionloss", Sdf.ValueTypeNames.Float)
        fl_attr.Set(_UNIFORM_FRICTIONLOSS)
        n_uniform_friction_set += 1

        if klass == "skip":
            continue
        counts[klass] = counts.get(klass, 0) + 1

        # Armature is authored universally above — the per-class
        # block below is a no-op since ``armature`` equals
        # ``_UNIFORM_ARMATURE``.  Kept so re-tiering armature per
        # class can drop in here without re-introducing the universal
        # authoring above.
        if armature is not None:
            attr = prim.GetAttribute("physxJoint:armature")
            if not attr or not attr.IsValid():
                attr = prim.CreateAttribute("physxJoint:armature", Sdf.ValueTypeNames.Float)
            attr.Set(armature)

        # Gripper-specific drive authoring.  Arm joints don't get a
        # drive override here — their kp/kd come from
        # ``physics_params.yaml:usd_drive_api.default_revolute`` (Kit path) or
        # ``MuJoCoWarpAdapter`` per-effort scaling (newton-standalone).
        if klass == "hand_master":
            drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
            drive.CreateStiffnessAttr().Set(_HAND_MASTER_USD_STIFFNESS)
            drive.CreateDampingAttr().Set(_HAND_MASTER_USD_DAMPING)
            # Defensive cleanup: clear any authored ``maxForce`` opinion
            # on the master so the URDF-derived ``forceLimit`` (≈5 N·m
            # via the importer's payload) is the value Newton's parser
            # reads.  USD overlay writes are sticky; this Clear() makes
            # the overlay deterministic — every run scrubs prior
            # opinions on this attribute.
            mf_attr = prim.GetAttribute("drive:angular:physics:maxForce")
            if mf_attr and mf_attr.IsValid() and mf_attr.HasAuthoredValue():
                mf_attr.Clear()
            # Joint-level viscous damping — see the
            # ``_HAND_JOINT_DAMPING_*`` block above for the full
            # rationale.  Master gets the same 0.05 N·m·s/rad as
            # the followers; the reference MJCF applies it
            # uniformly across the hand-joint class.
            dmp_attr = prim.GetAttribute("mjc:damping")
            if not dmp_attr or not dmp_attr.IsValid():
                dmp_attr = prim.CreateAttribute("mjc:damping", Sdf.ValueTypeNames.Float)
            dmp_attr.Set(_HAND_JOINT_DAMPING_USD)
        elif klass == "hand_follower":
            # Strip every DriveAPI attribute and the applied schema on
            # followers.  Why: Newton's USD→MJCF emitter creates one
            # ``<general>`` actuator per joint that has DriveAPI, even
            # when stiffness=damping=0 — those vestigial actuators
            # show up in ``robot_runtime.xml`` as
            # ``<general name="position_<follower>" gainprm="0" .../>``,
            # adding zero torque but consuming an actuator slot, taking
            # ``ctrl`` writes, and diverging from the hand-authored
            # reference G2 MJCF (which has no follower actuators at
            # all — the equality constraint alone propagates motion
            # from master to follower).  Static-diff tool flags this
            # as ``actuators only_in_a=10``.
            #
            # Authoring stiffness=damping=0 alone is insufficient
            # because the DriveAPI schema is still applied → Newton
            # still emits the actuator.  The schema must be removed
            # AND any authored attributes scrubbed (USD opinions are
            # sticky; re-running this overlay alone wouldn't undo
            # prior writes).
            #
            # Constraint-level mimic stays — see NewtonMimicAPI /
            # PhysxMimicJointAPI:angular below.
            for attr_name in (
                "drive:angular:physics:stiffness",
                "drive:angular:physics:damping",
                "drive:angular:physics:maxForce",
                "drive:angular:physics:targetPosition",
                "drive:angular:physics:targetVelocity",
                "drive:angular:physics:type",
            ):
                attr = prim.GetAttribute(attr_name)
                if attr and attr.IsValid() and attr.HasAuthoredValue():
                    attr.Clear()
                    n_stale_drive_cleared += 1
            applied_for_drive = prim.GetAppliedSchemas()
            if "PhysicsDriveAPI:angular" in applied_for_drive:
                prim.RemoveAppliedSchema("PhysicsDriveAPI:angular")
            # Joint-level viscous damping on the follower — critical
            # part of the gripper hold.  Followers are PD-free
            # (equality-constraint-only); without ``dof_damping``
            # the follower has ZERO dissipation against an external
            # perturbation.  When the master swings under arm
            # reaction torque, the equality constraint propagates
            # the motion straight through to the follower with no
            # damping anywhere in the loop → resonance amplification.
            dmp_attr = prim.GetAttribute("mjc:damping")
            if not dmp_attr or not dmp_attr.IsValid():
                dmp_attr = prim.CreateAttribute("mjc:damping", Sdf.ValueTypeNames.Float)
            dmp_attr.Set(_HAND_JOINT_DAMPING_USD)
            # Apply the per-engine mimic API instances on the follower
            # so the importer installs a CONSTRAINT-LEVEL equality at
            # finalize, instead of treating the ``newton:mimicJoint``
            # rel + ``newton:mimicCoef1/0`` attributes as inert
            # metadata.  The URDF→USD converter authors the rel +
            # coefs but does NOT apply the schema, which leaves
            # ``isaac_newton`` relying on the software-broadcast in
            # ``engine/_mimic.py:parse_mimic`` — that path runs only
            # when the wrapper's ``apply_action`` reaches it, and
            # whenever the controller writes directly to the tensor
            # handle the followers fall back to whatever target_pos
            # was last written → finger weight pulls them open →
            # gripper swings while the master holds.  Authoring the
            # schema here is the assemble-side fix surfaced by
            # ``test_assemble_robot.py --only-stage gripper``.
            #
            # ``AddAppliedSchema`` writes the token to the layer
            # without requiring the schema to be registered in the
            # local pxr build — Newton's runtime importer (which DOES
            # register ``NewtonMimicAPI``) reads it correctly.
            # ``PhysxMimicJointAPI:angular`` is added too for
            # isaac_physx parity.
            applied_now = prim.GetAppliedSchemas()
            if "NewtonMimicAPI" not in applied_now:
                if prim.AddAppliedSchema("NewtonMimicAPI"):
                    n_mimic_api_applied += 1
            if not any(s.startswith("PhysxMimicJointAPI:") for s in applied_now):
                prim.AddAppliedSchema("PhysxMimicJointAPI:angular")

    # Author the engine-facing gripper topology annotations
    # (``newton:isGripperPad`` on pad bodies, ``newton:isLoopCoupler`` on
    # 4-bar coupler joints) derived from the source URDF's <mimic> graph +
    # link topology.  Pure-add: it only *sets* two new boolean custom
    # attributes and never touches the mimic / armature / drive authoring
    # above.  The engine reads these instead of hard-coding G2 name
    # patterns; absent a URDF (or a URDF with no gripper) it is a no-op and
    # the engine falls back to its legacy literals.  See
    # ``_derive_gripper_annotations`` for the physical-truth derivation.
    if urdf_path is None:
        # Default to robot_raw.urdf staged alongside robot.usda (the
        # complete <mimic> graph lives in the RAW urdf — the converted
        # robot.usda keeps only 2 of 14 mimic edges).
        cand = os.path.join(os.path.dirname(usd_path), "robot_raw.urdf")
        if os.path.isfile(cand):
            urdf_path = cand
    annot_counts = {"pads_authored": 0, "couplers_authored": 0}
    if urdf_path and os.path.isfile(urdf_path):
        try:
            with open(urdf_path) as _fh:
                _urdf_str = _fh.read()
            annot_counts = _author_gripper_annotations(stage, _urdf_str, logger=logger)
        except Exception as exc:  # noqa: BLE001
            if logger is not None:
                logger.warn(f"[gripper-annot] annotation authoring skipped ({exc}); engine falls back to literals.")
    elif logger is not None:
        logger.warn(
            "[gripper-annot] no robot_raw.urdf next to robot.usda and no --urdf; "
            "isGripperPad / isLoopCoupler annotations NOT authored (engine falls back to literals)."
        )

    stage.GetRootLayer().Save()

    if logger is not None and (
        counts or n_stale_drive_cleared or n_mimic_api_applied or n_uniform_armature_set or n_uniform_friction_set
    ):
        cleared_str = (
            f"; cleared {n_stale_drive_cleared} stale drive:angular override(s) on "
            f"non-gripper joints (physics_params.yaml owns those at runtime)"
            if n_stale_drive_cleared > 0
            else ""
        )
        mimic_api_str = (
            f"; applied NewtonMimicAPI + PhysxMimicJointAPI:angular on "
            f"{n_mimic_api_applied} follower(s) (constraint-level mimic)"
            if n_mimic_api_applied > 0
            else ""
        )
        logger.info(
            f"[joint-overlay] universal passive baseline (x2 convention): "
            f"physxJoint:armature={_UNIFORM_ARMATURE} on {n_uniform_armature_set} joint(s), "
            f"mjc:frictionloss={_UNIFORM_FRICTIONLOSS} on {n_uniform_friction_set} joint(s); "
            f"per-class drive/mimic authoring: "
            f"hand_master={counts.get('hand_master', 0)}, "
            f"hand_follower={counts.get('hand_follower', 0)}, "
            f"arm_shoulder={counts.get('arm_shoulder', 0)}, "
            f"arm_mid={counts.get('arm_mid', 0)}, "
            f"arm_wrist={counts.get('arm_wrist', 0)}; "
            f"gripper masters got drive stiffness={_HAND_MASTER_FINAL_KP}/"
            f"damping={_HAND_MASTER_FINAL_KD}; gripper followers had DriveAPI "
            f"stripped (constraint-only — equality propagates motion from "
            f"master to follower; no vestigial follower actuators "
            f"in the MJCF dump)"
            f"{mimic_api_str}"
            f"{cleared_str}.  Per-joint kp/kd for arm/body/head/chassis stays "
            f"runtime-tuned via config/physics_params.yaml + kit/stage.py:"
            f"_configure_drives (Kit path) or MuJoCoWarpAdapter per-effort "
            f"scaling (newton-standalone)."
        )
    counts["isGripperPad_authored"] = annot_counts["pads_authored"]
    counts["isLoopCoupler_authored"] = annot_counts["couplers_authored"]
    return counts


def _apply_post_transformer_material_overrides(
    usd_path: str,
    overrides: list[dict],
    logger=None,
) -> int:
    """Run :func:`_apply_material_overrides` against the FINAL ``robot.usda``.

    Mirrors :func:`_apply_post_transformer_collision_policy`'s placement
    and rationale: editing the in-memory intermediate stage that
    ``urdf_usd_converter`` produced silently breaks the AS3 transformer's
    shape-discovery walk — the transformer returns "success" but
    ``robot.usda`` never gets written, so the launch fails with the
    cryptic ``[assemble_robot] ERROR: importer did not produce USD …``
    and no other stderr.  PBR overrides authored against the final
    ``robot.usda`` land in the AS3 root layer over the materials
    payload and compose at load time for both runtimes (PhysX via
    IsaacSimStage and Newton via ``add_usd``), so the override is
    effective everywhere without re-running the transformer.

    The actual material targeting and ``UsdShade`` writes are identical
    to the intermediate-stage version — only the stage opened differs.
    The robot prefix is derived from the stage's ``defaultPrim`` (always
    ``robot`` for the urdf_usd_converter output today, but reading it
    keeps this robust to renames at the converter layer).
    """
    if not overrides:
        return 0

    import gc as _gc

    from pxr import Sdf, Usd  # noqa: E402

    # ``Usd.Stage.Open`` with default load policy does NOT pull the
    # geometries / materials payloads that the AS3 transformer split
    # the asset into.  Without those, the link's visual wrapper has no
    # children and ``_collect_wrapper_materials`` finds zero
    # ``MaterialBindingAPI`` prims to patch — the override silently
    # no-ops despite parsing 16 blocks.  ``LoadAll`` brings every
    # payload into composition so the wrapper subtree (and its
    # bound materials in ``payloads/materials.usda``) is reachable
    # from ``Usd.PrimRange``.  Edits to those material prims still
    # land in ``robot.usda``'s root layer as overrides — the payloads
    # themselves stay untouched.
    from .physics_variant import select_physics_variant

    stage = Usd.Stage.Open(usd_path, load=Usd.Stage.LoadAll)
    if stage is None:
        if logger is not None:
            logger.warn(f"[material_override] post-transformer: could not open {usd_path}; " f"overrides not applied.")
        return 0

    select_physics_variant(stage, logger=logger)

    default_prim = stage.GetDefaultPrim()
    if default_prim is None or not default_prim.IsValid():
        if logger is not None:
            logger.warn(
                f"[material_override] post-transformer: {usd_path} has no defaultPrim; " f"overrides not applied."
            )
        # Still need to drop stage refs even on early-return — see
        # the cleanup section below.
        stage = None
        _gc.collect()
        return 0

    robot_prefix = default_prim.GetName()

    n = _apply_material_overrides(
        stage,
        overrides,
        robot_prefix,
        logger=logger,
    )
    if n:
        stage.GetRootLayer().Save()
    if logger is not None:
        logger.info(
            f"[material_override] patched {n} material(s) from "
            f"{len(overrides)} <material_override> block(s) (post-transformer)"
        )

    # CRITICAL hygiene: drop every reference this function held to
    # ``robot.usda`` and to the payloads loaded with
    # ``Usd.Stage.LoadAll`` so the next post-transformer step
    # (``_apply_mimic_joint_overlay``) opens a fresh stage and sees the
    # on-disk state, not a stale Sdf.Layer instance from the cache.
    #
    # Why it matters: ``Usd.Stage.Open`` registers each composed layer
    # with ``Sdf.Layer.FindOrOpen``'s process-global cache.  As long as
    # ANY Python object (incl. the ``stage`` variable, ``layer_stages``
    # in ``_apply_material_overrides``, or a stage built off a layer)
    # holds a reference, that ``Sdf.Layer`` instance stays cached.  The
    # next ``Usd.Stage.Open(usd_path)`` returns a stage backed by the
    # SAME layer object — including any in-memory edits not re-read
    # from disk.  The chain that motivates this cleanup is:
    #
    #   1. ``_apply_post_transformer_collision_policy``: writes
    #      overrides to ``robot.usda`` root layer, saves.
    #   2. this function: opens with ``LoadAll``, possibly writes to
    #      ``materials.usda`` source layer, saves root.
    #   3. ``_apply_mimic_joint_overlay``: opens ``robot.usda``,
    #      authors ``NewtonMimicAPI`` via ``AddAppliedSchema``, saves.
    #
    # Without this cleanup, step 3's
    # ``AddAppliedSchema("NewtonMimicAPI")`` returns True but the
    # schema doesn't make it into the saved ``robot.usda``, while
    # ``PhysxMimicJointAPI:angular`` (added on the same prims with
    # the same code) DOES land.  The cached ``Sdf.Layer`` for
    # ``robot.usda`` carries a pending listOp delta on
    # ``apiSchemas`` that the schema register treats as inconsistent
    # with the unregistered ``NewtonMimicAPI`` token, so the token
    # is silently dropped on save.
    #
    # Dropping refs + gc.collect() forces ``Sdf.Layer`` cleanup; the
    # next Open re-reads from disk and the mimic overlay step sees a
    # clean slate.  Mirrors the cleanup ``_convert_urdf_to_usd_60`` does
    # after step 3 of the converter (``stage = None; gc.collect()``).
    del default_prim
    stage = None
    _gc.collect()
    return n


def _apply_post_transformer_collision_policy(usd_path: str, logger=None) -> dict:
    """Run the selective collision policy AGAINST THE FINAL ``robot.usda``.

    Editing the final ``robot.usda`` after the AS3 transformer avoids
    interaction with the transformer's shape-discovery walk:

      * AS3 has finished its profile-driven asset split; what's on
        disk is the canonical robot definition for every runtime.
      * Edits land in ``robot.usda``'s root layer as overrides; the
        underlying payloads (``geometries.usd``, ``instances.usda``)
        stay untouched, which is the intent — only the
        ``CollisionAPI`` / ``MeshCollisionAPI`` attribute values need
        to change per the rules.
      * Both runtimes (PhysX via IsaacSimStage, Newton via
        ``add_usd``) compose the root layer's overrides on top of
        the payloads at load time, so the policy is effective for
        every engine.

    The actual decisions (gripper SDF / wheel convexHull / non-contact
    disabled) live in the per-link branches below.
    """
    from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

    counts = {
        "gripper_authored": 0,
        "gripper_visual_sdf": 0,
        "wheel_convex_hull": 0,
        "stripped": 0,
        "no_collision": 0,
        "skipped_no_visual": 0,
        "gripper_by_geometry": 0,
    }

    from genie_sim_engine_libuipc.assemble.physics_variant import select_physics_variant

    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        if logger is not None:
            logger.warn(f"[collision] post-transformer: could not open {usd_path}; " f"policy not applied.")
        return counts

    select_physics_variant(stage, logger=logger)

    # AS3 places collision meshes beneath instanceable geometry wrappers.
    # Default traversal hides their descendants, and USD forbids per-instance
    # collider overrides on proxies. Expand only geometry wrappers before the
    # existing policy writes; leave rigid-body and material instances intact.
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if not prim.IsInstance() or prim.IsInstanceProxy():
            continue
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        owner = prim.GetParent()
        while owner and owner.IsValid() and not owner.HasAPI(UsdPhysics.RigidBodyAPI):
            owner = owner.GetParent()
        if not owner or not owner.IsValid():
            continue
        descendants = list(Usd.PrimRange(prim, Usd.TraverseInstanceProxies()))
        # A component containing bodies is not a geometry wrapper, even when
        # that component is itself nested beneath another rigid body.
        if any(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in descendants):
            continue
        if any(p.IsA(UsdGeom.Mesh) for p in descendants):
            prim.SetInstanceable(False)

    # Two-pass design: pass 1 maps each CollisionAPI prim to its
    # nearest RigidBodyAPI ancestor; pass 2 applies the per-link
    # decision.
    link_to_colliders: dict[Usd.Prim, list[Usd.Prim]] = {}
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        ancestor = prim.GetParent()
        while ancestor and ancestor.IsValid() and ancestor.GetPath().pathString not in ("/", ""):
            if ancestor.HasAPI(UsdPhysics.RigidBodyAPI):
                link_to_colliders.setdefault(ancestor, []).append(prim)
                break
            ancestor = ancestor.GetParent()

    def _geometry_says_gripper(link_prim: Usd.Prim, prims: list[Usd.Prim]) -> bool:
        """True when a link's own geometry names it as gripper hardware.

        AS3 keeps each vendor's mesh names on the geometry beneath the link,
        and a hand may name the LINK anatomically while the geometry keeps a
        gripper word. The Wuji hand authors ``hand_l_thumb_link4`` over a
        collider named ``left_finger1_link4``: classifying on the link name
        alone leaves every finger in the non-contact branch, so its colliders
        are disabled and the hand cannot touch anything.

        Testing the geometry under the link uses the same family classifier
        instead of adding per-vendor name literals, which is the rule
        ``_derive_gripper_annotations`` already states.
        """
        base = link_prim.GetPath().pathString
        for geom in prims:
            tail = geom.GetPath().pathString[len(base) :]
            if _COLL_GRIPPER_RE.search(tail):
                return True
        return False

    def _direct_visuals(link_prim: Usd.Prim) -> list[Usd.Prim]:
        out: list[Usd.Prim] = []
        stack = list(link_prim.GetChildren())
        while stack:
            p = stack.pop()
            if p.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            if p.IsA(UsdGeom.Mesh) and not p.HasAPI(UsdPhysics.CollisionAPI):
                out.append(p)
            stack.extend(p.GetChildren())
        return out

    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        link_name = prim.GetName()
        coll_prims = link_to_colliders.get(prim, [])
        is_gripper = bool(_COLL_GRIPPER_RE.search(link_name))
        if not is_gripper:
            # Fall back to the geometry names under the link — see
            # ``_geometry_says_gripper``. Checked only when the link name
            # misses, so the common path costs nothing.
            if _geometry_says_gripper(prim, coll_prims) or _geometry_says_gripper(prim, _direct_visuals(prim)):
                is_gripper = True
                counts["gripper_by_geometry"] += 1
        is_wheel = bool(_COLL_WHEEL_RE.search(link_name))
        is_contact = is_gripper or is_wheel

        if not is_contact:
            if coll_prims:
                disabled = 0
                for d in coll_prims:
                    try:
                        UsdPhysics.CollisionAPI(d).CreateCollisionEnabledAttr().Set(False)
                        disabled += 1
                    except Exception as exc:
                        if logger is not None:
                            logger.warn(
                                f"[collision] {link_name!r}: could not disable "
                                f"collider {d.GetPath().pathString!r}: {exc}"
                            )
                counts["stripped"] += 1
                if logger is not None:
                    logger.info(
                        f"[collision] {link_name!r}: disabled {disabled}/{len(coll_prims)} "
                        f"collider(s) (non-contact link)"
                    )
            else:
                counts["no_collision"] += 1
            continue

        if coll_prims:
            approx = "sdf" if is_gripper else UsdPhysics.Tokens.convexHull
            for d in coll_prims:
                if d.IsA(UsdGeom.Mesh):
                    if not d.HasAPI(UsdPhysics.MeshCollisionAPI):
                        UsdPhysics.MeshCollisionAPI.Apply(d)
                    UsdPhysics.MeshCollisionAPI(d).CreateApproximationAttr().Set(approx)
            if is_gripper:
                counts["gripper_authored"] += 1
            else:
                counts["wheel_convex_hull"] += 1
            continue

        if is_gripper:
            visual_meshes = _direct_visuals(prim)
            if not visual_meshes:
                if logger is not None:
                    logger.warn(
                        f"[collision] gripper link {link_name!r}: no <collision> "
                        f"and no visual mesh; left without collider"
                    )
                counts["skipped_no_visual"] += 1
                continue
            for d in visual_meshes:
                UsdPhysics.CollisionAPI.Apply(d)
                if not d.HasAPI(UsdPhysics.MeshCollisionAPI):
                    UsdPhysics.MeshCollisionAPI.Apply(d)
                UsdPhysics.MeshCollisionAPI(d).CreateApproximationAttr().Set("sdf")
            counts["gripper_visual_sdf"] += 1
            continue

        counts["no_collision"] += 1

    # Save overrides to the root layer of robot.usda.
    try:
        stage.GetRootLayer().Save()
    except Exception as exc:
        if logger is not None:
            logger.warn(f"[collision] post-transformer: could not save {usd_path}: {exc}")
        return counts

    if logger is not None:
        logger.info(
            f"[collision] post-transformer policy applied: "
            f"gripper-authored(SDF)={counts['gripper_authored']}, "
            f"gripper-visual(SDF)={counts['gripper_visual_sdf']}, "
            f"wheel(convexHull)={counts['wheel_convex_hull']}, "
            f"stripped(non-contact)={counts['stripped']}, "
            f"no-collision={counts['no_collision']}, "
            f"skipped-no-visual={counts['skipped_no_visual']}, "
            f"gripper-by-geometry={counts['gripper_by_geometry']}"
        )
    return counts


def _convert_urdf_to_usd_60(
    urdf_path: str,
    usd_path: str,
    *,
    ros_packages: dict[str, str] | None = None,
    raw_urdf_path: str | None = None,
    config=None,  # URDFImporterConfig (6.2 API)
) -> None:
    """Isaac Sim 6.2+: URDF → Asset Structure 3.0 pipeline.

    Pipeline:
      1. FBX pre-conversion   FBX meshes → DAE via Blender (urdf_usd_converter
                              only supports STL/OBJ/DAE)
      2. urdf_usd_converter   URDF → flat intermediate USDC
      3. Schema post-processing  rigid body / joint / MJC+PhysX attrs
         (parameters from URDFImporterConfig)
      4. Asset transformer    intermediate → Asset Structure 3.0 package

    Output written into dirname(usd_path):
      robot.usda          ← root interface
      payloads/
        base.usda / geometries.usd / materials.usda / instances.usda / robot.usda
        Physics/  physx.usda / mujoco.usda / physics.usda
    """
    import gc
    import importlib
    import pathlib
    import shutil
    import tempfile

    pip_prebundle = (
        "/usr/local/lib/python3.12/dist-packages/isaacsim/exts/" "isaacsim.asset.importer.urdf/pip_prebundle"
    )
    if pip_prebundle not in sys.path:
        sys.path.append(pip_prebundle)

    urdf_usd_converter = importlib.import_module("urdf_usd_converter")

    from isaacsim.asset.importer.utils.impl import (  # noqa: E402
        importer_utils,
        stage_utils,
        urdf_to_mjc_physx_conversion_utils,
    )
    import isaacsim.asset.transformer.rules as _rules_mod  # noqa: E402

    ros_pkgs = [{"name": k, "path": v} for k, v in (ros_packages or {}).items()]
    source_urdf = raw_urdf_path or urdf_path
    robot_name = pathlib.Path(source_urdf).stem

    tmp_dir = tempfile.mkdtemp(prefix="assemble_robot_usd_")
    try:
        usdex_dir = os.path.join(tmp_dir, "usdex")
        intermediate_path = os.path.join(tmp_dir, "temp", f"{robot_name}.usd")
        output_package_root = os.path.join(tmp_dir, "pkg", robot_name)

        # Step 1: load URDF.  OBJ object-name normalization and
        # inertial-on-empty-link injection are NOT performed here at
        # runtime — both are offline mesh-developer steps owned by the
        # upstream robot-model package (the one named by
        # ``robot_source.package``):
        #   * scripts/normalize_obj_names.py
        #   * scripts/diagnose_urdf.py  ← inertial fix-up; runs against xacro
        # See that package's AGENTS.md for the workflow. Whatever
        # the URDF references is loaded as-is.
        with open(source_urdf) as fh:
            urdf_content = fh.read()

        # Parse inline <visual><material_override> blocks NOW (before the
        # converter sees the URDF). They patch metallic/roughness on the
        # enclosing visual's embedded material(s) in step 3. The converter
        # tolerates the unknown element (parses it as ElementUndefined), but
        # we strip the blocks from the URDF we feed it so the converted
        # asset carries no stray undefined elements. The strip regex
        # REQUIRES <roughness>/<metallic> children so it only matches real
        # element blocks — never the literal "<material_override>" token
        # that may appear in prose inside an XML comment.
        material_overrides = _parse_material_override_blocks(urdf_content)
        urdf_content = re.sub(
            r"[ \t]*<material_override\b[^>]*>"
            r"(?:\s*<(?:roughness|metallic)>[^<]*</(?:roughness|metallic)>)+"
            r"\s*</material_override>\s*",
            "",
            urdf_content,
            flags=re.DOTALL,
        )

        source_urdf = os.path.join(tmp_dir, "robot_preconverted.urdf")
        with open(source_urdf, "w") as fh:
            fh.write(urdf_content)

        # Step 2: raw conversion
        converter = urdf_usd_converter.Converter(layer_structure=False, scene=False, ros_packages=ros_pkgs)
        try:
            asset = converter.convert(source_urdf, usdex_dir)
        except Exception as exc:
            print(f"[assemble_robot] ERROR: urdf_usd_converter.Converter failed: {exc}", file=sys.stderr)
            sys.exit(1)

        # Step 3: schema post-processing (URDFImporterConfig parameters)
        allow_self_collision = config.allow_self_collision if config else False
        collision_from_visuals = config.collision_from_visuals if config else False
        collision_type = config.collision_type if config else "Convex Hull"
        merge_mesh = config.merge_mesh if config else False

        stage = stage_utils.open_stage(asset.path)
        importer_utils.remove_custom_scopes(stage)
        importer_utils.add_rigid_body_schemas(stage)
        importer_utils.add_joint_schemas(stage)
        importer_utils.enable_self_collision(stage, allow_self_collision)
        if collision_from_visuals:
            importer_utils.collision_from_visuals(stage, collision_type)
        urdf_to_mjc_physx_conversion_utils.convert_joints_attributes(stage)

        # Apply the default selective collision policy: keep gripper SDF +
        # wheel cylinder, strip everything else. See
        # ``_apply_post_transformer_collision_policy`` for the rationale —
        # that helper carries the same decision matrix authored against
        # the final ``robot.usda`` rather than the intermediate stage.
        class _LocalLogger:
            def info(self, m):
                print(f"[assemble_robot] {m}", file=sys.stderr)

            def warn(self, m):
                print(f"[assemble_robot] WARN: {m}", file=sys.stderr)

        _local_logger = _LocalLogger()
        # The selective collision policy and material overrides are
        # applied POST-transformer, against the final robot.usda — see
        # ``_apply_post_transformer_collision_policy`` and
        # ``_apply_post_transformer_material_overrides`` below. Editing
        # the in-memory intermediate stage at this point breaks the AS3
        # transformer's shape-discovery walk (the transformer returns
        # success but robot.usda is never written, with diagnostic
        # ``[assemble_robot] ERROR: importer did not produce USD …``
        # and no other stderr).
        os.makedirs(os.path.dirname(intermediate_path), exist_ok=True)
        stage_utils.save_stage(stage, intermediate_path)
        stage = None
        gc.collect()

        # Step 4: asset transformer → Asset Structure 3.0
        profile_json = str(pathlib.Path(_rules_mod.__file__).parents[4] / "data" / "isaacsim_structure.json")
        if not os.path.isfile(profile_json):
            print(f"[assemble_robot] ERROR: transformer profile not found: {profile_json}", file=sys.stderr)
            sys.exit(1)
        os.makedirs(output_package_root, exist_ok=True)
        try:
            importer_utils.run_asset_transformer_profile(
                input_stage_path=intermediate_path,
                output_package_root=output_package_root,
                profile_json_path=profile_json,
            )
        except Exception as exc:
            print(f"[assemble_robot] ERROR: asset transformer failed: {exc}", file=sys.stderr)
            sys.exit(1)

        final_path = os.path.join(output_package_root, f"{robot_name}.usda")
        if not os.path.isfile(final_path):
            print(f"[assemble_robot] ERROR: transformer did not produce {final_path}", file=sys.stderr)
            sys.exit(1)

        usd_out_dir = os.path.dirname(usd_path)
        os.makedirs(usd_out_dir, exist_ok=True)
        for entry in os.listdir(output_package_root):
            src = os.path.join(output_package_root, entry)
            dst = os.path.join(usd_out_dir, entry)
            if os.path.isdir(src):
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
        imported_root = os.path.join(usd_out_dir, f"{robot_name}.usda")
        if imported_root != usd_path and os.path.isfile(imported_root):
            os.replace(imported_root, usd_path)

        # Apply the selective collision policy AGAINST THE FINAL
        # robot.usda. Overrides authored here land in the robot.usda
        # root layer (AS3 root) and compose over the payloads at load
        # time, so both PhysX (via IsaacSimStage) and Newton's
        # add_usd see the policy without re-running the transformer.
        if os.path.isfile(usd_path):
            from pxr import Usd
            from .physics_variant import select_physics_variant, validate_urdf_joints

            final_stage = Usd.Stage.Open(usd_path)
            if final_stage is None:
                raise RuntimeError(f"[assemble_robot] cannot open converted robot {usd_path}")
            select_physics_variant(final_stage, logger=_local_logger)
            validate_urdf_joints(final_stage, raw_urdf_path or urdf_path)
            final_stage.GetRootLayer().Save()
            final_stage = None
            gc.collect()
            _apply_post_transformer_collision_policy(usd_path, logger=_local_logger)
            # Apply parsed <visual><material_override> blocks against
            # the FINAL robot.usda (root layer overrides over the
            # materials payload).
            if material_overrides:
                _apply_post_transformer_material_overrides(
                    usd_path,
                    material_overrides,
                    logger=_local_logger,
                )
            # Author hand-joint armature + master drive gains directly
            # in robot.usda's root layer so Newton's add_usd reads them
            # natively via SchemaResolverPhysx + UsdPhysics.DriveAPI.
            # See ``_apply_mimic_joint_overlay`` for the full schema /
            # provenance commentary.  Cheap (~12 attrs per gripper),
            # idempotent.  Passes the pristine raw URDF so the overlay can
            # derive the isGripperPad / isLoopCoupler annotations from the
            # complete <mimic> graph (the converted robot.usda keeps only a
            # subset of the mimic edges); falls back to the robot_raw.urdf
            # staged next to robot.usda when raw_urdf_path is unset.
            _apply_mimic_joint_overlay(usd_path, logger=_local_logger, urdf_path=raw_urdf_path)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _convert_urdf_to_usd_4x5x(
    simulation_app,
    urdf_path: str,
    usd_path: str,
) -> None:
    """Isaac Sim 4.x / 5.x: URDFParseAndImportFile kit command.

    Uses urdf_path with package:// URIs rewritten to relative paths.
    Requires SimulationApp (kit command + omni.usd stage context).

    ``fix_base`` is intentionally hard-coded to ``True`` here. The importer
    authors a PhysicsFixedJoint weld from the world to the URDF root link;
    the *runtime* (``runtime.stage._apply_fix_base_policy``) decides per
    scene whether to enable or disable that joint based on the manifest's
    ``fix_base`` flag. This keeps ``robot.usda`` scene-agnostic — the same
    cached USD serves both fixed-base and floating-base scenes.
    """
    import omni.kit.commands  # noqa: E402

    try:
        from isaacsim.asset.importer.urdf import _urdf  # type: ignore  # noqa: E402
    except ImportError:
        from omni.importer.urdf import _urdf  # type: ignore  # noqa: E402

    import_config = _urdf.ImportConfig()
    import_config.merge_fixed_joints = False
    import_config.fix_base = True  # see docstring — runtime toggles, not bake time
    import_config.import_inertia_tensor = True
    import_config.self_collision = False
    import_config.distance_scale = 1.0
    import_config.density = 0.0
    import_config.default_drive_strength = 1e7
    import_config.default_position_drive_damping = 1e5
    import_config.convex_decomp = True
    import_config.make_default_prim = True
    import_config.create_physics_scene = False

    result, _stage_path = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=urdf_path,
        import_config=import_config,
        dest_path=usd_path,
    )
    if not result:
        print(
            f"[assemble_robot] ERROR: URDFParseAndImportFile failed for {urdf_path}",
            file=sys.stderr,
        )
        sys.exit(1)
    for _ in range(10):
        simulation_app.update()
    import omni.usd  # noqa: E402

    usd_context = omni.usd.get_context()
    if not usd_context.open_stage(usd_path):
        print(f"[assemble_robot] ERROR: failed to open imported USD: {usd_path}", file=sys.stderr)
        sys.exit(1)
    for _ in range(5):
        simulation_app.update()
    if usd_context.save_stage() is False:
        print(f"[assemble_robot] ERROR: save_stage() failed for {usd_path}", file=sys.stderr)
        sys.exit(1)
    for _ in range(5):
        simulation_app.update()


def _convert_urdf_to_usd(
    urdf_path: str,
    usd_path: str,
    *,
    ros_packages: dict[str, str] | None = None,
    raw_urdf_path: str | None = None,
    success_stamp: tuple[str, str] | None = None,
) -> None:
    """Detect Isaac Sim version and dispatch to the appropriate importer branch.

    Both branches author the URDF root link as welded to the world via a
    PhysicsFixedJoint named ``root_joint`` (6.2) or the autogenerated 4.x/5.x
    equivalent. The *runtime* (``runtime.stage._apply_fix_base_policy``)
    decides per scene whether that joint is active, based on the manifest's
    ``fix_base`` flag. The cached ``robot.usda`` is therefore scene-agnostic.

    ``success_stamp`` is an optional ``(path, content)`` pair written iff the
    importer produces a valid USD. The launch's ``make_assemble_pipeline``
    reads ``<stage_dir>/urdf.sha256`` (the typical stamp content is
    ``sha256(urdf_text)``) to skip this script on subsequent runs when the
    URDF input hasn't changed.
    """
    import importlib.metadata

    try:
        _isaacsim_ver = tuple(int(x) for x in importlib.metadata.version("isaacsim").split(".")[:2])
    except Exception:
        _isaacsim_ver = (0, 0)

    from isaacsim import SimulationApp  # noqa: E402

    simulation_app = SimulationApp({"headless": True, "fast_shutdown": False})

    converted = False
    try:
        if _isaacsim_ver >= (6, 0):
            # 6.2 param injection: URDFImporterConfig (isaacsim.asset.importer.urdf API)
            from isaacsim.asset.importer.urdf import URDFImporterConfig  # noqa: E402

            config_60 = URDFImporterConfig()
            config_60.allow_self_collision = False
            config_60.collision_from_visuals = False
            config_60.collision_type = "Convex Hull"
            config_60.merge_mesh = False
            config_60.ros_package_paths = [{"name": k, "path": v} for k, v in (ros_packages or {}).items()]
            _convert_urdf_to_usd_60(
                urdf_path,
                usd_path,
                ros_packages=ros_packages,
                raw_urdf_path=raw_urdf_path,
                config=config_60,
            )
        else:
            # 4.x/5.x param injection: _urdf.ImportConfig (kit-command API)
            _convert_urdf_to_usd_4x5x(simulation_app, urdf_path, usd_path)
        converted = True
    finally:
        # Validate before exit — skip kit teardown (hangs/crashes) and exit
        # directly so ros2 launch sees the correct return code.
        if not converted:
            print(f"[assemble_robot] ERROR: conversion or post-processing failed: {sys.exc_info()[1]}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
            os._exit(1)
        if not os.path.isfile(usd_path):
            print(
                f"[assemble_robot] ERROR: importer did not produce USD at {usd_path}",
                file=sys.stderr,
            )
            os._exit(1)
        try:
            size = os.path.getsize(usd_path)
        except OSError:
            size = 0
        if size < 256:
            print(
                f"[assemble_robot] ERROR: produced USD looks empty ({size} bytes): {usd_path}",
                file=sys.stderr,
            )
            os._exit(1)
        if success_stamp is not None:
            stamp_path, stamp_body = success_stamp
            try:
                with open(stamp_path, "w") as fh:
                    fh.write(stamp_body)
                    if not stamp_body.endswith("\n"):
                        fh.write("\n")
            except OSError as exc:
                print(
                    f"[assemble_robot] WARN: failed to write success stamp " f"{stamp_path}: {exc}",
                    file=sys.stderr,
                )
        os._exit(0)


def assemble_robot(
    scene: str,
    output_dir: str = "/tmp/isaacsim_stage",
    urdf: str | None = None,
    robot_description_timeout_s: float = 30.0,
) -> None:
    cfg_path, config = _load_scene_yaml(scene)

    robot_section = config.get("robot")
    if not isinstance(robot_section, dict):
        print(f"[assemble_robot] ERROR: scene YAML missing 'robot' mapping: {cfg_path}", file=sys.stderr)
        sys.exit(1)

    source = robot_section.get("robot_source") or {}
    if not isinstance(source, dict):
        print(f"[assemble_robot] ERROR: 'robot.robot_source' must be a mapping: {cfg_path}", file=sys.stderr)
        sys.exit(1)

    # Gate: PRESENCE of the ``robot.robot_source.urdf`` key is the SOLE
    # signal that this scene wants URDF->USD conversion. Absent key ->
    # pre-baked USD route, no-op. The nested fields (xacro_entry,
    # robot_model, arm, gripper, ...) are consumed by the launch
    # composer's ``build_robot_description`` helper, which publishes
    # the resulting URDF on ``/robot_description``; this script only
    # needs the presence flag.
    if "urdf" not in source:
        print(
            f"[assemble_robot] robot.robot_source.urdf is absent; "
            f"scene targets pre-baked assets/robot/<robot_name>/robot.usda. No-op."
        )
        return

    # ``fix_base`` is read but NOT passed to the URDF→USD converter. The
    # converter always authors a fixed weld from world to the URDF root link
    # so the cached robot.usda can serve both fixed-base and floating-base
    # scenes. The decision of whether that joint is active happens at runtime
    # inside ``runtime.stage._apply_fix_base_policy``, driven by the
    # ``fix_base`` field in manifest.json (written by the isaac_scene stage).
    fix_base_raw = source.get("fix_base", True)
    if isinstance(fix_base_raw, bool):
        fix_base = fix_base_raw
    elif isinstance(fix_base_raw, str):
        fix_base = fix_base_raw.strip().lower() not in {"false", "0", "no", "off"}
    else:
        fix_base = bool(fix_base_raw)

    if urdf:
        urdf_file = os.path.abspath(urdf)
        if not os.path.isfile(urdf_file):
            print(f"[assemble_robot] ERROR: --urdf file not found: {urdf_file}", file=sys.stderr)
            sys.exit(1)
        with open(urdf_file) as fh:
            urdf_str = fh.read()
        urdf_origin = f"--urdf {urdf_file}"
    else:
        urdf_str = _fetch_urdf_from_topic(timeout_s=robot_description_timeout_s)
        urdf_origin = "/robot_description"

    os.makedirs(output_dir, exist_ok=True)
    urdf_out, raw_urdf_out, ros_pkgs = _stage_urdf_from_string(urdf_str, dest_dir=output_dir)
    usd_out = os.path.join(output_dir, "robot.usda")

    urdf_sha256 = hashlib.sha256(urdf_str.encode("utf-8")).hexdigest()
    stamp_path = os.path.join(output_dir, "urdf.sha256")

    print(f"[assemble_robot] scene:        {cfg_path}")
    print(f"[assemble_robot] urdf source:  {urdf_origin}")
    print(f"[assemble_robot] urdf sha256:  {urdf_sha256}")
    print(f"[assemble_robot] fix_base:     {fix_base} (decided at runtime, not bake time)")
    print(f"[assemble_robot] urdf staged:  {urdf_out}")
    print(f"[assemble_robot] usd output:   {usd_out}")
    print()

    _convert_urdf_to_usd(
        urdf_out,
        usd_out,
        ros_packages=ros_pkgs,
        raw_urdf_path=raw_urdf_out,
        success_stamp=(stamp_path, urdf_sha256),
    )

    print(f"[assemble_robot] robot USD written: {usd_out}")
    print("[assemble_robot] done!")


def main(argv: list[str] | None = None) -> None:
    """Subprocess entry — the argv :class:`UrdfToUsdStage` spawns with."""
    import argparse

    parser = argparse.ArgumentParser(prog="assemble_robot")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--output-dir", default="/tmp/isaacsim_stage")
    parser.add_argument("--base-path", default="")  # accepted for uniformity; unused
    parser.add_argument("--urdf", default=None)
    parser.add_argument("--robot-description-timeout", type=float, default=30.0)
    args = parser.parse_args(argv)
    assemble_robot(
        scene=args.scene,
        output_dir=args.output_dir,
        urdf=args.urdf,
        robot_description_timeout_s=args.robot_description_timeout,
    )


if __name__ == "__main__":
    main()

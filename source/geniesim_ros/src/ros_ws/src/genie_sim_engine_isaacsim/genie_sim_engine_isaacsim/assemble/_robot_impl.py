# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""URDF→USD assembler — impl module.

Exposes :func:`assemble_robot` for :class:`UrdfToUsdStage`. Isaac Sim / pxr / rclpy
imports remain gated inside the individual helper functions so
importing this module does NOT require a running Kit or a ROS node.

**Subprocess entry point:** this module also exposes
``python -m genie_sim_engine_isaacsim.assemble._robot_impl --scene … --output-dir …``
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
sys.stdout.reconfigure(line_buffering=True)  # flush each print immediately

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


def _isaacsim_version() -> tuple[int, int]:
    import importlib.metadata

    try:
        return tuple(int(x) for x in importlib.metadata.version("isaacsim").split(".")[:2])
    except Exception:
        return (0, 0)


def _check_assembly_output_permissions(output_dir: str) -> None:
    """Check the AS3 files we overwrite and directories we recursively replace.

    Isaac 6.2's transformer publishes a multi-file package. A writable scene
    directory alone is insufficient: deleting payloads needs write/search access
    inside every nested directory. Check before staging URDF or starting Kit.
    """
    import errno
    import stat
    from pathlib import Path

    output = Path(output_dir).absolute()
    ancestor = output
    while not ancestor.exists():
        ancestor = ancestor.parent

    def fail(path, reason):
        info = path.stat()
        return PermissionError(
            errno.EACCES,
            f"{reason}; effective UID={os.geteuid()}, owner={info.st_uid}:{info.st_gid}, "
            f"mode={stat.S_IMODE(info.st_mode):04o}. "
            "Repair ownership/permissions of this generated scene cache, then regenerate. "
            "Enter the container with `geniesim docker6.2 into` to use the host UID",
            str(path),
        )

    if os.geteuid() == 0:
        # The pipeline runner may already have created a root-owned scene
        # directory. Its user-owned parent still identifies the shared cache.
        for path in (ancestor, *ancestor.parents):
            if path.stat().st_uid != 0:
                raise fail(path, "Refusing to create root-owned assembly files in a user-owned cache")

    def check(path, mode):
        if not os.access(path, mode, effective_ids=True):
            raise fail(path, "Assembly output cannot be replaced by the current user")

    check(ancestor, os.R_OK | os.W_OK | os.X_OK)
    for name in ("robot_raw.urdf", "robot.urdf", "robot.usda", "urdf.sha256"):
        path = output / name
        if path.exists():
            check(path, os.W_OK)
    payloads = output / "payloads"
    if payloads.exists():

        def walk_error(error):
            raise error

        for directory, _, _ in os.walk(payloads, onerror=walk_error):
            check(Path(directory), os.R_OK | os.W_OK | os.X_OK)


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

    # Write the raw URDF (package:// URIs intact) for the 6.x importer.
    # The 6.x converter auto-discovers packages by walking up from the URDF
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


def _validated_pbr_entry(entry: dict, logger=None) -> dict[str, float]:
    """Clamp finite PBR scalars; ignore malformed/non-finite fields with a warning."""
    import math
    import warnings

    out = {}
    for field in ("roughness", "metallic"):
        if field not in entry:
            continue
        try:
            value = float(entry[field])
        except (ValueError, TypeError):
            value = math.nan
        message = None
        if not math.isfinite(value):
            message = f"[material_override] {field}={entry[field]!r} is not finite; ignored"
        elif not 0.0 <= value <= 1.0:
            message = f"[material_override] {field}={value} outside [0, 1]; clamped"
            out[field] = min(1.0, max(0.0, value))
        else:
            out[field] = value
        if message:
            if logger is not None:
                logger.warn(message)
            else:
                warnings.warn(message, RuntimeWarning, stacklevel=2)
    return out


def _parse_material_override_blocks(urdf_str: str, logger=None) -> list[dict]:
    """Read per-visual PBR overrides and assign collision-free converter names.

    Isaac 6.2 uses the URDF visual name for its geometry wrapper. Mesh stems
    and shape names cannot identify repeated visuals, so the converter copy
    receives a stable name for each overridden visual. The source URDF stays
    unchanged; diagnostic replays derive the same names from its visual indices.
    """
    import xml.etree.ElementTree as ET

    root = ET.fromstring(urdf_str)
    link_names = {link.get("name") for link in root.iter("link")}
    out = []
    for link in root.iter("link"):
        link_name = link.get("name")
        if not link_name:
            continue
        used = link_names | {child.get("name") for child in link if child.tag in ("visual", "collision")}
        for index, visual in enumerate(link.findall("visual")):
            block = visual.find("material_override")
            if block is None:
                continue
            entry = _validated_pbr_entry({child.tag: child.text for child in block}, logger=logger)
            if not entry:
                continue
            name = f"geniesim_visual_{index}"
            while name in used:
                name += "_"
            used.add(name)
            out.append({"link": link_name, "target": name, "visual_index": index, "entry": entry})
    return out


def _prepare_material_override_urdf(urdf_str: str, logger=None) -> tuple[str, list[dict]]:
    """Strip extension tags and name override targets in the 6.2 converter copy."""
    import xml.etree.ElementTree as ET

    overrides = _parse_material_override_blocks(urdf_str, logger=logger)
    root = ET.fromstring(urdf_str)
    links = {link.get("name"): link for link in root.iter("link")}
    for spec in overrides:
        links[spec["link"]].findall("visual")[spec["visual_index"]].set("name", spec["target"])
    for visual in root.iter("visual"):
        for block in visual.findall("material_override"):
            visual.remove(block)
    return ET.tostring(root, encoding="unicode"), overrides


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
    for field, value in _validated_pbr_entry(entry, logger=logger).items():
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
    prims, including private per-visual references after the override pass.
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


def _make_visual_materials_editable(wrapper):
    """Expand only this visual's instances so bindings can be changed locally."""
    from pxr import Usd

    while True:
        instances = [p for p in Usd.PrimRange(wrapper) if p.IsInstance()]
        if not instances:
            return
        for prim in instances:
            prim.SetInstanceable(False)


def _apply_material_overrides(stage, overrides: list[dict], robot_prefix: str, logger=None) -> int:
    """Patch private, per-visual material references in the generated root layer.

    Never edit shared material payloads. Internal references preserve shader
    networks, textures and unselected fields; local bindings isolate this visual
    from other users of the same material. Reapplying a patch reuses its clones.
    """
    from pxr import Usd, UsdGeom, UsdShade

    applied = 0
    for spec in overrides:
        link = _find_link_prim(stage, robot_prefix, spec["link"])
        wrappers = _link_visual_wrappers(link) if link else []
        matched = [p for name, p in wrappers if name == spec["target"]]
        if len(matched) != 1:
            raise ValueError(
                f"[material_override] {spec['link']}/{spec['target']}: expected one visual, "
                f"found {len(matched)}; regenerate robot USD"
            )
        wrapper = matched[0]
        _make_visual_materials_editable(wrapper)
        bindings = {}
        for prim in Usd.PrimRange(wrapper):
            if prim.HasAPI(UsdShade.MaterialBindingAPI):
                rel = UsdShade.MaterialBindingAPI(prim).GetDirectBindingRel()
                targets = rel.GetTargets() if rel else []
                if len(targets) == 1:
                    bindings.setdefault(targets[0], []).append(rel)
        scope_path = wrapper.GetPath().AppendChild("GenieMaterialOverrides")
        for source_path, rels in bindings.items():
            if source_path.HasPrefix(scope_path):
                material = UsdShade.Material(stage.GetPrimAtPath(source_path))
            else:
                source = UsdShade.Material(stage.GetPrimAtPath(source_path))
                if not source:
                    raise ValueError(f"[material_override] missing material {source_path}")
                UsdGeom.Scope.Define(stage, scope_path)
                suffix = hashlib.sha256(str(source_path).encode()).hexdigest()[:16]
                material = UsdShade.Material.Define(stage, scope_path.AppendChild(f"material_{suffix}"))
                material.GetPrim().GetReferences().AddInternalReference(source_path)
            if _set_pbr_inputs(material.GetPrim(), spec["entry"], logger=logger):
                for rel in rels:
                    rel.SetTargets([material.GetPath()])
                applied += 1
        if not bindings:
            raise ValueError(f"[material_override] {wrapper.GetPath()}: no bound visual materials")
    return applied


# Boolean custom-attribute names the engine reads to identify gripper
# topology WITHOUT the old hard-coded name patterns (``"link4"`` substring,
# ``gripper_[lr]_(inner|outer)_joint[034]$`` regex).  Authored at assemble
# time by ``_derive_gripper_annotations`` + ``_author_gripper_annotations``.
# See docs/PORT_ENGINE_HARDCODING_AUDIT.md (H1-H4) and
# docs/superpowers/plans/2026-07-21-engine-hardcoding-H1-H4-fix.md.
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
    mimic_followers: list[tuple[str, str | None, str | None]] = []  # (joint, parent, master)
    joint_parent: dict[str, str | None] = {}  # joint name -> parent link
    for j in root.iter("joint"):
        jname = j.get("name") or ""
        jtype = (j.get("type") or "").strip().lower()
        parent_el = j.find("parent")
        child_el = j.find("child")
        parent = parent_el.get("link") if parent_el is not None else None
        child = child_el.get("link") if child_el is not None else None

        # Coupler candidates are gripper-classified <mimic>-follower joints.
        # Whether each is a passive 4-bar coupler or the second DRIVEN joint
        # is decided structurally in the second pass below, once the movable
        # topology is complete.
        joint_parent[jname] = parent
        mimic_el = j.find("mimic")
        if mimic_el is not None and _is_gripper(jname):
            mimic_followers.append((jname, parent, mimic_el.get("joint")))

        if jtype in _URDF_MOVABLE_JOINT_TYPES and parent and child:
            movable_children.setdefault(parent, set()).add(child)
            movable_child_links.add(child)

    # Couplers, decided on TOPOLOGY rather than on a name suffix.
    #
    # The MASTER joint hangs off the gripper's base -- whatever that link is
    # called: ``gripper_[lr]_base_link`` on G2, the arm-provided ``tool0`` on
    # ur5. A follower sharing the master's parent is therefore the second
    # DRIVEN joint (G2's ``outer_joint1``, multiplier -1), while a follower
    # rooted deeper in the assembly is a passive 4-bar coupler.
    #
    # Keying on the master's parent rather than on ``parent.endswith(
    # "_base_link")`` is what makes this survive renaming the mount frame --
    # the old literal mis-split every arm the moment it became ``tool0``.
    # "Inside the movable chain" is NOT a usable test either: ur5's
    # ``*_tip_joint`` couplers hang off finger links reached by FIXED joints.
    for jname, parent, master in mimic_followers:
        if master is not None and master in joint_parent and parent == joint_parent[master]:
            continue  # second driven joint, not a passive coupler
        out["couplers"].add(jname)

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

    if n_pads != len(pads) or n_couplers != len(couplers):
        raise RuntimeError(
            f"[gripper-annot] inventory mismatch: pads={n_pads}/{len(pads)}, "
            f"couplers={n_couplers}/{len(couplers)}; check Physics variant and converted topology"
        )

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
    # Newton unit quirk — GONE as of 1.4.0, see below
    # ----------------------------------------------
    # ``mjc:damping`` (angular) *used* to be divided by π/180 at MJCF
    # write time, so this file pre-multiplied by 180/π in USD to make
    # the round-trip land the right value.  Newton removed that
    # conversion; measured absent on 1.4.0 and 1.6.1, so the
    # pre-multiplication is gone too and the physical value is
    # authored verbatim — see the ``_HAND_JOINT_DAMPING_*`` block
    # below and quirk 8 in
    # ``genie_sim_engine_newton/docs/newton_quirks.md``.
    # ``mjc:frictionloss`` never had the quirk — author verbatim.  See
    # ``MuJoCoWarpAdapter._per_class_passive_joint`` line 828: "No
    # quirk on frictionloss — author verbatim".
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
    # which Newton's mjwarp converter reads as
    # ``dof_passive_damping`` and writes to MJCF as ``damping=N``
    # on each joint).  Reference's 0.05 value was tuned against the
    # G2's hand-joint inertia and is adopted verbatim — independent
    # from the actuator force cap, so it composes cleanly with PD /
    # effort-limit tuning.
    #
    # Unit convention — IMPORTANT:
    #
    # Newton's mjwarp converter (in the container's
    # ``newton==1.15.0.dev20260526``) bakes a per-angle conversion
    # at MJCF WRITE TIME, on line 4531 of
    # ``newton/_src/solvers/mujoco/solver_mujoco.py``::
    #
    #     joint_params["damping"] = joint_damping[ai] * (np.pi / 180)
    #
    # for angular DOFs.  It does this regardless of the MJCF's own
    # ``<compiler angle="..."/>`` tag (which Newton hard-codes to
    # ``"radian"``).  Net effect: any value authored in USD as
    # ``mjc:damping`` is divided by 180/π before reaching the MJCF
    # — making the physical damping 60× weaker than intended.
    # Verified empirically: USD ``mjc:damping=0.05`` → MJCF
    # ``damping="0.000872665"`` in the launcher's dump and in
    # ``test_robot_xml_dynamic.py`` runs inside the container.
    #
    # To land at MJCF ``damping=0.05`` (the reference's per-radian
    # value), pre-multiply by 180/π so the converter's downstream
    # ``× π/180`` cancels it out::
    #
    #     mjc:damping = 0.05 × 180/π ≈ 2.8648
    #
    # THAT HAS NOW HAPPENED, and the pre-multiplication above is
    # gone as a result.  Everything from "To land at MJCF" down to
    # here is kept as history, because the diff that removes a
    # 180/π is unreadable without it.
    #
    # Measured 2026-10-18 on newton 1.4.0 (host) and 1.6.1: the
    # write-side ``× π/180`` is gone from both, so the authored value
    # now reaches ``joint_damping`` verbatim.  Two consequences, and
    # the fix needs both halves or it makes things worse:
    #
    #   * the pre-multiplied ``2.8648`` was 57.3× too stiff on 1.4.0
    #     (measured: ``joint_damping = 2.8648`` where ``0.05`` was
    #     intended) — fixed by authoring the physical value here;
    #   * on 1.6.1 ``mjc:damping`` is read by ``SchemaResolverMjc``
    #     alone (``dof_passive_damping`` was deleted), and this repo
    #     passed that resolver at zero call sites — so authoring
    #     ``0.05`` *by itself* would have produced NO damping at all
    #     on 1.6.1 and brought back the 1700× resonance this block
    #     exists to kill.  Fixed alongside, in
    #     ``genie_sim_engine_newton/engine/newton/setup/model.py``
    #     (``add_usd(schema_resolvers=...)``), which is now the
    #     load-bearing other half of this constant.
    #
    # Verified on the host's 1.4.0 with the quirk-8 repro: authored
    # ``0.05`` → ``joint_damping = 0.05`` both with and without the
    # Mjc resolver, so the resolver is a no-op there rather than a
    # second conversion.  Full measurements and the MJCF column:
    # quirk 8 in ``genie_sim_engine_newton/docs/newton_quirks.md``.
    #
    # Note the master's DriveAPI gains above (``_HAND_MASTER_USD_*``)
    # are NOT affected, and this was measured rather than assumed: on
    # 1.4.0, ``drive:angular:physics:stiffness = 5.0 * pi/180`` still
    # resolves to ``joint_target_ke = 5.0``, so the ``x 180/pi`` at read
    # is intact there.  The removal was specific to the ``mjc:*``
    # custom-attribute path.  Rule of thumb: ``mjc:*`` is per-radian
    # verbatim, ``drive:angular:physics:*`` is per-degree and gets
    # converted -- a ``pi/180`` is a bug on the first, required on the
    # second.
    _HAND_JOINT_DAMPING_FINAL = 0.05
    # Authored verbatim: the converter no longer scales it on the way
    # through.  ``_USD`` is kept as the name the write sites use.
    _HAND_JOINT_DAMPING_USD = _HAND_JOINT_DAMPING_FINAL

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

    from genie_sim_engine_isaacsim.assemble.physics_variant import select_physics_variant, validate_urdf_joints

    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise RuntimeError(f"[joint-overlay] failed to open {usd_path}")
    select_physics_variant(stage, logger=logger)
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
        except Exception as exc:
            raise RuntimeError(f"[gripper-annot] could not author expected annotations in {usd_path}") from exc
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
    """Apply 6.2 PBR patches after AS3 has emitted the final robot asset.

    Editing the intermediate converter stage disrupts transformer geometry
    discovery. Open the final asset with payloads loaded instead; only its root
    layer receives per-visual material references, scalar overrides and bindings.
    Shared material/geometry payloads remain unchanged.
    """
    if not overrides:
        return 0

    import gc as _gc

    from pxr import Usd  # noqa: E402

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
    from genie_sim_engine_isaacsim.assemble.physics_variant import select_physics_variant

    stage = Usd.Stage.Open(usd_path, load=Usd.Stage.LoadAll)
    if stage is None:
        if logger is not None:
            logger.warn(f"[material_override] post-transformer: could not open {usd_path}; " f"overrides not applied.")
        return 0

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

    select_physics_variant(stage, logger=logger)
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

    # Release composed layers before the next joint-overlay pass opens the
    # asset. Retained Sdf layers previously swallowed a NewtonMimicAPI listOp
    # during save; each post-transformer pass must reopen the saved root.
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

    from genie_sim_engine_isaacsim.assemble.physics_variant import select_physics_variant

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
        if any(p.IsA(UsdGeom.Mesh) for p in Usd.PrimRange(prim, Usd.TraverseInstanceProxies())):
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


def _convert_urdf_to_usd_as3(
    urdf_path: str,
    usd_path: str,
    *,
    ros_packages: dict[str, str] | None = None,
    raw_urdf_path: str | None = None,
    config=None,  # URDFImporterConfig (6.x API)
) -> None:
    """Isaac Sim 6.x: URDF → Asset Structure 3.0 pipeline.

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

    # Fail closed on an unresolvable mesh map. The 6.x converter resolves
    # ``package://`` mesh URIs through ``ros_packages`` and nothing else, and
    # when it cannot resolve them it reports success having imported no
    # geometry: AS3 then emits no geometries/instances payload, and the
    # collider and <material_override> policies correctly report zero edits
    # against an empty stage. Every downstream signal looks like a policy
    # regression while the real loss happened here, so refuse instead.
    with open(source_urdf) as _fh:
        _referenced_pkgs = sorted(set(re.findall(r"package://([A-Za-z_][A-Za-z0-9_]*)/", _fh.read())))
    _unmapped = [p for p in _referenced_pkgs if p not in (ros_packages or {})]
    if _unmapped:
        print(
            f"[assemble_robot] ERROR: {source_urdf} references package:// URIs for "
            f"{_unmapped} but ros_packages maps {sorted(ros_packages or {})}. "
            f"The converter would silently import zero meshes. Pass ros_packages "
            f"for every referenced package.",
            file=sys.stderr,
        )
        sys.exit(1)

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

        # Only the converter copy receives stable visual names and has the
        # extension stripped. robot_raw.urdf remains the authored source.
        urdf_content, material_overrides = _prepare_material_override_urdf(urdf_content)

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
            from genie_sim_engine_isaacsim.assemble.physics_variant import select_physics_variant, validate_urdf_joints

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


def _convert_urdf_to_usd_as2(
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
    PhysicsFixedJoint named ``root_joint`` (6.x) or the autogenerated 4.x/5.x
    equivalent. The *runtime* (``runtime.stage._apply_fix_base_policy``)
    decides per scene whether that joint is active, based on the manifest's
    ``fix_base`` flag. The cached ``robot.usda`` is therefore scene-agnostic.

    ``success_stamp`` is an optional ``(path, content)`` pair written iff the
    importer produces a valid USD. The launch's ``make_assemble_pipeline``
    reads ``<stage_dir>/urdf.sha256`` (the typical stamp content is
    ``sha256(urdf_text)``) to skip this script on subsequent runs when the
    URDF input hasn't changed.
    """
    _isaacsim_ver = _isaacsim_version()

    from isaacsim import SimulationApp  # noqa: E402

    from genie_sim_engine_isaacsim.kit.launch_config import prepare_launch_config

    simulation_app = SimulationApp(prepare_launch_config({"headless": True, "fast_shutdown": False}))

    converted = False
    try:
        if _isaacsim_ver >= (6, 0):
            # 6.x param injection: URDFImporterConfig (isaacsim.asset.importer.urdf API)
            from isaacsim.asset.importer.urdf import URDFImporterConfig  # noqa: E402

            config_60 = URDFImporterConfig()
            config_60.allow_self_collision = False
            config_60.collision_from_visuals = False
            config_60.collision_type = "Convex Hull"
            config_60.merge_mesh = False
            config_60.ros_package_paths = [{"name": k, "path": v} for k, v in (ros_packages or {}).items()]
            _convert_urdf_to_usd_as3(
                urdf_path,
                usd_path,
                ros_packages=ros_packages,
                raw_urdf_path=raw_urdf_path,
                config=config_60,
            )
        else:
            # 4.x/5.x param injection: _urdf.ImportConfig (kit-command API)
            _convert_urdf_to_usd_as2(simulation_app, urdf_path, usd_path)
        converted = True
    finally:
        # Validate before exit — skip kit teardown (hangs/crashes) and exit
        # directly so ros2 launch sees the correct return code.
        if _isaacsim_ver >= (6, 0) and not converted:
            print(f"[assemble_robot] ERROR: conversion or post-processing failed: {sys.exc_info()[1]}", file=sys.stderr)
            # os._exit bypasses Python's exception hook; print the original
            # traceback before exiting so recursive-copy failures are traceable.
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

    if _isaacsim_version() == (6, 2):
        _check_assembly_output_permissions(output_dir)

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

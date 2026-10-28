#!/usr/bin/env python3

# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""USD stage + articulation + snapshot helpers for ``genie_sim_engine``.

This module owns **everything** that touches the USD / Isaac-Sim runtime:

  - :class:`IsaacSimStage` — opens the scene, configures drives + articulation
    gains, applies joint commands.
  - :func:`wait_for_manifest` — blocks until the manifest JSON exists.
  - :func:`snapshot_joint_states` / :func:`snapshot_body_transforms` /
    :func:`snapshot_odom` — flat-numpy readers consumed by the C++ ROS
    publishers in ``genie_sim_engine_py``.

Render scheduling and per-tick stats live in the C++ extension.
"""

from __future__ import annotations

import math
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from genie_sim_engine_core.common.params import GripperDriveParams, JointInitSpec, PhysicsParams, default_physics_params
from genie_sim_engine_isaacsim.kit.bootstrap import configure_carb_settings

__all__ = [
    "IsaacSimStage",
    "wait_for_manifest",
    "snapshot_joint_states",
    "snapshot_body_transforms",
    "snapshot_odom",
]

_DEG2RAD = math.pi / 180.0
_RAD2DEG = 180.0 / math.pi


# ---------------------------------------------------------------------------
# ArticulationView numpy↔torch shim
#
# Isaac Sim's PhysX wrapper accepts numpy ``ndarray`` inputs to every
# ArticulationView setter (``set_gains``, ``set_max_efforts``,
# ``set_joint_positions``, ``set_joint_position_targets``,
# ``apply_action`` …) and returns numpy on readback.  The Newton wrapper
# (``isaacsim.physics.newton``) does NOT — its setters call ``.to(device)``
# on the incoming buffer and its getters return CUDA torch tensors. So a
# numpy input dies with::
#
#     AttributeError: 'numpy.ndarray' object has no attribute 'to'
#
# and the matching readback dies with::
#
#     TypeError: can't convert cuda:0 device type tensor to numpy.
#     Use Tensor.cpu() to copy the tensor to host memory first.
#
# The two helpers below paper over both directions. ``_view_input`` is a
# no-op for non-Newton engines (PhysX path stays zero-cost); the torch
# import is lazy so a torch-less PhysX install keeps booting.
# ``_view_readback`` is symmetric and engine-agnostic — torch tensors are
# detected via duck-typing on ``.cpu()`` so callers don't need to thread
# the engine string through every readback.
# ---------------------------------------------------------------------------


def _view_input(arr, *, engine: str):
    """numpy → CUDA torch tensor when ``engine == 'isaac_newton'``.

    Why torch and not warp: ``SimulationManager`` defaults
    ``_backend = "numpy"`` and auto-promotes to ``"torch"`` on CUDA —
    there's no ``"warp"`` auto-promote. So under Newton the active
    ``Articulation._backend_utils`` is ``isaacsim.core.utils.torch``,
    whose ``move_data`` / ``resolve_indices`` accept torch tensors and
    not warp arrays. Forcing the per-instance backend to ``"warp"`` was
    a dead end — warp's tensor utilities at this Isaac Sim version
    raise ``Invalid device identifier`` on the same device strings
    (``"cuda:0"`` / ``"cpu"``) that work in unit tests; the runtime
    state of ``warp.context.runtime`` differs from a fresh ``wp.init()``
    and the failure modes are opaque from outside the kit process.

    The torch path, in contrast, is well-behaved: torch tensors flow
    through ``Articulation.set_gains`` / ``apply_action`` /
    ``set_joint_position_targets`` cleanly, and the wrapper's internal
    ``wrap_input_tensor`` knows how to coerce torch → wp at the kernel
    boundary.

    Returns ``arr`` unchanged for any non-Newton engine, so PhysX call
    sites stay branch-free at runtime. Failure to import torch returns
    the original array — let the underlying error surface rather than
    silently masking it.
    """
    if engine != "isaac_newton":
        return arr
    try:
        import torch  # noqa: PLC0415 — lazy: PhysX path never pays
    except Exception:
        return arr
    return torch.as_tensor(arr, device="cuda:0")


def _view_readback(x):
    """ArticulationView readback → numpy regardless of backend.

    Three readback shapes show up:

    * ``np.ndarray`` (PhysX numpy frontend) — pass through.
    * ``torch.Tensor`` on CUDA (Newton path or PhysX torch frontend) —
      needs ``.cpu().numpy()`` because CUDA tensors can't go straight
      to numpy.
    * ``wp.array`` / ``wp.indexedarray`` — has a direct ``.numpy()``
      method that issues an internal sync.

    Tries ``.numpy()`` first (works for warp arrays AND CPU torch),
    falls back to ``.cpu().numpy()`` for CUDA torch, then ``np.asarray``
    for plain arrays.
    """
    if x is None:
        return None
    np_method = getattr(x, "numpy", None)
    if callable(np_method):
        try:
            return np_method()
        except Exception:
            pass  # CUDA torch — falls through to cpu()
    cpu = getattr(x, "cpu", None)
    if callable(cpu):
        return cpu().numpy()
    return np.asarray(x)


def _parse_mimic_from_stage(stage, logger) -> Dict[str, List[Tuple[str, float, float]]]:
    """Compatibility shim so the call site in
    ``_init_articulation`` doesn't need to know about ``engine._mimic``
    directly. The implementation lives there and is shared with
    ``NewtonStandaloneEngine``.
    """
    from genie_sim_engine_core.engine._mimic import parse_mimic

    return parse_mimic(stage, logger)


# ---------------------------------------------------------------------------
# Override logging
# ---------------------------------------------------------------------------


def _fmt_value(value) -> str:
    """Compact representation for override logs."""
    if value is None:
        return "<unset>"
    try:
        if isinstance(value, float):
            return f"{value:g}"
        if isinstance(value, (int, bool, str)):
            return repr(value)
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return "[]"
            return f"{float(value.item()):g}" if value.size == 1 else f"{value.tolist()}"
        return repr(value)
    except Exception:
        return repr(value)


def _values_equal(old, new) -> bool:
    """Treat ``None`` (unauthored) as different from any concrete new value."""
    if old is None and new is None:
        return True
    if old is None or new is None:
        return False
    try:
        if isinstance(old, float) or isinstance(new, float):
            return math.isclose(float(old), float(new), rel_tol=1e-9, abs_tol=1e-12)
        return old == new
    except Exception:
        return False


def _log_override(logger, kind: str, target: str, attr: str, old, new) -> None:
    """Emit a uniform '[override] <kind> <target>.<attr>: old -> new' line."""
    if _values_equal(old, new):
        return
    logger.info(f"[override] {kind} {target}.{attr}: {_fmt_value(old)} -> {_fmt_value(new)}")


def _safe_get(attr):
    """Return the authored value of a USD attribute, or ``None`` if unset/missing."""
    if attr is None:
        return None
    try:
        if not attr.HasAuthoredValue() and not attr.HasFallbackValue():
            return None
    except Exception:
        pass
    try:
        return attr.Get()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Joint classification
# ---------------------------------------------------------------------------

# The classifier (``JK_*`` constants + ``classify_joint``) lives in
# ``common.joint_classification`` so it can be shared with the
# newton-standalone ``MuJoCoWarpAdapter`` without dragging Pxr / Isaac
# Sim imports into that path.  Re-exported here at module scope so
# existing call sites (``_configure_drives``, ``_init_articulation``)
# don't need import-path edits and the public names are still
# discoverable on ``kit.stage``.
from genie_sim_engine_newton.common.joint_classification import (  # noqa: E402,F401
    JK_ARM,
    JK_ARM_MID,
    JK_ARM_SHOULDER,
    JK_ARM_WRIST,
    JK_BODY,
    JK_CHASSIS_DRIVE,
    JK_CHASSIS_STEER,
    JK_CHASSIS_WHEEL,
    JK_GRIPPER,
    JK_HEAD,
    JK_OTHER,
    classify_joint,
    classify_joint_by_name,
    is_chassis_wheel_free,
    strip_idx_prefix,
)


def _snapshot_gripper_state(prim, name: str) -> Dict[str, object]:
    """Read every drive/limit/dynamics attribute we'd otherwise overwrite.

    Used for the URDF-route gripper logging path: when the robot was built
    from a URDF (via the urdf_to_usd stage) the gripper joints often carry
    closed-loop / mimic / passive constructs that don't survive cleanly
    into PhysX. Rather than guess at a tuning, we dump the as-authored
    state so the user can decide.
    """
    info: Dict[str, object] = {"path": prim.GetPath().pathString}
    is_prismatic = prim.IsA(UsdPhysics.PrismaticJoint)
    info["type"] = "prismatic" if is_prismatic else "revolute"

    if is_prismatic:
        joint = UsdPhysics.PrismaticJoint(prim)
    else:
        joint = UsdPhysics.RevoluteJoint(prim)
    if joint:
        info["lowerLimit"] = _safe_get(joint.GetLowerLimitAttr())
        info["upperLimit"] = _safe_get(joint.GetUpperLimitAttr())

    token = "linear" if is_prismatic else "angular"
    info["driveToken"] = token
    info["hasDriveAPI"] = prim.HasAPI(UsdPhysics.DriveAPI, token)
    drive = UsdPhysics.DriveAPI.Get(prim, token) if info["hasDriveAPI"] else None
    if drive:
        info["driveType"] = _safe_get(drive.GetTypeAttr())
        info["stiffness"] = _safe_get(drive.GetStiffnessAttr())
        info["damping"] = _safe_get(drive.GetDampingAttr())
        info["maxForce"] = _safe_get(drive.GetMaxForceAttr())
        info["targetPosition"] = _safe_get(drive.GetTargetPositionAttr())
        info["targetVelocity"] = _safe_get(drive.GetTargetVelocityAttr())

    # Surface anything that looks like a mimic / passive / loop hint so the
    # user can grep their log without re-opening the USDA. We deliberately
    # cast to str because the value types vary (tokens, rels, custom).
    schemas = []
    try:
        schemas = list(prim.GetAppliedSchemas())
    except Exception:
        pass
    info["appliedSchemas"] = schemas
    return info


# ---------------------------------------------------------------------------
# Manifest wait
# ---------------------------------------------------------------------------


def wait_for_manifest(path: str, logger, retries: int = 30, interval: float = 1.0) -> None:
    """Block until ``path`` exists, logging once per second.

    On timeout the error message includes the symlink-resolved path
    and the parent directory contents, so mount / symlink mismatches
    ("assembler wrote to A, runtime reads from B") are diagnosable
    from the log alone.
    """
    raw = Path(path)
    for _ in range(retries):
        if raw.exists():
            return
        logger.info(f"Waiting for manifest: {path}")
        time.sleep(interval)
    try:
        resolved = raw.resolve()
    except OSError:
        resolved = raw
    parent = raw.parent
    try:
        siblings = sorted(p.name for p in parent.iterdir())
    except OSError:
        siblings = []
    raise FileNotFoundError(
        f"manifest not found after {retries}s: {path}\n"
        f"  resolved to: {resolved}\n"
        f"  parent exists: {parent.exists()}\n"
        f"  parent contents ({len(siblings)}): {siblings[:20]}{'…' if len(siblings) > 20 else ''}\n"
        "  → run 'geniesim assemble' (or the equivalent launch step) to generate it"
    )


# ---------------------------------------------------------------------------
# Stage construction helpers (private)
# ---------------------------------------------------------------------------


def _open_scene_with_references(
    scene_usda: str,
    robot_usda: str,
    render_layer_usda: str,
    robot_prefix: str,
    simulation_app,
    logger,
    newton_solvers_path: str = "",  # kept for API compat; not used here
    scene_cfg: Optional[dict] = None,
):
    from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage, open_stage

    logger.info(f"Opening scene USD: {scene_usda}")
    open_stage(scene_usda)
    for _ in range(10):
        simulation_app.update()

    stage = get_current_stage()

    robot_root = f"/{robot_prefix}"
    if robot_usda and Path(robot_usda).exists():
        logger.info(f"Adding robot visual: {robot_usda}")
        add_reference_to_stage(robot_usda, robot_root)
        from genie_sim_engine_isaacsim.assemble.physics_variant import select_physics_variant

        select_physics_variant(stage, robot_root, logger)
        # Apply ``init_base_pose`` BEFORE the simulation_app.update() loop
        # that triggers PhysX's rigid-body initialization.  If we wait
        # until after these updates, PhysX has already created the base
        # link at the cached pose (typically world origin); a subsequent
        # Xform translate then becomes a constraint mismatch the solver
        # tries to resolve over many ticks, producing the visible "swing"
        # at startup.  Authoring the translate now means PhysX scans the
        # already-translated prim during its first init pass and the
        # base spawns at the requested pose with zero residual energy.
        _apply_init_base_pose(stage, robot_prefix, scene_cfg, logger)
        for _ in range(5):
            simulation_app.update()
    if render_layer_usda and Path(render_layer_usda).exists():
        logger.info(f"Adding render layer: {render_layer_usda}")
        add_reference_to_stage(render_layer_usda, "/RenderOVRTX")
        for _ in range(5):
            simulation_app.update()

    # Newton extras layer: cloth/softbody USDs at /World/<name> on the physics
    # stage. Separate from render_layer.usda (which lives under /RenderOVRTX).
    newton_scene = Path(render_layer_usda).parent / "newton_scene.usda" if render_layer_usda else None
    if newton_scene and newton_scene.is_file():
        logger.info(f"Adding Newton scene layer: {newton_scene}")
        stage.GetRootLayer().subLayerPaths.append(str(newton_scene))
        for _ in range(3):
            simulation_app.update()

    return stage


def _apply_init_base_pose(stage, robot_prefix: str, scene_cfg, logger) -> None:
    """Apply ``scene.robot.init_base_pose`` to the robot Xform on the Kit stage.

    Mirrors ``engine/newton/setup/stage.py:_apply_init_base_pose`` so the
    Kit / Isaac PhysX path honours the same YAML knob as the Newton path.

    YAML shape (in the scene yaml ``robot:`` block)::

        robot:
          init_base_pose:
            x: 0.0          # world metres
            y: 0.0
            z: 0.05
            theta: 0.0      # yaw radians about world Z

    Authors a session-layer translate + orient op pair on ``/<robot_prefix>``,
    using unique suffixes so they coexist with any xform ops already
    authored in ``robot.usda``'s cached layer.  ``SetXformOpOrder`` then
    names ONLY our two ops as participating in the composed transform —
    the cached layer's ops drop out of the order and contribute nothing.

    Both ``pin_base_to_world`` modes consume this transparently:

      * ``pin_base_to_world=True`` — the world-weld FixedJoint pins
        ``base_link`` to ``/<robot_prefix>``, which is now translated/
        rotated to the requested pose; the welded base sits at
        ``init_base_pose`` in world.
      * ``pin_base_to_world=False`` — when PhysX adds the FREE base
        joint, base_link inherits ``/<robot_prefix>``'s composed pose.

    No-op when ``scene_cfg`` is missing or the ``init_base_pose`` block
    is absent.
    """
    if stage is None or not isinstance(scene_cfg, dict):
        return
    base_pose = (scene_cfg.get("robot") or {}).get("init_base_pose")
    if not isinstance(base_pose, dict):
        return
    try:
        x = float(base_pose.get("x", 0.0))
        y = float(base_pose.get("y", 0.0))
        z = float(base_pose.get("z", 0.0))
        theta = float(base_pose.get("theta", 0.0))
    except (TypeError, ValueError) as exc:
        logger.warn(f"[stage] init_base_pose: numeric parse failed ({exc!r}); skipping")
        return

    import math as _math

    from pxr import Gf, Sdf, UsdGeom

    root_path = Sdf.Path(f"/{robot_prefix}")
    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim or not root_prim.IsValid():
        logger.warn(f"[stage] init_base_pose: robot root {root_path} not on stage; skipping")
        return
    xformable = UsdGeom.Xformable(root_prim)
    if not xformable:
        logger.warn(f"[stage] init_base_pose: {root_path} is not Xformable; skipping")
        return

    t_op = xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble, "init_base")
    t_op.Set(Gf.Vec3d(x, y, z))
    half = 0.5 * theta
    quat = Gf.Quatd(_math.cos(half), Gf.Vec3d(0.0, 0.0, _math.sin(half)))
    o_op = xformable.AddOrientOp(UsdGeom.XformOp.PrecisionDouble, "init_base")
    o_op.Set(quat)
    xformable.SetXformOpOrder([t_op, o_op])

    logger.info(
        f"[stage] init_base_pose applied to {root_path}: " f"xyz=({x:.4f}, {y:.4f}, {z:.4f}) yaw={theta:.4f} rad"
    )


def _detect_asset_format(robot_usda: str) -> str:
    """Return 'as3' if robot_usda is an Asset Structure 3.0 package, 'flat' otherwise.

    Asset Structure 3.0 packages have a payloads/ directory next to robot.usda
    produced by the isaacsim_structure.json transformer (Isaac Sim 6.x).
    Flat assets are single-file USD outputs from URDFParseAndImportFile (4.x/5.x).
    """
    if not robot_usda:
        return "flat"
    payloads_dir = Path(robot_usda).parent / "payloads"
    return "as3" if payloads_dir.is_dir() else "flat"


def _apply_fix_base_policy(stage, robot_prefix: str, fix_base: bool, logger=None) -> int:
    """Enable or disable the URDF importer's world-weld joint at runtime.

    Both URDF importers (6.x ``urdf_usd_converter`` and 4.x/5.x
    ``URDFParseAndImportFile``) MAY author a ``PhysicsFixedJoint`` from
    world to the URDF root link so the cached ``robot.usda`` can serve
    fixed-base scenes out of the box. The 6.x importer names it
    ``root_joint`` explicitly; 4.x/5.x produces an importer-named
    equivalent. We detect the joint by topology, not by name.

    Distinguishing the world weld from internal fixed joints
    --------------------------------------------------------
    On a **non-ghost** URDF root, every internal joint's ``body0`` points
    at the actual parent link, and the converter emits a separate
    ``root_joint`` with ``body0 = articulation_root`` and ``body1 =
    root_link``. The world weld is the unique fixed joint matching that
    pattern.

    On a **ghost** URDF root (the URDF root link has no inertial /
    visual / collision — UR / Robotiq / aloha put inertia on a
    ``*_inertia`` sibling), the converter takes its ``is_ghost_link``
    branch: it emits NO ``root_joint`` AND every joint's ``body0`` is set
    to the articulation root. In this configuration the converter is
    Newton-only-correct; PhysX simulation needs a separate fix or the
    URDF root needs to stop being a ghost (see ``diagnose_urdf.py``'s
    ghost-root warning). The runtime cannot distinguish "world weld" from
    "internal fixed joint" by ``body0`` alone in this case — they all
    point at the root.

    To disambiguate, we additionally require that ``body1`` is **the
    URDF root link itself** (the unique direct child of the articulation
    root prim that carries ``RigidBodyAPI``). World welds anchor the
    root link to world; internal joints anchor deeper-nested prims to
    each other.

    Returns the number of world-weld joints found. ``0`` for ghost-root
    URDFs — in that case ``fix_base`` is silently inert for this robot
    (operator should fix the ghost root in the URDF).
    """
    robot_root_path = Sdf.Path(f"/{robot_prefix}")
    root_prim = stage.GetPrimAtPath(robot_root_path)
    if not root_prim or not root_prim.IsValid():
        if logger is not None:
            logger.warn(f"[fix_base] robot root prim {robot_root_path} not found")
        return 0

    # Identify the URDF root link's USD prim path. This is the topmost
    # ``RigidBodyAPI``-bearing descendant of the articulation root. We
    # require ``body1`` to match this exact path to qualify as a world
    # weld; any joint pointing at a deeper-nested rigid body is an
    # internal joint we must NOT touch.
    urdf_root_link_paths: List[Sdf.Path] = []
    for desc in Usd.PrimRange(root_prim):
        if desc == root_prim:
            continue
        if not desc.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        # Skip if any ancestor (excluding the prefix root) already carries
        # RigidBodyAPI — we only want the topmost one.
        ancestor = desc.GetParent()
        is_topmost = True
        while ancestor and ancestor.IsValid() and ancestor.GetPath() != robot_root_path:
            if ancestor.HasAPI(UsdPhysics.RigidBodyAPI):
                is_topmost = False
                break
            ancestor = ancestor.GetParent()
        if is_topmost:
            urdf_root_link_paths.append(desc.GetPath())

    if not urdf_root_link_paths:
        if logger is not None:
            logger.warn(
                f"[fix_base] no RigidBodyAPI descendant under {robot_root_path}; "
                f"cannot identify URDF root link, leaving joints untouched"
            )
        return 0

    found = 0
    for prim in Usd.PrimRange(root_prim):
        if not prim.IsA(UsdPhysics.FixedJoint):
            continue
        joint = UsdPhysics.FixedJoint(prim)
        b0 = joint.GetBody0Rel().GetTargets()
        b1 = joint.GetBody1Rel().GetTargets()
        if not b0 or not b1:
            continue
        if b0[0] != robot_root_path:
            continue
        if b1[0] not in urdf_root_link_paths:
            # Internal fixed joint — body1 is a deeper-nested rigid body.
            # Leave it strictly alone, regardless of fix_base.
            continue

        enabled_attr = joint.GetJointEnabledAttr()
        if not enabled_attr:
            enabled_attr = joint.CreateJointEnabledAttr()
        enabled_attr.Set(bool(fix_base))
        found += 1
        if logger is not None:
            logger.info(
                f"[fix_base] {'enabled' if fix_base else 'disabled'} world-weld "
                f"{prim.GetPath()} (body0={b0[0]}, body1={b1[0]})"
            )

    if logger is not None and found == 0:
        logger.info(
            f"[fix_base] no world-weld joint detected — likely a ghost-root URDF "
            f"(see diagnose_urdf.py). fix_base={fix_base} will not take effect "
            f"for this robot until the URDF is restructured."
        )
    return found


def _ensure_default_ground_plane(stage, robot_prefix: str, logger=None) -> bool:
    """Guarantee a static ground collider on the Isaac Sim backends.

    Neither ``isaac_physx`` (raw PhysX) nor ``isaac_newton`` (Isaac Sim's
    ``isaacsim.physics.newton`` wrapper) adds a ground plane on their
    own — both build physics purely from the authored USD stage. The
    standalone ``genie_sim_engine_newton`` package is the odd one out:
    it calls ``builder.add_ground_plane()`` unconditionally in
    ``newton/setup/model.py``.

    When the scene YAML leaves ``scene:`` empty,
    ``assemble/_scene_impl.py`` synthesises a stage carrying only a
    *visual* ``/World/GroundPlane`` mesh — no ``UsdPhysics.CollisionAPI``.
    Paired with ``pin_base_to_world: false`` (which disables the URDF
    importer's world weld in ``_apply_fix_base_policy``), the robot's
    base is a free rigid body over an empty physics world and falls
    through the visual ground.

    Policy:
      * If **any** prim outside the robot subtree already carries
        ``UsdPhysics.CollisionAPI``, the scene author owns physics — do
        nothing.
      * Otherwise upgrade ``/World/GroundPlane`` (created by the
        synthesised-scene branch) to a static collider by applying
        ``UsdPhysics.CollisionAPI`` + ``UsdPhysics.MeshCollisionAPI`` with
        ``approximation=none``. The synth mesh is 4 verts / 2 triangles,
        which is a trivial static tri-mesh for either backend.
      * If neither the synth ground nor any other collider exists, log a
        warning and leave the stage untouched — the scene is unusable
        but we don't want to silently invent geometry the operator
        didn't ask for.

    Returns ``True`` iff a new collider was authored.
    """
    robot_root = Sdf.Path(f"/{robot_prefix}")

    for prim in stage.Traverse():
        prim_path = prim.GetPath()
        if prim_path == robot_root or prim_path.HasPrefix(robot_root):
            continue
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            if logger is not None:
                logger.info(f"[default_ground] existing collider {prim_path} — skipping default ground")
            return False

    ground_path = Sdf.Path("/World/GroundPlane")
    ground = stage.GetPrimAtPath(ground_path)
    if not ground or not ground.IsValid():
        if logger is not None:
            logger.warn(
                "[default_ground] no collider found under stage and no "
                f"{ground_path} to upgrade — physics has no ground plane; "
                "robot will free-fall under gravity. Either point the "
                "scene YAML's ``scene:`` at a USD with a physics ground, "
                "or set ``pin_base_to_world: true`` to weld the base."
            )
        return False

    UsdPhysics.CollisionAPI.Apply(ground)
    mesh_api = UsdPhysics.MeshCollisionAPI.Apply(ground)
    approx_attr = mesh_api.CreateApproximationAttr()
    if not approx_attr.HasAuthoredValue():
        approx_attr.Set("none")
    if logger is not None:
        logger.info(
            f"[default_ground] applied CollisionAPI to {ground_path} "
            "(synthesised empty-scene default; shared by isaac_physx and "
            "isaac_newton. Standalone newton keeps using "
            "builder.add_ground_plane())"
        )
    return True


def _collect_body_paths(stage, robot_prefix: str) -> List[str]:
    """Return absolute USD prim paths for every rigid body the engine
    should track — robot links AND free-joint scene bodies.

    Three asset layouts must be supported:

    * Isaac Sim 4.x/5.x ``URDFParseAndImportFile`` — flat: every link is a
      direct child of ``/<robot_prefix>``.
    * Isaac Sim 6.x raw ``urdf_usd_converter`` — single-file flat: same as
      4.x/5.x in terms of layout.
    * Isaac Sim 6.x Asset Structure 3.0 — links live deep under
      ``/<robot_prefix>/Geometry/.../link``, with ``/<robot_prefix>``'s
      direct children being organisational scopes (``Geometry``, ``Physics``,
      ``Sensor``, …) rather than links.

    The unified discriminator is ``HasAPI(RigidBodyAPI)``: every URDF link
    that ends up as a real rigid body carries it, regardless of which
    converter authored the stage. Free-joint scene bodies authored under
    ``/World`` also carry it once the assemble stage marks them rigid —
    those must reach the renderer via ``/tf_scene`` so OVRTX tracks
    physics-driven motion. Kinematic / static scene bodies (no
    ``RigidBodyAPI``, or ``rigidBodyEnabled=false``) are skipped — they
    stay at their authored USD pose in the render stage.

    ``snapshot_body_transforms`` returns **world-frame** poses (see its
    docstring) so this list is safe to union across the robot and
    ``/World`` subtrees: both channels want world-frame.
    """
    paths: List[str] = []
    seen: set = set()

    def _walk(root_path: str) -> None:
        root_prim = stage.GetPrimAtPath(root_path)
        if not root_prim or not root_prim.IsValid():
            return
        for prim in Usd.PrimRange(root_prim):
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            # Skip explicitly-disabled rigid bodies (kinematic /
            # static-collider entries authored with the API for shape
            # purposes but ``physics:rigidBodyEnabled=false``). Only
            # bodies PhysX integrates are wire-worthy.
            enabled_attr = prim.GetAttribute("physics:rigidBodyEnabled")
            if enabled_attr and enabled_attr.HasAuthoredValue():
                if enabled_attr.Get() is False:
                    continue
            p = prim.GetPath().pathString
            if p in seen:
                continue
            seen.add(p)
            paths.append(p)

    _walk(f"/{robot_prefix}")
    _walk("/World")
    return paths


def _collect_joints(stage) -> Tuple[List[str], Dict[str, str]]:
    joint_names: List[str] = []
    joint_prim_map: Dict[str, str] = {}
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint):
            name = prim.GetName()
            joint_names.append(name)
            joint_prim_map[name] = prim.GetPath().pathString
    return joint_names, joint_prim_map


def _collect_joints_as3(stage, robot_usda: str) -> Tuple[List[str], Dict[str, str]]:
    """Read composed joints; never fabricate paths from an inactive payload.

    Isaac 6.2 moved common joint definitions into physics.usda, inherited by
    physx.usda. The Physics variant is selected before the first app update.
    """
    names, pmap = _collect_joints(stage)
    if not names:
        raise RuntimeError(f"AS3 robot {robot_usda} composed zero movable joints; check its Physics variant/payloads")
    return names, pmap


def _configure_drives(
    stage,
    joint_prim_map: Dict[str, str],
    logger,
    params: PhysicsParams,
    from_urdf: bool,
    physics_engine: str,
) -> Dict[str, str]:
    """Apply per-joint DriveAPI tuning, dispatching by classified joint role.

    Returns a ``{name: kind}`` map so the articulation-init pass can reuse
    the same classification (avoid double-walking the regex set).

    **Option C — USD writes as model-build seed:**

    ``_configure_drives`` writes DriveAPI attributes unconditionally for all
    physics engines.  The writes serve two distinct purposes depending on
    which backend is active:

    * **PhysX** — USD ``drive:angular:physics:*`` attributes are the
      *boot seed* that ``World.reset()``'s 5 init ticks consume before
      ``_init_articulation`` calls the tensor handle.  Without a seed
      PhysX initialises from the importer's too-soft defaults
      (kp=625, kd=0) and the gripper droops for the first ~80 ms.
    * **Newton wrapper** (``isaac_newton``) — Newton's USD importer
      reads the same DriveAPI to populate ``Model.joint_target_ke`` /
      ``joint_target_kd`` AND, critically, to decide whether to create a
      POSITION actuator for each DOF at ``ModelBuilder.finalize()`` time.
      ``JointTargetMode.from_gains(target_ke=0, target_kd=0, has_drive=True)``
      → ``EFFORT`` mode → no position actuator created → no force when
      ``apply_action`` later writes ``joint_target_pos``.  So a follower
      joint whose URDF→USD output left ``drive:angular:physics:stiffness``
      at 0 will not be drivable even after the runtime tensor handle's
      ``set_dof_stiffnesses`` updates ``joint_target_ke`` (the kernel
      only walks existing actuators).  Once the actuator exists, the
      runtime tensor handle remains the source of truth for the actual
      gain values at solve time.

    Gripper handling is route-aware:

    * URDF route — the importer authors a soft default DriveAPI on the
      single master joint.  The master is identified by the **absence**
      of a mimic schema (``PhysxMimicJointAPI:<axis>`` or
      ``NewtonMimicAPI``), not by ``HasAPI(DriveAPI)`` — every URDF
      revolute joint (mimics included) gets a vestigial DriveAPI, so a
      DriveAPI-based discriminator would try to apply master tuning to
      all followers and fight the mimic constraint.  We override the
      master (``stiffness=1e4, damping=10, armature=0.001``) and zero
      the mimics' natural-frequency / damping-ratio.  Under
      ``isaac_newton`` we ALSO author non-zero stiffness/damping on the
      mimic followers so Newton creates POSITION actuators for them
      at finalize time — without this seed the followers stay in
      ``EFFORT`` mode and the software-broadcast ``apply_action``
      targets have nothing to push against.  The follower DriveAPI write
      is gated to ``isaac_newton`` to avoid creating a parallel PhysX
      drive that would fight the ``PhysxMimicJointAPI`` constraint on
      the PhysX backend.
    * Pre-baked USD route — the gripper is already tuned in
      the USD authoring layer; we leave it alone and dump the snapshot
      for offline review.
    """
    drv_chassis = params.drive_chassis_drive_joint
    drv_steer = params.drive_chassis_steer_joint
    drv_revolute = params.drive_default_revolute
    drv_prismatic = params.drive_default_prismatic

    kinds: Dict[str, str] = {}
    counts: Dict[str, int] = {
        k: 0 for k in (JK_BODY, JK_ARM, JK_HEAD, JK_GRIPPER, JK_CHASSIS_DRIVE, JK_CHASSIS_STEER, JK_OTHER)
    }
    unknown_names: List[str] = []

    for name, path in joint_prim_map.items():
        prim = stage.GetPrimAtPath(path)
        if not prim:
            unknown_names.append(name)
            continue
        kind = classify_joint(name, prim)
        kinds[name] = kind
        counts[kind] = counts.get(kind, 0) + 1

        if kind == JK_CHASSIS_DRIVE:
            _apply_chassis_drive(prim, name, logger, drv_chassis)
            continue

        if kind == JK_CHASSIS_STEER:
            # Steering rack: position-controlled like body/arm/head, but
            # stiffer and snappier so it tracks setpoints fast & accurately.
            # Same drive shape as the regular path, just different params.
            _apply_regular_drive(prim, name, logger, drv_steer, drv_steer)
            continue

        if kind == JK_GRIPPER:
            # Two-track gripper handling — engine-free under Option C:
            #
            #   * URDF route — the importer authors the master joint with
            #     a too-soft DriveAPI (stiffness=625, damping=0) and the
            #     mimic followers with a PhysxMimicJointAPI whose default
            #     compliance lets the linkage flex under gravity. Identify
            #     the master by the **absence** of a mimic schema, not by
            #     ``HasAPI(DriveAPI)``: the URDF→USD converters all author
            #     DriveAPI on every revolute joint (mimics included,
            #     vestigially), so a DriveAPI-based discriminator gives the
            #     master tuning to all five followers, fights the mimic
            #     constraint, and freezes the gripper.
            #
            #     The mimic schema name varies by pipeline:
            #       - Isaac Sim 6.x after ``convert_joints_attributes`` —
            #         ``PhysxMimicJointAPI:<axis>`` (instance API).
            #       - Isaac Sim 6.x raw Newton output (no PhysX post-pass) —
            #         ``NewtonMimicAPI`` only.
            #       - Isaac Sim 4.x/5.x ``URDFParseAndImportFile`` — neither
            #         mimic schema is authored. On that pipeline mimic
            #         constraints aren't enforced at all unless the URDF
            #         itself authored the second joint with a parallel-jaw
            #         kinematic surrogate. Every gripper joint is
            #         independently driven; that's a known limitation of
            #         the older importer (not worked around here).
            #     Detect the first two; the third falls through to the
            #     master branch which is the correct behaviour for both
            #     6.x mimics and the 4.x/5.x no-mimic case.
            #
            #   * Pre-baked USD route — the gripper is already
            #     tuned in-USD (the genie crsB stage authors stiffness=100,
            #     damping=20 etc. directly), so we leave it alone and
            #     just dump the snapshot for offline review.
            #
            # Engine semantics under Option C:
            #
            #   * PhysX honors these USD writes — they're the boot seed
            #     that runs during ``World.reset()``'s 5 init ticks
            #     before ``_init_articulation`` calls the tensor handle.
            #     Without this seed PhysX would init with the importer's
            #     too-soft kp=625 / kd=0 and the gripper droops in the
            #     first 80 ms.
            #   * MuJoCo ignores them — its actuator config lives in
            #     the parallel ``mjc:gainPrm`` / ``mjc:biasPrm`` tree
            #     authored by the URDF→USD importer; PhysX-flavoured
            #     ``drive:angular:physics:*`` attrs are inert under
            #     MuJoCo. Boot seed = no-op = harmless.
            #
            # Either way, the runtime tensor handle in ``_init_articulation``
            # is the source of truth for both backends — see
            # ``set_dof_stiffnesses`` etc. there.
            snap = _snapshot_gripper_state(prim, name)
            route = "URDF" if from_urdf else "pre-baked-USD"
            if from_urdf:
                applied = prim.GetAppliedSchemas()
                has_mimic = any(s.startswith("PhysxMimicJointAPI:") for s in applied) or "NewtonMimicAPI" in applied
                if not has_mimic and (
                    prim.HasAPI(UsdPhysics.DriveAPI, "angular") or prim.HasAPI(UsdPhysics.DriveAPI, "linear")
                ):
                    logger.info(f"[gripper] {route} master joint {name}: applying drive override. state={snap}")
                    _apply_gripper_master_drive(prim, name, logger, params.drive_gripper)
                else:
                    logger.info(f"[gripper] {route} mimic joint {name}: zeroing compliance + armature. state={snap}")
                    _apply_gripper_mimic(prim, name, logger, params.drive_gripper, physics_engine)
            else:
                logger.info(f"[gripper] {route} joint {name}: leaving as-authored. state={snap}")
            continue

        if kind == JK_OTHER:
            unknown_names.append(name)
            continue

        # JK_BODY / JK_ARM / JK_HEAD: regular position-controlled joint.
        # Pick prismatic-vs-revolute drive params from the prim type (some
        # body joints are prismatic, e.g. linear waist columns).
        _apply_regular_drive(prim, name, logger, drv_revolute, drv_prismatic)

    # Hard-fail on any joint the name-driven classifier couldn't bucket.
    # Silent fall-through leaves these joints with the URDF
    # importer's default DriveAPI (kp=625, kd=0) and no per-class
    # tuning, which causes operator-visible failures (gripper droop,
    # arms not tracking commands) hours into a session.  Exit loudly
    # at startup with the joint list and a hint about updating the
    # ``_RE_*`` patterns in this file so the next pipeline rebuild has
    # the right classifier rules.
    if unknown_names:
        raise RuntimeError(
            "[classify] _configure_drives encountered "
            f"{len(unknown_names)} joint(s) the name-driven classifier "
            f"could not bucket: {unknown_names}. Add a regex to "
            f"``kit/stage.py`` (one of ``_RE_BODY`` / ``_RE_ARM`` / "
            f"``_RE_HEAD`` / ``_RE_GRIPPER`` / ``_RE_CHASSIS_WHEEL``) "
            f"so the joint gets a known kind, then re-launch.  "
            f"Aborting startup rather than silently leaving these "
            f"joints with the URDF importer's vestigial drive defaults."
        )

    logger.info(
        f"[classify] body={counts[JK_BODY]} arm={counts[JK_ARM]} head={counts[JK_HEAD]} "
        f"gripper={counts[JK_GRIPPER]} chassis_drive={counts[JK_CHASSIS_DRIVE]} "
        f"chassis_steer={counts[JK_CHASSIS_STEER]} other={counts[JK_OTHER]} "
        f"(from_urdf={from_urdf})"
    )
    if counts[JK_CHASSIS_STEER] == 0 and counts[JK_CHASSIS_DRIVE] > 0:
        logger.warn(
            "[classify] chassis_steer=0 but chassis_drive>0 — steering joints were not recognised. "
            "Check that joint names match the pattern chassis_*wheel*_joint* with bounded limits (<100 rad). "
            "Steering commands via /joint_command will be silently ignored."
        )
    if unknown_names:
        raise RuntimeError(
            f"[classify] {len(unknown_names)} joint(s) had no prim in stage "
            f"(missing USD paths — URDF→USD conversion may be incomplete): {unknown_names}"
        )
    return kinds


def _apply_chassis_drive(prim, name: str, logger, drv_chassis) -> None:
    """Free-spin road wheel: open limits, force-mode angular drive, kp=0."""
    revolute = UsdPhysics.RevoluteJoint(prim)
    if revolute and drv_chassis.free_limits:
        lo_attr = revolute.GetLowerLimitAttr()
        hi_attr = revolute.GetUpperLimitAttr()
        if lo_attr and hi_attr:
            lo, hi = _safe_get(lo_attr), _safe_get(hi_attr)
            lo_attr.Set(-1e20)
            hi_attr.Set(1e20)
            _log_override(logger, "joint", name, "lowerLimit", lo, -1e20)
            _log_override(logger, "joint", name, "upperLimit", hi, 1e20)

    had_drive = prim.HasAPI(UsdPhysics.DriveAPI, "angular")
    if not had_drive:
        UsdPhysics.DriveAPI.Apply(prim, "angular")
        logger.info(f"[override] joint {name}: applied DriveAPI:angular (was absent)")

    drive = UsdPhysics.DriveAPI.Get(prim, "angular")
    if not drive:
        return

    type_attr = drive.GetTypeAttr()
    if type_attr:
        old_type = _safe_get(type_attr)
        type_attr.Set("force")
        _log_override(logger, "drive", f"{name}:angular", "type", old_type, "force")

    stiff_attr = drive.CreateStiffnessAttr()
    damp_attr = drive.CreateDampingAttr()
    maxf_attr = drive.CreateMaxForceAttr()
    tvel_attr = drive.CreateTargetVelocityAttr()
    tpos_attr = drive.CreateTargetPositionAttr()
    old_stiff = _safe_get(stiff_attr) if had_drive else None
    old_damp = _safe_get(damp_attr) if had_drive else None
    old_maxf = _safe_get(maxf_attr) if had_drive else None
    old_tvel = _safe_get(tvel_attr) if had_drive else None
    old_tpos = _safe_get(tpos_attr) if had_drive else None
    stiff_attr.Set(drv_chassis.stiffness)
    damp_attr.Set(drv_chassis.damping)
    maxf_attr.Set(drv_chassis.max_force)
    tvel_attr.Set(0.0)
    tpos_attr.Set(0.0)
    _log_override(logger, "drive", f"{name}:angular", "stiffness", old_stiff, drv_chassis.stiffness)
    _log_override(logger, "drive", f"{name}:angular", "damping", old_damp, drv_chassis.damping)
    _log_override(logger, "drive", f"{name}:angular", "maxForce", old_maxf, drv_chassis.max_force)
    _log_override(logger, "drive", f"{name}:angular", "targetVelocity", old_tvel, 0.0)
    _log_override(logger, "drive", f"{name}:angular", "targetPosition", old_tpos, 0.0)


def _apply_regular_drive(prim, name: str, logger, drv_revolute, drv_prismatic) -> None:
    """Force-mode position drive shared by body / arm / head / fallback."""
    is_prismatic = prim.IsA(UsdPhysics.PrismaticJoint)
    token = "linear" if is_prismatic else "angular"
    had_drive = prim.HasAPI(UsdPhysics.DriveAPI, token)
    if not had_drive:
        UsdPhysics.DriveAPI.Apply(prim, token)
        logger.info(f"[override] joint {name}: applied DriveAPI:{token} (was absent)")

    drive = UsdPhysics.DriveAPI.Get(prim, token)
    if not drive:
        return
    type_attr = drive.GetTypeAttr()
    if type_attr:
        old_type = _safe_get(type_attr)
        type_attr.Set("force")
        _log_override(logger, "drive", f"{name}:{token}", "type", old_type, "force")
    tuning = drv_prismatic if is_prismatic else drv_revolute
    stiff_attr = drive.CreateStiffnessAttr()
    damp_attr = drive.CreateDampingAttr()
    maxf_attr = drive.CreateMaxForceAttr()
    old_stiff = _safe_get(stiff_attr) if had_drive else None
    old_damp = _safe_get(damp_attr) if had_drive else None
    old_maxf = _safe_get(maxf_attr) if had_drive else None
    stiff_attr.Set(tuning.stiffness)
    damp_attr.Set(tuning.damping)
    maxf_attr.Set(tuning.max_force)
    _log_override(logger, "drive", f"{name}:{token}", "stiffness", old_stiff, tuning.stiffness)
    _log_override(logger, "drive", f"{name}:{token}", "damping", old_damp, tuning.damping)
    _log_override(logger, "drive", f"{name}:{token}", "maxForce", old_maxf, tuning.max_force)


# ---------------------------------------------------------------------------
# Gripper tuning
#
# The Isaac Sim URDF→USD importer authors a too-soft DriveAPI on the gripper
# master joint (stiffness=625, damping=0, driveType=acceleration), which
# cannot resist gravity-induced moment on the finger linkage — the gripper
# droops in a pose-dependent way. We override the master only; mimic
# followers are PhysxMimicJointAPI-constrained and have no DriveAPI.
#
# The numeric values come from ``GripperDriveParams`` (loaded from
# ``physics_params.yaml`` ``usd_drive_api.gripper`` block, with built-in defaults that
# reproduce the hand-tuned 1e4 / 10 / 0.001 / 0 / 0 setup). Mimic
# compliance attributes (``physxMimicJoint:rotX:naturalFrequency`` /
# ``dampingRatio``) are forced to 0 by default: any non-zero value adds
# spring/damper compliance into the constraint that fights the master's
# drive and reintroduces the slack we just fixed.
# ---------------------------------------------------------------------------


def _set_armature(prim, name: str, logger, value: float) -> None:
    """Author ``physxJoint:armature`` on a joint prim, logging the override."""
    attr = prim.GetAttribute("physxJoint:armature")
    if not attr:
        attr = prim.CreateAttribute("physxJoint:armature", Sdf.ValueTypeNames.Float)
    old = _safe_get(attr)
    if _values_equal(old, value):
        return
    attr.Set(value)
    _log_override(logger, "joint", name, "physxJoint:armature", old, value)


def _apply_gripper_master_drive(prim, name: str, logger, tuning: GripperDriveParams) -> None:
    """Override the gripper master joint's DriveAPI to resist gravity droop.

    The master is the ONLY joint in a Robotiq-style gripper that carries a
    DriveAPI — mimic followers are constrained via ``PhysxMimicJointAPI``
    and have no drive. We therefore identify the master purely by
    ``HasAPI(DriveAPI, "angular")`` rather than by name, which works for
    both Robotiq 2F-85 and 2F-140 (and any other URDF gripper with one
    drive joint + N mimic joints).
    """
    is_prismatic = prim.IsA(UsdPhysics.PrismaticJoint)
    token = "linear" if is_prismatic else "angular"
    drive = UsdPhysics.DriveAPI.Get(prim, token)
    if not drive:
        return

    type_attr = drive.GetTypeAttr()
    if type_attr:
        old_type = _safe_get(type_attr)
        if old_type != "force":
            type_attr.Set("force")
            _log_override(logger, "drive", f"{name}:{token}", "type", old_type, "force")

    stiff_attr = drive.CreateStiffnessAttr()
    damp_attr = drive.CreateDampingAttr()
    old_stiff = _safe_get(stiff_attr)
    old_damp = _safe_get(damp_attr)
    stiff_attr.Set(tuning.master_stiffness)
    damp_attr.Set(tuning.master_damping)
    _log_override(logger, "drive", f"{name}:{token}", "stiffness", old_stiff, tuning.master_stiffness)
    _log_override(logger, "drive", f"{name}:{token}", "damping", old_damp, tuning.master_damping)

    _set_armature(prim, name, logger, tuning.armature)


def _apply_gripper_mimic(prim, name: str, logger, tuning: GripperDriveParams, physics_engine: str) -> None:
    """Zero out the PhysxMimicJoint compliance and set armature on a mimic.

    PhysX 5's ``PhysxMimicJointAPI`` exposes optional spring/damper
    compliance via ``physxMimicJoint:<axis>:naturalFrequency`` and
    ``physxMimicJoint:<axis>:dampingRatio``. Any non-zero value here turns
    the rigid kinematic constraint into a soft constraint that fights the
    master's drive — visible as residual slack even after the master is
    properly tuned. We force both to the configured values (default 0) so
    the constraint solver enforces the mimic relationship rigidly.

    The axis suffix is deduced from the applied schemas (PhysX exposes it
    per-axis: ``rotX`` / ``transX`` / etc). For Robotiq grippers the suffix
    is always ``rotX`` but we don't hard-code it.

    Newton wrapper note — under ``isaac_newton`` we ALSO author non-zero
    stiffness/damping on the follower's USD ``DriveAPI``.  Newton's USD
    importer maps ``DriveAPI`` ``(stiffness, damping)`` to a
    ``JointTargetMode`` via ``JointTargetMode.from_gains``; the URDF→USD
    converter leaves follower drives at ``stiffness=0, damping=0`` which
    parses as ``EFFORT`` mode, and ``ModelBuilder.finalize()`` then skips
    creating any POSITION actuator for the follower DOF.  Without a
    POSITION actuator the MuJoCo solver has nothing for
    ``update_axis_properties_kernel`` to update when
    ``set_dof_stiffnesses`` (the runtime tensor handle) writes new
    ``joint_target_ke`` values post-reset, and the software-broadcast
    ``apply_action`` position target has nothing to push against —
    followers swing freely with only the rigid ``mjEQ_JOINT`` mimic
    constraint trying to drag them along.  Writing the master's tuning
    onto the follower's DriveAPI before ``World.reset()`` makes Newton
    allocate a POSITION actuator for each follower DOF, which
    ``set_dof_stiffnesses`` then updates at runtime to the configured
    ``art_default`` gains.  The actuator's drive and the mimic
    constraint reinforce each other since ``_apply_joint_commands``
    broadcasts consistent targets to all gripper DOFs.

    Under PhysX this Newton-specific block is skipped — adding a force
    drive on a ``PhysxMimicJointAPI`` follower would author a parallel
    drive that competes with the kinematic mimic constraint.
    """
    schemas: List[str] = []
    try:
        schemas = list(prim.GetAppliedSchemas())
    except Exception:
        return

    mimic_axes: List[str] = [s.split(":", 1)[1] for s in schemas if s.startswith("PhysxMimicJointAPI:")]

    for axis in mimic_axes:
        for attr_suffix, target in (
            ("naturalFrequency", tuning.mimic_natural_frequency),
            ("dampingRatio", tuning.mimic_damping_ratio),
        ):
            attr_name = f"physxMimicJoint:{axis}:{attr_suffix}"
            attr = prim.GetAttribute(attr_name)
            if not attr:
                attr = prim.CreateAttribute(attr_name, Sdf.ValueTypeNames.Float)
            old = _safe_get(attr)
            if not _values_equal(old, target):
                attr.Set(target)
                _log_override(logger, "joint", name, attr_name, old, target)

    # Newton-only follower DriveAPI seed.  See docstring above for why
    # this is gated to ``isaac_newton`` and required only there.
    if physics_engine == "isaac_newton":
        is_prismatic = prim.IsA(UsdPhysics.PrismaticJoint)
        token = "linear" if is_prismatic else "angular"
        if not prim.HasAPI(UsdPhysics.DriveAPI, token):
            UsdPhysics.DriveAPI.Apply(prim, token)
            logger.info(f"[gripper-newton] follower {name}: applied DriveAPI:{token} (was absent)")
        drive = UsdPhysics.DriveAPI.Get(prim, token)
        if drive:
            type_attr = drive.GetTypeAttr()
            if type_attr:
                old_type = _safe_get(type_attr)
                if old_type != "force":
                    type_attr.Set("force")
                    _log_override(logger, "drive", f"{name}:{token}", "type", old_type, "force")
            stiff_attr = drive.CreateStiffnessAttr()
            damp_attr = drive.CreateDampingAttr()
            old_stiff = _safe_get(stiff_attr)
            old_damp = _safe_get(damp_attr)
            stiff_attr.Set(tuning.master_stiffness)
            damp_attr.Set(tuning.master_damping)
            _log_override(logger, "drive", f"{name}:{token}", "stiffness", old_stiff, tuning.master_stiffness)
            _log_override(logger, "drive", f"{name}:{token}", "damping", old_damp, tuning.master_damping)

    _set_armature(prim, name, logger, tuning.armature)


def _locate_physics_scene(stage, logger) -> str:
    scene_path = "/physicsScene"
    scene_prim = stage.GetPrimAtPath(scene_path)
    if not scene_prim or not scene_prim.IsValid():
        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.Scene):
                scene_path = prim.GetPath().pathString
                break
        logger.info(f"PhysX scene: {scene_path}")
    return scene_path


# ---------------------------------------------------------------------------
# Articulation init + command application (private)
# ---------------------------------------------------------------------------


def _seed_drive_target_positions(
    stage,
    joint_prim_map: Dict[str, str],
    robot_prefix: str,
    init_joint_pos: Optional[Dict[str, "JointInitSpec"]],
    logger,
) -> int:
    """Author USD ``drive:*:physics:targetPosition`` for joints in ``init_joint_pos``.

    WHY THIS EXISTS
    ---------------
    Two parallel target stores exist for every position-controlled joint:

      1. **USD layer** ``drive:angular:physics:targetPosition`` (degrees)
         or ``drive:linear:physics:targetPosition`` (metres). This is the
         authored value PhysX reads when ``World.reset()`` instantiates
         the articulation handle. It is the SEED.
      2. **Tensor handle** populated by
         ``ArticulationView.set_joint_position_targets`` (radians/metres
         for revolute/prismatic respectively). This is the runtime
         setpoint the controller uses each tick.

    The Isaac Sim URDF importer authors (1) as ``0`` for every joint.
    Without this function the seed says "go to 0", and on the very first
    physics tick(s) — which fire during ``World.reset()`` and the 5
    ``simulation_app.update()`` calls right after — the position
    controller drives the joint toward 0 even though we later push (2)
    to the requested ``init_joint_pos`` via the tensor API.

    Result the user sees: drive's ``targetPosition`` attribute (visible
    in /joint_states or in the USD inspector) does NOT match
    ``init_joint_pos``, and the robot snaps from the requested pose
    toward 0 between init and the first scheduler tick.

    The fix authors (1) to match (2) BEFORE ``World.reset()`` so the
    seed is correct. The tensor write in ``_init_articulation`` then
    only has to maintain coherence at runtime.

    Returns the number of joints seeded.
    """
    if not init_joint_pos:
        return 0

    seeded = 0
    unknown: List[str] = []
    for jname, spec in init_joint_pos.items():
        path = joint_prim_map.get(jname) or joint_prim_map.get(f"{robot_prefix}_{jname}")
        if not path:
            unknown.append(jname)
            continue
        prim = stage.GetPrimAtPath(path)
        if not prim:
            unknown.append(jname)
            continue
        is_prismatic = prim.IsA(UsdPhysics.PrismaticJoint)
        token = "linear" if is_prismatic else "angular"
        if not prim.HasAPI(UsdPhysics.DriveAPI, token):
            UsdPhysics.DriveAPI.Apply(prim, token)
        drive = UsdPhysics.DriveAPI.Get(prim, token)
        if not drive:
            unknown.append(jname)
            continue
        # USD drive units: linear→metres (verbatim), angular→degrees.
        # ``init_joint_pos`` YAML units: prismatic→metres, revolute→degrees.
        # So no unit conversion is needed here; the YAML is already in
        # USD's preferred unit on both sides.
        usd_target = float(spec.value)
        tpos_attr = drive.CreateTargetPositionAttr()
        old_tpos = _safe_get(tpos_attr) if tpos_attr else None
        tpos_attr.Set(usd_target)
        unit = "m" if is_prismatic else "deg"
        _log_override(logger, "drive", f"{jname}:{token}", f"targetPosition[{unit}]", old_tpos, usd_target)
        # Also seed the joint's INITIAL STATE position so PhysX
        # instantiates the joint AT the target instead of at zero.
        # Without this the drive target says "go to X" but the joint
        # starts at 0, and the controller drives the arm from 0 → X
        # across the first physics ticks (visible as a startup swing
        # before the user's first command).  Writing
        # ``state:angular:physics:position`` (revolute, degrees) /
        # ``state:linear:physics:position`` (prismatic, metres) makes
        # the init a pure teleport — the drive target equals the state
        # position, residual error is zero, no controller work on
        # tick 0.  Mirrors the drive-target unit convention above.
        state_attr_name = "state:linear:physics:position" if is_prismatic else "state:angular:physics:position"
        spos_attr = prim.GetAttribute(state_attr_name)
        if not spos_attr or not spos_attr.IsValid():
            spos_type = Sdf.ValueTypeNames.Float if is_prismatic else Sdf.ValueTypeNames.Float
            spos_attr = prim.CreateAttribute(state_attr_name, spos_type)
        old_spos = _safe_get(spos_attr)
        spos_attr.Set(usd_target)
        _log_override(logger, "joint-state", jname, f"position[{unit}]", old_spos, usd_target)
        seeded += 1

    if unknown:
        logger.warn(f"_seed_drive_target_positions: unknown joint(s) ignored: {unknown}")
    if seeded:
        logger.info(
            f"_seed_drive_target_positions: authored targetPosition + state:position on "
            f"{seeded} joint(s) (mimic followers inherit through the constraint)"
        )
    return seeded


def _init_articulation(
    stage,
    robot_prefix: str,
    logger,
    params: PhysicsParams,
    from_urdf: bool,
    init_joint_pos: Optional[Dict[str, JointInitSpec]] = None,
    physics_engine: str = "isaac_physx",
    joint_params=None,
) -> Tuple[Optional[object], Dict[str, int], Optional[np.ndarray], Dict[str, List[Tuple[str, float, float]]]]:
    art_default = params.art_default
    art_drive = params.art_chassis_drive
    art_steer = params.art_chassis_steer
    # Per-sub-class actuator gains.  ``art_default`` is the fallback
    # for any kind we can't sub-class (pre-baked USD URDFs, unknown joints
    # in the JK_ARM bucket).  Every other body / head / arm sub-class
    # has its own tiered values matching the reference G2 MJCF —
    # see ``common/params.py:_DEFAULTS``.
    art_body = params.art_body
    art_head = params.art_head
    art_arm_sh = params.art_arm_shoulder
    art_arm_md = params.art_arm_mid
    art_arm_wr = params.art_arm_wrist
    try:
        from isaacsim.core.prims import SingleArticulation as Articulation

        art_prim_path = f"/{robot_prefix}"
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                art_prim_path = prim.GetPath().pathString
                break
        logger.info(f"Articulation root: {art_prim_path}")

        articulation = Articulation(prim_path=art_prim_path, name="genie_art")
        articulation.initialize()
        dof_names = list(articulation.dof_names or [])
        art_joint_index = {name: i for i, name in enumerate(dof_names)}
        num_dof = articulation.num_dof
        logger.info(f"Articulation initialized with {num_dof} DOF: {dof_names}")

        art_view = articulation._articulation_view  # noqa: SLF001

        # Parse ``NewtonMimicAPI`` from the staged USD once — used to
        # broadcast master targets to follower DOFs on init pose AND in
        # ``_apply_joint_commands``. On PhysX this is redundant
        # (PhysxMimicJointAPI binds followers at the constraint level)
        # but harmless; on isaac_newton the Newton wrapper's
        # ``apply_action`` doesn't move followers from a master-only
        # command, so the software broadcast is the only thing that
        # keeps multi-finger grippers in sync.
        mimic_followers = _parse_mimic_from_stage(stage, logger)

        prev_kps: Optional[np.ndarray] = None
        prev_kds: Optional[np.ndarray] = None
        prev_max_efforts: Optional[np.ndarray] = None
        try:
            cur_kps, cur_kds = art_view.get_gains()
            prev_kps = _view_readback(cur_kps).reshape(-1).astype(np.float64, copy=False)
            prev_kds = _view_readback(cur_kds).reshape(-1).astype(np.float64, copy=False)
        except Exception as exc:
            logger.warn(f"could not read pre-existing articulation gains: {exc}")
        try:
            cur_max = art_view.get_max_efforts()
            prev_max_efforts = _view_readback(cur_max).reshape(-1).astype(np.float64, copy=False)
        except Exception as exc:
            logger.warn(f"could not read pre-existing articulation max_efforts: {exc}")

        # Seed everything with the regular-joint defaults; per-kind branches
        # below override only the DOFs that need different gains. DOFs we
        # want to leave untouched (e.g. grippers on the URDF route) get the
        # readback value rewritten into them so set_gains() is a no-op for
        # those slots.
        kps = np.full(num_dof, art_default.kp, dtype=np.float32)
        kds = np.full(num_dof, art_default.kd, dtype=np.float32)
        max_efforts = np.full(num_dof, art_default.max_effort, dtype=np.float32)

        # Per-DOF classification — same rules as _configure_drives so the
        # two passes can never disagree.
        dof_kinds: Dict[str, str] = {}
        # Cheap helper: turn a DOF name into its joint prim by walking
        # joint paths under the articulation root. We don't have direct
        # access here, so re-walk the stage.
        joint_prim_by_name: Dict[str, object] = {}
        spherical_dofs: set[str] = set()
        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint):
                joint_prim_by_name[prim.GetName()] = prim
            elif physics_engine == "isaac_newton" and prim.IsA(UsdPhysics.SphericalJoint):
                spherical_dofs.update(f"{prim.GetName()}:{axis}" for axis in range(3))

        skip_idxs: List[int] = []  # DOFs we'll restore to readback (no override)
        wheels: List[int] = []
        steers: List[int] = []
        others: List[int] = []
        gripper_masters: List[int] = []
        gripper_followers: List[int] = []
        unknown_dofs: List[str] = []
        drv_grip = params.drive_gripper
        for name, idx in art_joint_index.items():
            if name in spherical_dofs:
                # Newton exposes spherical axes separately; scalar robot tuning does not apply.
                for values, previous in ((kps, prev_kps), (kds, prev_kds), (max_efforts, prev_max_efforts)):
                    if previous is None or idx >= previous.size:
                        raise RuntimeError(f"Cannot read back spherical joint parameters for {name}")
                    values[idx] = previous[idx]
                skip_idxs.append(idx)
                continue
            jp = joint_prim_by_name.get(name)
            kind = classify_joint(name, jp) if jp is not None else JK_OTHER
            dof_kinds[name] = kind

            if kind == JK_CHASSIS_DRIVE:
                wheels.append(idx)
                kps[idx], kds[idx], max_efforts[idx] = art_drive.kp, art_drive.kd, art_drive.max_effort
            elif kind == JK_CHASSIS_STEER:
                steers.append(idx)
                kps[idx], kds[idx], max_efforts[idx] = art_steer.kp, art_steer.kd, art_steer.max_effort
            elif kind == JK_GRIPPER:
                # Master vs. follower discrimination is the whole point
                # of this branch.  Authoritative signal is the
                # ``newton:mimicJoint`` relationship that the URDF→USD
                # importer (and ``_apply_mimic_joint_overlay``) write
                # on followers — masters never carry it.  We don't trust
                # ``HasAPI(DriveAPI)`` alone because the importer authors
                # vestigial DriveAPIs on every gripper joint, and the
                # ``NewtonMimicAPI`` schema (added by the assemble overlay
                # so Newton's importer installs the equality constraint)
                # is registered only inside the kit/Newton runtime — its
                # absence in ``GetAppliedSchemas()`` is meaningless under
                # a vanilla pxr.  The mimic *relationship* is plain USD
                # and visible everywhere.
                #
                # Master  → use ``params.drive_gripper.master_stiffness``
                #           / ``master_damping`` (kp=1e4, kd=10 by
                #           default — the values the reference G2 MJCF
                #           uses for the inner_joint1 actuator).
                # Follower → kp=0, kd=0.  The equality constraint Newton
                #           installs from ``NewtonMimicAPI`` propagates
                #           the master's motion; a non-zero PD on the
                #           follower fights that constraint and produces
                #           the low-frequency limit cycle the operator
                #           sees as a "swinging" gripper.
                #
                # Both branches preserve the URDF effort cap from
                # ``prev_max_efforts`` (with a 0-guard — Newton's tensor
                # view returns 0 for follower joints whose DriveAPI was
                # zeroed by the overlay; ``art_default.max_effort`` is
                # the safe fallback in that case).
                is_follower = False
                if jp is not None:
                    rel = jp.GetRelationship("newton:mimicJoint")
                    if rel and rel.HasAuthoredTargets():
                        is_follower = True

                if is_follower:
                    kps[idx] = 0.0
                    kds[idx] = 0.0
                    gripper_followers.append(idx)
                else:
                    kps[idx] = drv_grip.master_stiffness
                    kds[idx] = drv_grip.master_damping
                    gripper_masters.append(idx)

                if prev_max_efforts is not None and idx < prev_max_efforts.size:
                    effort = float(prev_max_efforts[idx])
                    if effort > 0.0:
                        max_efforts[idx] = effort
            elif kind == JK_OTHER:
                # Hard-fail at the second classifier pass too: matches
                # the policy in ``_configure_drives``.  Silently
                # leaving an unrecognized DOF "as-authored" means
                # the URDF→USD importer's vestigial DriveAPI
                # (kp=625, kd=0) runs the joint, which manifests as
                # the kind of "this joint sometimes doesn't track
                # commands" symptoms that take hours to bisect.  We
                # collect every offender in one pass so the operator
                # sees the full list, then raise after the loop.
                unknown_dofs.append(name)
            elif kind == JK_BODY:
                kps[idx], kds[idx], max_efforts[idx] = art_body.kp, art_body.kd, art_body.max_effort
                others.append(idx)
            elif kind == JK_HEAD:
                kps[idx], kds[idx], max_efforts[idx] = art_head.kp, art_head.kd, art_head.max_effort
                others.append(idx)
            elif kind == JK_ARM_SHOULDER:
                kps[idx], kds[idx], max_efforts[idx] = art_arm_sh.kp, art_arm_sh.kd, art_arm_sh.max_effort
                others.append(idx)
            elif kind == JK_ARM_MID:
                kps[idx], kds[idx], max_efforts[idx] = art_arm_md.kp, art_arm_md.kd, art_arm_md.max_effort
                others.append(idx)
            elif kind == JK_ARM_WRIST:
                kps[idx], kds[idx], max_efforts[idx] = art_arm_wr.kp, art_arm_wr.kd, art_arm_wr.max_effort
                others.append(idx)
            else:
                # Generic ``JK_ARM`` (no shoulder/mid/wrist match)
                # plus pre-baked USD gripper paths.  Both keep the
                # fallback ``art_default`` gains the seed wrote at
                # ``np.full`` time — overriding here would erase
                # the hand-tuning pre-baked USDs depend on.
                others.append(idx)

        if unknown_dofs:
            raise RuntimeError(
                "[classify] _init_articulation encountered "
                f"{len(unknown_dofs)} articulation DOF(s) the name-driven "
                f"classifier could not bucket: {unknown_dofs}. Add a regex "
                f"to ``kit/stage.py`` (one of ``_RE_BODY`` / ``_RE_ARM`` / "
                f"``_RE_HEAD`` / ``_RE_GRIPPER`` / ``_RE_CHASSIS_WHEEL``) "
                f"so the joint gets a known kind, then re-launch.  "
                f"Aborting startup rather than silently leaving these "
                f"DOFs with the readback gains (typically the URDF "
                f"importer's vestigial defaults)."
            )

        from genie_sim_engine_isaacsim.kit.joint_params import apply_articulation_joint_params

        apply_articulation_joint_params(
            stage,
            {name: str(prim.GetPath()) for name, prim in joint_prim_by_name.items()},
            art_joint_index,
            joint_params,
            kps,
            kds,
            max_efforts,
        )

        for name, idx in art_joint_index.items():
            if idx in skip_idxs:
                # Don't emit override lines for joints we're preserving.
                continue
            if prev_kps is not None and idx < prev_kps.size:
                _log_override(logger, "articulation", name, "kp", float(prev_kps[idx]), float(kps[idx]))
            else:
                _log_override(logger, "articulation", name, "kp", None, float(kps[idx]))
            if prev_kds is not None and idx < prev_kds.size:
                _log_override(logger, "articulation", name, "kd", float(prev_kds[idx]), float(kds[idx]))
            else:
                _log_override(logger, "articulation", name, "kd", None, float(kds[idx]))
            if prev_max_efforts is not None and idx < prev_max_efforts.size:
                _log_override(
                    logger,
                    "articulation",
                    name,
                    "max_effort",
                    float(prev_max_efforts[idx]),
                    float(max_efforts[idx]),
                )
            else:
                _log_override(logger, "articulation", name, "max_effort", None, float(max_efforts[idx]))

        # Gain authoring branches by engine because ``Articulation.set_gains``
        # and ``set_max_efforts`` hardcode ``device="cpu"`` for indices, then
        # combine with the sim view's ``get_dof_stiffnesses()`` result. PhysX
        # returns CPU tensors there, so the combine works. Newton's torch
        # frontend returns CUDA tensors, so it explodes with::
        #
        #   set_gains failed: Expected all tensors to be on the same
        #   device, but found at least two devices, cuda:0 and cpu!
        #
        # Other ``Articulation`` methods (``set_joint_positions``,
        # ``set_joint_position_targets``, ``apply_action``) use
        # ``self._device == "cuda:0"`` everywhere and work fine — only
        # ``set_gains`` / ``set_max_efforts`` have the CPU pin. So under
        # Newton we bypass those wrappers and call the underlying
        # ``_physics_view.set_dof_stiffnesses`` /
        # ``set_dof_dampings`` / ``set_dof_max_forces`` directly. They
        # accept torch tensors via the wrapper's ``_wrap_input_tensor``
        # (which converts to wp internally) and ``set_dof_stiffnesses``
        # ALSO calls ``_notify_joint_dof_properties_changed`` on the
        # solver — meaning the new kp/kd land in MuJoCo-Warp's actuator
        # parameters on the same step. PhysX path keeps using the
        # higher-level wrapper (zero behaviour change).
        try:
            kps_2d = _view_input(np.expand_dims(kps, 0), engine=physics_engine)
            kds_2d = _view_input(np.expand_dims(kds, 0), engine=physics_engine)
            max_2d = _view_input(np.expand_dims(max_efforts, 0), engine=physics_engine)
            if physics_engine == "isaac_newton":
                import torch  # noqa: PLC0415

                arti_indices = torch.tensor([0], dtype=torch.int32, device="cuda:0")
                art_view._physics_view.set_dof_stiffnesses(kps_2d, arti_indices)  # noqa: SLF001
                art_view._physics_view.set_dof_dampings(kds_2d, arti_indices)  # noqa: SLF001
                art_view._physics_view.set_dof_max_forces(max_2d, arti_indices)  # noqa: SLF001
                logger.info(
                    f"Set gains via Newton _physics_view (bypassing the cpu/cuda set_gains pin): "
                    f"{len(wheels)} chassis_drive (kd={art_drive.kd:g}), "
                    f"{len(steers)} chassis_steer (kp={art_steer.kp:g}), "
                    f"{len(others)} body/arm/head tiered "
                    f"(body={art_body.kp:g}/{art_body.kd:g}, "
                    f"head={art_head.kp:g}/{art_head.kd:g}, "
                    f"arm_sh={art_arm_sh.kp:g}/{art_arm_sh.kd:g}, "
                    f"arm_md={art_arm_md.kp:g}/{art_arm_md.kd:g}, "
                    f"arm_wr={art_arm_wr.kp:g}/{art_arm_wr.kd:g}), "
                    f"{len(gripper_masters)} gripper_master "
                    f"(kp={drv_grip.master_stiffness:g},kd={drv_grip.master_damping:g}), "
                    f"{len(gripper_followers)} gripper_follower (kp=0,kd=0 — equality-constraint-only), "
                    f"{len(skip_idxs)} preserved spherical axes"
                )
            else:
                art_view.set_gains(kps=kps_2d, kds=kds_2d)
                art_view.set_max_efforts(values=max_2d)
                logger.info(
                    f"Set gains: {len(wheels)} chassis_drive (kd={art_drive.kd:g}), "
                    f"{len(steers)} chassis_steer (kp={art_steer.kp:g}), "
                    f"{len(others)} body/arm/head tiered "
                    f"(body={art_body.kp:g}/{art_body.kd:g}, "
                    f"head={art_head.kp:g}/{art_head.kd:g}, "
                    f"arm_sh={art_arm_sh.kp:g}/{art_arm_sh.kd:g}, "
                    f"arm_md={art_arm_md.kp:g}/{art_arm_md.kd:g}, "
                    f"arm_wr={art_arm_wr.kp:g}/{art_arm_wr.kd:g}), "
                    f"{len(gripper_masters)} gripper_master "
                    f"(kp={drv_grip.master_stiffness:g},kd={drv_grip.master_damping:g}), "
                    f"{len(gripper_followers)} gripper_follower (kp=0,kd=0 — equality-constraint-only), "
                    f"{len(skip_idxs)} preserved spherical axes"
                )
                logger.info("Set max_efforts on all DOFs")
        except Exception as exc:
            if joint_params:
                raise RuntimeError("Failed to apply sysid articulation gains") from exc
            logger.warn(f"set_gains/set_max_efforts failed: {exc}")

        # Apply scene-sourced initial joint pose BEFORE the hold-targets
        # block, so the readback below captures the requested pose and
        # ``hold_targets`` locks the controller setpoint to it (otherwise
        # the controller would snap the robot back to the USD-default zero
        # on the first physics tick).
        #
        # Unit convention:
        #   * RevoluteJoint  → degrees → radians (multiply by π/180)
        #   * PrismaticJoint → metres  → metres (verbatim)
        # DOF-name lookup falls back to the prefixed form so YAMLs can
        # use either ``arm_joint1`` or ``ur5_arm_joint1`` interchangeably.
        #
        # IMPORTANT: ``set_joint_positions`` is a tensor-API state write
        # that does not synchronously commit before the next physics tick,
        # so a readback via ``articulation.get_joint_positions()`` made
        # immediately afterwards may still return the USD-default (zero)
        # positions. If the downstream hold-targets block then derives
        # ``targets`` from that stale readback, the controller setpoint
        # is locked to zero and the robot snaps back from the just-set
        # init pose on tick 0 — visible as a sudden jump.
        # We avoid the race by keeping the array we wrote and reusing it
        # as the hold target via ``init_pose_for_hold``.
        init_pose_for_hold: Optional[np.ndarray] = None
        if init_joint_pos:
            try:
                import math

                _DEG2RAD = math.pi / 180.0

                current_for_init = _view_readback(articulation.get_joint_positions())
                if current_for_init is None:
                    current_for_init = np.zeros(num_dof, dtype=np.float32)
                pos_array = np.asarray(current_for_init, dtype=np.float32).reshape(-1).copy()

                applied: List[str] = []
                unknown: List[str] = []
                for jname, spec in init_joint_pos.items():
                    idx = art_joint_index.get(jname)
                    if idx is None:
                        idx = art_joint_index.get(f"{robot_prefix}_{jname}")
                    if idx is None:
                        unknown.append(jname)
                        continue
                    resolved_name = dof_names[idx]
                    jp = joint_prim_by_name.get(resolved_name)
                    is_prismatic = jp is not None and jp.IsA(UsdPhysics.PrismaticJoint)
                    new_value = float(spec.value) if is_prismatic else float(spec.value) * _DEG2RAD
                    old_value = float(pos_array[idx])
                    pos_array[idx] = new_value
                    unit = "m" if is_prismatic else "rad"
                    _log_override(logger, "articulation", resolved_name, f"position[{unit}]", old_value, new_value)
                    applied.append(f"{resolved_name}={spec.value}{'m' if is_prismatic else '°'}")

                # Broadcast through URDF mimic so follower joints (e.g.
                # the 5 Robotiq-85 followers of ``gripper_active_master_joint``)
                # land at the right relative position on tick 0. Without
                # this, ``set_joint_positions`` leaves followers at zero
                # and the first physics step rips them toward the
                # constraint or — under Newton — leaves them flailing.
                for master, followers in mimic_followers.items():
                    midx = art_joint_index.get(master)
                    if midx is None:
                        continue
                    mval = float(pos_array[midx])
                    for fname, mult, off in followers:
                        fidx = art_joint_index.get(fname)
                        if fidx is None:
                            continue
                        pos_array[fidx] = mult * mval + off

                if applied:
                    art_view.set_joint_positions(
                        positions=_view_input(np.expand_dims(pos_array, 0), engine=physics_engine)
                    )
                    # Also push the same array as the controller setpoint
                    # IMMEDIATELY, before any readback, so the controller
                    # cannot snap the robot back toward zero on tick 0.
                    art_view.set_joint_position_targets(
                        positions=_view_input(np.expand_dims(pos_array, 0), engine=physics_engine)
                    )
                    init_pose_for_hold = pos_array.copy()
                    logger.info(f"Applied init_joint_pos to {len(applied)} DOF(s): {', '.join(applied)}")
                if unknown:
                    logger.warn(f"init_joint_pos: unknown DOF(s) ignored: {unknown}")
            except Exception as exc:
                logger.warn(f"apply init_joint_pos failed: {exc}")

        hold_targets: Optional[np.ndarray] = None
        try:
            if init_pose_for_hold is not None:
                # Trust the array we just wrote — bypasses the tensor-API
                # readback race described above.
                targets = init_pose_for_hold.astype(np.float32, copy=True)
                art_view.set_joint_position_targets(
                    positions=_view_input(np.expand_dims(targets, 0), engine=physics_engine)
                )
                logger.info(
                    f"Holding all {num_dof} joints at init_joint_pos as targets " f"(bypassing readback to avoid jump)"
                )
                hold_targets = targets.copy()
            else:
                current = _view_readback(articulation.get_joint_positions())
                if current is not None:
                    targets = np.array(current, dtype=np.float32)
                    art_view.set_joint_position_targets(
                        positions=_view_input(np.expand_dims(targets, 0), engine=physics_engine)
                    )
                    logger.info(f"Holding all {num_dof} joints at current positions as targets")
                    hold_targets = targets.copy()
        except Exception as exc:
            logger.warn(f"initial hold failed: {exc}")

        try:
            art_view.set_joint_velocity_targets(
                velocities=_view_input(np.expand_dims(np.zeros(num_dof, dtype=np.float32), 0), engine=physics_engine)
            )
        except Exception as exc:
            logger.warn(f"set zero velocity targets failed: {exc}")

        return articulation, art_joint_index, hold_targets, mimic_followers
    except Exception as exc:
        logger.warn(f"Articulation init failed: {exc}; falling back to USD writes")
        return None, {}, None, {}


def _apply_joint_commands(
    *,
    articulation,
    art_joint_index: Dict[str, int],
    stage,
    joint_prim_map: Dict[str, str],
    cmd_positions: Dict[str, float],
    cmd_4ws_steer_pos: Dict[str, float],
    cmd_4ws_drive_vel: Dict[str, float],
    cmd_4ws_stamp: float,
    logger,
    cmd_4ws_timeout_s: float,
    chassis_drive_joint_names: frozenset,
    physics_engine: str = "isaac_physx",
    mimic_followers: Optional[Dict[str, List[Tuple[str, float, float]]]] = None,
    seen_joints: Optional[set] = None,
    unknown_joints: Optional[set] = None,
) -> None:
    elapsed = time.monotonic() - cmd_4ws_stamp if cmd_4ws_stamp else 999.0
    timeout = elapsed > cmd_4ws_timeout_s

    # Expand mimic relations (read from ``NewtonMimicAPI`` on the USD
    # stage) into the position dict. The Newton wrapper's apply_action
    # doesn't move followers from a master-only command; on PhysX this
    # is a no-op (followers are already constrained to the master so a
    # redundant target write is ignored).
    if mimic_followers and cmd_positions:
        from genie_sim_engine_core.engine._mimic import expand_targets

        extra = expand_targets(cmd_positions, mimic_followers)
        if extra:
            cmd_positions = {**cmd_positions, **extra}

    if articulation is not None and art_joint_index:
        pos_names: List[str] = []
        pos_values: List[float] = []
        vel_names: List[str] = []
        vel_values: List[float] = []
        unknown_this_call: List[str] = []

        for name, value in cmd_positions.items():
            if name in art_joint_index:
                pos_names.append(name)
                pos_values.append(float(value))
            else:
                unknown_this_call.append(name)

        if timeout:
            for name in art_joint_index:
                if name in chassis_drive_joint_names:
                    vel_names.append(name)
                    vel_values.append(0.0)
            cmd_4ws_steer_pos.clear()
            cmd_4ws_drive_vel.clear()
        else:
            for name, value in cmd_4ws_steer_pos.items():
                if name in art_joint_index:
                    pos_names.append(name)
                    pos_values.append(float(value))
                else:
                    unknown_this_call.append(name)
            for name, value in cmd_4ws_drive_vel.items():
                if name in art_joint_index:
                    vel_names.append(name)
                    vel_values.append(float(value))
                else:
                    unknown_this_call.append(name)

        try:
            from isaacsim.core.utils.types import ArticulationAction

            # ``SingleArticulation.apply_action`` delegates to
            # ``ArticulationController.apply_action``, which has a
            # NaN-defensive loop:
            #
            #     for i in range(...):
            #         if joint_positions[0][i] is None or np.isnan(
            #             self._articulation_view._backend_utils.to_numpy(
            #                 joint_positions[0][i]))
            #             ...
            #
            # Under the Newton wrapper, ``_backend_utils`` ends up as
            # the warp backend (not torch), and warp's ``to_numpy`` on
            # a CUDA tensor calls ``.numpy()`` directly without going
            # via ``.cpu()``.  The result is a flood of:
            #
            #     can't convert cuda:0 device type tensor to numpy.
            #     Use Tensor.cpu() to copy the tensor to host memory first.
            #
            # We side-step the controller's NaN loop entirely by going
            # to ``_articulation_view.apply_action`` directly.  That's
            # what the controller eventually calls after the NaN check
            # — we just skip the (broken-on-Newton) check.  Mirrors
            # what the Newton wrapper itself recommends for command-
            # heavy paths.  Falls back to the controller for non-Newton
            # engines (PhysX) where the controller path is fine.
            def _apply_action(action: ArticulationAction) -> None:
                if physics_engine == "isaac_newton":
                    av = getattr(articulation, "_articulation_view", None)
                    if av is not None and hasattr(av, "apply_action"):
                        # ArticulationView.apply_action takes
                        # ArticulationActions (plural) — wrap our
                        # single-instance ArticulationAction.
                        from isaacsim.core.utils.types import ArticulationActions

                        # Expand to (1, N) batched layout the view
                        # expects.  None fields stay None.
                        jp = action.joint_positions
                        jv = action.joint_velocities
                        je = action.joint_efforts
                        ji = action.joint_indices
                        # Add the batch dimension when the view
                        # expects it.  ``isaac_newton`` uses
                        # batched/world layout internally.
                        try:
                            import torch

                            def _bat(x):
                                if x is None:
                                    return None
                                if hasattr(x, "unsqueeze"):
                                    return x.unsqueeze(0)
                                return x.reshape(1, -1) if hasattr(x, "reshape") else x

                            jp = _bat(jp)
                            jv = _bat(jv)
                            je = _bat(je)
                        except Exception:
                            pass
                        av.apply_action(
                            ArticulationActions(
                                joint_positions=jp,
                                joint_velocities=jv,
                                joint_efforts=je,
                                joint_indices=ji,
                            )
                        )
                        return
                # Default path (PhysX): controller is fine here.
                articulation.apply_action(action)

            if pos_names:
                indices = np.array([art_joint_index[n] for n in pos_names], dtype=np.int32)
                _apply_action(
                    ArticulationAction(
                        joint_positions=_view_input(np.array(pos_values, dtype=np.float32), engine=physics_engine),
                        joint_indices=_view_input(indices, engine=physics_engine),
                    )
                )
            if vel_names:
                indices = np.array([art_joint_index[n] for n in vel_names], dtype=np.int32)
                _apply_action(
                    ArticulationAction(
                        joint_velocities=_view_input(np.array(vel_values, dtype=np.float32), engine=physics_engine),
                        joint_indices=_view_input(indices, engine=physics_engine),
                    )
                )

            # First-target diagnostic: log the FIRST command we see for each
            # unique joint name. Mirrors newton-standalone's apply_commands log
            # so the same probe works on both engine paths — confirms /joint_command
            # is reaching the engine and identifies which DOFs the publisher is
            # actually driving (often a partial gripper-only command on tick 1).
            if seen_joints is not None:
                new_pos = [(n, art_joint_index[n], v) for n, v in zip(pos_names, pos_values) if n not in seen_joints]
                new_vel = [(n, art_joint_index[n], v) for n, v in zip(vel_names, vel_values) if n not in seen_joints]
                if new_pos or new_vel:
                    seen_joints.update(n for n, _, _ in new_pos)
                    seen_joints.update(n for n, _, _ in new_vel)
                    parts = []
                    if new_pos:
                        parts.append("pos: " + ", ".join(f"{n}[dof={i}]={v:.4f}" for n, i, v in new_pos))
                    if new_vel:
                        parts.append("vel: " + ", ".join(f"{n}[dof={i}]={v:.4f}" for n, i, v in new_vel))
                    logger.info(
                        f"[stage] apply_action first-target for "
                        f"{len(new_pos) + len(new_vel)} new joint(s) — " + "; ".join(parts)
                    )
        except Exception as exc:
            warn_once = getattr(logger, "warn_once", None)
            (warn_once or logger.warn)(f"apply_action failed: {exc}")

        # Unknown-joint warning: log each name once across the lifetime of
        # this stage. Names that don't map to a DOF in the articulation
        # are silently dropped by the apply_action calls above, which made
        # joint-name typos invisible — the user just saw "command sent,
        # robot didn't move". The set is owned by IsaacSimStage so the
        # latch survives across calls without leaking into a global.
        if unknown_joints is not None and unknown_this_call:
            new_unknown = [n for n in unknown_this_call if n not in unknown_joints]
            if new_unknown:
                unknown_joints.update(new_unknown)
                logger.warn(
                    f"[stage] apply_action: unknown joint(s) in /joint_command: "
                    f"{new_unknown}; known DOFs: {list(art_joint_index.keys())}"
                )
        return

    # Fallback: direct DriveAPI writes. This branch only runs when the
    # articulation handle is unavailable (the normal path above uses
    # articulation.apply_action which already speaks the right units per
    # joint). Mirror the readback-side type branching so prismatic targets
    # are not mistakenly scaled by _RAD2DEG.
    def _drive_for(prim):
        if prim.IsA(UsdPhysics.PrismaticJoint):
            return UsdPhysics.DriveAPI.Get(prim, "linear"), 1.0
        return UsdPhysics.DriveAPI.Get(prim, "angular"), _RAD2DEG

    for name, value in cmd_positions.items():
        prim = stage.GetPrimAtPath(joint_prim_map.get(name, ""))
        if prim:
            drive, scale = _drive_for(prim)
            if drive:
                drive.GetTargetPositionAttr().Set(float(value) * scale)

    if timeout:
        for name in list(cmd_4ws_drive_vel):
            prim = stage.GetPrimAtPath(joint_prim_map.get(name, ""))
            if prim:
                drive, _ = _drive_for(prim)
                if drive:
                    drive.GetTargetVelocityAttr().Set(0.0)
        cmd_4ws_steer_pos.clear()
        cmd_4ws_drive_vel.clear()
    else:
        for name, value in cmd_4ws_steer_pos.items():
            prim = stage.GetPrimAtPath(joint_prim_map.get(name, ""))
            if prim:
                drive, scale = _drive_for(prim)
                if drive:
                    drive.GetTargetPositionAttr().Set(float(value) * scale)
        for name, value in cmd_4ws_drive_vel.items():
            prim = stage.GetPrimAtPath(joint_prim_map.get(name, ""))
            if prim:
                drive, scale = _drive_for(prim)
                if drive:
                    drive.GetTargetVelocityAttr().Set(float(value) * scale)


# ---------------------------------------------------------------------------
# IsaacSimStage
# ---------------------------------------------------------------------------


class IsaacSimStage:
    """Owns the Isaac Sim ``World``, USD stage, and articulation handle.

    Lifecycle::

        stage = IsaacSimStage(...)         # opens scene + sets drives
        stage.startup_loop_setup(headless) # call once before the physics loop
        while running:
            stage.apply_commands(...)
            stage.world.step(render=...)
        stage.shutdown()
    """

    def __init__(
        self,
        *,
        robot_prefix: str,
        scene_usda: str,
        robot_usda: str,
        render_layer_usda: str,
        physics_hz: float,
        render_hz: float = 30.0,
        simulation_app,
        logger,
        params: Optional[PhysicsParams] = None,
        robot_from_urdf: bool = False,
        init_joint_pos: Optional[Dict[str, JointInitSpec]] = None,
        runtime_usd_dump_path: Optional[str] = None,
        physics_engine: str = "isaac_physx",
        physics_solver_iterations: int = 0,
        fix_base: bool = True,
        newton_solvers_path: str = "",
        scene_cfg: Optional[dict] = None,
    ) -> None:
        self._logger = logger
        self._simulation_app = simulation_app
        self._robot_prefix = robot_prefix
        self._physics_hz = physics_hz
        self._render_hz = render_hz
        self._params: PhysicsParams = params if params is not None else default_physics_params()
        self._from_urdf = bool(robot_from_urdf)
        self._physics_engine = physics_engine.strip().lower()
        # Stash so the runtime-USD dump can identify which composition arc
        # to leave as a payload (and which spec edits to author as overs).
        self._robot_usda_path = robot_usda
        logger.info(f"[stage] robot_from_urdf={self._from_urdf}")

        self.stage = _open_scene_with_references(
            scene_usda=scene_usda,
            robot_usda=robot_usda,
            render_layer_usda=render_layer_usda,
            robot_prefix=robot_prefix,
            simulation_app=simulation_app,
            logger=logger,
            newton_solvers_path=newton_solvers_path,
            # Forward so init_base_pose is applied BEFORE the
            # simulation_app.update() loop that triggers PhysX rigid-body
            # init — pre-physics teleport instead of a post-init swing.
            scene_cfg=scene_cfg,
        )

        new_physics_dt = 1.0 / physics_hz
        new_rendering_dt = 1.0 / render_hz
        old_physics_dt = None
        for prim in self.stage.Traverse():
            if prim.IsA(UsdPhysics.Scene):
                ts_attr = prim.GetAttribute("physics:timeStepsPerSecond")
                ts = _safe_get(ts_attr) if ts_attr else None
                if ts and ts > 0:
                    old_physics_dt = 1.0 / float(ts)
                break
        _log_override(logger, "world", "/physicsScene", "physics_dt", old_physics_dt, new_physics_dt)
        _log_override(logger, "world", "/", "rendering_dt", None, new_rendering_dt)

        from isaacsim.core.api import World

        self._world: Optional[World] = World(
            stage_units_in_meters=1.0,
            physics_dt=new_physics_dt,
            rendering_dt=new_rendering_dt,
        )

        if self._physics_engine == "isaac_newton":
            from genie_sim_engine_isaacsim.kit.newton_config import configure_physics_scene

            configure_physics_scene(self.stage, physics_solver_iterations, logger)

        robot_root = f"/{robot_prefix}"
        if not self.stage.GetPrimAtPath(robot_root).IsValid():
            logger.warn(f"Robot root {robot_root} not found in stage")

        self._asset_format: str = _detect_asset_format(robot_usda)
        logger.info(f"[stage] asset_format={self._asset_format}")

        # Honor fix_base per scene YAML by toggling the URDF importer's
        # world-weld joint. See ``_apply_fix_base_policy`` for the
        # detection logic that works across 6.x and 4.x/5.x output.
        _apply_fix_base_policy(self.stage, robot_prefix, fix_base, logger)

        # Neither Isaac Sim backend (``isaac_physx`` raw PhysX, or
        # ``isaac_newton`` via ``isaacsim.physics.newton``) auto-adds a
        # ground plane — both build physics purely from the authored
        # stage. Only the standalone ``genie_sim_engine_newton`` engine
        # calls ``builder.add_ground_plane()`` unconditionally. When
        # ``scene:`` is empty in the scene YAML,
        # ``assemble/_scene_impl.py`` synthesises a stage with only a
        # visual ``/World/GroundPlane`` (no ``CollisionAPI``), and a
        # free-base robot free-falls. Upgrade that visual mesh to a
        # static collider so the empty-scene default is playable on the
        # Isaac Sim backends too. See ``_ensure_default_ground_plane``
        # for the scan/no-op policy that keeps user-authored scenes
        # untouched.
        if self._physics_engine in ("isaac_physx", "isaac_newton"):
            _ensure_default_ground_plane(self.stage, robot_prefix, logger)

        self.body_paths: List[str] = _collect_body_paths(self.stage, robot_prefix)
        if self._asset_format == "as3":
            self.joint_names, self._joint_prim_map = _collect_joints_as3(self.stage, robot_usda)
        else:
            self.joint_names, self._joint_prim_map = _collect_joints(self.stage)
        self._joint_kinds: Dict[str, str] = _configure_drives(
            self.stage,
            self._joint_prim_map,
            logger,
            self._params,
            self._from_urdf,
            self._physics_engine,
        )
        from genie_sim_engine_newton.common.joint_params import load_scene_joint_params
        from genie_sim_engine_isaacsim.kit.joint_params import apply_usd_joint_params

        self._joint_params = load_scene_joint_params(
            scene_cfg, engine="physx" if self._physics_engine == "isaac_physx" else "newton", logger=logger
        )
        apply_usd_joint_params(
            self.stage, self._joint_prim_map, self._joint_params, physics_engine=self._physics_engine, logger=logger
        )
        # Author USD ``drive:*:physics:targetPosition`` for every joint
        # named in ``init_joint_pos`` BEFORE ``world.reset()`` so PhysX
        # seeds the articulation handle with the requested target instead
        # of the importer-default 0. See _seed_drive_target_positions
        # docstring for the full rationale.
        _seed_drive_target_positions(self.stage, self._joint_prim_map, robot_prefix, init_joint_pos, logger)
        # Pre-compute the set of free-spin chassis joints for the cmd-timeout
        # zero-velocity broadcast in _apply_joint_commands. Frozen so the
        # hot path doesn't have to re-classify every tick.
        self._chassis_drive_joint_names: frozenset = frozenset(
            n for n, k in self._joint_kinds.items() if k == JK_CHASSIS_DRIVE
        )

        self._world.reset()
        for _ in range(5):
            simulation_app.update()

        (
            self._articulation,
            self._art_joint_index,
            self._hold_targets,
            self._mimic_followers,
        ) = _init_articulation(
            self.stage,
            robot_prefix,
            logger,
            self._params,
            self._from_urdf,
            init_joint_pos,
            physics_engine=self._physics_engine,
            joint_params=self._joint_params,
        )

        from isaacsim.core.simulation_manager import SimulationManager
        from genie_sim_engine_isaacsim.kit.episode_reset import EpisodeSnapshot, physics_paths

        if self._articulation is None:
            raise RuntimeError("Isaac episode reset requires a live robot articulation")
        self._episode_snapshot = EpisodeSnapshot(
            SimulationManager.get_physics_sim_view(),
            lambda value: _view_input(value, engine=self._physics_engine),
            _view_readback,
            self._articulation._articulation_view._physics_view,
            self._hold_targets,
            *physics_paths(self.stage),
        )

        # Debug snapshot of the runtime injection delta. Writes a thin
        # USD layer rooted at /<robot_prefix> that adds the original
        # robot.usda as a PAYLOAD and authors only the specs the runtime
        # touched on top (drive APIs, mass/inertia overrides, etc.).
        # Opening robot_runtime.usd in any editor shows exactly what the
        # simulator injected — zero importer-noise — and loading the
        # payload pulls in the original robot for context.
        if runtime_usd_dump_path:
            self._dump_robot_runtime_usd(runtime_usd_dump_path)

        self._physx_sim = None

        # Lifetime-scoped diagnostic sets for ``_apply_joint_commands``.
        # ``_cmd_seen_joints`` latches once per unique joint name we've
        # ever commanded — drives the "first-target" log line that
        # mirrors newton-standalone's identical probe. ``_cmd_unknown_joints``
        # latches names that DON'T map to any DOF so a typo gets one
        # warning instead of silent drop on every tick.
        self._cmd_seen_joints: set = set()
        self._cmd_unknown_joints: set = set()

    # --- debug helpers --------------------------------------------------

    def _dump_robot_runtime_usd(self, dump_path: str) -> None:
        """Export the runtime USD as a thin override on top of ``robot.usda``.

        Pipeline context
        ----------------
        ``robot_runtime.usda`` sits in the conceptual pipeline as::

            urdf  --[urdf importer]-->  robot.usda
                  --[genie_sim_engine param injection]-->  robot_runtime.usda
                  --[scene assemble]-->  scene.usda

        It is a **debug artifact only** (not consumed by ``assemble_scene``
        — see the README) but its file structure must mirror
        ``robot.usda`` exactly: a tiny shell file that composes the same
        ``./configuration/robot_base.usd``, ``./configuration/robot_physics.usd``
        and ``./configuration/robot_sensor.usd`` payloads/references via
        the same ``Physics`` / ``Sensor`` variantSets.

        Strategy
        --------
        1. **Filesystem-copy ``robot.usda`` to ``robot_runtime.usda``**.
           This preserves the layer's ``customLayerData``, ``defaultPrim``,
           every variantSet definition, and — crucially — the composition
           arcs to ``./configuration/*.usd``. The file stays a thin shell
           (~74 lines), not 800 MB of inlined geometry.
        2. **Layer the runtime parameter injections on top as overrides**.
           The injections all live on the live stage's **scene root
           layer** (because ``open_stage(scene_usda)`` makes that the
           edit target before ``_configure_drives`` runs). We walk the
           scene root layer's prim specs under ``/<robot_prefix>`` and
           ``Sdf.CopySpec`` each one into ``robot_runtime.usda`` at the
           **rebased path** ``/<robot.usda defaultPrim>/<rest>`` —
           because ``robot.usda`` roots its hierarchy at ``/robot``
           while the live stage uses the scene-yaml's ``robot_prefix``
           (``/aloha`` here), introduced by
           ``add_reference_to_stage(robot.usda, "/<prefix>")``.
        3. **Strip the root-prim ``references`` op** copied from the
           scene layer. That op was the original
           ``add_reference_to_stage`` arc pointing back at ``robot.usda``
           — keeping it would make ``robot_runtime.usda`` reference
           itself indirectly via its own root spec, which double-composes
           the variantSets.

        What survives the copy: per-joint ``DriveAPI`` overrides
        (stiffness / damping / max-force / target-position / target-
        velocity / drive-type), ``physxJoint:armature``,
        ``physxMimicJoint:<axis>:naturalFrequency`` /
        ``dampingRatio``, the ``fix_base`` joint authored on the scene
        root layer, and any other USD-layer attribute the runtime
        wrote.

        Out of scope (NOT captured): ``ArticulationView`` gains
        (``set_gains`` / ``set_max_efforts``) and joint-position resets
        (``set_joint_positions``) are PhysX tensor-handle state with no
        USD attribute backing.

        Why this beats the previous flatten-and-inline approach: the
        flatten path inlined every mesh and material from
        ``./configuration/*.usd`` into the dump (~800 MB), and the
        flatten algorithm aggressively factored duplicate sub-fragments
        into stage-root ``/Flattened_Prototype_<N>`` prims with
        self-asset references that broke under any path change. The
        override-on-shell approach keeps ``./configuration/*.usd`` as
        the geometry source of truth (one file edit propagates
        everywhere) and stays human-diffable.

        Why ``.usda`` (ASCII): the file is meant to be ``cat``'d,
        ``diff``'d across runs, and inspected by humans.

        Failures are logged but never raise: the snapshot is debug-only,
        not load-bearing.
        """
        try:
            import os
            import shutil

            from pxr import Sdf

            os.makedirs(os.path.dirname(dump_path) or ".", exist_ok=True)
            scene_prefix_path = Sdf.Path(f"/{self._robot_prefix}")

            scene_prim = self.stage.GetPrimAtPath(scene_prefix_path)
            if not scene_prim.IsValid():
                self._logger.warn(f"[stage] runtime USD dump skipped: prim {scene_prefix_path} not found in stage")
                return

            if not (self._robot_usda_path and os.path.exists(self._robot_usda_path)):
                self._logger.warn(
                    f"[stage] runtime USD dump skipped: robot.usda not found at "
                    f"{self._robot_usda_path!r}; cannot clone shell"
                )
                return

            # Step 1 — clone robot.usda verbatim. Filesystem copy is the
            # cleanest way to preserve customLayerData, defaultPrim, and
            # every variantSet definition exactly as the importer wrote
            # them. ``shutil.copyfile`` does NOT follow symlinks at the
            # destination and overwrites in place.
            shutil.copyfile(self._robot_usda_path, dump_path)

            # Step 2 — open the freshly cloned destination layer. We
            # need ``FindOrOpen`` (not ``CreateNew``) since the file
            # already exists on disk. The clone we just wrote may or
            # may not already be in pxr's layer registry depending on
            # whether the live stage opened robot.usda directly; either
            # way ``FindOrOpen`` returns a usable handle.
            dst_layer = Sdf.Layer.FindOrOpen(dump_path)
            if dst_layer is None:
                self._logger.warn(f"[stage] runtime USD dump: could not open cloned layer at {dump_path}")
                return

            # The destination's root prim path is whatever ``robot.usda``
            # uses as its defaultPrim (typically ``/robot``); the live
            # stage paths under ``/<robot_prefix>`` (e.g. ``/aloha``)
            # need to be rebased onto it.
            dst_root_name = dst_layer.defaultPrim or "robot"
            dst_root_path = Sdf.Path(f"/{dst_root_name}")

            # Step 3 — find the live stage's scene root layer (the
            # layer whose edit target ``_configure_drives`` was writing
            # into). It owns every USD-layer runtime injection.
            scene_layer = self.stage.GetRootLayer()

            # Step 4 — walk the scene root layer's prim specs under
            # ``/<robot_prefix>`` and copy each one to the destination
            # at its rebased path. ``Sdf.Layer.Traverse`` visits every
            # spec in the layer; we filter to the robot subtree.
            copied_count = 0

            # NEVER copy the scene-layer root spec onto dst_root_path.
            # ``Sdf.CopySpec`` is destructive — copying the scene's
            # ``/<robot_prefix>`` spec would overwrite ``robot.usda``'s
            # ``def Xform "robot"`` (with its variantSets, payloads to
            # ``./configuration/*.usd``, xformOps, etc.) with the
            # scene-layer spec, which only carries the
            # ``add_reference_to_stage(robot.usda)`` arc and an Xform
            # type. We want the cloned shell preserved verbatim and
            # only the *descendant* override specs layered on top.
            def _visit(spec_path: Sdf.Path) -> None:
                nonlocal copied_count
                if not spec_path.IsPrimPath():
                    return
                if not spec_path.HasPrefix(scene_prefix_path):
                    return
                if spec_path == scene_prefix_path:
                    # Skip the root spec — its only payload is the
                    # reference arc to robot.usda, which we already
                    # have via ``shutil.copyfile``.
                    return
                # Rebase: /<robot_prefix>/x/y -> /<dst_root_name>/x/y.
                rel = spec_path.MakeRelativePath(scene_prefix_path)
                rebased = dst_root_path.AppendPath(rel)

                Sdf.CreatePrimInLayer(dst_layer, rebased)
                if not Sdf.CopySpec(scene_layer, spec_path, dst_layer, rebased):
                    return
                copied_count += 1

            scene_layer.Traverse(Sdf.Path.absoluteRootPath, _visit)

            dst_layer.Save()
            # Drop the in-memory handle so a subsequent re-open from
            # the same process sees disk truth.
            del dst_layer

            # Make asset references relative to the dump's parent dir.
            # ``robot.usda`` already uses ``@./payloads/*.usda@`` so
            # the cloned shell starts clean — but ``Sdf.CopySpec`` of
            # scene-layer specs can drag in absolute references when
            # the live stage was opened with absolute paths (notably
            # the root prim's ``references = @/abs/path/robot.usda@``
            # arc, which we strip via the ``_visit`` skip-the-root
            # branch above, plus any per-prim references the runtime
            # added).  This pass is the safety net: ensure the final
            # dump has zero ``@/abs/...@`` paths regardless of what
            # the source layers carried.  Same logic the
            # newton-standalone path uses — see ``lifecycle.py:
            # _dump_runtime_usd``.
            try:
                from genie_sim_engine_newton.common.usd_path_helpers import (  # noqa: PLC0415
                    make_layer_asset_paths_relative,
                )

                make_layer_asset_paths_relative(dump_path, logger=self._logger)
            except Exception as exc:  # noqa: BLE001
                self._logger.warn(
                    f"[stage] runtime USD dump path-relativizer " f"failed ({exc!r}); dump kept absolute paths."
                )

            self._logger.info(
                f"[stage] robot runtime USD snapshot: {dump_path} "
                f"(cloned robot.usda + {copied_count} runtime-injection specs)"
            )
        except Exception as exc:  # noqa: BLE001 — debug-only, never fatal
            self._logger.warn(f"[stage] runtime USD dump failed ({exc}); continuing")

    # --- public properties ----------------------------------------------

    @property
    def robot_prefix(self) -> str:
        return self._robot_prefix

    @property
    def joint_prim_map(self) -> Dict[str, str]:
        return self._joint_prim_map

    @property
    def params(self) -> PhysicsParams:
        return self._params

    @property
    def physx_sim(self):
        return self._physx_sim

    @property
    def world(self):
        return self._world

    # --- lifecycle ------------------------------------------------------

    def startup_loop_setup(self, headless: bool) -> None:
        """One-time setup just before entering the physics loop."""
        # Log which engine is active via the unified interface (introspection only).
        # Newton replaces PhysX's solver but keeps the same stepping API, so
        # get_physx_simulation_interface() is always the correct object for
        # .simulate() / .fetch_results() regardless of which engine is active.
        try:
            from omni.physics.core import get_physics_interface

            phys_iface = get_physics_interface()
            for sid in phys_iface.get_simulation_ids():
                if phys_iface.is_simulation_active(sid):
                    self._logger.info(f"[stage] active physics engine: {phys_iface.get_simulation_name(sid)}")
                    break
        except Exception as exc:
            self._logger.warn(f"[stage] could not query active physics engine: {exc}")

        from omni.physx import get_physx_simulation_interface

        self._physx_sim = get_physx_simulation_interface()

        configure_carb_settings(headless, self._logger)
        _locate_physics_scene(self.stage, self._logger)

    def get_joint_states(self) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Read joint positions + velocities from the articulation handle.

        Returns ``(positions, velocities, efforts)`` in ``self.joint_names``
        order (the order the C++ ``/joint_states`` publisher consumes).
        Effort is ``None`` on this backend — Isaac Sim 6.x's articulation
        does not expose a stable measured-effort readback across the
        PhysX / Newton wrappers we support, so we signal "not measured"
        via an empty ``effort[]`` on the wire (standard
        ``sensor_msgs/JointState`` convention).  Consumers that need
        effort should read it from an actuator-side channel, not from
        the state feed.

        Why this exists alongside the module-level ``snapshot_joint_states``:
        the snapshot helper reads USD ``state:angular:physics:position``
        attributes, which PhysX writes back every tick but the Newton
        wrapper does NOT. On the isaac_newton path that leaves
        ``/joint_states`` at zero forever — RViz sees the robot at the
        origin even though Newton's articulation is happily tracking
        ``init_joint_pos`` and any subsequent ``/joint_command`` writes.

        The articulation handle, in contrast, exposes the same buffer
        Newton's solver actually integrates (returned as a CUDA torch
        tensor under Newton, numpy under PhysX — ``_view_readback``
        normalises both). Falls back to zeros for joints not in the
        articulation's DOF table (fixed joints, world joints).
        """
        n = len(self.joint_names)
        pos = np.zeros(n, dtype=np.float64)
        vel = np.zeros(n, dtype=np.float64)
        if n == 0 or self._articulation is None or not self._art_joint_index:
            return pos, vel, None
        try:
            all_pos = _view_readback(self._articulation.get_joint_positions())
            all_vel = _view_readback(self._articulation.get_joint_velocities())
        except Exception as exc:
            self._logger.warn(f"get_joint_states: articulation read failed: {exc}")
            return pos, vel, None
        if all_pos is None or all_vel is None:
            return pos, vel, None
        all_pos = np.asarray(all_pos).reshape(-1)
        all_vel = np.asarray(all_vel).reshape(-1)
        # NaN/inf one-shot guard — if the integrator blew up we want to
        # know it once, not from a flood of downstream TF_NAN errors in
        # RViz. Mirrors newton-standalone's identical guard in its own
        # ``get_joint_states``.
        if not getattr(self, "_joint_state_nan_warned", False):
            if not np.all(np.isfinite(all_pos)) or not np.all(np.isfinite(all_vel)):
                bad = [i for i in range(len(all_pos)) if not np.isfinite(all_pos[i])]
                self._logger.warn(
                    f"[stage] get_joint_states: non-finite values at DOF idx "
                    f"{bad[:8]}{'…' if len(bad) > 8 else ''} — the integrator "
                    f"likely blew up; check init pose, PD gains, and URDF "
                    f"mass/inertia."
                )
                self._joint_state_nan_warned = True
        for i, name in enumerate(self.joint_names):
            idx = self._art_joint_index.get(name)
            if idx is None or idx >= len(all_pos):
                continue
            pos[i] = float(all_pos[idx])
            vel[i] = float(all_vel[idx])
        return pos, vel, None

    def apply_commands(
        self,
        *,
        cmd_positions: Dict[str, float],
        cmd_4ws_steer_pos: Dict[str, float],
        cmd_4ws_drive_vel: Dict[str, float],
        cmd_4ws_stamp: float,
    ) -> None:
        _apply_joint_commands(
            articulation=self._articulation,
            art_joint_index=self._art_joint_index,
            stage=self.stage,
            joint_prim_map=self._joint_prim_map,
            cmd_positions=cmd_positions,
            cmd_4ws_steer_pos=cmd_4ws_steer_pos,
            cmd_4ws_drive_vel=cmd_4ws_drive_vel,
            cmd_4ws_stamp=cmd_4ws_stamp,
            logger=self._logger,
            cmd_4ws_timeout_s=self._params.cmd_4ws_timeout_s,
            chassis_drive_joint_names=self._chassis_drive_joint_names,
            physics_engine=self._physics_engine,
            mimic_followers=self._mimic_followers,
            seen_joints=self._cmd_seen_joints,
            unknown_joints=self._cmd_unknown_joints,
        )

    def shutdown(self) -> None:
        try:
            if self._world is not None:
                self._world.stop()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# USD → flat-numpy snapshots (consumed by the C++ publishers)
# ---------------------------------------------------------------------------


def snapshot_joint_states(
    stage,
    joint_names: List[str],
    joint_prim_map: Dict[str, str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (positions, velocities) as float64 numpy arrays.

    Units follow ROS sensor_msgs/JointState convention:
      * revolute  -> radians, rad/s   (USD reports degrees -> convert)
      * prismatic -> meters,  m/s     (USD reports meters  -> identity)

    USD exposes state under different attribute namespaces depending on the
    joint type:
      * revolute  -> ``state:angular:physics:{position,velocity}``
      * prismatic -> ``state:linear:physics:{position,velocity}``

    Reading a revolute attribute on a prismatic joint (or vice versa) silently
    returns ``None``, which leaves every gripper slot in
    ``/joint_states`` at 0.0 — so RViz's robot_state_publisher never animates
    the finger links even though Isaac's prismatic drives are tracking the
    command perfectly. Branch on the joint type to fix both axes.
    """
    n = len(joint_names)
    pos = np.zeros(n, dtype=np.float64)
    vel = np.zeros(n, dtype=np.float64)
    if n == 0:
        return pos, vel
    for i, name in enumerate(joint_names):
        prim = stage.GetPrimAtPath(joint_prim_map.get(name, ""))
        if not prim or not prim.IsValid():
            continue
        if prim.IsA(UsdPhysics.PrismaticJoint):
            pos_attr = "state:linear:physics:position"
            vel_attr = "state:linear:physics:velocity"
            scale = 1.0
        else:
            pos_attr = "state:angular:physics:position"
            vel_attr = "state:angular:physics:velocity"
            scale = _DEG2RAD
        p = prim.GetAttribute(pos_attr)
        if p and p.IsValid():
            v = p.Get()
            if v is not None:
                pos[i] = float(v) * scale
        w = prim.GetAttribute(vel_attr)
        if w and w.IsValid():
            v = w.Get()
            if v is not None:
                vel[i] = float(v) * scale
    return pos, vel


def _xform_to_xyzwxyz(prim) -> List[float]:
    """Return the prim's **world** pose as ``(x, y, z, qw, qx, qy, qz)``.

    Used for ``snapshot_odom`` and ``snapshot_body_transforms`` — both
    the /odom channel and the v2 ``/tf_scene`` wire expect world-frame
    poses (see DESIGN.DATA_EXCHANGE_PROTOCOL.md §Global frame convention).
    """
    world_xform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    t = Gf.Vec3d(world_xform[3][0], world_xform[3][1], world_xform[3][2])
    q = world_xform.ExtractRotationQuat()
    im = q.GetImaginary()
    return [float(t[0]), float(t[1]), float(t[2]), float(q.GetReal()), float(im[0]), float(im[1]), float(im[2])]


def snapshot_body_transforms(stage, body_paths: List[str]) -> Tuple[np.ndarray, List[str]]:
    """Return ``(Nx7 float64 array of (x,y,z,qw,qx,qy,qz), absolute_prim_paths)``.

    Each row is the body's **world-frame** pose (v2 wire protocol —
    see ``DESIGN.DATA_EXCHANGE_PROTOCOL.md``).  Pre-v2 this function
    returned parent-local poses because the renderer wrote each entry
    into ``xformOp:translate`` / ``xformOp:orient`` which composed
    through the render stage's kinematic hierarchy. Under v2:

    * Robot links reach the render process via ``/tf`` (RSP fan-out
      over URDF FK); they are filtered off ``/tf_scene`` by the
      ``body_is_robot`` mask in ``session.py``, so their entries here
      never reach a renderer over the wire — the frame convention
      chosen here does not affect them.
    * Scene links (free-joint props under ``/World/...``) reach the
      render via ``/tf_scene`` and are written with
      ``omni:resetXformStack=1``, which discards the parent xform
      stack.  Publishing them world-frame is what makes that write
      land at the right absolute location.

    The frame names are absolute USD prim paths so the renderer can
    locate the target prim in any layout (flat 4.x/5.x or AS3 6.x).
    """
    if not body_paths:
        return np.zeros((0, 7), dtype=np.float64), []
    rows: List[List[float]] = []
    names: List[str] = []
    for path in body_paths:
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            continue
        rows.append(_xform_to_xyzwxyz(prim))
        names.append(path)
    return np.asarray(rows, dtype=np.float64), names


def snapshot_odom(stage, robot_prefix: str, sim_time: float) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return ``(pose7, twist6)`` for ``<prefix>/base_link`` or ``None``.

    ``pose7`` is ``(x, y, z, qw, qx, qy, qz)`` in the world / odom frame.
    ``twist6`` is ``(vx, vy, vz, wx, wy, wz)`` expressed in the **base_link**
    (child) frame, matching ``nav_msgs/Odometry`` convention.

    The twist is computed by finite-differencing the world-frame pose against
    the previous call (cached on the function) and rotating the world-frame
    velocity into base_link via ``R_world2base.T``. On the first call (no
    cached previous sample) the twist is returned as zeros.

    Locates ``base_link`` by walking the robot subtree rather than assuming a
    fixed depth — flat 4.x/5.x exports place it at ``/<prefix>/base_link``
    while AS3 places it at ``/<prefix>/Geometry/base_link``. The first
    rigid-body prim named ``base_link`` (anywhere under the prefix) wins, and
    the resolved path is cached on the function for subsequent calls.
    """
    cached_path = getattr(snapshot_odom, "_base_link_path", None)
    cached_for_prefix = getattr(snapshot_odom, "_base_link_prefix", None)
    if cached_path is None or cached_for_prefix != robot_prefix:
        cached_path = None
        root = stage.GetPrimAtPath(f"/{robot_prefix}")
        if root and root.IsValid():
            for p in Usd.PrimRange(root):
                if p.GetName() == "base_link" and p.HasAPI(UsdPhysics.RigidBodyAPI):
                    cached_path = p.GetPath().pathString
                    break
        snapshot_odom._base_link_path = cached_path
        snapshot_odom._base_link_prefix = robot_prefix
    if not cached_path:
        return None
    prim = stage.GetPrimAtPath(cached_path)
    if not prim or not prim.IsValid():
        return None
    pose7 = np.asarray(_xform_to_xyzwxyz(prim), dtype=np.float64)

    twist6 = np.zeros(6, dtype=np.float64)
    prev = getattr(snapshot_odom, "_prev", None)
    snapshot_odom._prev = (sim_time, pose7.copy())
    if prev is not None:
        prev_t, prev_pose = prev
        dt = sim_time - prev_t
        if dt > 1e-9:
            # Linear velocity in world frame
            v_world = (pose7[0:3] - prev_pose[0:3]) / dt

            # Angular velocity from quaternion delta: q_curr = q_delta * q_prev
            # so q_delta = q_curr * q_prev^-1, then w = 2 * vec(q_delta) / dt
            qw0, qx0, qy0, qz0 = prev_pose[3], prev_pose[4], prev_pose[5], prev_pose[6]
            qw1, qx1, qy1, qz1 = pose7[3], pose7[4], pose7[5], pose7[6]
            # q_prev^-1 = (qw0, -qx0, -qy0, -qz0)  (unit quat)
            dw = qw1 * qw0 + qx1 * qx0 + qy1 * qy0 + qz1 * qz0
            dx = -qw1 * qx0 + qx1 * qw0 - qy1 * qz0 + qz1 * qy0
            dy = -qw1 * qy0 + qx1 * qz0 + qy1 * qw0 - qz1 * qx0
            dz = -qw1 * qz0 - qx1 * qy0 + qy1 * qx0 + qz1 * qw0
            # Shortest-arc: flip sign if dw < 0 so the small-angle approx is valid
            if dw < 0.0:
                dx, dy, dz = -dx, -dy, -dz
            w_world = np.array([2.0 * dx / dt, 2.0 * dy / dt, 2.0 * dz / dt], dtype=np.float64)

            # Rotate world-frame velocities into base_link frame: v_body = R^T * v_world
            # R(qw, qx, qy, qz) -> 3x3
            xx, yy, zz = qx1 * qx1, qy1 * qy1, qz1 * qz1
            xy, xz, yz = qx1 * qy1, qx1 * qz1, qy1 * qz1
            wx, wy, wz = qw1 * qx1, qw1 * qy1, qw1 * qz1
            r = np.array(
                [
                    [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
                    [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
                    [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
                ],
                dtype=np.float64,
            )
            v_body = r.T @ v_world
            w_body = r.T @ w_world
            twist6[0:3] = v_body
            twist6[3:6] = w_body

    return pose7, twist6

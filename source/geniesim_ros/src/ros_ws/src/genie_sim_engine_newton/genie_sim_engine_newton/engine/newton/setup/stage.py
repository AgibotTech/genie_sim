# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Build-phase 1: USD stage open + pre-``add_usd`` session-layer overrides.

Owns: stage open template, runtime fix policies (``pin_base_to_world``,
``convert_joints_to_fixed``, ``init_base_pose``), and the
RenderProduct disable cleanup.  Every method here mutates the
composed USD stage; ``add_usd`` runs in the next phase (Model).
"""

from __future__ import annotations

import json
import math
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import warp as wp

from genie_sim_engine_core.schema.scene import PinBaseMode


# --- Colliders that arrive at the solver as one solid lump ----------------
# Newton's USD importer turns ``physics:approximation`` into a remeshing
# method (``approximation_to_remeshing_method`` in
# newton/_src/utils/import_usd.py) and REPLACES the shape's mesh with the
# result.  Three separate ways that ends in a single convex hull for a
# concave object, all measured on this asset set:
#
#   * ``none`` / ``sdf`` -- PhysX's exact-mesh modes.  Newton's table has
#     neither, so it skips remeshing and the raw mesh is convexified by the
#     solver.  ``open_door``'s frame is a ring authored ``sdf``; the hull of
#     a ring is a slab flush with the wall and the doorway is sealed.
#   * ``convexDecomposition`` -- Newton runs CoACD at a hard-coded threshold
#     of 0.5, loose enough to return ONE piece.  Measured on the door frame:
#     0.5 -> 1 piece (still sealed), 0.1 -> 8, 0.05 -> 12.  Same for
#     ``stock_and_straighten_shelf``'s shopping cart.
#   * ``meshSimplification`` -- needs the optional ``fast_simplification``
#     package, which is not installed here.  Newton logs
#     "Falling back to bounding_box": that scene's supermarket shelf became a
#     solid BOX, worse than a hull.
#
# So the trigger is the OUTCOME, not the authored value: any collider whose
# mesh fills too little of its own convex hull is one we decompose ourselves,
# with a threshold we choose.  ``physics:approximation`` is blocked on those
# prims so Newton hands us the original mesh instead of a hull -- once its
# ``approximate_meshes`` has run, the concavity is gone and there is nothing
# left to decompose.
#
# Two measurements decide, and they answer different questions.
#
# Solidity = mesh volume / convex-hull volume: how wrong the single hull is.
# It only has to rule out colliders whose hull IS essentially the mesh, where
# cutting would just inflate them -- the room floor at 0.863 and a bottle's
# label at 0.697.  Everything this path exists for sits far below: shopping
# cart 0.012, supermarket shelf 0.038, parts bin 0.042, door frame 0.103, and
# the shape-sorter block box at 0.294 -- a thick-walled box is legitimately
# much more solid than a ring or a wire cart, which is why this line has to
# sit well above them.
_SCENE_DECOMP_MAX_SOLIDITY = 0.5

# Void volume = hull volume - mesh volume: how much empty space the collider
# encloses, which IS the thing that swallows other objects.  Cutting is not
# free -- CoACD's pieces come back slightly inflated, and a decomposed door
# handle grew into its own frame by 4.9mm where the single hull had touched
# nothing -- so anything that encloses nothing is left alone.
#
# In m3.  Must be cut: door frame 0.263, supermarket shelf ~0.79, shopping
# cart ~0.31, parts bin 0.0079, and the smallest real container, the
# shape-sorter block box, at 0.00216.  Must not: door handle 0.000243, its
# latch plate 0.000240, a bottle's label 0.000231.  The two groups are nearly
# nine-fold apart; a litre of enclosed void sits between them with 2.2x
# clearance below and 4.1x above.
#
# A floor on HULL volume instead would measure how big the object is rather
# than how much it encloses, and would silently exclude a small container
# whose hull is small but whose inside is the whole point.
_SCENE_DECOMP_MIN_VOID_VOLUME = 0.001   # m3

# CoACD is fast on the mesh but its merge pass is not: the shopping cart's
# 101682-vertex mesh takes ~18s, the shelf ~13s.  Results are cached on disk
# per (mesh, params) so only the first build of a given asset pays it.  The
# vertex cap is a guard against a pathological mesh, not a cost control.
_SCENE_DECOMP_MAX_VERTS = 200000

# A plain four-leg table has the SAME solidity/void signature as a door frame
# or a shopping cart -- big convex hull, little mesh volume, the space between
# the legs reads as a swallow-able cavity -- so the scan above sweeps tables
# in too, even though nothing about a table needed fixing.  Unlike a door
# frame or a cart, a table is the flat surface objects rest on for the whole
# episode: CoACD's pieces come back slightly inflated (see
# ``_SCENE_DECOMP_MIN_VOID_VOLUME``'s note on the door handle), and the seam
# between two decomposed pieces of a table can let a resting block sink
# through the top and keep drifting.  Excluded by name rather than fixing the
# seam itself: matched on the prim path so it does not depend on asset-library
# category metadata, which is not visible at this raw USD-traversal layer.
_SCENE_DECOMP_EXCLUDE_NAME_SUBSTRINGS = ("table",)


# The wrapper authors the task's room under this prim (task_loader's
# ``_wrap_scene_with_default_env``), which is what separates scenery from the
# foreground objects the task is actually about.
_BACKGROUND_SCOPE = "/Background"


def topmost_rigid_body_paths(stage, root_path) -> List[str]:
    """Paths of the topmost ``RigidBodyAPI`` prims under ``root_path``.

    These are the URDF root link(s): a rigid body with no rigid-body
    ancestor between it and the articulation root.  Pure and read-only, so
    both the root-weld diagnostic and the post-finalize pose readback can
    share one rule instead of each carrying a copy.

    ``PrimAllPrimsPredicate`` on purpose — an already-deactivated weld's
    sibling links must still be visible.

    ``_deactivate_root_joint`` keeps its own inline copy: changing what that
    method matches is a behaviour change, not a refactor.
    """
    from pxr import Usd, UsdPhysics  # noqa: PLC0415

    root_prim = stage.GetPrimAtPath(root_path) if stage is not None else None
    if not root_prim or not root_prim.IsValid():
        return []
    out: List[str] = []
    for desc in Usd.PrimRange(root_prim, Usd.PrimAllPrimsPredicate):
        if desc == root_prim or not desc.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        ancestor = desc.GetParent()
        is_topmost = True
        while ancestor and ancestor.IsValid() and ancestor.GetPath() != root_path:
            if ancestor.HasAPI(UsdPhysics.RigidBodyAPI):
                is_topmost = False
                break
            ancestor = ancestor.GetParent()
        if is_topmost:
            out.append(str(desc.GetPath()))
    return out


WELD_IDENT_TOL = 1.0e-6

# Marks a world weld whose ``localPose0`` is a deliberate world anchor
# authored by ``_repair_root_weld_anchor``, rather than a baked offset the
# converter left behind.  The two are structurally identical — body0 empty,
# localPose0 non-identity — and Newton treats both the same way (anchor =
# localPose0), which is the bug in one case and the fix in the other.  Only
# this annotation distinguishes them, so without it the classifier keeps
# reporting a repaired weld as broken and the repair never converges.
# ``newton:`` namespace follows ``common/gripper_annotations.py``.
WELD_ANCHOR_ANNOT = "newton:weldAnchorIsWorld"


def classify_world_weld(stage, prim, robot_root_path, urdf_root_link_paths) -> Optional[Dict[str, Any]]:
    """Classify one ``UsdPhysics.FixedJoint`` as a world-weld candidate.

    Returns ``None`` when the joint cannot be the world weld, else a record
    describing *how Newton will resolve its world side* and where the weld
    therefore ends up anchored.

    One shared classifier because three callers must agree: the diagnostic
    that reports the branch, the repair that rewrites a bad anchor, and the
    pose readback that confirms the outcome.  Three copies of this predicate
    would be three chances to disagree about which joint is the weld.

    A candidate must attach a topmost URDF root link as ``body1``: either
    Newton cannot resolve its world side (so it guesses), or ``body0`` names
    the articulation root itself. An internal joint through a non-rigid
    fixed frame still belongs to its ancestor rigid body, so it must never
    be rewritten as a world anchor. An internal rigid-to-rigid joint is
    never a world weld no matter how close to the root it sits — an earlier
    "touches a URDF root link on either side" rule flagged every one of
    them on a flat asset and fired a false warning for each.

    Branch semantics, all three measured end-to-end on the real asset:

      * ``body0`` resolves to a rigid body — no guess, Newton uses it.
      * guess + identity ``localPose0`` — the world frame is recovered from
        the child body's composed pose, so ``init_base_pose`` wins.  This
        is the branch our shipping asset takes.
      * guess + non-identity ``localPose0`` + a resolvable non-rigid
        ``body0`` prim — Newton reads that prim's *world* transform as the
        parent frame and composes ``localPose0`` on top, so a baked offset
        is applied TWICE (measured: stage z=0.04 + localPos0 z=0.04 ->
        anchor z=0.08).
      * guess + non-identity ``localPose0`` + nothing resolvable at all
        (``empty`` / ``dangling``) — the parent frame falls back to
        identity and ``localPose0`` becomes the anchor outright, dropping
        the stage pose (measured: requested x=0.3 z=0.04 -> anchor
        0, 0, 0.04).

    ``misplaces`` is true for the last two: the weld will not land where the
    stage asked.  Read-only; nothing here mutates the stage.
    """

    from pxr import UsdPhysics  # noqa: PLC0415

    def _targets(rel) -> List[str]:
        try:
            return [str(t) for t in (rel.GetTargets() or [])]
        except Exception:  # noqa: BLE001
            return []

    def _vec(attr) -> Optional[Tuple[float, float, float]]:
        if not attr:
            return None
        v = attr.Get()
        if v is None:
            return None
        return (float(v[0]), float(v[1]), float(v[2]))

    def _quat(attr) -> Optional[Tuple[float, float, float, float]]:
        if not attr:
            return None
        q = attr.Get()
        if q is None:
            return None
        im = q.GetImaginary()
        return (float(q.GetReal()), float(im[0]), float(im[1]), float(im[2]))

    def _is_zero(v) -> bool:
        return v is None or all(abs(c) <= WELD_IDENT_TOL for c in v)

    def _is_unit_quat(q) -> bool:
        return q is None or abs(q[0]) >= 1.0 - WELD_IDENT_TOL

    joint = UsdPhysics.FixedJoint(prim)
    b0 = _targets(joint.GetBody0Rel())
    b1 = _targets(joint.GetBody1Rel())
    if not b1 or b1[0] not in urdf_root_link_paths:
        return None

    # Newton's guess branch fires when the world side cannot be resolved to
    # a rigid body — body0 absent, OR present but pointing at a prim
    # without RigidBodyAPI (the /robot Xform case, which ``JointDesc.body0``
    # reports as "").
    if not b0:
        world_side = "empty"
    else:
        b0_prim = stage.GetPrimAtPath(b0[0])
        if b0_prim and b0_prim.IsValid() and b0_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            world_side = "rigid-body"
        elif b0_prim and b0_prim.IsValid():
            world_side = "non-rigid-prim"
        else:
            world_side = "dangling"
    newton_guesses = world_side in ("empty", "non-rigid-prim", "dangling")

    # ``b0[0]`` is safe here: an empty body0 is already covered by
    # ``newton_guesses``.
    if not (newton_guesses or b0[0] == str(robot_root_path)):
        return None

    p0, r0 = _vec(joint.GetLocalPos0Attr()), _quat(joint.GetLocalRot0Attr())
    p1, r1 = _vec(joint.GetLocalPos1Attr()), _quat(joint.GetLocalRot1Attr())
    pose0_identity = _is_zero(p0) and _is_unit_quat(r0)

    annot = prim.GetAttribute(WELD_ANCHOR_ANNOT)
    anchored_by_us = bool(annot and annot.Get())

    anchor_desc = ""
    if anchored_by_us:
        branch = f"world-anchored (localPose0 authored by the repair; {WELD_ANCHOR_ANNOT}=1)"
    elif not newton_guesses:
        branch = "body0 resolves to a rigid body — no guess"
    elif pose0_identity:
        branch = "recover-from-child (init_base_pose wins)"
    elif world_side == "non-rigid-prim":
        branch = "world-xform-of-non-rigid-prim (localPose0 stacks ON TOP of the stage pose)"
        anchor_desc = (
            f"the stage pose of {b0[0]} composed with localPose0 — i.e. "
            f"localPos0={p0} applied a second time on top of the pose the "
            f"stage already carries"
        )
    else:
        branch = "identity-fallback (init_base_pose DISCARDED, anchor=localPose0)"
        anchor_desc = f"exactly localPos0={p0}, with the requested pose dropped outright"

    return {
        "path": str(prim.GetPath()),
        "body0": b0,
        "body1": b1,
        "world_side": world_side,
        "newton_guesses": newton_guesses,
        "p0": p0,
        "r0": r0,
        "p1": p1,
        "r1": r1,
        "pose0_identity": pose0_identity,
        "branch": branch,
        "anchor_desc": anchor_desc,
        "anchored_by_us": anchored_by_us,
        "misplaces": bool(newton_guesses and not pose0_identity and not anchored_by_us),
        # ``_deactivate_root_joint`` requires body0 NON-empty; the upstream
        # anchor-fold requires body0 EMPTY.  Mutually exclusive on one prim,
        # so at most one can be live on any asset.
        "deactivate_match": bool(b0) and bool(b1) and b0[0] == str(robot_root_path) and b1[0] in urdf_root_link_paths,
        "fold_match": (not b0) and bool(b1) and not _is_zero(p0) and _is_unit_quat(r0),
    }

class _StageMixin:
    def _open_stage(self, newton_scene) -> None:
        """Open the USD stage.  MUST be overridden."""
        raise NotImplementedError("_open_stage must be implemented by the engine subclass")

    def _disable_local_render_products(self) -> None:
        """Deactivate UsdRender.Product prims in the session layer.

        Edits go to the session layer so they are scoped to this process only.
        Uses pure pxr APIs — safe in both headless and Kit-backed modes.
        In headless mode the render layer is never loaded so this is a no-op
        in practice; the call is retained as a defensive cleanup.
        """
        try:
            from pxr import Usd, UsdRender

            session_layer = self._stage.GetSessionLayer()
            disabled = []
            with Usd.EditContext(self._stage, session_layer):
                for prim in self._stage.Traverse():
                    if prim.IsA(UsdRender.Product):
                        prim.SetActive(False)
                        disabled.append(str(prim.GetPath()))
            if disabled:
                self._logger.info(
                    f"[newton-standalone] disabled {len(disabled)} RenderProduct(s) "
                    f"locally (session layer): {disabled}"
                )
            else:
                self._logger.info("[newton-standalone] no RenderProduct prims found to disable")
        except Exception as exc:
            self._logger.warn(f"[newton-standalone] _disable_local_render_products failed: {exc}")

    def _apply_init_base_pose(self) -> None:
        """Apply ``scene.robot.init_base_pose`` to the robot Xform.

        YAML shape::

            robot:
              init_base_pose:
                x: 0.0           # world meters
                y: 0.0
                z: 0.04
                theta: 0.0       # yaw radians about world Z

        Authors a translate + orient xform op on ``/<robot_prefix>`` at the
        stage's current edit target — which is the **root layer**, not the
        session layer: there is no ``Usd.EditContext`` here.  Nothing is
        ever saved, so this stays in memory, but it is not scoped away from
        anything else holding the same stage: ``Usd.Stage.Open`` caches by
        path, so a second open of the same scene within one process sees
        these ops and a second call raises "the xformOp ... already
        exists".  One engine, one call, so that is latent rather than live.

        How the two ``pin_base_to_world`` modes consume it:

          * ``pin_base_to_world: false`` — ``_deactivate_root_joint``
            drops the weld; Newton's ``parse_usd`` then adds a FREE base
            joint initialized to ``base_link``'s composed world pose,
            which is ``/<robot_prefix>``'s pose (since base_link itself
            has no local offset — measured true for the G2 asset, where
            the composed base_link is exactly the requested pose, but not
            guaranteed for a converter that emits one).
          * ``pin_base_to_world: true``  — the world-weld FixedJoint stays
            and pins ``base_link`` to ``/<robot_prefix>``.  This lands at
            ``init_base_pose`` **only when the weld's ``localPose0`` is
            identity**, which is what lets Newton recover the world frame
            from the child body.  A non-identity ``localPose0`` is either
            stacked on top of this pose or replaces it outright, depending
            on whether the joint's world side resolves to a stage prim —
            see ``_log_root_weld_diagnostics`` for the branch and
            ``_assert_base_pose_readback`` for the measured outcome.

        Existing xform ops authored on the cached ``robot.usda`` layer
        stay in their layer untouched; we append
        ``xformOp:translate:init_base`` + ``xformOp:orient:init_base``
        with a fresh xformOpOrder that names ONLY our two ops, so the
        composed transform is exactly the requested pose (the cached
        layer's ops are excluded from the order rather than overwritten).

        No-op when the yaml block is absent or non-mapping — callers
        get the cached ``robot.usda`` pose unchanged.
        """
        if self._stage is None:
            return
        base_pose = ((self._scene_cfg or {}).get("robot") or {}).get("init_base_pose")
        if not isinstance(base_pose, dict):
            return
        try:
            x = float(base_pose.get("x", 0.0))
            y = float(base_pose.get("y", 0.0))
            z = float(base_pose.get("z", 0.0))
            theta = float(base_pose.get("theta", 0.0))
        except (TypeError, ValueError) as exc:
            self._logger.warn(f"[newton-standalone] init_base_pose: numeric parse failed ({exc!r}); skipping")
            return
        try:
            import math  # noqa: PLC0415
            from pxr import Gf, Sdf, UsdGeom  # noqa: PLC0415
        except Exception as exc:  # noqa: BLE001
            self._logger.warn(f"[newton-standalone] init_base_pose: pxr import failed ({exc!r}); skipping")
            return

        root_path = Sdf.Path(f"/{self._robot_prefix_str}")
        root_prim = self._stage.GetPrimAtPath(root_path)
        if not root_prim or not root_prim.IsValid():
            self._logger.warn(f"[newton-standalone] init_base_pose: robot root {root_path} " f"not on stage; skipping")
            return

        xformable = UsdGeom.Xformable(root_prim)
        if not xformable:
            self._logger.warn(f"[newton-standalone] init_base_pose: {root_path} is not " f"Xformable; skipping")
            return

        # AddTranslateOp / AddOrientOp with a unique suffix author attributes
        # named ``xformOp:translate:init_base`` and ``xformOp:orient:init_base``
        # so they coexist with any existing ``xformOp:translate`` on a lower
        # layer.  SetXformOpOrder([t_op, o_op]) then names ONLY our two ops
        # as participating in the composed transform — the cached layer's
        # ops drop out of the order and contribute nothing.
        # Fold a baked world anchor out of the root weld and into this xform.
        #
        # A root FixedJoint with NO ``body0`` is the USD spelling of "welded to
        # the world", and newton's parse_usd has to guess where that world
        # frame is. It branches on whether the joint's world-side local pose is
        # identity (import_usd.py, "body0/body1 can resolve to world with an
        # empty path"):
        #
        #   identity     -> recover the world frame from the child body's
        #                   COMPOSED pose, which carries the xform authored
        #                   just below. The requested pose wins. Correct.
        #   non-identity -> assume USD baked the world anchor into localPose0
        #                   and pin the base there, never querying the prim.
        #                   The requested pose is ignored entirely.
        #
        # G2_omnipicker authors localPos0=(0,0,0) and lands correctly.
        # G2_90d authors (0, 0, 0.04) and lands at (0, 0, 0.04) no matter what
        # is asked for: a different requested base position is silently
        # ignored, and the robot ends up offset, inside the table.
        #
        # Nothing here compensates for that, deliberately. The repair that
        # does -- ``_repair_root_weld_anchor``, which clears ``body0`` and
        # rewrites ``localPose0`` to the pose the stage actually asks for --
        # reaches Newton's identity branch the same way a fold here would, so
        # the two must never coexist: a fold zeroes the very ``localPos0`` the
        # repair reads. The repair is the one kept, written up as quirk 7 of
        # ``docs/newton_quirks.md``. Where it is not present, an asset that
        # authors a non-identity ``localPos0`` lands offset until it arrives.
        # Either way, do not re-add a fold here.

        t_op = xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble, "init_base")
        t_op.Set(Gf.Vec3d(x, y, z))
        half = 0.5 * theta
        quat = Gf.Quatd(math.cos(half), Gf.Vec3d(0.0, 0.0, math.sin(half)))
        o_op = xformable.AddOrientOp(UsdGeom.XformOp.PrecisionDouble, "init_base")
        o_op.Set(quat)
        xformable.SetXformOpOrder([t_op, o_op])

        self._logger.info(
            f"[newton-standalone] init_base_pose applied to {root_path}: "
            f"xyz=({x:.4f}, {y:.4f}, {z:.4f}) yaw={theta:.4f} rad"
        )

    def _demote_kinematic_bodies(self) -> None:
        """Session-layer override: ``physics:rigidBodyEnabled = false`` on
        every prim that has ``physics:kinematicEnabled = true``.

        See the call site in ``_apply_runtime_fix_policies`` for the full
        rationale (kinematic bodies' free-joint/armature=1e10 encoding is
        numerically hostile to the mjwarp contact solver; a world-static
        collider is the faithful representation for scene furniture that
        never moves mid-episode).
        """
        if self._stage is None:
            return
        try:
            from pxr import UsdPhysics  # noqa: PLC0415
        except Exception as exc:  # noqa: BLE001
            self._logger.warn(f"[newton-standalone] kinematic demotion skipped (pxr import failed): {exc}")
            return
        demoted = []
        try:
            for prim in self._stage.Traverse():
                if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    continue
                api = UsdPhysics.RigidBodyAPI(prim)
                kin_attr = api.GetKinematicEnabledAttr()
                if not kin_attr or not bool(kin_attr.Get()):
                    continue
                api.CreateRigidBodyEnabledAttr(False)
                demoted.append(str(prim.GetPath()))
        except Exception as exc:  # noqa: BLE001
            self._logger.warn(f"[newton-standalone] kinematic demotion failed mid-walk: {exc!r}")
            return
        if demoted:
            self._logger.info(
                f"[newton-standalone] demoted {len(demoted)} kinematic rigid "
                f"body/bodies to STATIC collider(s): "
                f"{', '.join(p.rsplit('/', 2)[-2] for p in demoted[:6])}"
                f"{'…' if len(demoted) > 6 else ''} "
                f"(physics:kinematicEnabled=1 → rigidBodyEnabled=false; "
                f"collision geometry kept, dynamics dropped). These bodies can "
                f"no longer be MOVED at runtime — if a task needs to reposition "
                f"one mid-episode, this line is where that stopped working"
            )
        else:
            self._logger.info("[newton-standalone] kinematic demotion: no kinematic rigid bodies on stage")

    def _apply_runtime_fix_policies(self) -> None:
        """Apply ``pin_base_to_world`` + ``convert_joints_to_fixed``
        overrides on the composed stage before ``add_usd`` parses it.

        ``pin_base_to_world`` (refined by ``pin_base_mode`` — see
        ``genie_sim_engine_core.schema.scene.PinBaseMode``, the one place
        the mode vocabulary and the resolution rule are declared)
          IsaacSim's URDF→USD converter always emits a
          ``PhysicsFixedJoint "root_joint"`` between ``/robot`` (the
          Xform — Newton resolves as world) and ``base_link``, welding
          the base.  Three resolved modes:

            * ``weld``  (``pin_base_to_world=True``, no override) — leave
              the weld prim alone (exact prior ``True`` behavior).
            * ``free``  (``pin_base_to_world=False``) — deactivate the
              prim so Newton's importer adds a FREE joint instead and the
              base is mobile under physics (exact prior ``False``
              behavior).
            * ``mocap`` (``pin_base_to_world=True`` plus an explicit
              ``pin_base_mode: mocap``) — also deactivate the prim (same FREE joint
              as ``free``), but the adapter then clamps that joint every
              substep via the kinematic-pin machinery
              (``adapters/mujoco_warp_multi_env.py::_build_base_pin``) instead
              of leaving it to floor/wheel contact — so reset-time base
              qpos writes (benchmark generalization) stick.

        ``convert_joints_to_fixed: [base, head, body, arm, gripper, no_robot]``
          A list of sub-tree tokens.  Each present token triggers
          ``_replace_joints_with_fixed`` over a specific name-substring,
          except ``no_robot`` which deactivates the whole robot Xform:

            * ``"base"`` → joints whose name contains ``chassis``
              (chassis attach + every wheel steering / spin joint).
              Chassis link + wheel links stay rigid for visualization.
            * ``"head"`` → joints whose name contains ``head_joint``
              (3 head revolute joints).
            * ``"body"`` → joints whose name contains ``body_joint``
              (G2's 5-DOF torso chain).
            * ``"arm"`` → joints whose name contains ``arm_``
              (left + right arm revolute joints).  Welds the entire
              arm chain to its init pose — robot becomes a frozen
              statue from the shoulder down.
            * ``"gripper"`` → joints whose name contains ``gripper_``
              (gripper master + mimic followers).
            * ``"no_robot"`` → deactivates ``/<robot_prefix_str>``
              entirely on the session layer.  ``add_usd`` then never
              sees the robot at all — the model contains only scene
              entries (cloth, hanger, table, ground).  Useful for
              isolating "is the rigid solver even doing rigid contact
              correctly" tests, or for AVBD scenes where the robot
              would force unsupported joint types into the model.
              Other welding tokens become no-ops in this case (their
              substring matches return zero joints).

          Bodies on both sides stay in ``model.body_label`` so TF
          publishes them at their init pose; the joints contribute 0
          DOFs to Featherstone's M matrix / mjwarp's actuator count.

        Joints that are already typed ``PhysicsFixedJoint`` (e.g. the
        URDF-fixed ``idx100_chassis_base_joint``) get skipped by the
        replacement helper — they're already cost-free.

        All edits are USD session-layer overrides on the live composed
        stage; nothing touches the cached ``robot.usda`` payloads, so
        these flags toggle at zero rebuild cost.
        """
        if self._stage is None:
            return
        try:
            from pxr import UsdPhysics  # noqa: F401  (verify USD bindings available)
        except Exception as exc:  # noqa: BLE001
            self._logger.warn(f"[newton-standalone] runtime fix policies skipped (pxr import failed): {exc}")
            return

        # ``weld`` (pin_base_to_world=True, no mode override) is the ONLY
        # branch that leaves the URDF-authored world-weld FixedJoint alone.
        # ``free`` (pin_base_to_world=False) is the pre-existing behavior:
        # deactivate it so ``add_usd`` gives the base a FREE joint mobile
        # under ordinary physics.  ``mocap`` (see ``PinBaseMode``) ALSO
        # needs a FREE root joint — same deactivation call as ``free`` —
        # but the base is then clamped per-substep by the kinematic-pin
        # machinery (adapters/mujoco_warp_multi_env.py::_build_base_pin)
        # instead of being left to floor/wheel contact.  Asking
        # ``keeps_root_weld`` rather than comparing the mode string keeps
        # ``weld`` byte-for-byte and makes the question this branch is
        # really asking the thing written down.
        if not PinBaseMode.of(self).keeps_root_weld:
            self._deactivate_root_joint()

        # Demote kinematic rigid bodies to static colliders.
        #
        # Newton's importer represents a ``physics:kinematicEnabled`` rigid body
        # as a FREE joint whose DOFs carry armature = 1e10 — an "infinite
        # inertia" stand-in that MuJoCo-Warp's contact solve handles badly:
        # objects resting on such a body tumble microscopically forever, and
        # once the per-substep state push is removed the body itself drifts off
        # at constant velocity after any penetration-recovery impulse — nothing
        # can decelerate 1e10 kg. Patching it in place (mocap-style pinning plus
        # a scaled armature) drives the solve straight to NaN, because too many
        # precomputed quantities are entangled with that 1e10.
        #
        # A kinematic scene body never moves for a whole episode in our tasks,
        # so the faithful and numerically clean representation is a world-static
        # collider, exactly like the floor. Setting
        # ``physics:rigidBodyEnabled=false`` at the session layer keeps the
        # collision geometry and drops the rigid-body dynamics, so Newton
        # resolves it as a static shape.
        self._demote_kinematic_bodies()
        # Scenery that no joint drives costs 6 DOF and a contact set each, and
        # some rooms author it overlapping itself — see the method.
        self._demote_background_decor()

        # Map normalized sub-tree token -> joint-name substring used to
        # find matching joints in _replace_joints_with_fixed.  Add new
        # rows here when a robot has a new welded sub-chain (e.g. legs).
        #
        # ``arm`` / ``gripper`` exist primarily for the AVBD path: this
        # Newton version's SolverVBD doesn't support REVOLUTE rigid
        # joints (only CABLE/BALL/FIXED), so to test AVBD on a scene
        # that contains an articulated robot we have to weld every
        # revolute joint to FIXED.  The robot becomes a frozen statue
        # — visible in the viewer, present as a static collider, but
        # no actuation, no joint motion.  Use only when you specifically
        # need the AVBD comparison; the Featherstone+VBD path runs
        # revolute joints natively and shouldn't have these tokens set.
        _SUBSTRING_BY_TOKEN = {
            "base": "chassis",
            "head": "head_joint",
            "body": "body_joint",
            "arm": "arm_",
            "gripper": "gripper_",
        }
        tokens = list(getattr(self, "_convert_joints_to_fixed", []) or [])

        # ``no_robot`` is the only token that doesn't map to a joint-name
        # substring.  Handle it first — deactivating the robot's top-level
        # Xform makes the other tokens' joint substring matches return
        # nothing, which logs cleanly as "0 replaced".
        if "no_robot" in tokens:
            self._deactivate_robot_xform()
            tokens = [t for t in tokens if t != "no_robot"]

        for token in tokens:
            sub = _SUBSTRING_BY_TOKEN.get(token)
            if sub is None:
                self._logger.warn(
                    f"[newton-standalone] convert_joints_to_fixed: unknown "
                    f"token {token!r}; valid: {sorted(_SUBSTRING_BY_TOKEN) + ['no_robot']}.  "
                    f"Skipped."
                )
                continue
            self._replace_joints_with_fixed(name_substring=sub, label=f"convert_joints_to_fixed[{token}]")

    def _deactivate_robot_xform(self) -> None:
        """Deactivate ``/<robot_prefix_str>`` on the session layer so
        Newton's ``add_usd`` never registers any robot prim — bodies,
        joints, shapes — into the model.

        Used by ``convert_joints_to_fixed: [no_robot]`` to test
        cloth + scene-rigid pipelines in isolation, without the robot's
        joint types or contact pairs interfering.  Session-layer scoped
        — the cached ``robot.usda`` is untouched.
        """
        if self._stage is None:
            return
        path = f"/{self._robot_prefix_str}"
        prim = self._stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            self._logger.warn(
                f"[newton-standalone] convert_joints_to_fixed[no_robot]: "
                f"couldn't locate robot Xform at {path}; nothing to deactivate"
            )
            return
        prim.SetActive(False)
        self._logger.info(
            f"[newton-standalone] convert_joints_to_fixed[no_robot]: "
            f"deactivated {path} (robot will not appear in model — "
            f"scene runs with only cloth/hanger/static colliders)"
        )

    def _log_root_weld_diagnostics(self) -> None:
        """Read-only report on the URDF-authored root weld.

        Pure diagnostics — nothing on this path mutates the stage.  Runs
        BEFORE ``_apply_runtime_fix_policies`` so the weld prim is still
        active in every pin mode, and traverses with
        ``Usd.PrimAllPrimsPredicate`` so neither a future reordering nor a
        switch from ``SetActive(False)`` to ``physics:jointEnabled=false``
        can hide it.

        Why this exists: where a welded base actually ends up depends on
        how Newton's importer resolves the joint's WORLD side, and that
        resolution is invisible from here today.
        ``newton/_src/utils/import_usd.py`` reads ``body0`` through
        ``JointDesc.body0``, which returns ``""`` for any non-rigid target
        — and ``/<robot_prefix>`` is an ``ArticulationRootAPI`` Xform, not
        a rigid body, so an authored ``physics:body0`` does NOT guarantee
        Newton sees one.  An unresolved world side takes a guess branch
        with two very different outcomes:

          * ``localPose0`` identity → the world frame is recovered from the
            child body's composed pose, so ``init_base_pose`` wins.
          * ``localPose0`` non-identity → ``world_body_xform`` falls back
            to identity and the authored ``localPose0`` becomes the anchor,
            so ``init_base_pose`` is discarded entirely and the base lands
            at the raw offset.

        Logged per candidate weld: prim path, whether ``physics:body0``
        carries a target, whether that target bears ``RigidBodyAPI`` (the
        real condition for the guess branch), both local poses, and which
        branch each in-repo predicate would take — ``_deactivate_root_joint``
        (requires body0 NON-empty) versus the upstream anchor-fold
        (requires body0 EMPTY).  Those two predicates are mutually
        exclusive on one prim, so at most one can be live on any asset;
        this says which.  The requested ``init_base_pose`` and the resolved
        pin mode are logged alongside, so a misplaced base can be read off
        one log line instead of reproduced.
        """
        if self._stage is None:
            return
        try:
            from pxr import Sdf, Usd, UsdPhysics  # noqa: PLC0415

            mode = PinBaseMode.of(self)
            base_pose = ((self._scene_cfg or {}).get("robot") or {}).get("init_base_pose")
            if isinstance(base_pose, dict):
                requested = (
                    f"x={base_pose.get('x', 0.0)} y={base_pose.get('y', 0.0)} "
                    f"z={base_pose.get('z', 0.0)} theta={base_pose.get('theta', 0.0)}"
                )
            else:
                requested = "<absent — cached robot.usda pose>"

            robot_root_path = Sdf.Path(f"/{self._robot_prefix_str}")
            root_prim = self._stage.GetPrimAtPath(robot_root_path)
            self._logger.info(
                f"[newton-standalone] root-weld diag: pin_base_to_world="
                f"{getattr(self, '_pin_base_to_world', False)} resolved_mode={mode} "
                f"robot_root={robot_root_path} init_base_pose({requested})"
            )
            if not root_prim or not root_prim.IsValid():
                self._logger.warn(
                    f"[newton-standalone] root-weld diag: robot root {robot_root_path} "
                    f"not on stage; no weld analysis possible."
                )
                return

            # The URDF root link(s) — shared rule, see the module helper.
            urdf_root_link_paths = topmost_rigid_body_paths(self._stage, robot_root_path)
            # Capped: a flat asset (every link a direct child of the root)
            # has dozens of topmost rigid bodies and would print one unusable
            # line.  The count is the useful part; 4 samples place it.
            _shown = urdf_root_link_paths[:4]
            _more = len(urdf_root_link_paths) - len(_shown)
            self._logger.info(
                f"[newton-standalone] root-weld diag: "
                f"urdf_root_link(s)[{len(urdf_root_link_paths)}]={_shown or '<none>'}"
                + (f" (+{_more} more)" if _more > 0 else "")
            )

            candidates = 0
            for prim in Usd.PrimRange(root_prim, Usd.PrimAllPrimsPredicate):
                if not prim.IsA(UsdPhysics.FixedJoint):
                    continue
                rec = classify_world_weld(self._stage, prim, robot_root_path, urdf_root_link_paths)
                if rec is None:
                    continue
                candidates += 1

                self._logger.info(
                    f"[newton-standalone] root-weld diag: {rec['path']}\n"
                    f"    body0={rec['body0'] or '<none>'} ({rec['world_side']})  "
                    f"body1={rec['body1'] or '<none>'}\n"
                    f"    localPos0={rec['p0']} localRot0(w,x,y,z)={rec['r0']} "
                    f"identity={rec['pose0_identity']}\n"
                    f"    localPos1={rec['p1']} localRot1(w,x,y,z)={rec['r1']}\n"
                    f"    newton_world_side -> {rec['branch']}\n"
                    f"    predicate: _deactivate_root_joint={rec['deactivate_match']} "
                    f"anchor_fold={rec['fold_match']}"
                )
                if mode.keeps_root_weld and rec["misplaces"]:
                    self._logger.warn(
                        f"[newton-standalone] root-weld diag: {rec['path']} would MISPLACE "
                        f"the base under Newton — world side is '{rec['world_side']}' and "
                        f"localPose0 is non-identity, so the weld would be anchored at "
                        f"{rec['anchor_desc']}. Note the misplacement is NOT visible in "
                        f"model.body_q straight after finalize (that is the pose read off the "
                        f"stage, and it is correct); it lives in joint_X_p and appears once FK "
                        f"snaps the welded base onto the anchor. "
                        f"_repair_root_weld_anchor rewrites this weld before add_usd sees it; "
                        f"the base-pose readback confirms the outcome."
                    )
                if not mode.keeps_root_weld and not rec["deactivate_match"]:
                    self._logger.warn(
                        f"[newton-standalone] root-weld diag: mode={mode} wants this weld "
                        f"deactivated, but _deactivate_root_joint's predicate does NOT match "
                        f"{rec['path']} (body0={rec['body0'] or '<none>'}). "
                        f"The base may stay welded."
                    )

            if candidates == 0:
                self._logger.warn(
                    f"[newton-standalone] root-weld diag: no candidate world-weld FixedJoint "
                    f"under {robot_root_path}. Either a ghost-root URDF (no root_joint emitted "
                    f"at all) or the converter changed shape — see diagnose_urdf.py."
                )
        except Exception as exc:  # noqa: BLE001
            self._logger.warn(f"[newton-standalone] root-weld diag failed ({exc!r}); continuing")

    def _repair_root_weld_anchor(self) -> int:
        """Rewrite a world weld that Newton would anchor in the wrong place.

        The upstream behaviour, its two contradicting branches, and the
        measurements below are written up as quirk 7 in
        ``docs/newton_quirks.md``.

        ``weld`` mode keeps the URDF-authored world weld and relies on
        Newton *inferring* the world frame from it.  That inference is
        correct for exactly one authoring shape — identity ``localPose0``,
        which lets it recover the frame from the child body — and our
        shipping asset has that shape, so this method is a no-op on every
        scene in the repo today.  It exists because the inference is not
        ours: a converter revision that bakes the base offset into
        ``localPose0``, or emits ``body0`` empty, silently moves the robot
        (measured: 40 mm and 300 mm respectively, on an asset whose only
        change was those attributes).  ``_log_root_weld_diagnostics``
        predicts that; ``_assert_base_pose_readback`` measures it; this
        removes it.

        The repair replaces inference with a value we compute:

          * ``body0`` targets cleared, so the joint's parent frame is the
            world and Newton has nothing left to guess;
          * ``localPose0`` set to the composed world transform of
            ``body1`` — the pose the stage, and therefore
            ``init_base_pose``, actually asks for;
          * ``localPose1`` set to identity, so the child frame is the body
            frame and the anchor is ``localPose0`` outright.

        That lands on Newton's identity-fallback branch deliberately: with
        ``localPose0`` carrying the pose we want, "the anchor is
        ``localPose0``" is the correct answer rather than the bug.

        The rewritten weld is stamped ``newton:weldAnchorIsWorld`` so the
        classifier can tell a deliberate world anchor from the baked offset
        it looks exactly like; without it the predicate stays true and a
        second pass rewrites the same joint.

        MUST run after ``_apply_init_base_pose`` — it reads ``body1``'s
        composed world pose, which that method sets — and before
        ``add_usd``.  Edits go to the session layer, so the cached
        ``robot.usda`` payload stays scene-agnostic.

        Returns the number of welds rewritten.
        """
        if self._stage is None:
            return 0
        if not PinBaseMode.of(self).keeps_root_weld:
            # ``free`` / ``mocap`` deactivate the weld outright; there is no
            # anchor left to get wrong.
            return 0
        try:
            from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics  # noqa: PLC0415

            robot_root_path = Sdf.Path(f"/{self._robot_prefix_str}")
            root_prim = self._stage.GetPrimAtPath(robot_root_path)
            if not root_prim or not root_prim.IsValid():
                return 0
            urdf_root_link_paths = topmost_rigid_body_paths(self._stage, robot_root_path)

            repaired = 0
            session_layer = self._stage.GetSessionLayer()
            for prim in Usd.PrimRange(root_prim, Usd.PrimAllPrimsPredicate):
                if not prim.IsA(UsdPhysics.FixedJoint):
                    continue
                rec = classify_world_weld(self._stage, prim, robot_root_path, urdf_root_link_paths)
                if rec is None or not rec["misplaces"]:
                    continue
                if not rec["body1"]:
                    self._logger.warn(
                        f"[newton-standalone] root-weld repair: {rec['path']} would be "
                        f"misplaced but has no body1, so there is no body whose world pose "
                        f"could define the anchor. Left alone; expect a misplaced base."
                    )
                    continue

                child = self._stage.GetPrimAtPath(rec["body1"][0])
                xformable = UsdGeom.Xformable(child) if child and child.IsValid() else None
                if not xformable:
                    self._logger.warn(
                        f"[newton-standalone] root-weld repair: {rec['path']}'s body1 "
                        f"{rec['body1'][0]} is not Xformable; cannot compute the anchor. "
                        f"Left alone; expect a misplaced base."
                    )
                    continue
                # Gf.Transform decomposes, so a scaled ancestor still yields a
                # unit rotation quaternion.
                xf = Gf.Transform(xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
                t, q = xf.GetTranslation(), xf.GetRotation().GetQuat()

                with Usd.EditContext(self._stage, session_layer):
                    joint = UsdPhysics.Joint(prim)
                    joint.GetBody0Rel().SetTargets([])
                    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(float(t[0]), float(t[1]), float(t[2])))
                    joint.CreateLocalRot0Attr().Set(
                        Gf.Quatf(
                            float(q.GetReal()),
                            float(q.GetImaginary()[0]),
                            float(q.GetImaginary()[1]),
                            float(q.GetImaginary()[2]),
                        )
                    )
                    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
                    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
                    # Without this the rewritten weld is indistinguishable
                    # from the shape it just replaced, so the classifier
                    # would keep calling it broken and a second pass would
                    # rewrite it again.  Also shows up in the runtime USD
                    # dump, where it says why localPose0 looks like an
                    # offset.
                    prim.CreateAttribute(WELD_ANCHOR_ANNOT, Sdf.ValueTypeNames.Bool).Set(True)

                repaired += 1
                self._logger.info(
                    f"[newton-standalone] root-weld repair: rewrote {rec['path']} — "
                    f"branch was '{rec['branch']}', which would have anchored the base at "
                    f"{rec['anchor_desc']}. body0 cleared to world; localPose0 set to "
                    f"{rec['body1'][0]}'s composed world pose "
                    f"pos=({float(t[0]):.6f}, {float(t[1]):.6f}, {float(t[2]):.6f}) "
                    f"quat(w,x,y,z)=({float(q.GetReal()):.6f}, {float(q.GetImaginary()[0]):.6f}, "
                    f"{float(q.GetImaginary()[1]):.6f}, {float(q.GetImaginary()[2]):.6f}); "
                    f"localPose1 set to identity."
                )

            return repaired
        except Exception as exc:  # noqa: BLE001
            self._logger.warn(
                f"[newton-standalone] root-weld repair failed ({exc!r}); continuing with the "
                f"authored weld. Check the base-pose readback below."
            )
            return 0

    def _deactivate_root_joint(self) -> int:
        """Deactivate the URDF→USD-authored world-weld FixedJoint so
        Newton's ``parse_usd`` adds a FREE joint at the base instead,
        making the URDF root mobile under physics.

        Both URDF importers MAY author this weld but at different prim
        paths:

          * Isaac Sim 6.x (``urdf_usd_converter`` + Asset Structure 3.0):
            ``/<robot_prefix>/Joints/root_joint``
          * Isaac Sim 4.x/5.x (``URDFParseAndImportFile``): the path is
            importer-named and varies per layout, but the topology is
            the same — a ``UsdPhysics.FixedJoint`` whose ``body0`` is the
            articulation root prim and ``body1`` is the topmost
            ``RigidBodyAPI``-bearing descendant (the URDF root link).

        Detect by topology rather than by name, matching the same
        heuristic used by ``kit/stage.py::_apply_fix_base_policy``.
        Setting ``prim.SetActive(False)`` on every match removes them
        from the composed-stage view ``add_usd`` sees; Newton then
        treats the root link as an articulation root with no parent
        joint and ``_add_base_joint(floating=None)`` (the default)
        creates a FREE joint for it.

        Returns the number of welds deactivated.
        """
        if self._stage is None:
            return 0
        try:
            from pxr import Sdf, Usd, UsdPhysics  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            self._logger.warn(
                f"[newton-standalone] pin_base_to_world=False: pxr import failed ({exc}); "
                f"cannot locate URDF world-weld joint."
            )
            return 0

        robot_root_path = Sdf.Path(f"/{self._robot_prefix_str}")
        root_prim = self._stage.GetPrimAtPath(robot_root_path)
        if not root_prim or not root_prim.IsValid():
            self._logger.warn(
                f"[newton-standalone] pin_base_to_world=False: robot root prim "
                f"{robot_root_path} not found; cannot identify world-weld joint."
            )
            return 0

        # Topmost RigidBodyAPI descendants — the URDF root link(s).
        urdf_root_link_paths = []
        for desc in Usd.PrimRange(root_prim):
            if desc == root_prim:
                continue
            if not desc.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
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
            self._logger.warn(
                f"[newton-standalone] pin_base_to_world=False: no RigidBodyAPI "
                f"descendant under {robot_root_path}; cannot identify URDF root link. "
                f"Base will be free by Newton's default in that case."
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
                # Internal fixed joint (body1 is a deeper-nested rigid body) —
                # leave strictly alone, regardless of pin_base_to_world.
                continue
            prim.SetActive(False)
            found += 1
            self._logger.info(
                f"[newton-standalone] pin_base_to_world=False → deactivated "
                f"world-weld {prim.GetPath()} (body0={b0[0]}, body1={b1[0]}). "
                f"Newton will add a FREE base joint; pair with the plain "
                f"rigid-substep regime (real gravity + contacts)."
            )

        if found == 0:
            self._logger.warn(
                f"[newton-standalone] pin_base_to_world=False: no world-weld "
                f"FixedJoint detected under {robot_root_path} (body0=root, "
                f"body1=URDF root link). Likely a ghost-root URDF — see "
                f"diagnose_urdf.py. Base will be free by Newton's default."
            )
        return found

    def _replace_joints_with_fixed(self, *, name_substring: str, label: str) -> int:
        """Replace every joint whose name contains ``name_substring`` with
        a ``UsdPhysics.FixedJoint``.

        The original joint prim is deactivated; the new FixedJoint is
        authored as a sibling under the same ``/robot/Physics/...``
        scope.  ``add_usd`` only sees the new FixedJoint and treats the
        connection as a static weld:

          * bodies on both sides remain rigid bodies and STAY in
            ``model.body_label`` — TF publishing emits their poses every
            tick (poses are constant because the joint can't move).
          * the joint contributes ZERO DOFs to Featherstone's mass
            matrix.

        For revolute joints with a non-zero ``init_joint_pos`` entry,
        the init angle is BAKED INTO the FixedJoint's ``localRot0``.
        That preserves the user-requested pose without giving the joint
        a DOF.  The same value (in radians) is recorded into
        ``self._static_joint_q`` so ``/joint_states`` publishes it
        synthetically and ``robot_state_publisher``'s URDF FK matches
        Newton's body_q for everything downstream.

        Net effect: ``fix_*`` flags shrink Featherstone's M(q) and
        nothing else — URDF, /joint_states shape, TF tree, RViz, init
        pose semantics all behave as if the flag weren't there.

        Returns the count of joints replaced.
        """
        if self._stage is None:
            return 0
        try:
            from pxr import Gf, Sdf, UsdPhysics
        except Exception as exc:  # noqa: BLE001
            self._logger.warn(f"[newton-standalone] {label}: USD import failed: {exc}")
            return 0

        # Build a name → init-radians lookup once.  YAML stores revolute
        # init poses in degrees (matches PhysX convention); convert here
        # so the bake math is unit-clean.
        DEG2RAD = math.pi / 180.0
        init_rad_by_name: Dict[str, float] = {}
        ip = getattr(self, "_init_joint_pos", None) or {}
        if isinstance(ip, dict):
            for nm, raw in ip.items():
                try:
                    v = float(raw.value if hasattr(raw, "value") else raw)
                except (TypeError, ValueError):
                    continue
                # Revolute → degrees.  Prismatic init poses (metres) get
                # baked verbatim below as a translation; non-zero values
                # there would need a different code path (translation of
                # localPos0), which we don't currently exercise — flagged
                # in the bake site below.
                init_rad_by_name[nm] = v * DEG2RAD

        _AXIS_VEC = {
            "X": Gf.Vec3f(1.0, 0.0, 0.0),
            "Y": Gf.Vec3f(0.0, 1.0, 0.0),
            "Z": Gf.Vec3f(0.0, 0.0, 1.0),
        }

        n = 0
        n_baked = 0
        sample = []
        baked_sample = []
        try:
            # Two-step: collect first, mutate after — mutating mid-traverse
            # can invalidate the prim iterator on some USD versions.
            candidates = []
            for prim in self._stage.Traverse():
                if not (prim.IsA(UsdPhysics.Joint) or prim.HasAPI(UsdPhysics.Joint)):
                    continue
                if name_substring not in prim.GetName():
                    continue
                # Skip if already a FixedJoint or if the prim is somehow inactive.
                if prim.GetTypeName() == "PhysicsFixedJoint":
                    continue
                if not prim.IsActive():
                    continue
                candidates.append(prim)

            for prim in candidates:
                joint_name = prim.GetName()

                joint = UsdPhysics.Joint(prim)
                body0_targets = list(joint.GetBody0Rel().GetTargets())
                body1_targets = list(joint.GetBody1Rel().GetTargets())
                if not body0_targets or not body1_targets:
                    # Defensive — a degenerate joint with no bodies isn't
                    # useful to replace.  Just deactivate so add_usd
                    # doesn't get confused later.
                    prim.SetActive(False)
                    continue

                lp0 = joint.GetLocalPos0Attr().Get()
                lr0 = joint.GetLocalRot0Attr().Get()
                lp1 = joint.GetLocalPos1Attr().Get()
                lr1 = joint.GetLocalRot1Attr().Get()

                # Bake the init pose into the FixedJoint's local frame
                # for revolute joints.  Math:
                #   At joint angle θ, body1's frame in body0's frame is
                #     localTransform0 * Rotate_in_joint_frame(axis, θ)
                #     * localTransform1.Inverse()
                #   So the equivalent FixedJoint has
                #     new_localRot0 = localRot0 * Quat(axis, θ)
                #   while localPos0 / localPos1 / localRot1 stay
                #   unchanged.  Pixar Gf.Quatf uses (q1 * q2)(v) = q1(q2(v))
                #   composition — see ``Gf.Quatf.__mul__`` — so the
                #   left-multiply applies the existing localRot0 last,
                #   matching the joint's "rotate around the joint axis
                #   in body0's joint frame" semantics.
                init_rad = init_rad_by_name.get(joint_name, 0.0)
                if init_rad != 0.0 and prim.GetTypeName() == "PhysicsRevoluteJoint":
                    rev = UsdPhysics.RevoluteJoint(prim)
                    axis_token = rev.GetAxisAttr().Get()
                    axis = _AXIS_VEC.get(axis_token) if axis_token else None
                    if axis is not None:
                        half = init_rad * 0.5
                        bake_q = Gf.Quatf(
                            math.cos(half),
                            axis * math.sin(half),
                        )
                        lr0 = (Gf.Quatf(lr0) * bake_q) if lr0 is not None else bake_q
                        n_baked += 1
                        if len(baked_sample) < 4:
                            baked_sample.append(f"{joint_name}@{math.degrees(init_rad):+.1f}°")

                # New FixedJoint as a sibling under the same parent scope
                # (typically /robot/Physics/).
                src_path = prim.GetPath()
                new_path = Sdf.Path(f"{src_path.GetParentPath().pathString}/{src_path.name}__fixed")
                new_joint = UsdPhysics.FixedJoint.Define(self._stage, new_path)
                new_joint.GetBody0Rel().SetTargets(body0_targets)
                new_joint.GetBody1Rel().SetTargets(body1_targets)
                if lp0 is not None:
                    new_joint.GetLocalPos0Attr().Set(lp0)
                if lr0 is not None:
                    new_joint.GetLocalRot0Attr().Set(lr0)
                if lp1 is not None:
                    new_joint.GetLocalPos1Attr().Set(lp1)
                if lr1 is not None:
                    new_joint.GetLocalRot1Attr().Set(lr1)

                prim.SetActive(False)
                n += 1
                # Synthetic /joint_states value.  Always the init pose
                # value (radians) — for joints without an init_pose entry,
                # this is 0 and matches the un-rotated FixedJoint we just
                # authored; for joints with one, this matches the angle
                # we baked into ``localRot0`` so the URDF FK in
                # robot_state_publisher reproduces Newton's body_q
                # transform exactly.
                self._static_joint_q[joint_name] = init_rad
                if len(sample) < 4:
                    sample.append(src_path.name)

            if n > 0:
                msg = (
                    f"[newton-standalone] {label}=True: replaced {n} joint(s) "
                    f"containing {name_substring!r} with UsdPhysics.FixedJoint "
                    f"(bodies retained for TF; {n} DOF(s) removed from M); "
                    f"sample={sample}"
                )
                if n_baked > 0:
                    msg += (
                        f"; baked init pose into {n_baked} FixedJoint local "
                        f"frame(s) (sample={baked_sample}) — published "
                        f"synthetically via /joint_states"
                    )
                self._logger.info(msg)
            else:
                self._logger.info(
                    f"[newton-standalone] {label}=True: no active joints containing "
                    f"{name_substring!r} on the stage (no-op)"
                )
        except Exception as exc:  # noqa: BLE001
            self._logger.warn(f"[newton-standalone] {label} policy failed: {exc}")
        return n

    def _collect_scene_colliders_to_decompose(self) -> None:
        """Note the colliders that would reach the solver as one solid lump.

        See ``_SCENE_DECOMP_MAX_SOLIDITY`` for the three ways Newton's importer
        ends up there and why the trigger is measured concavity rather than the
        authored ``physics:approximation`` value.

        Two things happen per hit: the path is recorded for
        ``_decompose_scene_colliders`` (which runs after ``add_usd``, where a
        CoACD threshold is ours to choose), and the prim's
        ``physics:approximation`` is BLOCKED. The block is the load-bearing
        half -- Newton's ``approximate_meshes`` replaces the shape's mesh with
        its own result, so without it we would be handed a convex hull and have
        nothing left to decompose. Blocking is a session-time edit on the
        composed stage; the assets are untouched.

        Not the same thing as ``disable_swallowing_room_shells`` in
        ``task_loader``: that one *removes* collision from room shells whose
        hull swallows the robot. Prims it already disabled are skipped here.

        Colliders whose path matches ``_SCENE_DECOMP_EXCLUDE_NAME_SUBSTRINGS``
        (default: tables) are skipped before the solidity/void measurement
        runs at all -- see that constant for why tables trip the same test as
        the door frames and carts this path exists for, and what it cost one
        of them. Override per scene with
        ``scene_colliders.decomposition_exclude_names`` (an empty list clears
        the exclusion).
        """
        self._scene_decomp_paths: set = set()
        if self._stage is None:
            return
        _sc = getattr(getattr(self, "_params", None), "scene_colliders", None)
        if _sc is None or not bool(getattr(_sc, "decomposition", False)):
            self._logger.info(
                "[newton-standalone] scene-collider decomposition off "
                "(scene_colliders.decomposition.enabled) — concave colliders "
                "stay single convex hulls"
            )
            return
        try:
            import numpy as _np  # noqa: PLC0415
            from pxr import UsdGeom  # noqa: PLC0415

            from .model import mesh_solidity  # noqa: PLC0415
        except Exception as exc:  # noqa: BLE001
            self._logger.warn(
                f"[newton-standalone] scene-collider scan skipped "
                f"(import failed): {exc}"
            )
            return

        max_verts = int(getattr(_sc, "decomposition_max_verts",
                                _SCENE_DECOMP_MAX_VERTS))
        max_sol = float(getattr(_sc, "decomposition_max_solidity",
                                _SCENE_DECOMP_MAX_SOLIDITY))
        min_void = float(getattr(_sc, "decomposition_min_void",
                                 _SCENE_DECOMP_MIN_VOID_VOLUME))
        # Name-based exclusion, read the same way as the thresholds above (the
        # scene config is where this branch keeps these knobs -- it deliberately
        # moved off RLINF_SCENE_DECOMP_* env vars).  ``None`` (attribute absent,
        # or present-but-None) keeps the default table skip; an explicit empty
        # list clears it entirely.
        _excl = getattr(_sc, "decomposition_exclude_names", None)
        if _excl is None:
            _excl = _SCENE_DECOMP_EXCLUDE_NAME_SUBSTRINGS
        # Lowercased here rather than relying on the constant already being
        # lowercase: the match below runs against ``path_str.lower()``, so a
        # future edit to the constant that used capitals would silently stop
        # matching anything -- the failure mode this whole change exists to fix.
        exclude_subs = tuple(str(s).strip().lower() for s in _excl if str(s).strip())

        def _collision_enabled(prim) -> bool:
            """Only this prim's own opinion counts.

            Each prim with ``PhysicsCollisionAPI`` is its own collider, so a
            parent that says ``collisionEnabled=false`` does not disable its
            children -- and ``disable_swallowing_room_shells`` writes its
            override on the MESH prim, never on an ancestor, so there is
            nothing an ancestor check would catch. Reading the parent too
            silently dropped ``supermarket_shelf_002``: its
            ``entity/body`` carries ``collisionEnabled=false`` while the
            ``entity/body/visual`` under it collides for real, and skipping it
            left five bottles up to 168mm inside a solid shelf.
            """
            attr = prim.GetAttribute("physics:collisionEnabled")
            return not (attr and attr.IsValid() and attr.Get() is False)

        def _triangles(counts, indices):
            """Face-vertex arrays -> (N, 3) triangle indices, fan-triangulated.

            Vectorised for the all-triangle and all-quad cases, which is every
            benchmark mesh of any size; the loop is the correctness fallback
            for mixed topology and only ever runs on small meshes here.
            """
            c = _np.asarray(counts)
            i = _np.asarray(indices)
            if len(c) == 0:
                return _np.zeros((0, 3), dtype=_np.int64)
            if (c == 3).all():
                return i.reshape(-1, 3)
            if (c == 4).all():
                q = i.reshape(-1, 4)
                return _np.vstack([q[:, [0, 1, 2]], q[:, [0, 2, 3]]])
            out, off = [], 0
            for k in c:
                f = i[off:off + k]
                out += [(f[0], f[j], f[j + 1]) for j in range(1, k - 1)]
                off += k
            return _np.asarray(out, dtype=_np.int64)

        found, too_big, excluded = [], [], []
        try:
            for prim in self._stage.Traverse():
                mesh = UsdGeom.Mesh(prim)
                if not mesh or not _collision_enabled(prim):
                    continue
                path_str = str(prim.GetPath())
                if exclude_subs and any(s in path_str.lower() for s in exclude_subs):
                    excluded.append(path_str)
                    continue
                attr = prim.GetAttribute("physics:approximation")
                declared = (str(attr.Get() or "")
                            if (attr and attr.IsValid() and attr.HasAuthoredValue())
                            else "<absent>")
                points = mesh.GetPointsAttr().Get()
                n_verts = len(points) if points else 0
                if n_verts < 4:
                    continue
                if n_verts > max_verts:
                    too_big.append((str(prim.GetPath()), n_verts))
                    continue
                sol, void = mesh_solidity(
                    points,
                    _triangles(mesh.GetFaceVertexCountsAttr().Get(),
                               mesh.GetFaceVertexIndicesAttr().Get()))
                if sol > max_sol or void < min_void:
                    continue
                if attr and attr.IsValid() and attr.HasAuthoredValue():
                    attr.Block()
                self._scene_decomp_paths.add(str(prim.GetPath()))
                found.append((str(prim.GetPath()), declared, n_verts, sol))
        except Exception as exc:  # noqa: BLE001
            self._logger.warn(
                f"[newton-standalone] scene-collider scan failed mid-walk: "
                f"{exc!r}"
            )
            self._scene_decomp_paths = set()
            return

        if found:
            sample = ", ".join(
                f"{'/'.join(p.split('/')[-3:])}({d},solidity {s:.3f})"
                for p, d, _n, s in sorted(found, key=lambda r: r[3])[:5])
            self._logger.info(
                f"[newton-standalone] {len(found)} concave collider(s) taken "
                f"over from Newton (solidity ≤ {max_sol}): {sample}"
                f"{'…' if len(found) > 5 else ''} — each would have reached "
                f"the solver as a single convex hull"
            )
        else:
            self._logger.info(
                "[newton-standalone] scene-collider scan: every collider's "
                "convex hull fits its mesh; nothing to take over"
            )
        if too_big:
            worst = ", ".join(
                f"{'/'.join(p.split('/')[-3:])}({n}v)" for p, n in too_big[:4])
            self._logger.warn(
                f"[newton-standalone] {len(too_big)} collider(s) left to "
                f"Newton — over the {max_verts}-vertex cap: {worst}"
                f"{'…' if len(too_big) > 4 else ''}"
            )
        if excluded:
            sample = ", ".join('/'.join(p.split('/')[-3:]) for p in excluded[:4])
            self._logger.info(
                f"[newton-standalone] {len(excluded)} collider(s) skipped by "
                f"name (matched {list(exclude_subs)}), never measured for "
                f"decomposition: {sample}{'…' if len(excluded) > 4 else ''}"
            )

    def _demote_background_decor(self) -> None:
        """Session-layer override: make free-floating background props STATIC.

        The room a task loads is scenery, but some rooms ship their props as
        full dynamic rigid bodies, authored overlapping each other. One room
        ships 70 such props: most of the compiled model's startup contacts end
        up being background props penetrating each other by tens of
        millimetres. PhysX tolerates that -- it clamps de-penetration velocity
        -- while mjwarp solves it every step, so physics can run at a small
        fraction of its target rate and a reset never returns inside the
        client's timeout window. Other rooms carry far fewer such bodies, so
        this one is an outlier rather than the norm.

        Only bodies **no joint refers to** are demoted. That line matters:
        ``open_door``'s door lives in the background too, and its panel, latch,
        handle and frame are all joint-referenced, so the whole door stays
        dynamic and the task still works. Anything articulated is a mechanism
        the task may be about; anything free-floating is decor.

        A body is also spared when another body in the same asset is
        joint-referenced (same ``.../entity`` root), so a mechanism cannot be
        half-frozen.

        Collision geometry is kept -- this drops dynamics only, exactly like
        ``_demote_kinematic_bodies``. ``RLINF_BACKGROUND_STATIC=0`` turns it
        off.
        """
        if self._stage is None:
            return
        _sc = getattr(getattr(self, "_params", None), "scene_colliders", None)
        if _sc is None or not bool(getattr(_sc, "background_decor_static", False)):
            self._logger.info(
                "[newton-standalone] background decor stays dynamic "
                "(scene_colliders.background_decor_static)"
            )
            return
        try:
            from pxr import UsdPhysics  # noqa: PLC0415
        except Exception as exc:  # noqa: BLE001
            self._logger.warn(
                f"[newton-standalone] background decor demotion skipped "
                f"(pxr import failed): {exc}")
            return

        def _asset_root(path: str) -> str:
            """The prim that owns this body, so a mechanism is kept whole.

            These assets nest their parts under ``<asset>/entity/<part>``; the
            asset prim is what a joint's two ends have in common.
            """
            marker = "/entity/"
            return path.split(marker)[0] if marker in path else path.rsplit("/", 1)[0]

        try:
            jointed_roots, jointed = set(), set()
            for prim in self._stage.Traverse():
                tname = str(prim.GetTypeName())
                if not (tname.startswith("Physics") and "Joint" in tname):
                    continue
                for rel_name in ("physics:body0", "physics:body1"):
                    rel = prim.GetRelationship(rel_name)
                    if not rel:
                        continue
                    for target in rel.GetTargets():
                        jointed.add(str(target))
                        jointed_roots.add(_asset_root(str(target)))

            demoted, kept = [], 0
            for prim in self._stage.Traverse():
                path = str(prim.GetPath())
                if not path.startswith(_BACKGROUND_SCOPE):
                    continue
                if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    continue
                api = UsdPhysics.RigidBodyAPI(prim)
                enabled = api.GetRigidBodyEnabledAttr()
                if enabled and enabled.Get() is False:
                    continue
                if path in jointed or _asset_root(path) in jointed_roots:
                    kept += 1
                    continue
                api.CreateRigidBodyEnabledAttr(False)
                demoted.append(path)
        except Exception as exc:  # noqa: BLE001
            self._logger.warn(
                f"[newton-standalone] background decor demotion failed "
                f"mid-walk: {exc!r}")
            return

        if demoted:
            sample = ", ".join("/".join(p.split("/")[-3:]) for p in demoted[:4])
            self._logger.info(
                f"[newton-standalone] {len(demoted)} free-floating background "
                f"prop(s) demoted to STATIC collider(s) ({kept} left dynamic "
                f"because a joint refers to them): {sample}"
                f"{'…' if len(demoted) > 4 else ''} — scenery costs 6 DOF and "
                f"a contact set each, and these rooms author them overlapping"
            )
        elif kept:
            self._logger.info(
                f"[newton-standalone] background decor: all {kept} rigid "
                f"body/bodies are joint-driven mechanisms; none demoted")

    def _phase_open_stage_and_overrides(self) -> None:
        """Phase 1: open the USD stage and apply pre-``add_usd`` overrides.

        Edits land on the session layer so the cached ``robot.usda``
        payloads stay scene-agnostic.  Order matters: fix policies first
        (drop / weld sub-trees), then ``init_base_pose`` (translate the
        whole robot Xform), then ``RenderProduct`` disable.
        """
        self._logger.info("[newton-standalone] opening stage…")
        newton_scene = Path(self._render_layer_usda).parent / "newton_scene.usda" if self._render_layer_usda else None

        self._open_stage(newton_scene)

        # Runtime fix_base / fix_head / fix_body policies — drop or freeze
        # sub-trees from the composed stage so Newton's ``add_usd`` either
        # skips them entirely (fix_base → chassis: zero presence) or sees
        # them as Fixed welds (fix_head / fix_body: bodies stay in
        # body_label for TF, joints contribute 0 DOFs to Featherstone's M).
        # Cached robot.usda stays scene-agnostic — these are pure
        # session-layer overrides on the live composed stage.
        # Read-only: report how Newton will resolve the root weld's world
        # side BEFORE any mode-dependent mutation hides the prim.
        self._log_root_weld_diagnostics()
        self._apply_runtime_fix_policies()

        # Note the colliders whose ``physics:approximation`` Newton cannot
        # read (PhysX's ``none`` / ``sdf``) -- they would each become a
        # single convex hull.  Must run while the stage is still open;
        # the decomposition itself happens after ``add_usd``.
        self._collect_scene_colliders_to_decompose()

        # Apply ``robot.init_base_pose`` to the robot Xform on the composed
        # stage BEFORE ``add_usd`` parses it.  Authoring at session-layer
        # strength so the cached ``robot.usda`` stays untouched.  Works
        # uniformly for pin_base_to_world true/false: the welded path
        # follows the Xform; the FREE-joint path initialises base_link at
        # the Xform's world pose.
        self._apply_init_base_pose()

        # Only now can the world weld's anchor be checked against the pose
        # the stage actually composes — ``init_base_pose`` just set it.
        # No-op unless Newton would misplace the base (see the method).
        self._repair_root_weld_anchor()

        # NOTE: we deliberately do NOT add render_layer.usda to the engine's
        # stage. That layer carries Camera prims (Head_Camera, FreeCam) and
        # RenderProduct prims used by genie_sim_render_node (a separate
        # process running OVRTX, the canonical camera renderer in our
        # pipeline). The engine doesn't need them — Newton physics doesn't
        # consume cameras, and adding them only:
        #   * clutters Isaac's viewport camera dropdown
        #   * forces Hydra to walk those prims each tick
        #   * caused the "double-rendering" perf concern
        # The render_node opens its own stage view in its own process; our
        # not-loading-here is invisible to it.
        if self._render_layer_usda and Path(self._render_layer_usda).exists():
            self._logger.info(
                f"[newton-standalone] skipping render_layer.usda ({self._render_layer_usda}) — "
                f"OVRTX render_node loads it independently"
            )
        # _disable_local_render_products is now a no-op since we never loaded
        # the prims, but keep the call as a defensive cleanup if some other
        # code path adds them.
        self._disable_local_render_products()

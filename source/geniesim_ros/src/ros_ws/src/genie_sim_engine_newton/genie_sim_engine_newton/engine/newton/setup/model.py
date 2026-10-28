# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Build-phases 2-3: scene resolution + ModelBuilder + ``add_usd`` + finalize.

Owns scene-plugin / cloth / solver-preference resolution and the
ModelBuilder lifecycle through ``finalize``.  ``self._model`` is
populated by the end of this mixin's phase calls.
"""

from __future__ import annotations

import json
import math
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import warp as wp


def _generated_ground_visual_paths(stage, scene_cfg) -> list[str]:
    """Exclude the assembler's visual-only floor from Newton's model import.

    Empty scenes synthesize /World/GroundPlane for USD renderers. Newton adds
    its own visible collision plane at the same z=0, so importing both causes
    depth fighting in ViewerGL. Keep the USD untouched for Isaac and OVRtx.
    Authored worlds and physics-bearing geometry must keep their own floor.
    """
    from pxr import UsdGeom, UsdPhysics

    scene = (scene_cfg or {}).get("scene")
    if scene is not None and not (isinstance(scene, str) and not scene.strip()):
        return []
    prim = stage.GetPrimAtPath("/World/GroundPlane")
    if not prim or not prim.IsA(UsdGeom.Mesh):
        return []
    if prim.HasAPI(UsdPhysics.CollisionAPI) or prim.HasAPI(UsdPhysics.RigidBodyAPI):
        return []
    return [str(prim.GetPath())]


def _strip_disabled_physics_schemas(stage) -> None:
    """De-apply ``PhysicsRigidBodyAPI`` (and ``PhysicsMassAPI``) on prims
    whose composed ``physics:rigidBodyEnabled`` is ``False``, and clear the
    ``physics:approximation`` opinion on disabled colliders so Newton's GL
    viewer renders their actual mesh instead of the convex-decomposition
    proxy.

    Why this is necessary:

    1.  Newton's ``ModelBuilder.add_usd`` visual-loader pass
        (``newton/_src/utils/import_usd.py:_load_visual_shapes_impl``
        line 788) early-returns on any prim that has
        ``UsdPhysics.RigidBodyAPI`` applied, *regardless* of whether the
        rigid body is enabled.  Many of our scene props (e.g. the
        ``GDK/demo.usda`` shelves) compose ``RigidBodyAPI`` from the leaf
        USD and turn it off in the scene's per-instance ``over``;
        without the strip the visual is missing entirely.

    2.  When a mesh has ``PhysicsCollisionAPI`` with
        ``collisionEnabled=False`` *and* an ``physics:approximation``
        opinion (``convexDecomposition`` / ``convexHull`` / etc.), Newton
        still adds the shape to the builder (``COLLIDE_SHAPES`` flag
        cleared) and the GL viewer renders the *approximation* — which
        looks like the convex hull rather than the underlying mesh.
        Clearing the approximation token forces the viewer to render the
        mesh as authored.

    We deliberately do NOT strip ``PhysicsCollisionAPI`` itself: static
    props (like ``market_00``) that are NOT under a ``RigidBodyAPI``
    parent rely on the collision-shape pipeline to register their mesh
    as a (non-colliding) builder shape so the GL viewer renders it at
    all — Newton's visual-loader pass is only invoked from rigid-body
    parsing, so removing the schema would make those static props
    disappear from the viewport.

    The fix only touches the live stage in this process; on-disk USDs
    are unaffected, and other consumers (IsaacSim, OVRtx) read the
    canonical layers as authored.
    """
    try:
        from pxr import Sdf, UsdPhysics  # noqa: PLC0415
    except ImportError:
        return

    _RB_SCHEMAS = ("PhysicsRigidBodyAPI", "PhysicsMassAPI")
    stripped_rb = cleared_approx = 0
    for prim in stage.Traverse():
        applied = list(prim.GetAppliedSchemas())
        remove: list[str] = []

        rba = UsdPhysics.RigidBodyAPI(prim)
        if rba and rba.GetRigidBodyEnabledAttr().Get() is False:
            remove.extend(s for s in _RB_SCHEMAS if s in applied)

        col = UsdPhysics.CollisionAPI(prim)
        if col and col.GetCollisionEnabledAttr().Get() is False:
            approx_attr = prim.GetAttribute("physics:approximation")
            if approx_attr and approx_attr.HasAuthoredValue():
                approx_attr.Block()
                cleared_approx += 1

        if remove:
            cleaned = [s for s in applied if s not in remove]
            prim.SetMetadata("apiSchemas", Sdf.TokenListOp.CreateExplicit(cleaned))
            stripped_rb += sum(1 for s in remove if s in _RB_SCHEMAS)

    if stripped_rb or cleared_approx:
        print(
            f"[newton-standalone] reauthored disabled-physics opinions: "
            f"{stripped_rb} RigidBody/Mass schemas removed, "
            f"{cleared_approx} approximation tokens cleared "
            f"(restores visuals on rigidBodyEnabled=0 / collisionEnabled=0 props)",
            flush=True,
        )


def _pin_convex_decomposition_imports(stage) -> int:
    """Import every enabled ``convexDecomposition`` collider as one convex hull.

    The engine relies on Newton 1.0's import, whose CoACD threshold of 0.5
    gives one hull per such mesh (see ``stage.py``); ``_decompose_gripper_pads``
    then splits each pad hull itself. Newton 1.4 and later decompose at
    ``default_mesh_approximation_cfg.coacd_threshold`` (0.05), which turns each
    G2 fingertip pad into hundreds of pieces and the pad pass into a startup
    timeout. ``convexHull`` keeps the Newton 1.0 import on the session stage.
    Colliders already taken over by the scene-collider scan are blocked and
    skipped. Returns the number of colliders changed."""
    pinned = 0
    for prim in stage.Traverse():
        attr = prim.GetAttribute("physics:approximation")
        if not (attr and attr.HasAuthoredValue() and str(attr.Get()).lower() == "convexdecomposition"):
            continue
        enabled = prim.GetAttribute("physics:collisionEnabled")
        if enabled and enabled.IsValid() and enabled.Get() is False:
            continue
        attr.Set("convexHull")
        pinned += 1
    return pinned


def _strip_loop_spherical_joints(stage) -> list:
    """Deactivate closed-loop ``PhysicsSphericalJoint`` prims so ``add_usd``
    imports an open kinematic tree, and return what's needed to re-close each
    loop as a Newton *connect* equality after import:
    ``[(body0_path, body1_path, anchor_on_body0), ...]``.

    mjwarp-only fix: Newton's MuJoCo-Warp converter turns these loop joints
    into MuJoCo equality constraints (no qpos), but Newton's own model keeps
    counting their ball-joint coords — so the Newton↔MuJoCo qpos sizes disagree
    and ``SolverMuJoCo`` asserts (``mj qpos < joint_coord_count``). Importing
    them as connect equalities instead keeps both sides at the same coord count
    while preserving the closed loop. Featherstone keeps the ball joints (it
    handles loops natively), so callers gate this on the mjwarp adapter.

    ⚠ The selection is BROADER than the name suggests: every
    ``PhysicsSphericalJoint`` with both bodies set is taken, without testing
    whether it actually closes a loop. An open-tree ball joint would be
    converted too, and that is NOT a no-op — the joint's 3 rotational coords
    leave ``joint_q`` and position becomes a solver-enforced (soft) constraint
    rather than a hard coordinate.

    Kept this way deliberately: across the whole asset library today every
    ``PhysicsSphericalJoint`` is a gripper loop joint (``gripper_[lr]_
    {inner,outer}_joint2`` on G1_omnipicker / G1_120s), so there is no case
    where the broad rule fires spuriously, and cycle detection on the joint
    graph would be real work for no present benefit. The paths are logged
    below precisely so a future asset that breaks that assumption leaves a
    trace instead of silently losing a joint.
    """
    from pxr import UsdPhysics  # noqa: PLC0415

    loops: list = []
    stripped_paths: list = []
    for prim in stage.Traverse():
        if str(prim.GetTypeName()) != "PhysicsSphericalJoint":
            continue
        j = UsdPhysics.Joint(prim)
        b0 = [t.pathString for t in j.GetBody0Rel().GetTargets()]
        b1 = [t.pathString for t in j.GetBody1Rel().GetTargets()]
        if not b0 or not b1:
            continue
        lp = prim.GetAttribute("physics:localPos0").Get()
        anchor = (float(lp[0]), float(lp[1]), float(lp[2])) if lp is not None else (0.0, 0.0, 0.0)
        loops.append((b0[0], b1[0], anchor))
        stripped_paths.append(str(prim.GetPath()))
        prim.SetActive(False)  # excluded from add_usd's traversal
    if loops:
        print(
            f"[mjwarp-loopfix] stripped {len(loops)} SphericalJoint(s) for "
            f"re-add as connect equalities: "
            f"{', '.join(p.rsplit('/', 1)[-1] for p in stripped_paths)}",
            flush=True,
        )
    return loops


def _connect_equality_adder(builder):
    """Return ``add(body1=, body2=, anchor=, label=)`` for a CONNECT equality.

    Newton 1.0 exposes ``ModelBuilder.add_equality_constraint_connect``. Newton
    1.4 removed it; from then on its own USD/MJCF importers append the
    ``mujoco:equality_constraint`` row through ``_add_equality_constraint``."""
    legacy = getattr(builder, "add_equality_constraint_connect", None)
    if legacy is not None:
        return legacy
    from newton import EqType  # noqa: PLC0415
    from newton._src.solvers.mujoco.equality import _add_equality_constraint  # noqa: PLC0415

    def add(*, body1, body2, anchor, label):
        return _add_equality_constraint(
            builder, EqType.CONNECT, body1=body1, body2=body2, anchor=anchor, label=label
        )

    return add


def _add_loop_connect_equalities(builder, loops: list) -> int:
    """Re-close stripped gripper loops as Newton connect equalities. Returns
    the number added. A CONNECT equality pins body2 to ``anchor`` expressed in
    body1's frame — so body1 := USD body0 (anchor = its ``localPos0``),
    body2 := USD body1."""
    labels = list(getattr(builder, "body_label", []) or [])
    by_path = {p: i for i, p in enumerate(labels)}
    # leaf-name fallback (only when the short name is unambiguous)
    leaf_counts: dict = {}
    for p in labels:
        leaf_counts[p.split("/")[-1]] = leaf_counts.get(p.split("/")[-1], 0) + 1
    by_leaf = {p.split("/")[-1]: i for i, p in enumerate(labels) if leaf_counts[p.split("/")[-1]] == 1}
    add_connect = _connect_equality_adder(builder)

    def _resolve(path):
        if path in by_path:
            return by_path[path]
        return by_leaf.get(path.split("/")[-1])

    added = 0
    for b0, b1, anchor in loops:
        i0, i1 = _resolve(b0), _resolve(b1)
        if i0 is None or i1 is None:
            print(f"[mjwarp-loopfix] WARN: body not found for loop " f"{b0}->{i0} {b1}->{i1}; skipping", flush=True)
            continue
        add_connect(body1=i0, body2=i1, anchor=anchor, label=f"loop:{b1.split('/')[-1]}")
        added += 1
    if added:
        print(
            f"[mjwarp-loopfix] added {added} connect equality constraint(s) " f"to re-close gripper loops", flush=True
        )
    return added


def _gripper_pad_bodies(labels, pad_body_names=None):
    """Return ``[(body_index, leaf_name), ...]`` for the fingertip pad
    bodies among ``labels`` (a ``builder.body_label`` list).

    ``pad_body_names`` is the pad-body NAME set resolved by the
    assemble-time ``newton:isGripperPad`` annotation reader
    (``common/gripper_annotations.read_pad_body_names``) — the
    physical-truth 4-pad set (``gripper_[lr]_(inner|outer)_link4``).  This
    replaces the old hard-coded ``_PAD_LEAVES`` tuple that named those four
    bodies literally; the reader path additionally keeps
    ``arm_l/r_link4`` / ``body_link4`` out even on the fallback (its
    gripper-classifier restriction), so a re-assembled or new robot needs
    no code change here.
    When ``None`` the reader's gripper-classified ``link4`` fallback is
    resolved here (stage-free), so an un-wired caller still lands on the
    same 4 pads on G2.  Body index is the position into ``labels`` (matches
    ``builder.shape_body`` values), so the caller can iterate a body's
    shapes directly.
    """
    from genie_sim_engine_newton.common.gripper_annotations import (  # noqa: PLC0415
        pad_body_indices,
        read_pad_body_names,
    )

    labels = list(labels or [])
    if pad_body_names is None:
        pad_body_names = read_pad_body_names(stage=None, labels=labels)
    idx = pad_body_indices(labels, pad_body_names)
    return [(i, labels[i].split("/")[-1]) for i in sorted(idx)]


def _quat_xyzw_to_mat(q):
    """3x3 rotation matrix from an xyzw quaternion (numpy).

    Standalone numeric helper — the one piece of the reverted box-fitting
    collider neighborhood (``_obb_fit``/``_pad_face_slab``, not ported)
    that ``_decompose_gripper_pads`` below actually needs, to bake each
    pad shape's authored xform into its body frame before running CoACD.
    """
    import numpy as np  # noqa: PLC0415

    x, y, z, w = (float(v) for v in q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


_COACD_WORKER_SRC = r"""
import pickle
import sys

import coacd

try:
    coacd.set_log_level("error")
except Exception:
    pass

v, f, th, maxh = pickle.load(sys.stdin.buffer)
parts = coacd.run_coacd(coacd.Mesh(v, f), threshold=th, max_convex_hull=maxh)
pickle.dump(parts, sys.stdout.buffer)
"""


def _run_coacd_isolated(v, f, threshold: float, max_convex_hull: int, retries: int = 3):
    """Run ``coacd.run_coacd`` in a throwaway child process, retrying on crash.

    Rolling-specific hardening, not in the ported reference commit: CoACD's
    OpenMP-parallel search has an upstream, currently-unfixed thread-safety
    bug (SarahWeiii/CoACD PR #104 — a data race in a thread-unsafe helper
    touched during parallel execution) that intermittently SIGSEGVs on
    multi-core hosts. Verified empirically on this container (32 cores):
    roughly 1-in-8 to 1-in-18 ``run_coacd`` calls crash the whole process —
    which, in-process, would take down engine boot on ~1/3 of cold builds
    (4 pads/build). Forcing ``OMP_NUM_THREADS=1`` avoids the race but was
    measured to blow up per-pad decomposition time from under a second to
    minutes (single-threaded MCTS search), an unacceptable build-time cost.
    Isolating each call in a subprocess costs nothing on the (fast, fully
    parallel) happy path and turns a process-killing crash into a plain
    non-zero exit code the parent can retry — a fresh subprocess is an
    independent roll of the same low-probability dice, so 3 attempts drive
    the effective failure probability low enough to be negligible, and the
    existing "keep single hull" fallback (see caller) still catches the
    residual case where all retries are exhausted.
    """
    import pickle  # noqa: PLC0415
    import subprocess  # noqa: PLC0415
    import sys  # noqa: PLC0415

    payload = pickle.dumps((v, f, float(threshold), int(max_convex_hull)))
    last_err: Exception | None = None
    for attempt in range(max(1, retries)):
        try:
            proc = subprocess.run(
                [sys.executable, "-c", _COACD_WORKER_SRC],
                input=payload,
                capture_output=True,
                timeout=180,
            )
        except subprocess.TimeoutExpired as exc:
            last_err = exc
            continue
        if proc.returncode != 0:
            last_err = RuntimeError(
                f"coacd worker exited {proc.returncode} "
                f"(attempt {attempt + 1}/{retries}): "
                f"{proc.stderr[-500:]!r}"
            )
            continue
        return pickle.loads(proc.stdout)
    raise last_err or RuntimeError("coacd worker failed with no captured error")


def _decompose_gripper_pads(builder, pad_body_names=None, contact=None, logger=None) -> int:
    """Replace each fingertip's single collision hull with a CoACD convex
    DECOMPOSITION (multiple tight convex pieces on the same body).

    Why: the benchmark asset authors ``physics:approximation =
    "convexDecomposition"`` on the pad/knuckle meshes, but Newton's USD
    importer only builds ONE convex hull per mesh — and mjwarp emits a
    single contact point per convex pair.  One point per pad cannot
    transmit a contact couple: under inertial disturbance the grasped
    object rolls about the pinch axis and walks out (user-visible
    "shake the arm and the block flies out"; doc §7.10/7.14).  N tight
    pieces -> N contact points per pad -> a poor-man's manifold that
    locks rotation, with NO shape inflation (unlike the reverted box
    approach: decomposition hugs the true surface, so no table float
    and no visual gaps).

    ``pad_body_names`` is the pad-body NAME set from the assemble-time
    ``newton:isGripperPad`` annotation reader (physical-truth 4 pads); the
    caller in ``setup/model.py`` — which holds the live USD stage — resolves
    it and passes it in.  ``None`` -> the reader's gripper-classified
    ``link4`` fallback (stage-free) is used, so an un-wired caller still
    lands on the same 4 pads on G2 (replacing the old ``_PAD_LEAVES``
    literal that hard-coded the four names).

    Gated by ``gripper_contact.pad_decomposition.enabled`` in the physics
    profile, and by nothing else — the ``RLINF_PAD_DECOMP`` env gate this
    docstring used to name was removed on 2026-08-19 (see
    ``physics_knobs.md``, removed-knobs table).  Shipped value is off.
    Results are cached by content hash under /tmp/rlinf_pad_decomp.
    Returns the number of pads decomposed."""
    import hashlib  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    import warp as wp  # noqa: PLC0415
    from newton import GeoType, Mesh, ShapeFlags  # noqa: PLC0415

    # Value comes from ``gripper_contact.pad_decomposition`` in the profile
    # file and from nowhere else — there is no env back door; see
    # ``config/physics_params.yaml``. The shipped value is "off";
    # ``contact=None`` (which only happens in tests that call this
    # module-level function directly) is treated as off too.
    if not (bool(getattr(contact, "pad_decomposition", False)) if contact else False):
        print("[mjwarp-paddecomp] disabled " "(gripper_contact.pad_decomposition)", flush=True)
        return 0

    labels = list(getattr(builder, "body_label", []) or [])
    pad_bodies = _gripper_pad_bodies(labels, pad_body_names)
    mesh_types = {int(GeoType.MESH), int(GeoType.CONVEX_MESH)}
    cache_dir = "/tmp/rlinf_pad_decomp"
    os.makedirs(cache_dir, exist_ok=True)
    # CoACD params: fingertip meshes are ~4-6 cm; threshold 0.03 with a
    # small piece cap yields 3-8 pieces per pad — enough for a manifold
    # without exploding the contact count.
    _TH = float(getattr(contact, "pad_decomposition_threshold", 0.01)) if contact else 0.01
    _MAXH = int(getattr(contact, "pad_decomposition_max_hulls", 16)) if contact else 16

    def _decompose(v, f, tag=""):
        key = hashlib.sha1(v.tobytes() + f.tobytes() + f"{_TH}/{_MAXH}".encode()).hexdigest()
        cpath = os.path.join(cache_dir, key + ".npz")
        if os.path.exists(cpath):
            z = np.load(cpath, allow_pickle=True)
            return list(zip(z["verts"], z["faces"]))
        parts = _run_coacd_isolated(v, f, _TH, _MAXH)
        parts = [(np.asarray(pv, dtype=np.float64), np.asarray(pf, dtype=np.int32)) for pv, pf in parts]
        # keep the INPUT too — offline tuning + "was the input already a
        # convex hull?" debugging (doc §7.15)
        np.savez(
            cpath,
            in_verts=v,
            in_faces=f,
            tag=tag,
            verts=np.array([p[0] for p in parts], dtype=object),
            faces=np.array([p[1] for p in parts], dtype=object),
        )
        return parts

    cfg = builder.default_shape_cfg.copy()
    cfg.density = 0.0  # donors keep contributing mass; pieces must not
    done = 0
    for body, leaf in pad_bodies:
        n_parts = 0
        for s in range(len(builder.shape_body)):
            if (
                builder.shape_body[s] != body
                or not (builder.shape_flags[s] & int(ShapeFlags.COLLIDE_SHAPES))
                or int(builder.shape_type[s]) not in mesh_types
            ):
                continue
            src = builder.shape_source[s]
            v = np.asarray(getattr(src, "vertices", None), dtype=np.float64)
            f = np.asarray(getattr(src, "indices", None), dtype=np.int32)
            if v is None or f is None or not len(v):
                continue
            f = f.reshape(-1, 3)
            v = v * np.asarray(builder.shape_scale[s], dtype=np.float64)
            xf = builder.shape_transform[s]
            R = _quat_xyzw_to_mat([xf.q[0], xf.q[1], xf.q[2], xf.q[3]])
            t = np.array([xf.p[0], xf.p[1], xf.p[2]], dtype=np.float64)
            v = v @ R.T + t  # bake scale+xform into body frame
            try:
                parts = _decompose(v, f, tag=leaf)
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[mjwarp-paddecomp] WARN: {leaf} decomposition failed " f"({exc!r}); keeping single hull",
                    flush=True,
                )
                continue
            builder.shape_flags[s] &= ~int(ShapeFlags.COLLIDE_SHAPES)
            for i, (pv, pf) in enumerate(parts):
                builder.add_shape_mesh(
                    body=body,
                    xform=wp.transform(),
                    mesh=Mesh(pv, pf.flatten()),
                    cfg=cfg,
                    label=f"{leaf}_decomp{i}",
                )
                n_parts += 1
        if n_parts:
            done += 1
            print(f"[mjwarp-paddecomp] {leaf}: hull -> {n_parts} convex piece(s)", flush=True)
    if done:
        print(
            f"[mjwarp-paddecomp] {done} fingertip collider(s) decomposed "
            f"(multi-point pinch manifold, benchmark parity)",
            flush=True,
        )
    return done


def mesh_solidity(points, faces):
    """``(solidity, void_volume)``; returns ``(0.0, 0.0)`` when it can't be
    computed.

    ``solidity`` = mesh volume / convex-hull volume (how far off a single
    convex hull would be); ``void_volume`` = hull volume - mesh volume (how
    much empty space the hull encloses) -- the latter is what actually
    decides whether a collider can swallow another object. See
    ``stage.py``'s ``_SCENE_DECOMP_MAX_SOLIDITY``/``_SCENE_DECOMP_MIN_VOID_VOLUME``
    for the calibrated gate values.

    A solidity of 0.0 pushes the caller toward decomposing (the
    conservative choice for a collider we can't vouch for); the
    accompanying void_volume of 0.0 then holds that back, since a collider
    whose volume can't even be computed isn't something to reshape.

    Shared by ``stage.py`` (reading the USD mesh before ``add_usd``) and
    this module (reading the mesh inside the builder after ``add_usd``) so
    the two call sites can't drift apart.
    """
    from scipy.spatial import ConvexHull  # noqa: PLC0415

    try:
        V = np.asarray(points, dtype=float)
        F = np.asarray(faces)
        if len(F) < 4:
            return 0.0, 0.0
        a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        mesh_vol = abs(float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0))
        # The convex hull is decided only by the shell vertices, so a
        # fixed-seed subsample barely changes the result, while a
        # room-scale mesh can have six-figure point counts -- running qhull
        # on the full set can add up to tens of seconds across one scene.
        H = V
        if len(H) > 20000:
            H = H[np.random.default_rng(0).choice(len(H), 20000, replace=False)]
        hull_vol = float(ConvexHull(H).volume)
    except Exception:  # noqa: BLE001
        return 0.0, 0.0
    if hull_vol <= 0.0:
        return 0.0, 0.0
    return mesh_vol / hull_vol, max(hull_vol - mesh_vol, 0.0)


# --- Voxel box cover: a second backend for shell-like colliders -----------
# CoACD's pieces are an outer approximation and always eat into the rim
# around a hole; tuning its concavity threshold down helps but has a
# floor -- pushed as far as CoACD stays stable, a round hole still came
# back measurably narrower than its true diameter. That is a limit of
# convex decomposition as a method, not a tuning miss.
#
# This backend instead voxelizes the mesh and greedily covers the solid
# part with axis-aligned boxes. Empty cells can't grow anything, so holes
# are preserved by construction, and boxes are also MuJoCo's cheapest
# collision primitive, at a small (~1%) step-time cost.
#
# It only suits shells whose walls are thicker than the voxel and whose
# face count isn't too high, so four gates automatically fall back to
# CoACD for unsuitable objects:
#   * a cap on voxel count -- a fine pitch over a large object can need
#     tens of millions of cells;
#   * a cap on total work (voxel cells x face count), since the
#     ray-casting check used to voxelize scales with that product;
#   * a cap on box count, since too many boxes would blow up the
#     contact-pair count;
#   * voxel volume must match mesh volume within a ratio, gated on both
#     sides -- too much means the ray-casting check judged a degenerate,
#     non-watertight mesh (e.g. a thin ring) solid as a whole; too little
#     means the wall is thinner than the voxel pitch and the cover leaks.
_BOXCOVER_PITCH = 0.004  # m, voxel edge length
# trimesh.contains is a ray-casting check whose cost scales with query
# points x face count, not point count alone (measured roughly 70
# bytes/pair, 1.4e7 pairs/second). A cap on point count alone is not
# enough: a high-face-count mesh can still blow up memory or time. Two
# guards cover this: a cap on total work below (bounds time), and the
# face-count-adaptive chunk size in _contains_chunk (bounds memory).
_BOXCOVER_MAX_CELLS = 500_000
# Caps voxel cells x face count, bounding time: at roughly 1.4e7
# pairs/second, 300M pairs takes about 21 seconds. Meshes whose work
# exceeds this fall back to CoACD.
_BOXCOVER_MAX_WORK = 300_000_000
# The cap on how many points to feed ``contains`` at once, plus the two
# constants used to convert that into memory. After chunking, peak memory
# only depends on "this chunk's point count x face count", not on the total
# cell count.
_BOXCOVER_CHUNK = 100_000
_BOXCOVER_CHUNK_BYTES = 500_000_000  # peak memory budget allowed per chunk
_BOXCOVER_BYTES_PER_PAIR = 70  # measured 69-71


def _contains_chunk(n_faces):
    """How many points can be fed to ``trimesh.contains`` at once without
    blowing up memory."""
    if n_faces <= 0:
        return _BOXCOVER_CHUNK
    fit = int(_BOXCOVER_CHUNK_BYTES / (_BOXCOVER_BYTES_PER_PAIR * n_faces))
    return max(512, min(_BOXCOVER_CHUNK, fit))


_BOXCOVER_MAX_BOXES = 400
_BOXCOVER_MAX_VOLUME_RATIO = 1.5  # voxel volume / mesh volume


def _greedy_box_cover(occ, max_boxes=_BOXCOVER_MAX_BOXES):
    """Greedily cover a 3D bool voxel array with axis-aligned boxes, returning
    ``[(lo_idx, hi_idx)]``.

    Each step takes one not-yet-covered voxel and grows it as far as
    possible along x, then y, then z, as long as the whole block stays
    solid. Greedy doesn't guarantee an optimal cover, but it's good enough
    for shell-like shapes, and it's deterministic.
    """
    import numpy as np  # noqa: PLC0415

    rem = occ.copy()
    n = occ.shape
    out = []
    while rem.any():
        i, j, k = np.argwhere(rem)[0]
        i2, j2, k2 = i + 1, j + 1, k + 1
        while i2 < n[0] and rem[i2, j:j2, k:k2].all():
            i2 += 1
        while j2 < n[1] and rem[i:i2, j2, k:k2].all():
            j2 += 1
        while k2 < n[2] and rem[i:i2, j:j2, k2].all():
            k2 += 1
        rem[i:i2, j:j2, k:k2] = False
        out.append(((int(i), int(j), int(k)), (int(i2), int(j2), int(k2))))
        if len(out) > max_boxes:
            return out  # already over budget; the caller will discard this
    return out


def _voxel_grid(v, f, pitch, max_cells, max_ratio=_BOXCOVER_MAX_VOLUME_RATIO):
    """``(solid bool 3D array, per-axis coordinates)``; returns
    ``(None, reason)`` when it can't compute or the result isn't trustworthy.

    Voxelization uses ``trimesh.contains`` (is the voxel center inside the
    mesh), fed in **chunks**: measured at 3.2 GB peak memory per million
    points, so feeding it all at once would OOM the engine.
    """
    import numpy as np  # noqa: PLC0415

    try:
        import trimesh  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        return None, f"trimesh is not available ({exc!r})"

    lo, hi = v.min(0), v.max(0)
    axes = [np.arange(lo[i] + pitch / 2, hi[i], pitch) for i in range(3)]
    n = [len(a) for a in axes]
    if min(n) < 2:
        return None, "voxel is coarser than the object"
    cells = int(np.prod(n))
    if cells > max_cells:
        return None, f"voxel count {cells} exceeds {max_cells}"
    work = cells * len(f)
    if work > _BOXCOVER_MAX_WORK:
        return None, (
            f"voxels {cells} x faces {len(f)} = {work:.1e} pairs, exceeds "
            f"{_BOXCOVER_MAX_WORK:.0e} (ray casting would need roughly {work / 1.4e7:.0f}s)"
        )

    mesh = trimesh.Trimesh(v, f, process=False)
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    flat = np.empty(len(grid), dtype=bool)
    chunk = _contains_chunk(len(f))
    for start in range(0, len(grid), chunk):
        stop = start + chunk
        flat[start:stop] = mesh.contains(grid[start:stop])
    occ = flat.reshape(n)
    if not occ.any():
        return None, "voxelization found no solid cells at all"

    vox_vol = float(occ.sum()) * pitch**3
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    mesh_vol = abs(float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0))
    # Gated on both sides. Too much volume match means the ray-casting
    # check judged a degenerate, non-watertight mesh solid as a whole
    # (e.g. a thin ring judged solid end to end). Too little means the
    # wall is thinner than the voxel pitch and the voxel model leaks (a
    # 2mm wall against a 4mm voxel pitch was measured recovering only
    # about 40% of the material). Either case falls back to CoACD.
    ratio = vox_vol / mesh_vol if mesh_vol > 0.0 else 0.0
    if not (1.0 / max_ratio <= ratio <= max_ratio):
        return None, (
            f"voxel volume {vox_vol * 1e6:.0f}cm3 does not match mesh volume "
            f"{mesh_vol * 1e6:.0f}cm3 (ratio {ratio:.2f})"
        )
    return (occ, axes), ""


def _cover_to_boxes(cover, axes, pitch):
    """Voxel index ranges -> ``[(center, half_extent)]`` (world coordinates,
    in meters)."""
    out = []
    for (i, j, k), (i2, j2, k2) in cover:
        centre = [float((axes[t][[i, j, k][t]] + axes[t][[i2, j2, k2][t] - 1]) / 2.0) for t in range(3)]
        half = [float(([i2, j2, k2][t] - [i, j, k][t]) * pitch / 2.0) for t in range(3)]
        out.append((centre, half))
    return out


def _box_cover(v, f, pitch, max_cells, max_boxes, max_ratio):
    """``[(center, half_extent)]``; returns ``(None, reason)`` when this
    path isn't a good fit.

    Voxelization is **not bit-for-bit deterministic** on a non-watertight
    mesh (box count varies run to run), but the variance falls on
    interior voxels only -- outer hole/boundary dimensions were measured
    stable across runs. Combined with the caller's disk cache, the first
    build's result is reused afterward, so it stays stable within a given
    deployment.
    """
    grid, why = _voxel_grid(v, f, pitch, max_cells, max_ratio)
    if grid is None:
        return None, why
    occ, axes = grid
    cover = _greedy_box_cover(occ, max_boxes)
    if len(cover) > max_boxes:
        return None, f"needs {len(cover)} boxes, exceeds {max_boxes}"

    return _cover_to_boxes(cover, axes, pitch), ""


def _decompose_scene_colliders(builder, paths, logger=None, scene=None) -> int:
    """Convex-decompose the scene colliders the USD asked to keep exact.

    ``_collect_scene_colliders_to_decompose`` collects the prim paths
    whose ``physics:approximation`` is PhysX-only (``none`` or ``sdf``),
    which Newton's importer cannot read, so each arrives here as a single
    convex hull. ``builder.shape_label`` is the prim path, which is what
    makes the hand-off possible; it is also what MuJoCo ends up naming the
    geom.

    This does not just rewrite the USD attribute to ``convexDecomposition``
    and let the importer do it, because Newton hard-codes CoACD's
    concavity threshold at 0.5, which is too coarse for some colliders:
    on one measured door-frame mesh, 0.5 returns a single piece that
    leaves a doorway sealed, while a finer threshold opens it into
    multiple pieces in well under half a second. This function picks the
    threshold instead.

    Runs through ``_run_coacd_isolated`` for the same reason the gripper
    pads do: CoACD crashes the whole process often enough to matter, and
    it asserts outright on zero-area triangles, which some meshes contain.
    Degenerate triangles are dropped first; anything that still fails
    keeps its single hull.
    """
    import hashlib  # noqa: PLC0415

    from newton import GeoType, Mesh, ShapeFlags  # noqa: PLC0415

    if not paths:
        return 0

    def _say(msg, warn=False):
        if logger is not None:
            (logger.warn if warn else logger.info)(msg)
        else:
            print(msg, flush=True)

    # Finer than the gripper pads' threshold (rooms are metre-scale, and a
    # finer threshold trades a few extra pieces for fidelity margin on
    # colliders that have not been individually tuned).
    _TH = float(getattr(scene, "decomposition_threshold", 0.05)) if scene else 0.05
    _MAXH = int(getattr(scene, "decomposition_max_hulls", 32)) if scene else 32
    # Same gate the earlier pass used to pick these paths (see
    # _SCENE_DECOMP_MAX_SOLIDITY there). Kept here too because decomposing
    # a collider whose hull already fits it is not neutral: CoACD remeshes,
    # and its pieces come back inflated -- a floor slab was measured coming
    # back noticeably thicker, with its top surface and an adjoining door
    # frame pushed out of alignment.
    _MAXSOL = float(getattr(scene, "decomposition_max_solidity", 0.5)) if scene else 0.5
    # Defaults ON here: this function only runs on colliders already
    # identified as shell-like, and for those the box cover is the fix
    # (see the module comment above). A profile that explicitly sets
    # box_cover: false still turns it off.
    _BOXCOVER = bool(getattr(scene, "box_cover", True)) if scene else True
    # pitch / max_boxes carry calibration that belongs with the algorithm, so a
    # profile leaving them at 0 means "use the measured constants".
    _BC_PITCH = float(getattr(scene, "box_cover_pitch", 0.0) or 0.0) if scene else 0.0
    _BC_PITCH = _BC_PITCH if _BC_PITCH > 0.0 else _BOXCOVER_PITCH
    _BC_MAXBOX = int(getattr(scene, "box_cover_max_boxes", 0) or 0) if scene else 0
    _BC_MAXBOX = _BC_MAXBOX if _BC_MAXBOX > 0 else _BOXCOVER_MAX_BOXES

    mesh_types = {int(GeoType.MESH), int(GeoType.CONVEX_MESH)}
    cfg = builder.default_shape_cfg.copy()
    cfg.density = 0.0  # the donor shape keeps its mass; the pieces must not
    labels = list(getattr(builder, "shape_label", []) or [])
    # Same disk cache the pad path uses, for the same reason: CoACD's merge
    # pass can cost over ten seconds on a room-scale mesh, and a task
    # rebuilds its model on every reload. Keyed on the mesh bytes AND the
    # params, so a knob change misses the cache rather than serving a
    # stale shape.
    cache_dir = "/tmp/rlinf_scene_decomp"
    os.makedirs(cache_dir, exist_ok=True)
    memo: dict = {}
    bmemo: dict = {}
    done = pieces = cached = boxed = 0
    box_skips: dict = {}

    def _skip_box(reason, label=""):
        box_skips.setdefault(reason, []).append(label)

    # Every path out of the loop is counted and reported below: a path
    # that quietly never reaches a terminal branch would leave a collider
    # as a solid lump with nothing in the logs to explain why.
    skips: dict = {}

    def _skip(reason, label=""):
        skips.setdefault(reason, []).append(label)

    def _apply_boxes(s, label, boxes):
        """Install a box cover onto the builder: the original mesh stops
        participating in collision, replaced by this pile of boxes."""
        nonlocal done, pieces, boxed
        builder.shape_flags[s] &= ~int(ShapeFlags.COLLIDE_SHAPES)
        for i, (centre, half) in enumerate(boxes):
            builder.add_shape_box(
                body=builder.shape_body[s],
                xform=wp.transform(wp.vec3(*[float(c) for c in centre]), wp.quat_identity()),
                hx=float(half[0]),
                hy=float(half[1]),
                hz=float(half[2]),
                cfg=cfg,
                label=f"{label}_box{i}",
            )
        done += 1
        pieces += len(boxes)
        boxed += 1

    seen_labels = set()
    for s in range(len(builder.shape_body)):
        label = labels[s] if s < len(labels) else ""
        if label not in paths:
            continue
        seen_labels.add(label)
        if not (builder.shape_flags[s] & int(ShapeFlags.COLLIDE_SHAPES)):
            _skip("collision disabled", label)
            continue
        if int(builder.shape_type[s]) not in mesh_types:
            _skip("not a mesh shape", label)
            continue
        src = builder.shape_source[s]
        v = getattr(src, "vertices", None)
        f = getattr(src, "indices", None)
        if v is None or f is None or not len(v):
            _skip("shape carries no mesh data", label)
            continue
        v = np.asarray(v, dtype=np.float64) * np.asarray(builder.shape_scale[s], dtype=np.float64)
        f = np.asarray(f, dtype=np.int32).reshape(-1, 3)
        xf = builder.shape_transform[s]
        R = _quat_xyzw_to_mat([xf.q[0], xf.q[1], xf.q[2], xf.q[3]])
        v = v @ R.T + np.array([xf.p[0], xf.p[1], xf.p[2]], dtype=np.float64)
        # CoACD aborts the process on zero-area triangles rather than skipping
        # them, and retrying an isolated child just aborts again.
        e0 = v[f[:, 1]] - v[f[:, 0]]
        e1 = v[f[:, 2]] - v[f[:, 0]]
        f = f[np.linalg.norm(np.cross(e0, e1), axis=1) > 2e-12]
        if len(f) < 4:
            _skip("mesh is degenerate", label)
            continue
        if mesh_solidity(v, f)[0] > _MAXSOL:
            _skip(f"hull already fits the mesh (solidity > {_MAXSOL})", label)
            continue
        # The key must carry everything that can change the result,
        # including the gates that are not exposed as env vars (voxel
        # budget, volume fidelity): otherwise a rule change could still
        # serve a stale cached answer computed under the old rule.
        key = hashlib.sha1(
            v.tobytes() + f.tobytes() + f"{_TH}/{_MAXH}/box{int(_BOXCOVER)}:{_BC_PITCH}:{_BC_MAXBOX}"
            f":{_BOXCOVER_MAX_CELLS}:{_BOXCOVER_MAX_VOLUME_RATIO}"
            f":{_BOXCOVER_MAX_WORK}".encode()
        ).hexdigest()
        cpath = os.path.join(cache_dir, key + ".npz")
        # Something already judged "use the box cover" last time: reuse the
        # result verbatim. Voxelization itself takes a few seconds, and a
        # task rebuilds its model on every reload.
        bpath = os.path.join(cache_dir, key + ".box.npz")
        nopath = os.path.join(cache_dir, key + ".nobox")
        if _BOXCOVER and (key in bmemo or os.path.exists(bpath)):
            if key not in bmemo:
                z = np.load(bpath)
                bmemo[key] = list(zip(z["centre"], z["half"]))
                cached += 1
            _apply_boxes(s, label, bmemo[key])
            continue
        # Box cover is tried before CoACD (see the module comment above for
        # why CoACD alone loses holes) and falls back to CoACD the moment
        # any of the four gates (voxel budget, total work, volume
        # fidelity, box-count budget) fails. The order can't be reversed:
        # once CoACD has run, the holes are already eaten into, and boxing
        # afterward can't restore them.
        if _BOXCOVER and not os.path.exists(nopath):
            boxes, why = _box_cover(v, f, _BC_PITCH, _BOXCOVER_MAX_CELLS, _BC_MAXBOX, _BOXCOVER_MAX_VOLUME_RATIO)
            if boxes:
                np.savez(bpath, centre=np.array([b[0] for b in boxes]), half=np.array([b[1] for b in boxes]))
                bmemo[key] = boxes
                _apply_boxes(s, label, boxes)
                _say(f"[scene-decomp] {label}: hull -> {len(boxes)} box(es) " f"(voxel {_BC_PITCH * 1000:.0f}mm)")
                continue
            # Record the verdict once it's made: voxelization costs seconds,
            # and the model rebuilds on every reload.
            with open(nopath, "w") as fh:
                fh.write(why)
            _skip_box(why, label)

        if key in memo:
            parts = memo[key]
        elif os.path.exists(cpath):
            z = np.load(cpath, allow_pickle=True)
            parts = list(zip(z["verts"], z["faces"]))
            memo[key] = parts
            cached += 1
        else:
            try:
                parts = _run_coacd_isolated(v, f, _TH, _MAXH)
            except Exception as exc:  # noqa: BLE001
                _say(f"[scene-decomp] WARN: {label} decomposition failed " f"({exc!r}); keeping single hull", warn=True)
                _skip("CoACD failed", label)
                continue
            parts = [(np.asarray(pv, dtype=np.float64), np.asarray(pf, dtype=np.int32)) for pv, pf in parts]
            np.savez(
                cpath,
                verts=np.array([p[0] for p in parts], dtype=object),
                faces=np.array([p[1] for p in parts], dtype=object),
            )
            memo[key] = parts
        if len(parts) < 2:
            # One piece back means CoACD judged the mesh convex enough. The
            # single hull we already have is that same answer -- swapping it
            # for CoACD's remesh would only move vertices around.
            _skip("CoACD returned a single piece", label)
            continue
        builder.shape_flags[s] &= ~int(ShapeFlags.COLLIDE_SHAPES)
        for i, (pv, pf) in enumerate(parts):
            builder.add_shape_mesh(
                body=builder.shape_body[s],
                xform=wp.transform(),
                mesh=Mesh(pv, pf.flatten()),
                cfg=cfg,
                label=f"{labels[s]}_decomp{i}",
            )
        done += 1
        pieces += len(parts)
        _say(f"[scene-decomp] {labels[s]}: hull -> {len(parts)} convex piece(s)")

    missing = set(paths) - seen_labels
    if missing:
        skips["no builder shape carries this path"] = sorted(missing)
    for reason, who in sorted(box_skips.items()):
        sample = ", ".join("/".join(w.split("/")[-3:]) for w in who[:3] if w)
        _say(
            f"[scene-decomp] {len(who)} collider(s) fell back to CoACD — " f"{reason}{': ' + sample if sample else ''}"
        )
    for reason, who in sorted(skips.items()):
        sample = ", ".join("/".join(w.split("/")[-4:]) for w in who[:4] if w)
        _say(
            f"[scene-decomp] {len(who)} collider(s) NOT decomposed — "
            f"{reason}{': ' + sample if sample else ''}"
            f"{'…' if len(who) > 4 else ''}"
        )
    if done:
        _say(
            f"[scene-decomp] {done} scene collider(s) -> {pieces} piece(s) "
            f"({boxed} via box cover, threshold {_TH}, {cached} from cache); "
            f"Newton would "
            f"have used one hull each, which seals doorways and fills the "
            f"inside of carts and shelves"
        )
    return done


class _ModelMixin:
    # Post-finalize base-pose readback tolerances.  Loose enough that float32
    # round-trips through USD double -> wp.transform float32 never trip them,
    # tight enough that any real placement divergence does: the failure this
    # catches is a whole baked offset (centimetres) or a discarded yaw, not
    # a last-bit difference.
    _BASE_POSE_POS_TOL_M = 1.0e-3
    _BASE_POSE_ROT_TOL_DEG = 0.1

    def _assert_base_pose_readback(self) -> None:
        """Check the *post-FK* base pose against what the stage asked for.

        Newton's USD importer can silently discard the authored base pose.
        A root ``FixedJoint`` whose world side it cannot resolve to a rigid
        body takes a guess, and both guess branches can anchor the weld
        somewhere other than the composed stage pose — see
        ``_log_root_weld_diagnostics``, which reports *which* branch fires.
        This reports what actually came out.

        Nothing downstream notices such a divergence today: the robot is
        simply somewhere else, and every joint reading, camera pose and
        contact is consistently wrong by the same rigid offset.

        **Why this runs FK first.**  ``model.body_q`` straight out of
        ``finalize`` is the pose the *importer read off the stage*, so it
        matches the stage by construction and tells us nothing — measured on
        the real scene, all three branches agree there to 0.000 mm.  The
        divergence lives in the weld's world anchor ``joint_X_p``, and only
        forward kinematics materialises it, by snapping the welded base onto
        that anchor.  So the check evaluates FK into a throwaway state and
        compares *that*.  Measured on the real scene with the offset baked
        into ``localPos0`` (``z`` 0.04 -> anchor 0.08, a 40 mm error) and with
        ``body0`` empty (anchor = ``localPos0``, losing ``x``/``y``
        entirely), both of which ``body_q`` at finalize reports as clean.

        The comparison target is the *composed stage pose*, not the raw
        ``init_base_pose`` yaml: the base link may carry a local offset under
        ``/<robot_prefix>``, in which case the two differ legitimately and
        the stage pose is the correct expectation.  All of requested, stage,
        import-time and post-FK are logged, so a gap between any pair is
        visible.

        Read-only and non-fatal — the throwaway state is discarded, nothing
        the engine goes on to use is touched.  This makes a silent failure
        loud, it does not gate the build.
        """
        try:
            if self._model is None or getattr(self, "_stage", None) is None:
                return
            import newton  # noqa: PLC0415
            from pxr import Gf, Sdf, Usd, UsdGeom  # noqa: PLC0415

            from genie_sim_engine_newton.engine.newton.setup.stage import (  # noqa: PLC0415
                topmost_rigid_body_paths,
            )

            # num_envs>1 finalizes a replicated builder whose body labels no
            # longer correspond 1:1 to stage prims; the single-env template
            # is what this check is about.
            num_envs = int(getattr(self, "_num_envs", 1) or 1)
            if num_envs > 1:
                self._logger.info(
                    "[newton-standalone] base-pose readback: skipped " f"(num_envs={num_envs}, replicated labels)"
                )
                return

            body_q = getattr(self._model, "body_q", None)
            labels = list(getattr(self._model, "body_label", []) or [])
            if body_q is None or not labels:
                return
            bq_import = body_q.numpy()

            # The actual measurement: FK into a throwaway state.  Without
            # this the check compares the import against itself.  If FK is
            # unavailable for any reason, fall back to the import pose and
            # say so rather than reporting a clean result we did not measure.
            bq = bq_import
            fk_ran = False
            try:
                probe = self._model.state()
                newton.eval_fk(self._model, self._model.joint_q, self._model.joint_qd, probe)
                bq = probe.body_q.numpy()
                fk_ran = True
            except Exception as exc:  # noqa: BLE001
                self._logger.warn(
                    f"[newton-standalone] base-pose readback: eval_fk failed ({exc!r}); "
                    "falling back to the import-time pose, which cannot see a "
                    "bad weld anchor."
                )

            robot_root = Sdf.Path(f"/{getattr(self, '_robot_prefix_str', '') or ''}")
            root_links = topmost_rigid_body_paths(self._stage, robot_root)
            if not root_links:
                self._logger.warn(
                    "[newton-standalone] base-pose readback: no topmost rigid "
                    f"body under {robot_root}; cannot verify the base pose."
                )
                return

            # Same resolution strategy as ``_add_loop_connect_equalities``:
            # exact prim path, then an unambiguous leaf name.
            by_path = {p: i for i, p in enumerate(labels)}
            leaf_counts: Dict[str, int] = {}
            for p in labels:
                leaf = p.split("/")[-1]
                leaf_counts[leaf] = leaf_counts.get(leaf, 0) + 1
            by_leaf = {p.split("/")[-1]: i for i, p in enumerate(labels) if leaf_counts[p.split("/")[-1]] == 1}

            # ``add_usd`` rotates the whole stage when the stage's up axis
            # differs from the builder's, so ``body_q`` is NOT necessarily in
            # the stage's frame.  Both are "Z" on every scene we ship (the
            # ModelBuilder default and ``upAxis = "Z"`` in the scene layer),
            # and then the comparison below is apples-to-apples.  When they
            # differ, the rotation Newton applied is ``quat_between_axes`` --
            # private, and its module path already moved between 1.4 and 1.6,
            # so rather than reimplement a convention we cannot pin, fall back
            # to the distance from the origin, which that rotation preserves.
            # That still catches a baked translation offset, which is the
            # failure mode this check exists for.
            stage_up = str(UsdGeom.GetStageUpAxis(self._stage)).strip().upper()
            model_up = str(getattr(self._model, "up_axis", stage_up)).strip().upper()
            axes_agree = stage_up == model_up
            if not axes_agree:
                self._logger.info(
                    "[newton-standalone] base-pose readback: stage up axis "
                    f"{stage_up!r} != model up axis {model_up!r}; add_usd rotated "
                    "the stage, so comparing distance from origin only."
                )

            base_pose = ((self._scene_cfg or {}).get("robot") or {}).get("init_base_pose")
            requested = (
                "x={} y={} z={} theta={}".format(
                    base_pose.get("x", 0.0),
                    base_pose.get("y", 0.0),
                    base_pose.get("z", 0.0),
                    base_pose.get("theta", 0.0),
                )
                if isinstance(base_pose, dict)
                else "<absent — cached robot.usda pose>"
            )

            for link_path in root_links:
                idx = by_path.get(link_path, by_leaf.get(link_path.split("/")[-1]))
                if idx is None:
                    self._logger.warn(
                        "[newton-standalone] base-pose readback: "
                        f"{link_path} is not in model.body_label "
                        f"({len(labels)} bodies); cannot verify its pose."
                    )
                    continue
                if idx >= len(bq):
                    continue

                # wp.transform layout is (px, py, pz, qx, qy, qz, qw).
                got_p = (float(bq[idx][0]), float(bq[idx][1]), float(bq[idx][2]))
                got_q = (float(bq[idx][3]), float(bq[idx][4]), float(bq[idx][5]), float(bq[idx][6]))
                imp_p = (float(bq_import[idx][0]), float(bq_import[idx][1]), float(bq_import[idx][2]))

                prim = self._stage.GetPrimAtPath(link_path)
                xformable = UsdGeom.Xformable(prim) if prim and prim.IsValid() else None
                if not xformable:
                    continue
                m = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                # Gf.Transform decomposes, so a scaled matrix still yields a
                # unit rotation quaternion.
                xf = Gf.Transform(m)
                t = xf.GetTranslation()
                gq = xf.GetRotation().GetQuat()
                im = gq.GetImaginary()
                want_p = (float(t[0]), float(t[1]), float(t[2]))
                want_q = (float(im[0]), float(im[1]), float(im[2]), float(gq.GetReal()))

                if axes_agree:
                    d_pos = math.sqrt(sum((a - b) ** 2 for a, b in zip(got_p, want_p)))
                    # Quaternion geodesic angle; abs() folds the q/-q double cover.
                    dot = min(1.0, abs(sum(a * b for a, b in zip(got_q, want_q))))
                    d_deg = math.degrees(2.0 * math.acos(dot))
                    delta = f"delta pos={d_pos * 1000.0:.3f} mm  rot={d_deg:.4f} deg"
                else:
                    d_pos = abs(math.sqrt(sum(v * v for v in got_p)) - math.sqrt(sum(v * v for v in want_p)))
                    d_deg = 0.0
                    delta = f"delta |pos|={d_pos * 1000.0:.3f} mm (rotated frame; orientation not compared)"

                msg = (
                    f"[newton-standalone] base-pose readback: {link_path}\n"
                    f"    requested init_base_pose({requested})\n"
                    f"    stage composed   pos={want_p} quat(xyzw)={want_q}\n"
                    f"    imported body_q  pos={imp_p}\n"
                    f"    post-FK body_q[{idx}] pos={got_p} quat(xyzw)={got_q}"
                    f"{'' if fk_ran else '  (FK UNAVAILABLE — import pose)'}\n"
                    f"    {self._base_weld_anchor_str(idx)}\n"
                    f"    {delta}"
                )
                if d_pos > self._BASE_POSE_POS_TOL_M or d_deg > self._BASE_POSE_ROT_TOL_DEG:
                    why = (
                        "FK snapped the base away from where the stage asked, so the "
                        "weld is anchored in the wrong place."
                        if fk_ran
                        else "the importer did not place the base where the stage asked "
                        "(measured without FK, so a bad weld anchor may ALSO be present "
                        "on top of this)."
                    )
                    self._logger.warn(
                        msg + f"\n    MISMATCH — {why} See the root-weld diag above for "
                        "which branch of Newton's world-frame resolution fired."
                    )
                else:
                    self._logger.info(msg)
        except Exception as exc:  # noqa: BLE001
            self._logger.warn(f"[newton-standalone] base-pose readback failed ({exc!r}); continuing")

    def _base_weld_anchor_str(self, body_index: int) -> str:
        """Describe the world anchor of the root joint welding ``body_index``.

        This is the array the divergence actually lives in: for a root
        ``FixedJoint`` (``joint_parent == -1``) ``joint_X_p`` *is* the world
        pose FK snaps the child body to, which is why a bad anchor is
        invisible until FK runs.  Reported alongside the pose comparison so a
        mismatch names the thing to fix rather than only its symptom.

        A ``FREE`` root joint also has ``joint_parent == -1`` and a
        ``joint_X_p`` (identity in practice), but it anchors nothing -- the
        base floats, and ``joint_q`` carries its pose.  Labelled as such
        rather than reported as a weld at the origin.  Best-effort: returns a
        short note instead of raising when the arrays are not as expected.
        """
        try:
            import newton  # noqa: PLC0415

            parent = self._model.joint_parent.numpy()
            child = self._model.joint_child.numpy()
            jtype = self._model.joint_type.numpy()
            x_p = self._model.joint_X_p.numpy()
            for j in range(len(child)):
                if int(child[j]) != int(body_index) or int(parent[j]) != -1:
                    continue
                if int(jtype[j]) != int(newton.JointType.FIXED):
                    return (
                        f"weld anchor: none -- root joint[{j}] is "
                        f"{self._joint_type_name(int(jtype[j]))}, so the base is not "
                        "welded to the world and FK takes its pose from joint_q"
                    )
                a = x_p[j]
                return f"weld anchor joint_X_p[{j}] pos=({float(a[0]):.6f}, " f"{float(a[1]):.6f}, {float(a[2]):.6f})"
            return "weld anchor: none (base has no root joint; it is a free body)"
        except Exception as exc:  # noqa: BLE001
            return f"weld anchor: unavailable ({exc!r})"

    @staticmethod
    def _joint_type_name(value: int) -> str:
        """``newton.JointType`` value -> its name, or the raw int."""
        try:
            import newton  # noqa: PLC0415

            for name in dir(newton.JointType):
                if name.isupper() and int(getattr(newton.JointType, name)) == value:
                    return name
        except Exception:  # noqa: BLE001
            pass
        return str(value)

    def _phase_resolve_scene(self) -> Tuple[List[Any], str, bool, bool]:
        """Phase 2: resolve cloth entries, scene plugin, and solver preference.

        Returns ``(cloth_entries, solver_pref, plugin_has_build,
        needs_particle_solver)`` — locals consumed by later build phases.
        Also stashes ``self._cloth_solver_pref`` for the substep selector.
        """
        self._logger.info("[newton-standalone] building Newton Model…")

        cloth_entries = self._load_cloth_entries()
        # Resolve scene-plugin early so its presence is part of the
        # particle-vs-rigid-only decision below.  A plugin-driven particle
        # scene (e.g. chow-mein FEM noodles) declares
        # ``newton.solver.prefer`` directly in the yaml even with no
        # ``newton.entries``; we treat that the same as a cloth entry for
        # adapter gating + solver-attribute registration.
        self._load_scene_plugin()
        plugin = getattr(self, "_scene_plugin", None)
        plugin_has_build = plugin is not None and callable(getattr(plugin, "on_build", None))

        # Cloth gate — an adapter that doesn't support cloth raises early
        # instead of silently running rigid-only on a cloth scene.
        if cloth_entries and not self._adapter.supports_cloth:
            raise RuntimeError(
                f"physics_solver={self._adapter.name} does not support cloth: found "
                f"{len(cloth_entries)} cloth entry/entries in the scene yaml. "
                f"Use physics_solver:=fsvbd for cloth tasks (launcher_newton_fsvbd.yaml)."
            )
        # Load the solver preference up-front so we can register the
        # matching custom attributes on the builder. The attributes
        # solvers register on the builder MUST match the solver we then
        # instantiate after finalize — VBD reads ``edge_rest_angle`` etc.
        # that XPBD doesn't author, and vice versa.  Plugin-driven scenes
        # also need this even with no cloth entries.
        needs_particle_solver = bool(cloth_entries) or plugin_has_build
        solver_pref = self._load_solver_preference() if needs_particle_solver else "none"
        # Stash for the substep — _substep_body_franka_vbd_cloth needs to
        # know whether to wrap cloth_solver.step() in a body-state freeze
        # (XPBD re-integrates rigid bodies; VBD/Style3D's
        # ``integrate_with_external_rigid_solver=True`` mode does not).
        self._cloth_solver_pref = solver_pref

        return cloth_entries, solver_pref, plugin_has_build, needs_particle_solver

    def _phase_build_model(
        self,
        cloth_entries: List[Any],
        solver_pref: str,
        plugin_has_build: bool,
        needs_particle_solver: bool,
    ) -> None:
        """Phase 3: ModelBuilder + ``add_usd`` + extras + ``finalize``.

        Builds the ModelBuilder, registers the matching custom-attribute
        set for the chosen solver, parses USD, adds cloth / ground /
        plugin bodies, applies pre-finalize mjc_contact tuning, then
        finalises onto cuda:0.  ``self._model`` is populated by the end
        of this phase; ``on_model_ready`` plugin hook also runs here.
        """
        import newton  # noqa: PLC0415

        # ``SchemaResolverMjc`` is REQUIRED, not an extra.  On newton 1.6.1
        # ``dof_passive_damping`` was deleted and this resolver is the only
        # remaining reader of ``mjc:damping`` — the attribute the gripper
        # hand joints carry (``_HAND_JOINT_DAMPING_USD`` in
        # ``genie_sim_engine_isaacsim/assemble/_robot_impl.py``).  Without it
        # those joints import with zero damping and the 1700x gripper
        # resonance comes back.  On 1.4.0 it is a measured no-op —
        # ``SchemaResolverNewton`` already reads the attribute and the value
        # resolves identically with and without it — so passing all three is
        # correct on both builds.  Quirk 8 in ``docs/newton_quirks.md``.
        #
        # Public ``newton.usd`` first: the private ``newton._src`` reach was
        # R4 in the 6.X audit, and all three symbols are re-exported there on
        # 1.4.0.  The fallback covers a build whose public surface predates
        # the re-export; a build with NEITHER raises, because a silently
        # dropped resolver is a wrong run, not a degraded one.
        try:
            from newton.usd import (  # noqa: PLC0415
                SchemaResolverMjc,
                SchemaResolverNewton,
                SchemaResolverPhysx,
            )

            _resolver_module = "newton.usd"
        except ImportError:
            from newton._src.usd.schemas import (  # noqa: PLC0415
                SchemaResolverMjc,
                SchemaResolverNewton,
                SchemaResolverPhysx,
            )

            _resolver_module = "newton._src.usd.schemas"

        # --- Build ModelBuilder (gravity from stage, default -9.81 m/s²) ---
        builder = newton.ModelBuilder()
        self._adapter.register_custom_attributes(builder)
        if needs_particle_solver:
            # AVBD's adapter already registered SolverVBD's custom
            # attributes above (the unified solver handles cloth too),
            # so the cloth-pref branch is a no-op on that path.
            if getattr(self._adapter, "name", "") == "avbd":
                pass
            elif solver_pref == "vbd":
                newton.solvers.SolverVBD.register_custom_attributes(builder)
            elif solver_pref == "xpbd":
                newton.solvers.SolverXPBD.register_custom_attributes(builder)
            elif solver_pref == "style3d":
                newton.solvers.SolverStyle3D.register_custom_attributes(builder)
            else:
                # Defensive — _load_solver_preference already raises on
                # unknown values, but if it returns "none" with cloth /
                # plugin particles present that's a wiring bug we want
                # to surface here.
                raise RuntimeError(
                    f"particle entries present (cloth={bool(cloth_entries)}, "
                    f"plugin={plugin_has_build}) but solver preference is "
                    f"{solver_pref!r}. Set newton.solver.prefer in the "
                    f"scene yaml."
                )

        # Robot from USD.  Note: we do NOT override the importer's
        # ``floating`` flag here — IsaacSim's URDF→USD converter
        # always authors a ``PhysicsFixedJoint "root_joint"`` between
        # ``/robot`` and ``base_link``, which Newton parses as a
        # body-to-world FixedJoint and uses as the root of the
        # articulation.  Passing ``floating=False`` on top would tell
        # ``_add_base_joint`` to ALSO add an auto-fixed-root-joint,
        # producing two joints with the same ``child=base_link`` and
        # tripping mjwarp's directed ``topological_sort`` with
        # ``"Multiple joints lead to body 0"``.
        # Plugins may author an anonymous session-layer overlay before import.
        # The chef USDA path uses this to turn food instances from authored
        # rigid bodies into source meshes without modifying the shared asset.
        # Keep this before the disabled-schema normalizer: it must observe the
        # composed false opinions before ``add_usd`` examines applied schemas.
        self._call_plugin("on_stage", self._stage, self._plugin_ctx())
        _strip_disabled_physics_schemas(self._stage)
        # mjwarp can't sync coordinates for closed-loop ball joints; strip them
        # to equality constraints (see _strip_loop_spherical_joints). Gate on the
        # mjwarp adapter so Featherstone (handles loops natively) is unaffected.
        _loop_eqs: list = []
        if getattr(self._adapter, "name", "").startswith("mujoco-warp"):
            _loop_eqs = _strip_loop_spherical_joints(self._stage)
        # Newton 1.0 has no import-time approximation config and keeps its
        # one-hull CoACD outcome; see _pin_convex_decomposition_imports.
        if hasattr(builder, "default_mesh_approximation_cfg"):
            _pinned = _pin_convex_decomposition_imports(self._stage)
            self._logger.info(f"[newton-model] {_pinned} convexDecomposition collider(s) imported as convexHull")
        builder.add_usd(
            source=self._stage,
            verbose=False,
            ignore_paths=_generated_ground_visual_paths(self._stage, self._scene_cfg),
            collapse_fixed_joints=False,
            schema_resolvers=[SchemaResolverNewton(), SchemaResolverPhysx(), SchemaResolverMjc()],
        )
        self._logger.info(
            f"[newton-model] schema_resolvers=Newton+Physx+Mjc from {_resolver_module} "
            f"(Mjc is load-bearing: sole reader of mjc:damping on newton >= 1.6.1)"
        )
        if _loop_eqs:
            _add_loop_connect_equalities(builder, _loop_eqs)
        # Colliders the USD asked to keep exact (PhysX ``none`` / ``sdf``)
        # arrive from ``add_usd`` as one convex hull each -- see
        # ``_collect_scene_colliders_to_decompose``, which listed them while
        # the stage was open.  Split them here, where CoACD's threshold is
        # ours to choose.
        _decompose_scene_colliders(
            builder,
            getattr(self, "_scene_decomp_paths", None),
            self._logger,
            scene=getattr(getattr(self, "_params", None), "scene_colliders", None),
        )
        # Fingertip pads are convex-decomposed with CoACD
        # (_decompose_gripper_pads): this gives both a multi-point contact
        # manifold and a close fit to the real surface, matching how the
        # benchmark's PhysX assets convex-decompose the same meshes.
        # ``gripper_contact.pad_decomposition.enabled: false`` falls back to a
        # single hull.
        #
        # Pad identity is read from the robot USD's own ``newton:isGripperPad``
        # annotation and has two consumers: the decomposition here, and the
        # ``newton.mjc_contact.gripper_pad:`` refinement below -- which needs
        # only the NAME set, because it re-resolves names to body indices
        # against the builder as it stands at that point (this call runs
        # before the ground plane, the cloth entries and the scene plugin,
        # any of which may add bodies). Older assets without the annotation
        # fall back to gripper-class name matching.
        from genie_sim_engine_newton.common.gripper_annotations import (  # noqa: PLC0415
            read_pad_body_names as _read_pad_body_names,
        )

        _pad_names = _read_pad_body_names(
            self._stage,
            labels=list(getattr(builder, "body_label", []) or []),
            logger=self._logger,
        )
        if getattr(self._adapter, "name", "").startswith("mujoco-warp"):
            _decompose_gripper_pads(
                builder,
                pad_body_names=_pad_names,
                contact=getattr(getattr(self, "_params", None), "gripper_contact", None),
                logger=self._logger,
            )

        # Ground plane — cloth needs it
        builder.add_ground_plane()

        # Cloth meshes
        for entry in cloth_entries:
            self._inject_cloth(builder, entry)

        # Scene-plugin: optional Python file referenced from the scene
        # yaml as ``newton.scene_plugin``.  Lets a scene author add bodies
        # / soft-grids / shapes / kinematic colliders that have no USD
        # schema (e.g. FEM tet bars, runtime-meshed wok colliders) without
        # extending the cloth-entry vocabulary.  Plugin was loaded above so
        # solver-attribute registration could see it; we run on_build now,
        # after USD parse + ground plane + cloth injection.
        self._call_plugin("on_build", builder, self._plugin_ctx())

        # VBD requires coloring; XPBD/Style3D do not.  Coloring is also
        # safe-but-wasteful on a non-VBD particle scene, so gate it.
        if cloth_entries or (plugin_has_build and solver_pref == "vbd"):
            builder.color()

        # MuJoCo-style per-class contact compliance + friction injection.
        # Runs PRE-FINALIZE so it writes into the builder's shape arrays
        # AND the ``mujoco:geom_solimp`` custom-attribute store — both of
        # which Newton then propagates into the runtime model AND MJW's
        # MJCF compile path.  Single source of truth: the MJCF dump and
        # the live contact kernel always agree.
        #
        # The previous post-finalize approach wrote only to runtime
        # buffers and missed the custom-attribute store, producing the
        # misleading "MJCF shows defaults, runtime uses overrides" split
        # that drove the gripper-vs-box tunneling debug.  Module's
        # docstring at ``engine/newton/mjc_contact.py`` records the
        # asymmetry and why this is now the right cut.
        #
        # Every override VALUE comes from ONE place -- the active scene's
        # ``newton.mjc_contact:`` yaml block, merged onto
        # ``common/object_classification.py::MJC_CONTACT_DEFAULTS``. Nothing is
        # passed in from ``physics_params``; if a knob is not in that yaml
        # block it does not exist.
        #
        # ``pad_body_names`` is not a value, it is MEMBERSHIP: the yaml says
        # what the fingertip pads should be tuned to, the robot's own USD says
        # which bodies those are. It is consulted only when the scene names a
        # ``gripper_pad:`` block.
        from genie_sim_engine_newton.engine.newton.mjc_contact import apply_pre_finalize as _mjc_apply  # noqa: PLC0415

        _mjc_apply(
            builder,
            scene_cfg=self._scene_cfg,
            robot_prefix=str(getattr(self, "_robot_prefix_str", "") or ""),
            adapter_name=getattr(self._adapter, "name", ""),
            physics_hz=float(getattr(self, "_physics_hz", 0) or 0),
            sim_substeps=int(getattr(self, "_sim_substeps", 0) or 0),
            logger=self._logger,
            pad_body_names=_pad_names,
        )

        # Native multi-world: when num_envs>1 the single-env ``builder`` above
        # is a TEMPLATE; replicate it into N worlds and finalize the replica.
        # spacing=(0,0,0) keeps all worlds at the origin (numerically stable
        # per Newton docs); visual separation is a render-side concern, not
        # physics. num_envs==1 keeps the original single-builder path
        # byte-for-byte.
        _num_envs = int(getattr(self, "_num_envs", 1) or 1)
        if _num_envs > 1:
            if cloth_entries or plugin_has_build:
                raise NotImplementedError(
                    "multi-env (num_envs>1) native replicate does not yet handle "
                    "cloth / scene-plugin builds (coloring + per-world plugin "
                    "state need post-replicate handling); tracked for a later "
                    "milestone."
                )
            _main = newton.ModelBuilder()
            self._adapter.register_custom_attributes(_main)
            _main.replicate(builder, world_count=_num_envs, spacing=(0.0, 0.0, 0.0))
            self._logger.info(
                f"[multi-env] replicated single-env template into {_num_envs} " f"worlds (native Newton replicate)"
            )
            self._model = _main.finalize(device="cuda:0")
        else:
            self._model = builder.finalize(device="cuda:0")

        # Outcome verification: did the importer actually place the base
        # where the stage asked?  Right after finalize, before anything
        # downstream can mask a divergence.
        self._assert_base_pose_readback()

        # Engine soft-contact defaults (franka cloth values) — set BEFORE the
        # plugin hook so a scene plugin's on_model_ready can OVERRIDE them.
        # These MUST precede _call_plugin("on_model_ready"): FEM scenes
        # (chow-mein noodles) set their own leak-proof ke/kd/mu/margin in
        # on_model_ready, and if the engine wrote these AFTER the plugin
        # (as _phase_normalize_model used to) it would clobber the plugin's
        # tuning and the particles would tunnel straight through the wok.
        self._model.soft_contact_ke = self._SOFT_CONTACT_KE
        self._model.soft_contact_kd = self._SOFT_CONTACT_KD
        self._model.soft_contact_mu = self._SOFT_CONTACT_MU
        # Cloth ↔ rigid narrow-phase margin.  Newton's default is sized for
        # cm-scale assets; at meter scale it generates many spurious close
        # pairs and pumps cloth-rigid narrow-phase cost.  The franka cloth
        # demo creates a CollisionPipeline with an explicit
        # ``soft_contact_margin = particle_radius`` (0.8 cm); we set the
        # same on ``model.soft_contact_margin`` so ``model.collide()`` uses
        # it without us having to switch to the explicit pipeline.
        if hasattr(self._model, "soft_contact_margin"):
            self._model.soft_contact_margin = self._CLOTH_BODY_CONTACT_M

        # Plugin's chance to write into model.particle_* / soft_contact_*
        # before the solver is constructed.  Common use: the chow-mein
        # plugin sets particle_mu / soft_contact_radius / soft_contact_margin
        # (and ke/kd/mu) for FEM noodles, OVERRIDING the engine defaults set
        # just above.  Solver is built after this in _build's tail.
        self._call_plugin("on_model_ready", self._model, self._plugin_ctx())

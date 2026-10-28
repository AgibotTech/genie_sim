# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Joint name classification — shared by every PD-authoring path.

Both layers of the Isaac-Sim drive stack (``usd_drive_api`` /
``articulation_view_runtime`` in ``config/physics_params.yaml``) bucket
joints by the same rules so the per-class gains in those blocks land on
the same DOFs.  This module is where that rule lives.

It also exists so newton-standalone (``engine/newton/adapters/
mujoco_warp.py``) can use the same classifier without depending on
``kit/stage.py`` (which would drag in Pxr + Isaac Sim imports the
standalone path doesn't have).  Callers without USD prim handles use
``classify_joint_by_name`` + ``is_chassis_wheel_free``; callers with
prims use the convenience wrapper ``classify_joint``.

Joint kinds (``JK_*`` string constants — string instead of an Enum
because YAML / JSON / log lines all consume them as strings, and we
don't want every consumer to import an enum class just to compare):

  - ``JK_BODY``           — torso / body chain
  - ``JK_ARM``            — arm-shoulder / mid / wrist
  - ``JK_HEAD``           — head / neck
  - ``JK_GRIPPER``        — gripper master + mimics
  - ``JK_CHASSIS_DRIVE``  — free-spin road wheel (huge limits)
  - ``JK_CHASSIS_STEER``  — bounded-limit steering joint
  - ``JK_CHASSIS_WHEEL``  — chassis wheel before drive/steer split
                            (returned by name-only classification)
  - ``JK_OTHER``          — anything that didn't match a regex

The "drive vs steer" split needs a numeric joint-limit check, so
``classify_joint_by_name`` returns ``JK_CHASSIS_WHEEL`` and lets the
caller refine with ``is_chassis_wheel_free`` once it has the limits.
The Pxr wrapper ``classify_joint`` does this refinement automatically.
"""

from __future__ import annotations

import re
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Joint kinds
# ---------------------------------------------------------------------------

JK_BODY = "body"
JK_ARM = "arm"  # fallback / generic arm — kept for unmatched arm joint numbers
JK_ARM_SHOULDER = "arm_shoulder"  # arm joint indices 1, 2 — proximal / heavy
JK_ARM_MID = "arm_mid"  # arm joint indices 3, 4, 5 — elbow region
JK_ARM_WRIST = "arm_wrist"  # arm joint indices 6, 7 — distal / light
JK_HEAD = "head"
JK_GRIPPER = "gripper"
JK_CHASSIS_DRIVE = "chassis_drive"  # free-spinning road wheel (huge limits)
JK_CHASSIS_STEER = "chassis_steer"  # bounded-limit steering joint
JK_CHASSIS_WHEEL = "chassis_wheel"  # pre-split — caller refines via is_chassis_wheel_free
JK_PASSIVE = "passive"  # auto-created free joints for scene rigid bodies — no PD, just integrate
JK_SCENE_ARTICULATION = (
    "scene_articulation"  # non-free scene joint (door hinge / drawer slide) — no PD + passive damping
)
JK_OTHER = "other"

ALL_JOINT_KINDS = (
    JK_BODY,
    JK_ARM,
    JK_ARM_SHOULDER,
    JK_ARM_MID,
    JK_ARM_WRIST,
    JK_HEAD,
    JK_GRIPPER,
    JK_CHASSIS_DRIVE,
    JK_CHASSIS_STEER,
    JK_CHASSIS_WHEEL,
    JK_PASSIVE,
    JK_SCENE_ARTICULATION,
    JK_OTHER,
)

# ---------------------------------------------------------------------------
# Name patterns
# ---------------------------------------------------------------------------

# URDF importers prepend a numeric ``idxNN_`` prefix (e.g.
# ``idx21_arm_l_joint3``).  We strip it before matching so
# hand-authored USDs without the prefix (``arm_l_joint3``) match the
# same patterns.
_RE_IDX_PREFIX = re.compile(r"^idx\d+_")

# Each pattern is anchored to a name component so partial collisions
# cannot leak (``foo_arm_l_joint1`` does NOT match ``_RE_ARM`` because
# the leading ``foo`` would have to be the start-of-string or follow
# an underscore, not be a free-form prefix).  ``\b`` at the end keeps
# us tolerant of trailing decorations (``arm_l_joint3_extra`` won't
# match, ``arm_l_joint3`` will).
_RE_BODY = re.compile(r"(^|_)body_joint\d*\b")
_RE_ARM = re.compile(r"(^|_)arm(_[a-z0-9]+)?_joint(\d+)\b")
# Arm sub-classes — distinguish by joint index (proximal -> distal).
# References the same ``arm_*_joint<N>`` shape; the trailing capture
# group on ``_RE_ARM`` lets the classifier grab N without re-matching.
_ARM_SHOULDER_INDICES = frozenset({1, 2})
_ARM_MID_INDICES = frozenset({3, 4, 5})
_ARM_WRIST_INDICES = frozenset({6, 7})
_RE_HEAD = re.compile(r"(^|_)head_joint\d*\b")
_RE_GRIPPER = re.compile(r"(^|_)gripper(_[a-z0-9_]+)?_joint\d*\b")
# Dexterous multi-finger hands (e.g. wuji) name their DOFs
# ``<side>_finger<F>_joint<J>`` instead of ``gripper_*_joint``.  They are
# end-effector / gripper DOFs, so they classify as ``JK_GRIPPER`` and
# receive the gripper PD block.  Each finger joint is independently
# actuated (no mimic linkage); the gripper drive path already handles
# that via its "no mimic schema -> master drive" branch.
_RE_GRIPPER_FINGER = re.compile(r"(^|_)finger\d+_joint\d*\b")
# Alternate dexterous-hand convention where fingers are named
# (``hand_l_index_joint1``, ``hand_r_thumb_joint4``, …) instead of
# numbered.  Same PD story as ``_RE_GRIPPER_FINGER``: classify as
# ``JK_GRIPPER``.  Optional ``_l|_r`` side keeps single-hand rigs matched
# (``hand_index_joint1``).  Constrained to a fixed vocabulary of finger
# tokens so unrelated ``hand_*`` links (e.g. ``hand_link``) do not leak in.
_RE_GRIPPER_HAND = re.compile(r"(^|_)hand(_[lr])?_(index|middle|ring|pinky|little|thumb)_joint\d*\b")
# chassis_*wheel*joint* — wheel/steer share the same regex.  The
# drive/steer split is decided by the actual joint limit width (see
# ``is_chassis_wheel_free``), not the name.
_RE_CHASSIS_WHEEL = re.compile(r"(^|_)chassis_[a-z0-9_]*wheel[a-z0-9_]*_joint\d*\b")

# Anything wider than this (in degrees — USD stores revolute limits in
# degrees) is treated as an unbounded "free-spin" road wheel rather
# than a steering joint.  Sentinel limits of +/- 1e20 appear on
# free-spin joints; any sane steering axis is well under +/- 360 deg
# (the URDF importer converts +/- 2.97 rad → +/- 170 deg, giving a
# range of ~340 deg).  Threshold must exceed that but stay far below
# 2e20.
_FREE_LIMIT_THRESHOLD_DEG = 700.0

# Newton's ``add_usd`` auto-creates a 6-DOF free joint for any
# free-floating ``RigidBodyAPI`` prim and names it ``joint_<N>`` (e.g.
# ``joint_53``).  These are passive scene rigid bodies (hangers, dropped
# objects) — NOT robot joints.  No regex above matches them by design.
# Recognise them here so they're classified as ``JK_PASSIVE`` and routed
# through a no-PD path in the adapter.
_RE_PASSIVE_AUTO = re.compile(r"^joint_\d+$")

# USD's default leaf-name for an un-renamed joint prim — bare schema type
# (``RevoluteJoint`` / ``PrismaticJoint`` / ``D6Joint`` / ``SphericalJoint``
# / ``FixedJoint`` / ``DistanceJoint``), optionally with a ``_<N>`` de-dup
# suffix.  Treated as passive, same as ``_RE_PASSIVE_AUTO`` above.
_RE_PASSIVE_DEFAULT = re.compile(r"^(Revolute|Prismatic|Spherical|D6|Fixed|Distance)Joint_?\d*$")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def strip_idx_prefix(name: str) -> str:
    """Strip the ``idxNN_`` prefix that URDF importers prepend (if any).

    Used as the first step in every classifier so the same regexes
    match both URDF-imported and hand-authored joint names.
    """
    return _RE_IDX_PREFIX.sub("", name)


def classify_joint_by_name(name: str) -> str:
    """Return one of the ``JK_*`` constants for ``name`` using name
    pattern matching only — no Pxr / USD / Newton dependency.

    For chassis wheel joints this returns ``JK_CHASSIS_WHEEL``
    *unsplit*; callers that need the drive-vs-steer distinction must
    feed the joint's lower/upper limit to ``is_chassis_wheel_free``
    and refine the result.

    Use this from contexts where you don't have a USD prim (e.g. the
    newton-standalone ``MuJoCoWarpAdapter`` working purely from
    ``model.joint_label``).  Callers WITH a prim should use
    ``classify_joint`` instead, which does the chassis split
    automatically.
    """
    stripped = strip_idx_prefix(name)
    if _RE_BODY.search(stripped):
        return JK_BODY
    m_arm = _RE_ARM.search(stripped)
    if m_arm:
        # Sub-class by joint index when the name carries one.  Falls
        # back to plain ``JK_ARM`` for arm joints whose index doesn't
        # land in the shoulder / mid / wrist buckets (e.g. ``joint0``
        # or ``joint8+``).  The adapter's per-class table still has
        # ``JK_ARM`` as a safe default mapping in that case.
        try:
            n = int(m_arm.group(3))
        except (TypeError, ValueError):
            n = -1
        if n in _ARM_SHOULDER_INDICES:
            return JK_ARM_SHOULDER
        if n in _ARM_MID_INDICES:
            return JK_ARM_MID
        if n in _ARM_WRIST_INDICES:
            return JK_ARM_WRIST
        return JK_ARM
    if _RE_HEAD.search(stripped):
        return JK_HEAD
    if _RE_GRIPPER.search(stripped) or _RE_GRIPPER_FINGER.search(stripped) or _RE_GRIPPER_HAND.search(stripped):
        return JK_GRIPPER
    if _RE_CHASSIS_WHEEL.search(stripped):
        return JK_CHASSIS_WHEEL
    if _RE_PASSIVE_AUTO.match(stripped) or _RE_PASSIVE_DEFAULT.match(stripped):
        return JK_PASSIVE
    return JK_OTHER


def classify_scene_joint(
    *,
    is_free: bool,
    parent_body_label: Optional[str],
    child_body_label: Optional[str],
    robot_prefix: str,
) -> Optional[str]:
    """Ownership-based scene-joint routing (Task 2 — articulated scene
    objects).  Returns the joint kind for a joint whose parent AND child
    bodies are both OUTSIDE the robot prefix, or ``None`` when the joint is
    robot-owned / ownership is undecidable.

    Returns
    -------
    * ``JK_SCENE_ARTICULATION`` — a non-free scene joint (a door hinge or
      drawer slide connecting two scene bodies).  The adapter gives it zero
      PD gains (passive, no ghost drive) PLUS the ``scene_articulation``
      passive ``dof_damping`` / ``dof_frictionloss`` so it settles instead
      of swinging frictionlessly forever.
    * ``JK_PASSIVE`` — a free-floating scene body's auto-created 6-DOF FREE
      joint.  Zero PD, and (unlike scene articulation) NO artificial passive
      damping, so it free-falls and settles naturally — identical to the
      pre-Task-2 behaviour for table/dropped-block scene props.
    * ``None`` — the joint is robot-owned (at least one body under the robot
      prefix) OR ``robot_prefix`` is empty.  The caller keeps its
      byte-identical name-based classification (``classify_joint_by_name``)
      for these, so robot joints are entirely unaffected.

    This is the joint-level analogue of ``object_classification.classify_shape``
    — same body-prefix ownership concept, one level up.  Keying on ownership
    (not on the joint name) is what lets an articulated prop authored with a
    CUSTOM joint name (``door_hinge_left``) be recognised without a per-asset
    name regex and without tripping the ``unknown_joint_names`` hard-fail.
    """
    # Local import keeps ``object_classification`` (typing-only) off this
    # module's import cost for callers that never touch scene joints, and
    # avoids any import-order coupling between the two ``common`` modules.
    from genie_sim_engine_newton.common.object_classification import is_scene_joint  # noqa: PLC0415

    if not is_scene_joint(
        parent_body_label=parent_body_label, child_body_label=child_body_label, robot_prefix=robot_prefix
    ):
        return None
    return JK_PASSIVE if is_free else JK_SCENE_ARTICULATION


def is_chassis_wheel_free(
    low: Optional[float],
    high: Optional[float],
    *,
    threshold: float = _FREE_LIMIT_THRESHOLD_DEG,
) -> bool:
    """Decide if a chassis wheel joint is a free-spin drive wheel.

    Drive wheels use sentinel limits (effectively +/- inf).  Steering
    joints have bounded limits (typically a few radians, or a few
    hundred degrees once USD converts radians to its native degree
    storage).

    Falls back to ``True`` (drive) when limits are missing or
    unreadable — chassis joints with the wheel suffix are typically
    drive wheels, so "unknown chassis wheel" defaults to the more
    common case.

    Unit handling
    -------------
    The default threshold is 700.0, calibrated for USD revolute joints
    which store limits in DEGREES even though the rest of USD physics
    is radians.  Callers reading from a Newton ``model.joint_limit_*``
    array (radians) should pass ``threshold=12.0`` (= 700 / 57.3) so
    the same drive-vs-steer split applies in either unit.  Prismatic
    joints use meters and ``threshold`` should reflect the scene scale
    (the default is unlikely to be meaningful there).
    """
    if low is None or high is None:
        return True
    try:
        return abs(float(high) - float(low)) > threshold
    except (TypeError, ValueError):
        return True


def classify_joint(name: str, prim: Any) -> str:
    """Return one of the ``JK_*`` constants for ``name``, splitting
    chassis wheels into drive/steer by reading the prim's joint limits.

    Convenience wrapper for callers that have a Pxr USD prim.  When
    the prim has no readable limits the chassis wheel defaults to
    ``JK_CHASSIS_DRIVE`` (matches ``is_chassis_wheel_free``'s fallback).

    Pxr is imported lazily so importing this module does NOT pull in
    Pxr — newton-standalone can ``from common.joint_classification
    import classify_joint_by_name`` without any USD dependency.
    """
    kind = classify_joint_by_name(name)
    if kind != JK_CHASSIS_WHEEL:
        return kind

    # Pxr-using branch — chassis wheel needs limit lookup.
    try:
        from pxr import UsdPhysics  # noqa: PLC0415
    except ImportError:
        # No Pxr available; fall through with the safe default.
        return JK_CHASSIS_DRIVE

    revolute = UsdPhysics.RevoluteJoint(prim)
    prismatic = UsdPhysics.PrismaticJoint(prim)
    if revolute:
        lo_attr = revolute.GetLowerLimitAttr()
        hi_attr = revolute.GetUpperLimitAttr()
    elif prismatic:
        lo_attr = prismatic.GetLowerLimitAttr()
        hi_attr = prismatic.GetUpperLimitAttr()
    else:
        return JK_CHASSIS_DRIVE

    def _safe_get(attr):
        try:
            return attr.Get()
        except Exception:  # noqa: BLE001
            return None

    return JK_CHASSIS_DRIVE if is_chassis_wheel_free(_safe_get(lo_attr), _safe_get(hi_attr)) else JK_CHASSIS_STEER

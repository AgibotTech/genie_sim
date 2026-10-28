# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Engine-side reader for the assemble-time gripper semantic annotations.

Task 2 of the engine-hardcoding H1-H4 fix (see
``docs/superpowers/plans/2026-07-21-engine-hardcoding-H1-H4-fix.md`` and
``docs/PORT_ENGINE_HARDCODING_AUDIT.md``).

The physics engine used to identify the gripper's fingertip "pad" bodies,
its 4-bar "coupler" joints, and its drive "master" joints with G2-specific
name literals / regexes baked into the hot path
(``"link4" in label``, ``gripper_[lr]_(inner|outer)_joint[034]$``,
``gripper_[lr]_outer_joint1$``).  Task 1 moved that knowledge to ASSEMBLE
TIME: when ``robot.usda`` is built from the URDF it authors two boolean
custom attributes so the engine can read the gripper topology from the
robot's own USD instead of hard-coding it:

  * ``newton:isGripperPad``  (bool) on each fingertip pad body prim
  * ``newton:isLoopCoupler`` (bool) on each 4-bar coupler joint prim

This module is the ENGINE-SIDE consumer.  It follows the same proven path
``genie_sim_engine_core.engine._mimic.parse_mimic`` uses: **take the live
USD stage, traverse it, collect the prims carrying the annotation** —
rather than trying to pull custom attributes back out of the finalized
Newton model (which does not preserve them).

ANNOTATION-FIRST with LITERAL FALLBACK.  Existing G2 ``robot.usda`` files
predate Task 1 and carry NO annotations; re-generating them is optional.
So every reader here degrades to the LEGACY literal/regex behaviour when
the stage is missing or carries no annotation, which keeps un-annotated
scenes working with zero risk.  The fallbacks are the physical-truth
variants (see the plan's "reconnaissance correction" note):

  * pad fallback  — gripper-CLASSIFIED ``link4`` leaves only.  The old
    ``"link4" in label`` substring test also hit ``arm_l/r_link4`` and
    ``body_link4`` (a latent bug); the fallback restricts to the gripper
    family so it lands on the 4 real pads and nothing else.
  * coupler fallback — the legacy regex ``gripper_[lr]_(inner|outer)_joint[034]$``.

The "master" set needs NO annotation — it is exactly
``mimic_followers.keys()`` (the joints that ``<mimic>`` followers point
at).  Per the plan's reconnaissance-correction note the URDF truth is
that every follower references ``inner_joint1``, so the master set is the two
``inner_joint1`` joints — NOT the ``outer_joint1`` the old H4 regex
picked.  H4 (master force-cap) switches to this set.

No ``pxr`` import at module load: the stage object is passed in and only
its duck-typed ``Traverse`` / ``GetAttribute`` methods are used, so this
module imports cleanly anywhere (including hosts without USD).
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

# ----------------------------------------------------------------------
# Annotation attribute names — MUST match assemble/_robot_impl.py's
# ``_ANNOT_GRIPPER_PAD`` / ``_ANNOT_LOOP_COUPLER``.  Keep verbatim.
# ----------------------------------------------------------------------
ANNOT_GRIPPER_PAD = "newton:isGripperPad"
ANNOT_LOOP_COUPLER = "newton:isLoopCoupler"

# ----------------------------------------------------------------------
# Name-matching fallbacks — consulted ONLY when the stage carries no
# annotations.  The patterns below are DEFAULTS describing the G2
# omnipicker; ``configure_naming`` replaces them from
# ``physics_params.yaml``'s ``gripper_naming:`` block so a differently
# named gripper needs no code change.
#
#   family        marks a joint/link as gripper-related at all.  The pad
#                 fallback intersects the ``link4`` substring with this so
#                 it does not also hit ``arm_l/r_link4`` / ``body_link4``.
#   loop_coupler  the passive four-bar joints closing each finger loop.
#   driven        driven joints; consulted only when no mimic map parsed.
#   master        master joint; consulted only when no mimic map parsed
#                 (with a mimic map the masters come from it instead).
# ----------------------------------------------------------------------
DEFAULT_NAMING = {
    "family": r"gripper|finger|jaw|knuckle",
    "loop_coupler": r"gripper_[lr]_(inner|outer)_joint[034]$",
    "driven": r"gripper_[lr]_(inner|outer)_joint1$",
    "master": r"gripper_[lr]_outer_joint1$",
}

_GRIPPER_CLASSIFY_RE = re.compile(DEFAULT_NAMING["family"], re.IGNORECASE)
_COUPLER_FALLBACK_RE = re.compile(DEFAULT_NAMING["loop_coupler"])
_DRIVEN_FALLBACK_RE = re.compile(DEFAULT_NAMING["driven"])
_LEGACY_MASTER_RE = re.compile(DEFAULT_NAMING["master"])


def configure_naming(naming: Any = None, logger: Any = None) -> None:
    """Install the fallback name patterns from config.

    ``naming`` is anything with ``family`` / ``loop_coupler`` / ``driven``
    / ``master`` attributes (``params.GripperNaming``) or the equivalent
    mapping.  Missing or unreadable entries keep their default, and an
    invalid regex is reported and skipped rather than killing the build.
    Call once per engine build, before any lookup runs.
    """
    global _GRIPPER_CLASSIFY_RE, _COUPLER_FALLBACK_RE
    global _DRIVEN_FALLBACK_RE, _LEGACY_MASTER_RE
    if naming is None:
        return
    get = naming.get if isinstance(naming, dict) else lambda k, d: getattr(naming, k, d)
    targets = {
        "family": ("_GRIPPER_CLASSIFY_RE", re.IGNORECASE),
        "loop_coupler": ("_COUPLER_FALLBACK_RE", 0),
        "driven": ("_DRIVEN_FALLBACK_RE", 0),
        "master": ("_LEGACY_MASTER_RE", 0),
    }
    for key, (var, flags) in targets.items():
        pat = str(get(key, DEFAULT_NAMING[key]) or DEFAULT_NAMING[key])
        try:
            globals()[var] = re.compile(pat, flags)
        except re.error as exc:
            if logger is not None:
                logger.warn(
                    f"[gripper-annot] gripper_naming.{key} is not a valid "
                    f"regex ({exc}); keeping the default {DEFAULT_NAMING[key]!r}"
                )


# The importer prefixes every joint name with this in the compiled MJCF
# (e.g. ``_genie_joints_idx31_gripper_l_inner_joint1``).  Annotation prim
# names carry the un-prefixed short form (``idx31_gripper_l_inner_joint1``),
# so name matching strips this to compare like-for-like.
_JOINT_NAME_PREFIX_SEP = "_genie_joints_"


def _leaf(label: Any) -> str:
    """Last ``/``-separated segment of a body/prim path label."""
    return str(label).split("/")[-1]


def _short_joint(name: Any) -> str:
    """Un-prefixed joint short-name (strip the ``_genie_joints_`` prefix)."""
    return str(name).split(_JOINT_NAME_PREFIX_SEP)[-1]


# ======================================================================
# master set (H4) — no annotation needed
# ======================================================================
def gripper_masters(
    mimic_followers: Optional[Dict[str, List[Tuple[str, float, float]]]],
) -> Set[str]:
    """Return the gripper MASTER joint names.

    The master set is exactly the keys of the mimic map
    (``{master_name: [(follower, coef1, coef0), ...]}``) produced by
    ``genie_sim_engine_core.engine._mimic.parse_mimic`` — i.e. every
    joint that at least one ``<mimic>`` follower references.  On G2 this
    is the two ``inner_joint1`` joints (one per gripper); see the plan's
    reconnaissance-correction note.  Returns an empty set when the mimic map
    is empty/None (callers fall back to the legacy regex — see
    ``joint_is_master``).
    """
    if not mimic_followers:
        return set()
    return set(mimic_followers.keys())


def joint_is_master(
    joint_name: str,
    masters: Set[str],
    *,
    fallback_regex: Optional["re.Pattern[str]"] = _LEGACY_MASTER_RE,
) -> bool:
    """Is ``joint_name`` a gripper drive master?

    ANNOTATION/MIMIC-FIRST: when ``masters`` is non-empty, ``joint_name``
    is a master iff it matches one of them.  Matching is prefix-tolerant
    (the compiled MJCF prefixes joints with ``_genie_joints_`` while the
    mimic map keys are un-prefixed prim short-names), mirroring how
    ``_apply_mimic_dof_damping`` matches short names against full ones.

    FALLBACK: when ``masters`` is empty (no mimic map parsed) and a
    ``fallback_regex`` is given, fall back to the legacy regex so an
    un-mimic'd scene still identifies a master.  Pass ``fallback_regex=None``
    to disable the fallback entirely.
    """
    jn = str(joint_name)
    if masters:
        short = _short_joint(jn)
        return any(jn.endswith(m) or short == m for m in masters)
    if fallback_regex is not None:
        return bool(fallback_regex.search(jn))
    return False


def master_actuator_names(
    joint_names: Iterable[str],
    masters: Set[str],
    *,
    fallback_regex: Optional["re.Pattern[str]"] = _LEGACY_MASTER_RE,
) -> Set[str]:
    """Subset of ``joint_names`` that are gripper masters.

    Convenience wrapper over :func:`joint_is_master` for the H4 force-cap
    loop: pass every actuator's joint name and get back the ones to cap.
    """
    return {jn for jn in joint_names if joint_is_master(jn, masters, fallback_regex=fallback_regex)}


# ======================================================================
# stage-based annotation readers (H1/H2 pads, H3 couplers)
# ======================================================================
def _annotated_prim_names(stage: Any, attr_name: str) -> Optional[Set[str]]:
    """Traverse ``stage`` and collect short-names of prims carrying
    ``attr_name == True``.

    Returns ``None`` when the stage is unusable (missing, no ``Traverse``,
    or traversal raises) so callers can distinguish "no stage" from "stage
    present but no annotations" (empty set) and fall back accordingly.
    Mirrors ``_mimic.parse_mimic``'s traversal exactly.
    """
    if stage is None or not hasattr(stage, "Traverse"):
        return None
    try:
        names: Set[str] = set()
        for prim in stage.Traverse():
            attr = prim.GetAttribute(attr_name)
            if attr and attr.IsValid() and attr.HasAuthoredValue() and bool(attr.Get()):
                names.add(prim.GetName())
        return names
    except Exception:
        return None


def read_pad_body_names(
    stage: Any = None,
    *,
    labels: Optional[Iterable[Any]] = None,
    logger: Any = None,
) -> Set[str]:
    """Fingertip pad body NAMES (leaf/prim short-names).

    ANNOTATION-FIRST: every prim on ``stage`` carrying
    ``newton:isGripperPad = True``.  FALLBACK (stage absent / no
    annotation): the gripper-CLASSIFIED ``link4`` leaves of ``labels``
    (body path labels, e.g. ``model.body_label``).  The gripper-class
    restriction is what keeps the fallback off the old
    ``arm_l/r_link4`` / ``body_link4`` mis-hits.
    """
    annotated = _annotated_prim_names(stage, ANNOT_GRIPPER_PAD)
    if annotated:
        if logger is not None:
            logger.info(f"[gripper-annot] isGripperPad: {len(annotated)} pad body(ies) from USD annotation")
        return annotated
    out: Set[str] = set()
    for label in labels or ():
        leaf = _leaf(label)
        if "link4" in leaf and _GRIPPER_CLASSIFY_RE.search(leaf):
            out.add(leaf)
    if logger is not None:
        logger.info(f"[gripper-annot] isGripperPad: no annotation; fallback matched {len(out)} link4 pad(s)")
    return out


def read_loop_coupler_joint_names(
    stage: Any = None,
    *,
    joint_names: Optional[Iterable[Any]] = None,
    logger: Any = None,
) -> Set[str]:
    """4-bar loop coupler joint NAMES.

    ANNOTATION-FIRST: every prim on ``stage`` carrying
    ``newton:isLoopCoupler = True`` (prim short-names, e.g.
    ``idx32_gripper_l_inner_joint3``).  FALLBACK (stage absent / no
    annotation): the entries of ``joint_names`` whose (prefix-stripped)
    name matches the legacy regex ``gripper_[lr]_(inner|outer)_joint[034]$``;
    returned verbatim so the caller can match them against whatever
    namespace it holds.
    """
    annotated = _annotated_prim_names(stage, ANNOT_LOOP_COUPLER)
    if annotated:
        if logger is not None:
            logger.info(f"[gripper-annot] isLoopCoupler: {len(annotated)} coupler joint(s) from USD annotation")
        return annotated
    out: Set[str] = set()
    for jn in joint_names or ():
        if _COUPLER_FALLBACK_RE.search(_short_joint(jn)):
            out.add(jn)
    if logger is not None:
        logger.info(f"[gripper-annot] isLoopCoupler: no annotation; fallback matched {len(out)} coupler joint(s)")
    return out


# ======================================================================
# pure name -> engine-identifier mappers (consumed by Tasks 3-4)
# ======================================================================
def pad_body_indices(labels: Iterable[Any], pad_names: Set[str]) -> Set[int]:
    """Map pad body NAMES to body INDICES against ``labels``.

    ``labels`` is the per-body path list (``model.body_label``); a body
    is a pad iff its leaf name is in ``pad_names``.  Compose with
    :func:`read_pad_body_names` at the H1/H2 call sites::

        labels = list(model.body_label)
        pads = read_pad_body_names(stage, labels=labels)
        idx = pad_body_indices(labels, pads)
    """
    return {i for i, label in enumerate(labels) if _leaf(label) in pad_names}


def coupler_joint_labels(joint_labels: Iterable[str], coupler_names: Set[str]) -> Set[str]:
    """Select the entries of ``joint_labels`` that are loop couplers.

    Prefix-tolerant: a label matches when it ends with, equals, or has
    the leaf/short-name of one of ``coupler_names`` (which may be authored
    prim short-names or already-full names).  Compose with
    :func:`read_loop_coupler_joint_names` at the H3 call site.
    """
    out: Set[str] = set()
    for jl in joint_labels:
        s = str(jl)
        short = _short_joint(s)
        if any(s.endswith(c) or short == c or _short_joint(c) == short for c in coupler_names):
            out.add(jl)
    return out


def gripper_loop_constraint_labels(
    joint_labels: Iterable[str],
    coupler_names: Set[str],
    follower_names: Set[str],
    master_names: Set[str],
    *,
    driven_fallback_regex: Optional["re.Pattern[str]"] = _DRIVEN_FALLBACK_RE,
) -> Tuple[Set[str], Set[str]]:
    """Map the gripper loop-closure name sets onto the adapter's joint labels.

    This is the H3 (``_fix_gripper_loop_constraints``) consumer: the
    adapter iterates the compiled MJCF joint names (``mj_id2name`` — e.g.
    ``_genie_joints_idxNN_gripper_l_inner_joint3``) and needs, in that same
    label namespace, the two sets the three H3 sites act on.  Returns
    ``(coupler_labels, driven_labels)``, both subsets of ``joint_labels``:

      * ``coupler_labels`` — the passive in-loop 4-bar COUPLERS
        (``newton:isLoopCoupler`` truth, or the legacy ``joint[034]`` regex
        fallback — see :func:`read_loop_coupler_joint_names`).  H3 site 1
        (coupler actuator UNLOCK) and site 3 (passive damping **1.0**) act
        on this set — 12 joints on G2.
      * ``driven_labels`` — the DRIVEN gripper joints (H3 site 2, passive
        damping **8.0**): ``masters ∪ (followers − couplers)``.  On G2 that
        is the two ``inner_joint1`` masters plus the two ``outer_joint1``
        followers (which are DRIVEN, not couplers) = 4 joints.

    ``coupler_names`` / ``follower_names`` / ``master_names`` are joint
    NAMES (authored prim short-names, or the fallback's verbatim entries);
    ``coupler_joint_labels`` / :func:`joint_is_master` map them onto
    ``joint_labels`` prefix-tolerantly, so the set operations happen in one
    consistent (label) namespace.

    STRUCTURAL primary path (mimic present): ``driven_labels`` comes purely
    from ``masters ∪ (followers − couplers)`` — no name regex.  When NO
    mimic map was parsed (``master_names`` AND ``follower_names`` both empty
    — never on a live G2 scene) it degrades to ``driven_fallback_regex``
    (the legacy ``gripper_[lr]_(inner|outer)_joint1$``) so un-mimic'd scenes
    keep pre-fix behaviour, mirroring the coupler reader's stage-absent
    fallback.  Pass ``driven_fallback_regex=None`` to disable that degrade.
    """
    labels = list(joint_labels)
    coupler_labels = coupler_joint_labels(labels, coupler_names)
    if master_names or follower_names:
        # STRUCTURAL: masters (mimic-map keys) ∪ (all followers − couplers).
        # ``fallback_regex=None`` keeps the master match off the legacy
        # ``outer_joint1`` regex — masters come only from the mimic map here.
        master_labels = {jl for jl in labels if joint_is_master(jl, master_names, fallback_regex=None)}
        follower_labels = coupler_joint_labels(labels, follower_names)
        driven_labels = master_labels | (follower_labels - coupler_labels)
    elif driven_fallback_regex is not None:
        driven_labels = {jl for jl in labels if driven_fallback_regex.search(_short_joint(str(jl)))}
    else:
        driven_labels = set()
    return coupler_labels, driven_labels

# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Object (shape) name classification — shared by every contact-param
authoring path on the MJW side.

Companion to :mod:`common.joint_classification`.  The joint classifier
buckets actuator DOFs by class so per-class PD gains land on the right
DOFs; this one buckets *colliders* by class so per-class contact
compliance (solref / solimp) lands on the right shapes.

Why a Python-side classifier instead of USD ``mjc:*`` attributes:

  * Scenes here are authored in IsaacSim, not MuJoCo.  Authoring
    ``mjc:solref`` per prim during IsaacSim composition is awkward and
    bleeds MJ-specific knobs into asset files that other solvers ignore.
  * Most tuning is reproduced from pure-python ``mujoco.MjModel``
    experiments.  Keeping the canonical values in Python — alongside
    diff-able default tables — lets the same constants flow back into
    those scripts.
  * One central classifier means a new scene only needs to label its
    prims (``/World/<name>`` or ``<robot_prefix>/<link>``) for the
    classifier to bucket them automatically.  No per-asset USD edits.

Object kinds (``OK_*`` string constants — string instead of an Enum
for the same reason as joint kinds):

  - ``OK_ROBOT``        — any collider under the active robot's prim
                          path (matched by ``robot_prefix``).
  - ``OK_FLOOR``        — the ground plane added by
                          ``builder.add_ground_plane()`` (no USD prim;
                          identified by ``GeoType.PLANE`` + ``body == -1``).
  - ``OK_PASSIVE``      — passive scene rigid bodies (hanger, dropped
                          objects).  Identified by ``body`` index that
                          isn't part of the robot's articulation but
                          isn't world-static either.
  - ``OK_STATIC_PROP``  — world-static USD shapes that aren't the ground
                          plane (cube props, tables, walls in
                          ``/World/*``).
  - ``OK_OTHER``        — anything that didn't match.

The classifier is name + topology based (label, body index, shape type)
so it works on Newton's flat model arrays after ``builder.finalize()``,
without re-walking the USD stage.
"""

from __future__ import annotations

from typing import Any, Optional

# ---------------------------------------------------------------------------
# Object kinds
# ---------------------------------------------------------------------------

OK_ROBOT = "robot"
OK_FLOOR = "floor"
OK_PASSIVE = "passive"
OK_STATIC_PROP = "static_prop"
OK_OTHER = "other"

ALL_OBJECT_KINDS = (
    OK_ROBOT,
    OK_FLOOR,
    OK_PASSIVE,
    OK_STATIC_PROP,
    OK_OTHER,
)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def classify_shape(
    *,
    shape_label: str,
    body_index: int,
    body_label: Optional[str],
    shape_type_int: int,
    robot_prefix: str,
    plane_geo_type: int,
) -> str:
    """Classify a single shape into one of ``OK_*``.

    Parameters
    ----------
    shape_label
        The ``model.shape_label[i]`` string.  Usually the USD prim path
        (``/World/hanger/geometry``, ``/genie/.../link_mesh``) or a
        synthetic label like ``ground_plane`` for ``add_ground_plane``.
    body_index
        The ``model.shape_body[i]`` value.  ``-1`` means world-static.
    body_label
        ``model.body_label[body_index]`` when ``body_index >= 0``.
        Used for prefix matching the robot.
    shape_type_int
        ``int(model.shape_type[i])`` — compared against ``plane_geo_type``
        to recognise the ground plane without importing Newton enums
        in this module.
    robot_prefix
        The active robot's USD prefix (e.g. ``"genie"``).  Shapes whose
        owning body's label starts with ``/<robot_prefix>/`` are robot.
    plane_geo_type
        ``int(newton.GeoType.PLANE)`` — passed by the caller to avoid an
        import on this module.

    Returns
    -------
    One of the ``OK_*`` string constants.
    """
    # 1. Ground plane — fastest path, identified by geometry type alone.
    #    ``builder.add_ground_plane`` always emits a PLANE bound to body -1
    #    (world-static).  Anything else is not the floor.
    if shape_type_int == plane_geo_type and body_index < 0:
        return OK_FLOOR

    # 2. Robot — match by owning body's prim path prefix.  We use the
    #    body label rather than shape label because shape labels can be
    #    geometry leaves (e.g. ``/genie/Robot/arm_l_link3/visuals/mesh``)
    #    where only the body label cleanly carries the prefix.
    if robot_prefix and body_index >= 0 and body_label:
        prefix = f"/{robot_prefix}/"
        if body_label.startswith(prefix) or body_label == f"/{robot_prefix}":
            return OK_ROBOT

    # 3. World-static + not the ground plane → static prop (cube, table,
    #    wall — anything authored in /World with a CollisionAPI but no
    #    RigidBodyAPI).
    if body_index < 0:
        return OK_STATIC_PROP

    # 4. Body-bound but not under the robot prefix → passive scene rigid
    #    (hanger, dropped object).  Newton auto-FREE-joints these.
    return OK_PASSIVE


# ---------------------------------------------------------------------------
# Joint-level ownership (Task 2: articulated scene objects)
# ---------------------------------------------------------------------------
#
# The shape classifier above buckets COLLIDERS by robot-prefix ownership.
# Scene *articulation* (door hinges, drawer slides) needs the same
# ownership concept one level up — at the JOINT level — so an articulated
# prop's joints are recognised structurally, independent of their name.
# ``classify_shape``'s OK_ROBOT test is ``body_label.startswith(
# f"/{robot_prefix}/") or body_label == f"/{robot_prefix}"``; ``body_is_robot``
# is that exact predicate, factored out so the joint path shares it.


def body_is_robot(body_label: Optional[str], robot_prefix: str) -> bool:
    """True iff a body's Newton ``body_label`` places it under the active
    robot's USD prefix.

    Mirrors the OK_ROBOT prefix test in :func:`classify_shape` — a body is
    robot-owned when its label starts with ``/<robot_prefix>/`` or equals
    ``/<robot_prefix>`` exactly (the robot root).  The world body (Newton
    joint parent ``-1``, no label) and any ``/World/...`` scene body return
    ``False``.  Returns ``False`` when ``robot_prefix`` is empty (ownership
    undecidable) or ``body_label`` is falsy.
    """
    if not robot_prefix or not body_label:
        return False
    prefix = f"/{robot_prefix}/"
    return body_label.startswith(prefix) or body_label == f"/{robot_prefix}"


def is_scene_joint(
    *,
    parent_body_label: Optional[str],
    child_body_label: Optional[str],
    robot_prefix: str,
) -> bool:
    """True iff a joint is a non-robot **SCENE** articulation joint.

    A joint is a scene joint when NEITHER its parent NOR its child body is
    robot-owned (:func:`body_is_robot`).  This catches:

      * a door hinge / drawer slide between two scene bodies, and
      * a free-floating scene body's auto-created 6-DOF FREE joint
        (parent = world, child = scene body).

    It deliberately does NOT catch the robot's own floating-base FREE joint
    (parent = world, child = robot base_link) — its child is robot-owned, so
    it stays robot-side.

    Returns ``False`` when ``robot_prefix`` is empty (ownership undecidable),
    so callers on the no-params / standalone path fall through to their
    name-based classification unchanged.

    Parameters
    ----------
    parent_body_label / child_body_label
        ``model.body_label[model.joint_parent[j]]`` /
        ``model.body_label[model.joint_child[j]]``.  Pass ``None`` for the
        world body (joint parent/child index ``< 0``).
    robot_prefix
        The active robot's USD prefix (e.g. ``"genie"``).
    """
    if not robot_prefix:
        return False
    return not (body_is_robot(parent_body_label, robot_prefix) or body_is_robot(child_body_label, robot_prefix))


# ---------------------------------------------------------------------------
# Default contact-compliance table (MuJoCo-style solref / solimp)
# ---------------------------------------------------------------------------
#
# Every class uses ``solref=(0.002, 1.0)`` — 2 ms time constant at
# critical damping.  The choice is driven by two MuJoCo-side facts that
# together fix the dampratio at exactly 1.0:
#
#   1. Stability floor: MJW's contact integrator requires
#      ``timeconst >= 2 * sub_dt``.  At ``physics_hz=100`` with
#      ``physics_solver_substep=10`` the sub-dt is 1 ms, so 2 ms is the
#      tightest stable timeconst.
#
#   2. Spring constant inversion: MJW recomputes contact stiffness from
#      ``shape_material_ke / kd`` each step via
#      ``ke = (1/(timeconst·dampratio))²`` (see ``mjc_contact.py``).
#      Holding timeconst constant and raising dampratio DROPS ke as the
#      square — at timeconst=2 ms, dampratio=1 gives ke=250 000 while
#      dampratio=5 gives ke=10 000 (25× softer spring).  Steady-state
#      penetration under a load F scales as ``F·(timeconst·dampratio)²``,
#      so an "overdamped" contact paradoxically sinks deeper, not less.
#
#   Critical damping (dampratio=1) is the unique sweet spot: maximum
#   spring stiffness at a given timeconst with no oscillation.  Below 1
#   the contact bounces; above 1 the spring weakens.
#
# MuJoCo's pair mix rule for solref is MAX of the two timeconsts (softer
# side wins), so every class has to be tight — leaving any one at MJ's
# stock ``(0.02, 1.0)`` drags the pair to 20 ms timeconst and ~4 mm
# steady-state penetration regardless of how tight the other side is.
#
# The reference G2 MJCF keeps the floor soft (``<geom name="floor"
# solref="0.02 1"/>``) for parity with upstream samples; we diverge
# here because the visible penetration is more confusing than the
# MJCF compatibility is useful.
#
# ``solimp`` is left at MuJoCo's stock ``(0.9, 0.95, 0.001, 0.5, 2.0)``
# for every class.
#
# Override per-class via the scene yaml ``newton.mjc_contact:`` block.
# Any class omitted there falls back to the value here.  This module is
# the in-code source of truth for every consumer.

_DEFAULT_SOLREF: tuple[float, float] = (0.002, 1.0)
# ⚠️ The RL path overrides the ROBOT class to ``solref = [0.002, 5.0]``
# (ke 250000 -> 10000, ζ=5), because at ke=250000 the gripper LAUNCHES 0.1 kg
# objects on contact (escape velocity v = δ·√(ke/m)); the softer value measured
# better on both eval and teleop feel.  This table stays at dampratio=1.0 — the
# analysis above (max spring stiffness at the sub_dt-bounded timeconst, hence
# minimum steady-state penetration) is what other consumers want, and every
# scene yaml that cares already overrides per class.  So: LOWEST penetration
# here, LOWEST flyaway on the RL robot side; the two goals genuinely pull
# opposite ways at fixed timeconst, and the split is a decision, not a drift.
_DEFAULT_SOLIMP: tuple[float, float, float, float, float] = (0.9, 0.95, 0.001, 0.5, 2.0)
# Friction: (tangential, torsional, rolling) — MuJoCo's geom_friction
# layout.  Defaults match MuJoCo stock:
#   tangential = 1.0   Coulomb sliding μ (along the contact tangent plane)
#   torsional  = 0.005 twisting friction at the contact patch (per radius)
#   rolling    = 0.0001 rolling friction (per radius)
#
# The tangential default of 1.0 alone is *not enough to grasp round
# objects*: a finger closing on a hanger rod with μ_torsional=0.005 lets
# the rod twist in the jaw; μ_rolling=0.0001 lets it roll out.  Override
# per-class via ``newton.mjc_contact.<kind>.friction`` — typical rubber
# tip on metal rod: ``[1.5, 0.1, 0.01]``.  MJW's pair mix rule for
# friction is the MAX of both surfaces' values, so bumping just the
# robot side is usually enough.
_DEFAULT_FRICTION: tuple[float, float, float] = (1.0, 0.005, 0.0001)

#: Upstream / shipped table.  Every consumer that does not opt into the
#: grasp-fidelity profile below gets exactly these values, which are
#: MuJoCo's own defaults — so an engine carrying this module behaves like
#: one without it.
MJC_CONTACT_DEFAULTS: dict[str, dict[str, tuple[float, ...]]] = {
    kind: {
        "solref": _DEFAULT_SOLREF,
        "solimp": _DEFAULT_SOLIMP,
        "friction": _DEFAULT_FRICTION,
    }
    for kind in (OK_ROBOT, OK_FLOOR, OK_PASSIVE, OK_STATIC_PROP, OK_OTHER)
}

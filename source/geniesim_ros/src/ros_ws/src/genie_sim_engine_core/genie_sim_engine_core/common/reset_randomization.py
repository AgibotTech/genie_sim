# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0
"""Seeded reset randomization shared by the Newton engine and the fused PhysX node.

A reset seed >= 0 asks for the physics-side generalization configured in
``scene_cfg.randomization``: object xy offsets, arm joint offsets and a robot
base xy offset. ``randomization.joint_pd`` separately carries per-env PD gain
multipliers drawn once at build time. Both engines take their draws from here,
so a seed means the same thing on either.

``draw_reset`` keeps the draw the Newton engine has always made -- same
defaults, same rejection sampling, same rng consumption order (objects, then
arm joints, then base). On top of it, when a per-instance ``layout`` is given
(object boxes and what carries each object, recorded when the instance's rest
state is baked), the object offsets are constrained:

* objects that were apart stay apart: no new box overlap, with a clearance;
* an object resting on or in another object, or touching it, moves with it
  (one rigid group, offset by its lowest member's draw);
* an object resting on a surface (the table) keeps its footprint on it, and
  never moves into a surface it does not rest on.

A group whose draw breaks a constraint is redrawn from a separate rng stream,
and left in place if no redraw fits, so the main stream -- and with it every
unconstrained object and the arm and base draws -- is unchanged.

Layout format (env-local metres, per instance)::

    {"objects":  {name: {"aabb": [x0, y0, z0, x1, y1, z1], "carrier": name | None}},
     "surfaces": {name: {"aabb": [x0, y0, z0, x1, y1, z1]}}}

``carrier`` names the object or surface directly underneath (or around) the
object, or None when it rests on something the layout does not know about.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

#: Newton's defaults. The object radius applies even when nothing is
#: configured: a seed >= 0 is the request.
DEFAULT_OBJECT_XY_RADIUS = 0.03
DEFAULT_EXCLUDE = ("table",)
DEFAULT_MIN_GAP = 0.05
MAX_TRIES = 20

#: Footprints keep this far inside the surface they rest on.
SURFACE_MARGIN = 0.005
#: Boxes that were apart keep at least this gap.
CLEARANCE = 0.002
#: Boxes closer than this are touching and move as one group.
CONTACT_TOL = 0.002
#: Redraws for a group whose draw breaks a constraint.
MAX_REDRAWS = 20

_OBJECTS_PREFIX = "World/Objects/"
_ELIGIBLE_KINDS = ("body", "arm", "arm_shoulder", "arm_mid", "arm_wrist", "head")


@dataclass
class ResetDraw:
    """One env's offsets for one reset.

    ``object_offsets`` only names objects that actually move. ``arm_offsets``
    lines up with the arm joints passed to ``draw_reset`` and is all zeros when
    arm randomization is off. ``base_offset`` is ``None`` when base
    randomization is off. ``constrained`` lists, per changed group, what the
    constraints did (``"grouped"``, ``"redrawn"`` or ``"kept in place"``).
    """

    object_offsets: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    arm_offsets: np.ndarray = field(default_factory=lambda: np.zeros(0))
    base_offset: Optional[Tuple[float, float]] = None
    constrained: List[str] = field(default_factory=list)


def _thresholds(spec: dict):
    radius = float(spec.get("object_xy_radius", DEFAULT_OBJECT_XY_RADIUS))
    jt = float(spec.get("init_joint_thresh", 0.0) or 0.0)
    bt = spec.get("init_base_xy_thresh")
    return radius, jt, bt


def is_active(spec: dict) -> bool:
    """Whether a seeded reset changes anything under ``spec``."""
    radius, jt, bt = _thresholds(spec or {})
    return bool(radius > 0 or jt > 0 or bt)


def object_name(rel_path: str) -> Optional[str]:
    """The task object a body belongs to (``[/]World/Objects/<name>/...``)."""
    head, sep, tail = rel_path.partition(_OBJECTS_PREFIX)
    if not sep or head not in ("", "/"):
        return None
    return tail.split("/", 1)[0] or None


def draw_reset(seed: int, spec: dict, objects: Sequence[Tuple[str, Tuple[float, float]]],
               n_arm_joints: int, *, has_base: bool, layout: Optional[dict] = None) -> ResetDraw:
    """Draw one env's reset offsets from ``seed``.

    ``objects`` is ``[(name, (x, y)), ...]`` in a fixed order, with each
    object's reset position; excluded objects may be included and are skipped
    here. ``n_arm_joints`` is how many arm joints get an offset, in the
    caller's order. ``has_base`` says whether the robot base can be moved at
    all. ``layout`` is the instance's layout (see the module docstring); the
    constraints apply when it is given, unless
    ``spec["object_constraints"]`` is false.
    """
    if seed < 0:
        raise ValueError(f"seed {seed} < 0 is the verbatim-replay sentinel; there is nothing to draw")
    spec = spec or {}
    draw = ResetDraw(arm_offsets=np.zeros(int(n_arm_joints)))
    radius, jt, bt = _thresholds(spec)
    if not (radius > 0 or jt > 0 or bt):
        return draw
    rng = np.random.default_rng(int(seed))

    # 1. Objects: uniform +-radius, rejecting a trial that lands closer to an
    #    already placed object than the pair's original distance (capped at
    #    min_gap). Excluded objects draw nothing.
    exclude = [str(x) for x in spec.get("exclude", list(DEFAULT_EXCLUDE))]
    min_gap = float(spec.get("min_gap", DEFAULT_MIN_GAP))
    cand = [(name, np.asarray(xy, dtype=np.float64)) for name, xy in objects if not any(x in name for x in exclude)]
    if radius > 0:
        placed: list = []
        orig = [c[1] for c in cand]
        for name, oxy in cand:
            new_xy = oxy
            for _ in range(MAX_TRIES):
                trial = oxy + rng.uniform(-radius, radius, 2)
                ok = True
                for j, pxy in enumerate(placed):
                    lim = min(float(np.linalg.norm(oxy - orig[j])), min_gap)
                    if float(np.linalg.norm(trial - pxy)) < lim:
                        ok = False
                        break
                if ok:
                    new_xy = trial
                    break
            placed.append(new_xy)
            d = new_xy - oxy
            if abs(d[0]) > 1e-12 or abs(d[1]) > 1e-12:
                draw.object_offsets[name] = (float(d[0]), float(d[1]))
        if layout and spec.get("object_constraints", True):
            draw.object_offsets, draw.constrained = constrain_object_offsets(
                draw.object_offsets, layout, [c[0] for c in cand], radius, int(seed))

    # 2. Arm joints: U(-jt, jt) radians each.
    if jt > 0:
        for k in range(int(n_arm_joints)):
            draw.arm_offsets[k] = float(rng.uniform(-jt, jt))

    # 3. Robot base: U(-x, x), U(-y, y) metres.
    if bt and has_base:
        tx, ty = float(bt[0]), float(bt[1])
        draw.base_offset = (float(rng.uniform(-tx, tx)), float(rng.uniform(-ty, ty)))
    return draw


# ---------------------------------------------------------------------------
# Layout constraints
# ---------------------------------------------------------------------------

def _box(entry) -> np.ndarray:
    return np.asarray(entry["aabb"], dtype=np.float64)


def _shift(box: np.ndarray, off) -> np.ndarray:
    return box + np.array([off[0], off[1], 0.0, off[0], off[1], 0.0])


def _overlap(a: np.ndarray, b: np.ndarray, gap: float) -> bool:
    """Boxes closer than ``gap`` on every axis (overlapping when gap = 0)."""
    return bool(np.all(a[:3] < b[3:] + gap) and np.all(b[:3] < a[3:] + gap))


def layout_groups(layout: dict, movable: Iterable[str]) -> List[List[str]]:
    """Movable objects that move together: carried by one another, or touching."""
    names = sorted(n for n in movable if n in (layout.get("objects") or {}))
    objs = layout["objects"]
    parent = {n: n for n in names}

    def find(n):
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for n in names:
        carrier = objs[n].get("carrier")
        if carrier in parent:
            union(n, carrier)
    boxes = {n: _box(objs[n]) for n in names}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if _overlap(boxes[a], boxes[b], CONTACT_TOL):
                union(a, b)
    groups: Dict[str, List[str]] = {}
    for n in names:
        groups.setdefault(find(n), []).append(n)
    return [groups[k] for k in sorted(groups)]


def constrain_object_offsets(offsets: Dict[str, Tuple[float, float]], layout: dict, movable: Sequence[str],
                             radius: float, seed: int):
    """Apply the layout constraints to plain object offsets.

    Returns ``(offsets, notes)``. Objects absent from the layout keep their
    plain offset; see the module docstring for the rules.
    """
    objs = layout.get("objects") or {}
    surfaces = {n: _box(s) for n, s in (layout.get("surfaces") or {}).items()}
    groups = layout_groups(layout, movable)
    in_layout = {n for g in groups for n in g}
    boxes = {n: _box(objs[n]) for n in in_layout}
    final: Dict[int, Tuple[float, float]] = {}
    notes: List[str] = []

    def base_of(group):
        roots = [n for n in group if objs[n].get("carrier") not in in_layout] or group
        return min(roots, key=lambda n: (boxes[n][2], n))

    def fits(gi, off) -> bool:
        for m in groups[gi]:
            box = _shift(boxes[m], off)
            carrier = objs[m].get("carrier")
            if carrier in surfaces:
                s = surfaces[carrier]
                inner = s + np.array([SURFACE_MARGIN, SURFACE_MARGIN, 0, -SURFACE_MARGIN, -SURFACE_MARGIN, 0])
                was_inside = bool(np.all(boxes[m][:2] >= inner[:2]) and np.all(boxes[m][3:5] <= inner[3:5]))
                if was_inside:
                    if not (np.all(box[:2] >= inner[:2]) and np.all(box[3:5] <= inner[3:5])):
                        return False
                else:
                    c = (box[:2] + box[3:5]) / 2
                    if not (np.all(c >= s[:2]) and np.all(c <= s[3:5])):
                        return False
            for sname, s in surfaces.items():
                if sname != carrier and _overlap(box, s, 0.0) and not _overlap(boxes[m], s, 0.0):
                    return False
            for gj, other in enumerate(groups):
                if gj == gi:
                    continue
                o_off = final.get(gj, (0.0, 0.0))
                for n in other:
                    if _overlap(box, _shift(boxes[n], o_off), CLEARANCE) and not _overlap(boxes[m], boxes[n], CLEARANCE):
                        return False
        return True

    for gi, group in enumerate(groups):
        base = base_of(group)
        off = offsets.get(base, (0.0, 0.0))
        changed = len(group) > 1 and any(offsets.get(m, (0.0, 0.0)) != off for m in group)
        note = "grouped" if changed else ""
        if not fits(gi, off):
            rng = np.random.default_rng([int(seed), 0x5EED, gi])
            off = (0.0, 0.0)
            note = "kept in place"
            for _ in range(MAX_REDRAWS):
                trial = tuple(float(v) for v in rng.uniform(-radius, radius, 2))
                if fits(gi, trial):
                    off, note = trial, "redrawn"
                    break
        final[gi] = off
        if note:
            notes.append(f"{'+'.join(group)}: {note}")

    out = {n: v for n, v in offsets.items() if n not in in_layout}
    for gi, group in enumerate(groups):
        off = final[gi]
        if abs(off[0]) > 1e-12 or abs(off[1]) > 1e-12:
            for m in group:
                out[m] = off
    return out, notes


# ---------------------------------------------------------------------------
# joint_pd
# ---------------------------------------------------------------------------

def joint_pd_eligible(joint_names: Sequence[str], classify: Callable[[str], str]) -> np.ndarray:
    """DOFs whose gains ``joint_pd`` may scale: arm, body and head.

    Grippers and the chassis keep their gains. ``classify`` is
    ``joint_classification.classify_joint_by_name``.
    """
    return np.array([classify(n) in _ELIGIBLE_KINDS for n in joint_names], dtype=bool)


def scale_joint_pd(jpd_map: dict, env_ids: Sequence[int], base_kp: np.ndarray, base_kd: np.ndarray,
                   eligible: np.ndarray):
    """Scale per-env gain rows from ``randomization.joint_pd``.

    ``jpd_map`` is ``{"<env>": {"kp_scale": .., "kd_scale": ..}}``, already
    drawn at build time. ``base_kp``/``base_kd`` are ``(num_envs, num_dofs)``
    gains to scale from (never the scaled ones, so a second call does not
    compound). Returns ``(kp, kd, requested, applied, touched)`` with the
    Newton engine's meaning: ``kp``/``kd`` are ``None`` when no env needs a
    real (non-identity) scale; ``requested`` counts envs with an entry;
    ``applied`` counts envs that were scaled or whose entry is the identity;
    ``touched`` lists the envs whose rows changed.
    """
    entries = [(e, jpd_map.get(str(e))) for e in env_ids]
    entries = [(e, entry) for e, entry in entries if entry]
    if not entries:
        return None, None, 0, 0, []
    eligible = np.asarray(eligible, dtype=bool)
    kp = kd = None
    applied = 0
    touched: list = []
    for e, entry in entries:
        kp_scale = float(entry.get("kp_scale", 1.0))
        kd_scale = float(entry.get("kd_scale", 1.0))
        if kp_scale == 1.0 and kd_scale == 1.0:
            applied += 1
            continue
        if kp is None:
            kp, kd = np.array(base_kp, copy=True), np.array(base_kd, copy=True)
        if e >= len(base_kp) or not eligible.any():
            continue
        kp[e] = np.where(eligible, base_kp[e] * kp_scale, kp[e])
        kd[e] = np.where(eligible, base_kd[e] * kd_scale, kd[e])
        touched.append(e)
        applied += 1
    return kp, kd, len(entries), applied, touched

# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Shared mimic-joint utilities for both physics-engine paths.

Newton's articulation does not honor multi-finger gripper ``<mimic>`` tags
the way PhysX does (PhysX has ``PhysxMimicJointAPI`` enforced at the
constraint level; Newton's wrapper has ``NewtonMimicAPI`` declared in
USD but its ``apply_action`` still requires every follower DOF to be
explicitly commanded). Both ``IsaacNewtonEngine`` and
``NewtonStandaloneEngine`` therefore have to broadcast a master target
across its followers in software.

The two engines diverged on this for a while:

  * isaac_newton (``runtime.stage._apply_joint_commands``) parsed
    ``NewtonMimicAPI`` from the staged USD.
  * newton-standalone (``engine.newton_standalone.topology._build_mimic_map``)
    re-parsed ``robot.urdf`` from disk via ``xml.etree``.

This module unifies both on the USD path. ``robot.urdf`` is upstream of
``assemble_robot`` and may not exist at runtime; the USD on the live
stage IS the source of truth (``NewtonMimicAPI`` is what the importer
authors and what the Newton wrapper actually reads).

The data shape is unchanged so call sites only need a one-line swap:

    {master_joint_name: [(follower_name, coef1, coef0), …]}

with the relation ``follower_q = coef0 + coef1 · master_q`` (matches
both Newton's USD schema and URDF's ``multiplier``/``offset`` — coef1
IS the URDF multiplier, coef0 IS the URDF offset).
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Tuple


def parse_mimic(stage, logger) -> Dict[str, List[Tuple[str, float, float]]]:
    """Read ``NewtonMimicAPI`` from every joint prim on a USD stage.

    Returns ``{master_name: [(follower_name, coef1, coef0), …]}``. The
    URDF→USD importer authors ``newton:mimicJoint`` (rel pointing at the
    master prim), ``newton:mimicCoef1`` (multiplier; default 1.0 — when
    the URDF mimic was ``+1`` the importer omits the attribute), and
    ``newton:mimicCoef0`` (offset; default 0.0).

    Logs each master→followers chain for visibility.
    """
    from pxr import UsdPhysics

    followers: Dict[str, List[Tuple[str, float, float]]] = {}
    for prim in stage.Traverse():
        if not (prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint)):
            continue
        rel = prim.GetRelationship("newton:mimicJoint")
        phys = False
        if not rel or not rel.HasAuthoredTargets():
            # Fallback: some robot USDs carry PhysX's ``PhysxMimicJointAPI:rotX``
            # (built for the PhysX path — a hard constraint) but NO
            # ``NewtonMimicAPI``. Same data: ``referenceJoint`` = master,
            # ``gearing``/``offset`` = coef1/coef0 (schema defaults 1.0 / 0.0).
            # Reading it here is what lets Newton's manual broadcast couple e.g.
            # the omnipicker gripper (inner_joint1 mimics outer_joint1) — without
            # it the mimic map is empty and the gripper never closes.
            rel = prim.GetRelationship("physxMimicJoint:rotX:referenceJoint")
            phys = True
            if not rel or not rel.HasAuthoredTargets():
                continue
        targets = rel.GetTargets()
        if not targets:
            continue
        master_name = targets[0].name  # last segment of the prim path
        coef1 = 1.0
        coef0 = 0.0
        c1_attr = "physxMimicJoint:rotX:gearing" if phys else "newton:mimicCoef1"
        c0_attr = "physxMimicJoint:rotX:offset" if phys else "newton:mimicCoef0"
        a1 = prim.GetAttribute(c1_attr)
        if a1 and a1.HasAuthoredValue():
            try:
                coef1 = float(a1.Get())
            except (TypeError, ValueError):
                pass
        a0 = prim.GetAttribute(c0_attr)
        if a0 and a0.HasAuthoredValue():
            try:
                coef0 = float(a0.Get())
            except (TypeError, ValueError):
                pass
        if phys:
            # PhysX's mimic constraint is q_follower = -gearing*q_ref - offset
            # (opposite sign to Newton's coef convention). Negate so the broadcast
            # AGREES with what the physical linkage produces (empirically the
            # gripper's inner_joint tracks -outer_joint) instead of fighting it.
            coef1, coef0 = -coef1, -coef0
        followers.setdefault(master_name, []).append((prim.GetName(), coef1, coef0))
    if followers:
        for master, foll in followers.items():
            logger.info(f"mimic: {master} → " + ", ".join(f"{n}({c1:+g}·q{c0:+g})" for n, c1, c0 in foll))
    return followers


def expand_targets(
    cmd_positions: Mapping[str, float],
    mimic_followers: Dict[str, List[Tuple[str, float, float]]],
) -> Dict[str, float]:
    """Compute follower targets implied by the master commands.

    Returns ONLY the new follower entries (does NOT echo masters back),
    so callers can decide how to merge — ``stage.py`` does
    ``{**cmd_positions, **expand_targets(cmd_positions, mimic)}``;
    ``newton_standalone.control`` walks the result and writes via DOF index.

    Idempotent: if the caller already includes a follower in
    ``cmd_positions``, the mimic-derived value will overwrite the
    explicit one in the merged dict — which is the correct behaviour
    (the master's command IS the canonical source for the follower).
    """
    if not (mimic_followers and cmd_positions):
        return {}
    extra: Dict[str, float] = {}
    for name, val in cmd_positions.items():
        for fname, mult, off in mimic_followers.get(name, ()):
            extra[fname] = mult * float(val) + off
    return extra

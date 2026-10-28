#!/usr/bin/env python3

# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Scene-YAML readers shared between assemble_scene and genie_sim_engine.

The ``manifest.json`` written by ``assemble_scene`` is intentionally limited
to **static** fields — the ones that determine which assets get loaded and
where prims live in USD namespace (``scene_usda``, ``robot_usda``,
``robot_prefix``, etc.). Anything that's a runtime *behavior* —
``pin_base_to_world``, ``pin_base_mode``, ``convert_joints_to_fixed``, init joint poses,
overrides — is re-read live from the scene YAML at engine startup so the
operator can tweak behavior without invalidating the bake cache.

The canonical typed reader lives in
:mod:`genie_sim_engine_core.schema.scene`; the two functions here are
thin BC shims retained so callers who imported them from this module
before the schema landed keep working. New code should read
:class:`~genie_sim_engine_core.schema.scene.RobotSource` attributes
directly.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..schema.scene import RobotSource as _SchemaRobotSource


def parse_pin_base_to_world(robot_section: Dict[str, Any]) -> bool:
    """Read ``robot.robot_source.pin_base_to_world``.  Defaults to ``False``.

    Controls whether the base is HELD to world at all.  It does not by itself
    say *how*: with the default mode the URDF→USD ``root_joint`` FixedJoint is
    kept (base_link welded), but under ``pin_base_mode: mocap`` the weld is
    deactivated too and the solver clamps the free root joint instead.  See
    :class:`~genie_sim_engine_core.schema.scene.PinBaseMode`.

    Default is **False**: a scene that wants the base held says so
    explicitly, as every arm-only and stationary scene does; mobile-robot
    scenes leave it off so wheel-on-floor contact drives the base.
    """
    source = robot_section.get("robot_source") if isinstance(robot_section, dict) else None
    if not isinstance(source, dict):
        return False
    return _SchemaRobotSource.from_dict(source).pin_base_to_world


def parse_pin_base_mode(robot_section: Dict[str, Any]) -> str:
    """Read ``robot.robot_source.pin_base_mode``.  Defaults to ``""``.

    Selects HOW a pinned base is held.  The vocabulary, the default, and the
    rule that pairs this with ``pin_base_to_world`` are declared once in
    :class:`~genie_sim_engine_core.schema.scene.PinBaseMode` — read that
    rather than restating it here.  Returns the raw (normalized) field; call
    ``PinBaseMode.resolve(pin_base_to_world, mode)`` to get the resolved mode,
    which is what every consumer actually routes on.
    """
    source = robot_section.get("robot_source") if isinstance(robot_section, dict) else None
    if not isinstance(source, dict):
        return ""
    return _SchemaRobotSource.from_dict(source).pin_base_mode


def parse_convert_joints_to_fixed(robot_section: Dict[str, Any]) -> List[str]:
    """Read ``robot.robot_source.convert_joints_to_fixed``.

    Returns a normalized list of sub-tree names whose articulated joints
    should be REPLACED with ``UsdPhysics.FixedJoint`` at runtime to
    shrink Featherstone's mass matrix (or trim mjwarp's actuator
    count).  Supported tokens:

      * ``"base"``  — joints whose name contains ``chassis``
      * ``"head"``  — joints whose name contains ``head_joint``
      * ``"body"``  — joints whose name contains ``body_joint`` (G2's
                      5-DOF torso chain)

    Accepts a YAML list or a single string.  Defaults to an empty list.
    """
    source = robot_section.get("robot_source") if isinstance(robot_section, dict) else None
    if not isinstance(source, dict):
        return []
    return list(_SchemaRobotSource.from_dict(source).convert_joints_to_fixed)

# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Shared USD-walk helpers used by every backend's assemble stage.

The v2 data-exchange protocol (DESIGN.DATA_EXCHANGE_PROTOCOL.md) requires
each stage dir to carry a ``robot.body_paths`` and ``scene.body_paths``
table so the renderer can resolve short link names / MJCF body names to
their absolute USD prim paths. Both tables are produced by walking a USD
file and matching the same set of link names against it; the walk is
identical, only the input file and name-set differ.

``build_body_paths`` is the single implementation both robot and scene
maps go through. Any backend can call it: the MuJoCo assemble reads
MJCF body names and matches them against ``robot.usda`` and
``scene.usda`` separately; the Newton assemble hands the same helper
the abs-prim-path list its state carries.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


def build_body_paths(
    usda_path: Path,
    body_names: Iterable[str],
    *,
    logger=None,
) -> Dict[str, str]:
    """Map body names to their absolute USD prim path in ``usda_path``.

    Walks the USD once, collecting every ``UsdGeom.Xform`` whose leaf
    name is in ``body_names``. Tiebreaker for ambiguous leaves (multiple
    prims share the same short name — common when URDF-Converter
    authored nested ``base_link/base_link/…`` composites):

    1. Prefer the prim with ``PhysicsRigidBodyAPI`` applied.
    2. Otherwise, prefer the longest path (innermost — matches how the
       physics engine actually interprets the hierarchy).
    3. If two candidates tie on both rules, log a WARN listing all
       candidates and drop the name from the result. A missing entry
       fails loudly downstream rather than silently mapping to the
       wrong prim.

    Absent USDA files or names with no candidates are silently omitted
    — callers that care about coverage should compare
    ``result.keys()`` against the input set and warn themselves.

    ``logger`` is optional; when provided it must have ``.warn`` and
    ``.info`` methods matching the ROS logger interface. When absent,
    warnings go to stderr via ``print``.
    """
    from pxr import Usd, UsdGeom, UsdPhysics

    name_set: Set[str] = set(body_names)
    if not name_set:
        return {}

    if not usda_path.is_file():
        _warn(logger, f"build_body_paths: USDA not found/openable: {usda_path}")
        return {}
    stage = Usd.Stage.Open(str(usda_path))
    if stage is None:
        _warn(logger, f"build_body_paths: USDA not found/openable: {usda_path}")
        return {}

    candidates: Dict[str, List[Tuple[str, bool]]] = {}
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Xform):
            continue
        leaf = prim.GetName()
        if leaf not in name_set:
            continue
        has_rb_api = prim.HasAPI(UsdPhysics.RigidBodyAPI)
        candidates.setdefault(leaf, []).append((str(prim.GetPath()), has_rb_api))

    result: Dict[str, str] = {}
    for name, options in candidates.items():
        if len(options) == 1:
            result[name] = options[0][0]
            continue
        chosen = _resolve_tiebreaker(name, options, usda_path, logger=logger)
        if chosen is not None:
            result[name] = chosen
    return result


def _resolve_tiebreaker(
    name: str,
    options: List[Tuple[str, bool]],
    usda_path: Path,
    *,
    logger,
) -> Optional[str]:
    with_api = [o for o in options if o[1]]
    if len(with_api) == 1:
        return with_api[0][0]
    ranked = with_api if with_api else options
    # Longest path first — the innermost prim in a nested composite is
    # what the physics engine's body table actually references. Length
    # is measured in path segments (count of '/') to match how the URDF
    # nesting depth accumulates.
    ranked = sorted(ranked, key=lambda o: (-o[0].count("/"), o[0]))
    top = ranked[0]
    if len(ranked) > 1 and ranked[0][0].count("/") == ranked[1][0].count("/"):
        _warn(
            logger,
            f"build_body_paths: ambiguous leaf {name!r} in {usda_path} — "
            f"candidates {[o[0] for o in ranked]}; dropping (add "
            f"PhysicsRigidBodyAPI to the intended prim to disambiguate)",
        )
        return None
    _info(
        logger,
        f"build_body_paths: {name!r} resolved to {top[0]} "
        f"(picked from {len(ranked)} candidates by "
        f"{'PhysicsRigidBodyAPI + ' if with_api else ''}longest-path)",
    )
    return top[0]


def _warn(logger, msg: str) -> None:
    if logger is not None and hasattr(logger, "warn"):
        logger.warn(msg)
    else:
        import sys

        print(f"[usd_walk] WARNING: {msg}", file=sys.stderr)


def _info(logger, msg: str) -> None:
    if logger is not None and hasattr(logger, "info"):
        logger.info(msg)

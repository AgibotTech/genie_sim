# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""World AABBs for STATIC scene props, for the rlinf eval checkers.

Why this exists
---------------
Checkers like ``Inside`` / ``InBBox`` ask for the bounding box of a
*passive* object — typically a background prop such as a storage bin::

    "Inside": "benchmark_beverage_bottle_35b3d04a|/World/storage_bin|1.0"

rlinf answers box queries out of ``get_body_aabbs()``, which is indexed by
Newton **rigid body**. Background props are static scene geometry and have
no rigid body, so that lookup misses and the shim returns an all-zero box.
A zero box contains no point, so the checker can never pass and the task
scores 0% no matter how well the policy performs — silently, because
backend AABBs are documented as optional (``Follow`` / ``PickUpOnGripper``
compute their boxes from the pose matrix and never need them).

These props never move, so their world AABB is a build-time constant.
This module computes it once from the composed USD stage; the caller
publishes it latched (see ``visualizers/markers.publish_static_prop_aabb``)
and rlinf's shim falls back to it when the body-index lookup misses.

Path dialect
------------
Checkers name props in the ``/World/<name>`` dialect, but the composed
stage puts background props under ``/Background/<name>``. Paths are
therefore resolved by exact match FIRST and by trailing prim name second
(shallowest match wins; ties are refused rather than guessed — handing a
checker the wrong prop's box would produce plausible-looking but wrong
scores, which is harder to catch than no box at all).

The returned names are the paths as REQUESTED, not as resolved, so the
consumer can look them up with the string the checker already emits.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = ["compute_static_prop_aabbs"]


def _tail(path: str) -> str:
    return str(path).rstrip("/").rsplit("/", 1)[-1]


def _build_tail_index(stage: Any) -> Dict[str, List[str]]:
    """Map trailing prim name -> every stage path ending in that name."""
    index: Dict[str, List[str]] = {}
    for prim in stage.Traverse():
        index.setdefault(prim.GetName(), []).append(str(prim.GetPath()))
    return index


def _resolve(stage: Any, want: str, index: Dict[str, List[str]],
             logger: Optional[Any]) -> Optional[Any]:
    """Resolve one requested path to a prim, or None (having warned)."""
    prim = stage.GetPrimAtPath(want)
    if prim and prim.IsValid():
        return prim

    candidates = index.get(_tail(want), [])
    if not candidates:
        _warn(logger, f"static prop {want!r} not found on the stage "
                      f"(no prim named {_tail(want)!r}) — checkers "
                      f"referencing it will get a zero box")
        return None

    # Shallowest wins: '/Background/storage_bin' beats the mesh prims nested
    # under it ('/Background/storage_bin/entity/body/...'), which share the
    # bin's bound anyway but are an arbitrary pick among siblings.
    depth = min(p.count("/") for p in candidates)
    shallowest = sorted(p for p in candidates if p.count("/") == depth)
    if len(shallowest) > 1:
        _warn(logger, f"static prop {want!r} is ambiguous — {len(shallowest)} "
                      f"prims share the name {_tail(want)!r} at the same depth "
                      f"({', '.join(shallowest[:4])}); refusing to guess, "
                      f"checkers referencing it will get a zero box")
        return None
    return stage.GetPrimAtPath(shallowest[0])


def _warn(logger: Optional[Any], msg: str) -> None:
    text = f"[static-props] {msg}"
    if logger is not None:
        logger.warn(text)
    else:
        print(text, flush=True)


def compute_static_prop_aabbs(
    stage: Any,
    requested: Sequence[str],
    logger: Optional[Any] = None,
) -> Tuple[List[str], List[List[float]]]:
    """Compute world AABBs for ``requested`` static prop paths.

    Args:
        stage: the composed USD stage (``sim.stage``).
        requested: prim paths in checker dialect, e.g. ``/World/storage_bin``.
        logger: optional ROS logger; ``print`` is used when absent.

    Returns:
        ``(names, boxes)`` where ``names[i]`` is the path as REQUESTED and
        ``boxes[i]`` is ``[min_x, min_y, min_z, max_x, max_y, max_z]`` in
        world coordinates — the same min/max convention
        ``get_body_aabbs()`` returns, so both feed the shim unchanged.

    Props that cannot be resolved, or whose bound is empty, are skipped
    (with a warning) rather than emitted as a zero box: a missing entry
    lets the shim keep its own "unresolved" warning, while a zero entry
    would look like a real answer.

    Never raises — a prop box is a nice-to-have for eval, never worth
    failing engine startup over.
    """
    names: List[str] = []
    boxes: List[List[float]] = []
    wanted = [str(w) for w in (requested or ()) if str(w).strip()]
    if not wanted:
        return names, boxes

    try:
        from pxr import Usd, UsdGeom  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        _warn(logger, f"pxr unavailable ({exc!r}); no prop boxes published")
        return names, boxes

    try:
        # 'default' + 'render' purposes: props are authored as render geometry;
        # 'guide'/'proxy' would drag in helper geometry that isn't the prop.
        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        )
        index = _build_tail_index(stage)
    except Exception as exc:  # noqa: BLE001
        _warn(logger, f"stage traversal failed ({exc!r}); no prop boxes published")
        return names, boxes

    for want in wanted:
        try:
            prim = _resolve(stage, want, index, logger)
            if prim is None:
                continue
            rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            if rng.IsEmpty():
                _warn(logger, f"static prop {want!r} has an empty bound "
                              f"(no renderable geometry?) — skipped")
                continue
            lo, hi = rng.GetMin(), rng.GetMax()
            names.append(want)
            boxes.append([float(lo[0]), float(lo[1]), float(lo[2]),
                          float(hi[0]), float(hi[1]), float(hi[2])])
        except Exception as exc:  # noqa: BLE001
            _warn(logger, f"static prop {want!r} failed ({exc!r}) — skipped")

    return names, boxes

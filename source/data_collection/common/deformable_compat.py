# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0
"""Isaac Sim 6 deformable schema and tensor-view compatibility helpers.

This module deliberately has no Isaac/pxr imports so hierarchy decisions can be
unit tested without starting SimulationApp.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, Literal, Mapping, NamedTuple


SURFACE_SIM_SCHEMA = "OmniPhysicsSurfaceDeformableSimAPI"
VOLUME_SIM_SCHEMA = "OmniPhysicsVolumeDeformableSimAPI"

DEFORMABLE_BODY_SCHEMA_NAMES = (
    "OmniPhysicsDeformableBodyAPI",
    "PhysxAutoDeformableBodyAPI",
    # Legacy assets still need cleanup during replay. Selection remains based
    # on applied-schema strings and never imports a removed Python API class.
    "PhysxSurfaceDeformableBodyAPI",
    "PhysxDeformableBodyAPI",
)

DEFORMABLE_SCHEMA_NAMES = (
    "OmniPhysicsDeformableBodyAPI",
    SURFACE_SIM_SCHEMA,
    VOLUME_SIM_SCHEMA,
    "OmniPhysicsSurfaceDeformableMaterialAPI",
    "OmniPhysicsVolumeDeformableMaterialAPI",
    "OmniPhysicsDeformableMaterialAPI",
    "PhysxAutoDeformableBodyAPI",
    "PhysxSurfaceDeformableBodyAPI",
    "PhysxDeformableBodyAPI",
    "PhysxDeformableSurfaceAPI",
    "PhysxAutoAttachmentAPI",
)

COLLISION_SCHEMA_NAMES = (
    "PhysicsCollisionAPI",
    "PhysxCollisionAPI",
)

# Older surface recordings can put this API directly on the mesh without a
# body root. Recognize it for replay cleanup, not Isaac 6 tensor-view creation.
REPLAY_ROOT_SCHEMA_NAMES = DEFORMABLE_BODY_SCHEMA_NAMES + (
    "PhysxDeformableSurfaceAPI",
)


class DeformableBodySpec(NamedTuple):
    root_path: str
    simulation_path: str
    kind: Literal["surface", "volume"]


PrimSchemas = tuple[str, Sequence[str]]


def _is_same_or_descendant(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip("/") + "/")


def select_deformable_bodies(
    prims: Iterable[PrimSchemas],
) -> list[DeformableBodySpec]:
    """Resolve each applied-schema body hierarchy to one tensor-view spec."""
    records = [(str(path), frozenset(schemas)) for path, schemas in prims]
    body_roots = sorted(
        {
            path
            for path, schemas in records
            if schemas.intersection(DEFORMABLE_BODY_SCHEMA_NAMES)
        },
        key=lambda path: (path.count("/"), path),
    )
    specs: list[DeformableBodySpec] = []
    for root in body_roots:
        # A nested body owns its own simulation subtree, not its ancestor's.
        nested_roots = [
            candidate
            for candidate in body_roots
            if candidate != root and _is_same_or_descendant(candidate, root)
        ]
        candidates: list[tuple[int, str, Literal["surface", "volume"]]] = []
        for path, schemas in records:
            if not _is_same_or_descendant(path, root):
                continue
            if any(_is_same_or_descendant(path, nested) for nested in nested_roots):
                continue
            if SURFACE_SIM_SCHEMA in schemas:
                candidates.append((path.count("/") - root.count("/"), path, "surface"))
            if VOLUME_SIM_SCHEMA in schemas:
                candidates.append((path.count("/") - root.count("/"), path, "volume"))
        if not candidates:
            continue
        candidates.sort()
        _, simulation_path, kind = candidates[0]
        specs.append(DeformableBodySpec(root, simulation_path, kind))
    return specs


def create_deformable_body_view(simulation_view: Any, spec: DeformableBodySpec):
    """Create the Isaac 6 tensor view on the body root selected by ``spec``."""
    root_path, _, kind = spec
    if kind == "surface":
        return simulation_view.create_surface_deformable_body_view(root_path)
    if kind == "volume":
        return simulation_view.create_volume_deformable_body_view(root_path)
    raise ValueError(f"unsupported deformable kind: {kind!r}")


def paths_in_deformable_subtrees(
    prims: Iterable[PrimSchemas], entry_paths: Iterable[str] | None = None
) -> set[str]:
    """Return prim paths whose collision schemas are safe to remove for replay."""
    records = [(str(path), frozenset(schemas)) for path, schemas in prims]
    roots = {
        path
        for path, schemas in records
        if schemas.intersection(REPLAY_ROOT_SCHEMA_NAMES)
    }
    if entry_paths is not None:
        entries = tuple(str(path) for path in entry_paths)
        roots = {
            root
            for root in roots
            if any(
                _is_same_or_descendant(entry, root)
                or _is_same_or_descendant(root, entry)
                for entry in entries
            )
        }
    return {
        path
        for path, _ in records
        if any(_is_same_or_descendant(path, root) for root in roots)
    }


def unique_views_by_identity(views: Mapping[str, Any]) -> list[tuple[str, Any]]:
    """Keep the first (canonical) alias for each physical tensor view."""
    result = []
    seen: set[int] = set()
    for path, view in views.items():
        identity = id(view)
        if identity in seen:
            continue
        seen.add(identity)
        result.append((path, view))
    return result

# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Single-world episode snapshots using Isaac's live physics tensor views."""

from __future__ import annotations

import numpy as np


def physics_paths(stage):
    """Resolve exact USD paths: PhysX wildcard matching does not recurse through ancestors."""
    from pxr import Usd, UsdPhysics

    roots, bodies = [], []
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            if UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get() is not False:
                bodies.append(str(prim.GetPath()))
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            descendants = [
                p for p in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()) if p.HasAPI(UsdPhysics.RigidBodyAPI)
            ]
            if descendants and all(
                UsdPhysics.RigidBodyAPI(p).GetRigidBodyEnabledAttr().Get() is False for p in descendants
            ):
                continue
            enabled = prim.GetAttribute("physxArticulation:articulationEnabled")
            if enabled and enabled.Get() is False:
                continue
            roots.append(str(prim.GetPath()))
    return roots, bodies


def validate_reset(env_ids, seeds, instances):
    """Reject unsupported requests before any state writes, rather than acknowledge a no-op."""
    selected = {0} if env_ids is None else set(env_ids)
    if selected - {0}:
        raise ValueError("Isaac Kit backends contain only env 0")
    if any(value >= 0 for env, value in (seeds or {}).items() if env in selected):
        raise NotImplementedError("Isaac Kit episode reset does not implement seeded randomization")
    if any(value >= 0 for env, value in (instances or {}).items() if env in selected):
        raise NotImplementedError("Isaac Kit episode reset does not implement instance switching")
    return bool(selected)


class EpisodeSnapshot:
    """Capture every live articulation and non-articulated body before the first engine step.

    Use the tensor inventory, including its actual row/DOF order. USD poses are
    stale under Newton; World.reset also recreates handles and loses runtime gains.
    Views retain their buffers, so every cached array must be an owned copy.
    """

    def __init__(self, simulation_view, to_backend, readback, robot_view, hold_targets, articulation_paths, body_paths):
        self._to_backend = to_backend
        self._articulations = []
        inventory = simulation_view.create_articulation_view(articulation_paths)
        if inventory is None:
            raise RuntimeError(f"Isaac reset could not resolve articulation paths: {articulation_paths}")
        links = {path for row in inventory.link_paths for path in row}
        for path in inventory.prim_paths:
            view = simulation_view.create_articulation_view(path)
            roots = readback(view.get_root_transforms()).copy()
            joints = readback(view.get_dof_positions()).copy()
            targets = readback(view.get_dof_position_targets()).copy()
            if path in robot_view.prim_paths and hold_targets is not None:
                # Initial pose setters can have delayed readback. Reuse the
                # exact requested array, including the resolved mimic followers.
                if joints.shape != (1, len(hold_targets)):
                    raise RuntimeError("Isaac reset robot DOF inventory differs from initial hold targets")
                joints[0] = hold_targets
                targets[0] = hold_targets
            self._articulations.append((view, roots, joints, targets))

        free_paths = [path for path in body_paths if path not in links]
        self._bodies = simulation_view.create_rigid_body_view(free_paths) if free_paths else None
        self._body_poses = readback(self._bodies.get_transforms()).copy() if self._bodies is not None else None

    def restore(self):
        indices = self._to_backend(np.array([0], dtype=np.int32))
        for view, roots, joints, targets in self._articulations:
            zero = self._to_backend(np.zeros_like(joints))
            view.set_root_transforms(self._to_backend(roots.copy()), indices)
            view.set_root_velocities(self._to_backend(np.zeros((1, 6), dtype=np.float32)), indices)
            view.set_dof_positions(self._to_backend(joints.copy()), indices)
            view.set_dof_velocities(zero, indices)
            view.set_dof_actuation_forces(zero, indices)
            view.set_dof_position_targets(self._to_backend(targets.copy()), indices)
            view.set_dof_velocity_targets(zero, indices)
        if self._bodies is not None:
            indices = self._to_backend(np.arange(len(self._body_poses), dtype=np.int32))
            self._bodies.set_transforms(self._to_backend(self._body_poses.copy()), indices)
            self._bodies.set_velocities(
                self._to_backend(np.zeros((len(self._body_poses), 6), dtype=np.float32)), indices
            )

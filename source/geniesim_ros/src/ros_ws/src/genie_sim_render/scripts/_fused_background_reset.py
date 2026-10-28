# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Episode reset and live joint readback for background articulations.

Foreground body tables only cover World/Objects. Background mechanisms keep
their own startup state, read from PhysX immediately after initialization and
before camera updates or simulation steps. They do not add robot action DOFs.
"""
from __future__ import annotations

import numpy as np
from pxr import Usd, UsdPhysics


def _array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=np.float32).copy()


class BackgroundArticulations:
    def __init__(self, stage, num_envs):
        from isaacsim.core.prims import Articulation

        groups = {}
        for env in range(num_envs):
            prefix = f"/World/envs/env_{env}/"
            # compose_background_into_scene puts the room below the
            # foreground's /World; GridCloner references that at env_i/World.
            # Explicit/per-env background references instead use env_i/Background.
            for relative_root in ("World/Background", "Background"):
                background = stage.GetPrimAtPath(prefix + relative_root)
                if not background:
                    continue
                for prim in Usd.PrimRange(background, Usd.TraverseInstanceProxies()):
                    if not prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                        continue
                    enabled = prim.GetAttribute("physxArticulation:articulationEnabled")
                    if enabled and enabled.Get() is False:
                        continue
                    # Rooms can freeze an articulated prop by overriding all
                    # its bodies (pour_workpiece's laptops) while retaining
                    # the asset's root API. PhysX creates no articulation for
                    # it. Only skip this proven static case; a missing tensor
                    # handle for a live mechanism must still fail loudly.
                    bodies = [p for p in Usd.PrimRange(prim, Usd.TraverseInstanceProxies())
                              if p.HasAPI(UsdPhysics.RigidBodyAPI)]
                    if bodies and all(UsdPhysics.RigidBodyAPI(p).GetRigidBodyEnabledAttr().Get()
                                      is False for p in bodies):
                        continue
                    path = str(prim.GetPath())
                    groups.setdefault(path[len(prefix):], {})[path] = env

        self._states = []
        self.joint_paths = []
        self._joint_slices = []
        for paths in groups.values():
            view = Articulation(list(paths), reset_xform_properties=False)
            view.initialize()
            if view.count != len(paths) or set(view.prim_paths) != set(paths):
                raise RuntimeError("fused: background articulation view did not match its USD roots")
            # Use PhysX's actual row order, which need not match input paths.
            envs = np.array([paths[p] for p in view.prim_paths], dtype=np.int64)
            # Isaac's initialized _dof_paths comes from the PhysX tensor
            # view, in the SAME column order as get_joint_positions(). Bare
            # dof_names cannot identify an owning object for the eval shim.
            relative_paths = []
            for env, row in zip(envs, view._dof_paths):
                prefix = f"/World/envs/env_{env}/"
                if any(not p.startswith(prefix) for p in row):
                    raise RuntimeError("fused: background joint path belongs to another env")
                relative_paths.append([p[len(prefix):] for p in row])
            if (len(relative_paths) != view.count
                    or any(row != relative_paths[0] for row in relative_paths)
                    or len(relative_paths[0]) != len(view.dof_names)):
                raise RuntimeError("fused: background joint paths do not align across envs")
            begin = len(self.joint_paths)
            self.joint_paths.extend(relative_paths[0])
            self._joint_slices.append(slice(begin, len(self.joint_paths)))
            positions, orientations = view.get_world_poses()
            actions = view.get_applied_actions()
            self._states.append((view, envs, {
                "positions": _array(positions),
                "orientations": _array(orientations),
                "joints": _array(view.get_joint_positions()),
                "position_targets": _array(actions.joint_positions),
                "velocity_targets": _array(actions.joint_velocities),
            }))

    def read_joint_positions(self, out):
        """Fill (env, scene DOF), in joint_paths order; absent envs are NaN."""
        out.fill(np.nan)
        for (view, envs, _), columns in zip(self._states, self._joint_slices):
            out[envs, columns] = _array(view.get_joint_positions())

    def reset(self, env_ids, to_backend):
        for view, envs, initial in self._states:
            rows = np.flatnonzero(np.isin(envs, env_ids))
            if not len(rows):
                continue
            indices = to_backend(rows, long=True)
            view.set_world_poses(to_backend(initial["positions"][rows]),
                                 to_backend(initial["orientations"][rows]), indices=indices)
            view.set_velocities(to_backend(np.zeros((len(rows), 6), dtype=np.float32)), indices=indices)
            view.set_joint_positions(to_backend(initial["joints"][rows]), indices=indices)
            zero = to_backend(np.zeros_like(initial["joints"][rows]))
            view.set_joint_velocities(zero, indices=indices)
            view.set_joint_efforts(zero, indices=indices)
            view.set_joint_position_targets(to_backend(initial["position_targets"][rows]), indices=indices)
            view.set_joint_velocity_targets(to_backend(initial["velocity_targets"][rows]), indices=indices)

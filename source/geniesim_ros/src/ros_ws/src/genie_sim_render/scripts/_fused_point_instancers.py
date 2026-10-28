# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Rigid PointInstancer compatibility for the fused GPU/Fabric backend.

The installed omni.physx.fabric does not update rigid PointInstancer instances.
PhysX simulates them, but their visible poses stay frozen; toggling USD writeback
also fails with GPU dynamics. Expand only rigid prototypes into referenced prims
on the runtime stage. Ordinary rigid bodies have working tensor/Fabric updates.
Keep these bodies out of the task schema and reset them through a separate view.

This covers authored rigid-root prototypes such as scoop_popcorn. Animated PI
arrays and material bindings authored on the PI itself are not migrated.
"""

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics


BODY_MARKER = "rlinf:pointInstancerBody"


class PointInstancerRigids:
    def __init__(self, stage):
        self.paths = []
        env_ids, positions, orientations = [], [], []
        self._view = None
        envs = stage.GetPrimAtPath("/World/envs")
        if envs:
            for env in envs.GetChildren():
                if not env.GetName().startswith("env_"):
                    continue
                env_id = int(env.GetName()[4:])
                # Snapshot before adding prims/deactivating the instancers.
                instancers = [p for p in Usd.PrimRange(env)
                              if p.IsA(UsdGeom.PointInstancer)]
                for prim in instancers:
                    paths = self._expand(stage, UsdGeom.PointInstancer(prim))
                    cache = UsdGeom.XformCache()
                    for path in paths:
                        world = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path))
                        q = Gf.Transform(world).GetRotation().GetQuat().GetNormalized()
                        self.paths.append(path)
                        env_ids.append(env_id)
                        positions.append(list(world.ExtractTranslation()))
                        orientations.append([q.GetReal(), *q.GetImaginary()])
        self._env_ids = np.asarray(env_ids, dtype=np.int64)
        # Capture authored poses BEFORE SimulationContext.reset() starts PhysX.
        self._positions = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
        self._orientations = np.asarray(orientations, dtype=np.float32).reshape(-1, 4)
        if self.paths:
            print(f"[fused] expanded {len(self.paths)} PointInstancer rigid bodies "
                  "for GPU/Fabric pose updates (separate from task objects)", flush=True)

    @staticmethod
    def _expand(stage, pi):
        targets = pi.GetPrototypesRel().GetTargets()
        # Decorative instancers and disabled/kinematic prototypes retain their
        # original representation. The compatibility path is for dynamic bodies.
        if not targets:
            return []
        for target in targets:
            prim = stage.GetPrimAtPath(target)
            body = UsdPhysics.RigidBodyAPI(prim)
            if (not body or not body.GetRigidBodyEnabledAttr().Get()
                    or body.GetKinematicEnabledAttr().Get()):
                return []
        indices = list(pi.GetProtoIndicesAttr().Get() or [])
        if not indices:
            return []
        transforms = pi.ComputeInstanceTransformsAtTime(
            Usd.TimeCode.Default(), Usd.TimeCode.Default(),
            UsdGeom.PointInstancer.IncludeProtoXform, UsdGeom.PointInstancer.IgnoreMask)
        mask = list(pi.ComputeMaskAtTime(Usd.TimeCode.Default()))
        if len(transforms) != len(indices):
            raise ValueError(f"fused: invalid PointInstancer transforms at {pi.GetPath()}")
        group_path = pi.GetPath().GetParentPath().AppendChild(pi.GetPrim().GetName() + "_rigids")
        if stage.GetPrimAtPath(group_path):
            raise ValueError(f"fused: PointInstancer expansion path already exists: {group_path}")
        group = UsdGeom.Xform.Define(stage, group_path)
        group.AddTransformOp().Set(pi.GetLocalTransformation())
        group.SetResetXformStack(pi.GetResetXformStack())
        group.CreateVisibilityAttr(pi.GetVisibilityAttr().Get())
        group.CreatePurposeAttr(pi.GetPurposeAttr().Get())
        paths = []
        for i, proto_index in enumerate(indices):
            if mask and not mask[i]:
                continue
            path = group_path.AppendChild(f"instance_{i:06d}")
            prim = stage.DefinePrim(path)
            # References retain prototype geometry, collision/mass/material
            # settings. IncludeProtoXform above already includes its transform.
            prim.GetReferences().AddInternalReference(targets[proto_index])
            prim.SetInstanceable(False)
            prim.SetCustomDataByKey(BODY_MARKER, True)
            xform = UsdGeom.Xformable(prim)
            xform.ClearXformOpOrder()
            xform.AddTransformOp().Set(transforms[i])
            paths.append(str(path))
        pi.GetPrim().SetActive(False)
        return paths

    def initialize(self):
        if not self.paths:
            return
        from isaacsim.core.prims import RigidPrim

        # The constructor's defaults rewrite the xform stack (losing the
        # instance scale in our matrix) and disable sleeping for contact
        # sensors. We only need tensor pose/velocity setters.
        self._view = RigidPrim(self.paths, reset_xform_properties=False,
                              prepare_contact_sensors=False)
        self._view.initialize()
        if self._view.count != len(self.paths):
            raise RuntimeError("fused: PointInstancer rigid view did not match all expanded bodies")

    def reset(self, env_ids, to_backend):
        if self._view is None:
            return
        rows = np.flatnonzero(np.isin(self._env_ids, env_ids))
        if not len(rows):
            return
        indices = to_backend(rows, long=True)
        self._view.set_world_poses(to_backend(self._positions[rows]),
                                   to_backend(self._orientations[rows]), indices=indices)
        self._view.set_velocities(to_backend(np.zeros((len(rows), 6), dtype=np.float32)),
                                  indices=indices)

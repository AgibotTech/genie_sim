# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Rule evaluation for scoop_popcorn; no camera or model-service dependencies.

The two equally weighted points remain: at least one grain in the bucket,
then no grain on the tabletop outside both the machine and bucket. Source
contents, airborne grains and floor drops do not count as tabletop spills.
"""

import numpy as np

from geniesim_benchmark.plugins.ader.action.common_actions import EvaluateAction
from geniesim_benchmark.plugins.ader.action.custom.popcorn_geometry import (
    BucketCavity,
    score_counts,
    transform,
)
from geniesim_benchmark.plugins.logger import Logger

logger = Logger()


class ScoopPopcorn(EvaluateAction):
    def __init__(
        self,
        env,
        bucket="benchmark_popcorn_bucket_fb1efa0f",
        instancer="/World/background/popcornInstancer",
        machine="/World/background/benchmark_popcorn_machine_001/entity/body/collider_body_003",
        tabletop="/World/background/benchmark_popcorn_car_000/entity/body/collider_body_000",
        interval=10,
        expected_count=1200,
        surface_tolerance=0.002,
    ):
        super().__init__(env)
        self.bucket = bucket
        self.instancer = instancer
        self.machine = machine
        self.tabletop = tabletop
        self.interval = int(interval)
        self.expected_count = int(expected_count)
        self.surface_tolerance = float(surface_tolerance)
        if (
            self.interval <= 0
            or self.expected_count <= 0
            or not np.isfinite(self.surface_tolerance)
            or self.surface_tolerance < 0
        ):
            raise ValueError("ScoopPopcorn requires positive interval/count and nonnegative surface tolerance")
        self._updates = 0
        self._best = 0.0
        self._done_flag = False
        self._cavity = None
        self._counts = {"inside": 0, "spilled": 0, "total": 0}

    def _path(self, name):
        return self.api_core._env_path(self._analyze_obj_name(name))

    def _sample(self):
        """Take one consistent snapshot on the physics thread, including instance transforms.

        These are rigid-body PointInstancer instances, not PBD fluid particles.
        Include the prototype's bind transform captured before physics starts.
        Its independent live transform must not shift the other instances.
        """
        from pxr import Usd, UsdGeom, UsdPhysics

        stage = self.api_core._stage
        cache = UsdGeom.XformCache()
        bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.proxy])

        def prim(name):
            p = stage.GetPrimAtPath(self._path(name))
            if not p.IsValid():
                raise ValueError(f"ScoopPopcorn missing prim: {name}")
            return p

        bucket = prim(self.bucket + "/entity")
        if self._cavity is None:
            mesh_prim = prim(self.bucket + "/entity/body/collider_body_00")
            mesh = UsdGeom.Mesh(mesh_prim)
            matrix, _ = cache.ComputeRelativeTransform(mesh_prim, bucket)
            vertices = transform(mesh.GetPointsAttr().Get(), matrix)
            indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
            triangles, offset = [], 0
            for count in mesh.GetFaceVertexCountsAttr().Get():
                face = indices[offset : offset + count]
                triangles.extend(vertices[[face[0], face[i], face[i + 1]]] for i in range(1, count - 1))
                offset += count
            # Aligned.usda makes the bucket Z-up with a rotation on /entity;
            # the simulated entity itself has its opening along local +Y.
            self._cavity = BucketCavity(np.asarray(triangles)[:, :, [0, 2, 1]])

        instancer = UsdGeom.PointInstancer(prim(self.instancer))
        if not instancer or len(instancer.GetProtoIndicesAttr().Get() or []) != self.expected_count:
            raise ValueError("ScoopPopcorn population is missing or changed")
        matrices = instancer.ComputeInstanceTransformsAtTime(
            Usd.TimeCode.Default(),
            Usd.TimeCode.Default(),
            UsdGeom.PointInstancer.ExcludeProtoXform,
        )
        if len(matrices) != self.expected_count:
            raise ValueError("ScoopPopcorn has inactive or missing instances")
        prototype_paths = instancer.GetPrototypesRel().GetTargets()
        if len(prototype_paths) != 1:
            raise ValueError("ScoopPopcorn expects one popcorn prototype")
        snapshots = getattr(self.api_core, "_rigid_instancer_prototypes", {})
        bind = snapshots.get(str(instancer.GetPath()))
        if bind is None or len(bind) != 1:
            raise ValueError("ScoopPopcorn prototype bind transform was not captured before physics")
        matrices = [bind[0] * m for m in matrices]
        prototype = stage.GetPrimAtPath(prototype_paths[0])
        prototype_bounds = bounds.ComputeUntransformedBound(prototype).ComputeAlignedRange()
        center = (prototype_bounds.GetMin() + prototype_bounds.GetMax()) * 0.5
        radius = max(prototype_bounds.GetSize()) * 0.5
        instancer_world = cache.GetLocalToWorldTransform(instancer.GetPrim())
        centers = np.asarray([list((m * instancer_world).Transform(center)) for m in matrices])
        # The prototype's collision approximation is boundingSphere.
        radii = np.asarray(
            [radius * np.linalg.norm(np.asarray(m * instancer_world)[:3, :3], axis=1).max() for m in matrices]
        )
        # This asset keeps its prototype outside the instancer as a visible,
        # independently simulated grain as well. Do not overlook its spills.
        if (
            not prototype.GetPath().HasPrefix(instancer.GetPath())
            and prototype.HasAPI(UsdPhysics.RigidBodyAPI)
            and UsdGeom.Imageable(prototype).ComputeVisibility() != UsdGeom.Tokens.invisible
        ):
            prototype_world = cache.GetLocalToWorldTransform(prototype)
            centers = np.vstack([centers, prototype_world.Transform(center)])
            radii = np.append(
                radii,
                radius * np.linalg.norm(np.asarray(prototype_world)[:3, :3], axis=1).max(),
            )

        # Only the pedestal covers the table. The whole machine's AABB also
        # includes its roof and tray overhangs, hiding real tabletop spills.
        # Source contents sit above the tabletop-height band independently.
        machine = prim(self.machine)
        machine_bounds = bounds.ComputeUntransformedBound(machine).ComputeAlignedRange()
        machine_local = transform(centers, cache.GetLocalToWorldTransform(machine).GetInverse())
        in_machine = ((machine_local >= machine_bounds.GetMin()) & (machine_local <= machine_bounds.GetMax())).all(
            axis=1
        )
        table_bounds = bounds.ComputeWorldBound(prim(self.tabletop)).ComputeAlignedRange()
        lower, upper = np.asarray(table_bounds.GetMin()), np.asarray(table_bounds.GetMax())
        if not np.isfinite(centers).all() or not np.isfinite([*lower, *upper, *radii]).all():
            raise ValueError("ScoopPopcorn geometry contains nonfinite values")
        bucket_local = transform(centers, cache.GetLocalToWorldTransform(bucket).GetInverse())[:, [0, 2, 1]]
        return centers, bucket_local, radii, in_machine, lower, upper

    def update(self, delta_time):
        if not self.is_running():
            return 0.0
        self._updates += 1
        if self._updates % self.interval == 0:
            centers, local, radii, in_machine, lower, upper = self.api_core.run_on_physics_loop(self._sample)
            inside, envelope = self._cavity.classify(local)
            on_table = ((centers[:, :2] >= lower[:2]) & (centers[:, :2] <= upper[:2])).all(axis=1)
            on_table &= np.abs(centers[:, 2] - upper[2]) <= radii + self.surface_tolerance
            spilled = on_table & ~in_machine & ~envelope
            self._counts = {
                "inside": int(inside.sum()),
                "spilled": int(spilled.sum()),
                "total": len(centers),
            }
            self._best = max(
                self._best,
                score_counts(self._counts["inside"], self._counts["spilled"]),
            )
            self._done_flag = self._best == 1.0
            logger.info(f"[ScoopPopcorn] {self._counts} score={self._best}")
        return super().update(delta_time)

    def _is_done(self):
        return self._done_flag

    def update_progress(self):
        self.progress_info["SCORE"] = self._best
        self.progress_info["POPCORN_COUNTS"] = dict(self._counts)
        if self._done_flag:
            self.progress_info["STATUS"] = "SUCCESS"

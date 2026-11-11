# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Optional live Isaac 6.2 smoke: python3.12 tests/isaac_popcorn_smoke.py.

Loads the real background and bucket, verifies rigid instance readback after
120 physics steps, checks controlled 0/.5/1 layouts on the paused stage and
runs six physical drop scenarios to verify entry and spill counts after motion.
Set GENIESIM_ASSETS_PATH to the asset checkout. No inference endpoint is used.
"""


def main():
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        import os
        import importlib.metadata
        from pathlib import Path
        from types import SimpleNamespace

        import numpy as np
        import omni.physx

        print(
            f"POPCORN_SMOKE isaac_version={importlib.metadata.version('isaacsim')}",
            flush=True,
        )
        from pxr import Gf, UsdGeom, UsdPhysics, Vt
        from isaacsim.core.api import World

        from geniesim_benchmark.plugins.ader.action.custom.popcorn import ScoopPopcorn
        from geniesim_benchmark.rigid_instancer import (
            snapshot_rigid_instancer_prototypes,
        )
        from geniesim_benchmark.plugins.ader.action.custom.popcorn_geometry import (
            transform,
        )

        assets = Path(os.environ["GENIESIM_ASSETS_PATH"])
        world = World(physics_dt=1 / 60)
        stage = world.stage
        UsdGeom.Xform.Define(stage, "/World").GetPrim().GetReferences().AddReference(
            str(assets / "background/popcorn/popcorn_1/background.usda")
        )
        bucket_path = "/Workspace/Objects/benchmark_popcorn_bucket_fb1efa0f"
        bucket = UsdGeom.Xform.Define(stage, bucket_path)
        bucket.GetPrim().GetReferences().AddReference(
            str(assets / "objects/benchmark/popcorn_bucket/benchmark_popcorn_bucket_003/Aligned.usda")
        )
        bucket.AddTranslateOp().Set(Gf.Vec3d(0.7, 0.117, 1.092))
        env = SimpleNamespace(
            api_core=SimpleNamespace(_stage=stage, _env_path=lambda p: p, run_on_physics_loop=lambda f: f()),
            init_task_config={"sub_task_name": "scoop_popcorn"},
            task=SimpleNamespace(update_progress=lambda *args: None),
        )
        env.api_core._rigid_instancer_prototypes = snapshot_rigid_instancer_prototypes(stage)
        before = ScoopPopcorn(env)._sample()[0]
        world.reset()
        for _ in range(120):
            world.step(render=False)
        world.pause()
        checker = ScoopPopcorn(env)
        after = checker._sample()[0]
        assert len(after) == 1201
        assert not np.allclose(before, after), "Rigid PointInstancer readback did not change"
        checker.start()
        for _ in range(9):
            checker.update(0.01)
        assert checker.progress_info["SCORE"] == 0, checker.progress_info
        print("POPCORN_SMOKE live_readback=PASS count=1201 baseline_score=0", flush=True)

        instancer = UsdGeom.PointInstancer(stage.GetPrimAtPath(checker.instancer))
        prototype = stage.GetPrimAtPath(instancer.GetPrototypesRel().GetTargets()[0])
        bucket_entity = stage.GetPrimAtPath(bucket_path + "/entity")
        table = (
            UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_])
            .ComputeWorldBound(stage.GetPrimAtPath(checker.tabletop))
            .ComputeAlignedRange()
        )
        table_z = table.GetMax()[2]
        for expected in [0, 0.5, 1]:
            cache = UsdGeom.XformCache()
            # Keep the untouched source grains in the machine. Author one
            # receiver grain and one tabletop grain for the controlled cases.
            centers = before[:1200].copy()
            if expected:
                centers[0] = cache.GetLocalToWorldTransform(bucket_entity).Transform(Gf.Vec3d(0))
            if expected == 0.5:
                centers[1] = [0.65, 0.5, table_z + 0.011]
            matrix = env.api_core._rigid_instancer_prototypes[str(instancer.GetPath())][0]
            positions = transform(centers, matrix.GetInverse())
            instancer.GetPositionsAttr().Set(Vt.Vec3fArray.FromNumpy(positions.astype(np.float32)))
            instancer.GetOrientationsAttr().Set(Vt.QuathArray([Gf.Quath(1)] * 1200))
            checker = ScoopPopcorn(env)
            checker.start()
            for _ in range(9):
                checker.update(0.01)
            assert checker.progress_info["SCORE"] == expected, checker.progress_info
            print(
                f"POPCORN_SMOKE controlled_score={expected} PASS {checker._counts}",
                flush=True,
            )

        # Isolate real drops with one instance plus the standalone prototype.
        # Every case starts a fresh physics scene and a fresh score history.
        # The full 1200-instance readback was checked above.
        cases = [
            ("source_only", [0.98, -0.23, 1.15], [0.93, -0.20, 1.2], 0, 0, 0),
            ("bucket_and_floor", [0.7, 0.117, 1.25], [2, 2, 1.1], 1, 0, 1),
            ("bucket_and_table", [0.7, 0.117, 1.25], [0.65, 0.5, 1.1], 1, 1, 0.5),
            (
                "bucket_and_overhang_table",
                [0.7, 0.117, 1.25],
                [1.14, 0.025, 0.97],
                1,
                1,
                0.5,
            ),
            ("empty_bucket_and_table", [2, 2, 1.1], [0.65, 0.5, 1.1], 0, 1, 0),
            ("bucket_corner_miss", [2, 2, 1.1], [0.80, 0.217, 1.25], 0, 1, 0),
        ]
        results = []
        for (
            name,
            prototype_at,
            instance_at,
            expected_inside,
            expected_spilled,
            expected_score,
        ) in cases:
            world.stop()
            prototype.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*prototype_at))
            body = UsdPhysics.RigidBodyAPI(prototype)
            body.CreateVelocityAttr().Set(Gf.Vec3f(0))
            body.CreateAngularVelocityAttr().Set(Gf.Vec3f(0))
            matrix = UsdGeom.XformCache().GetLocalToWorldTransform(prototype)
            positions = transform([instance_at], matrix.GetInverse())
            instancer.GetPositionsAttr().Set(Vt.Vec3fArray.FromNumpy(positions.astype(np.float32)))
            instancer.GetProtoIndicesAttr().Set(Vt.IntArray([0]))
            instancer.GetOrientationsAttr().Set(Vt.QuathArray([Gf.Quath(1)]))
            instancer.GetVelocitiesAttr().Set(Vt.Vec3fArray([Gf.Vec3f(0)]))
            instancer.GetAngularVelocitiesAttr().Set(Vt.Vec3fArray([Gf.Vec3f(0)]))
            env.api_core._rigid_instancer_prototypes = snapshot_rigid_instancer_prototypes(stage)
            world.reset()
            world.play()
            for _ in range(180):
                world.step(render=False)
            world.pause()
            checker = ScoopPopcorn(env, expected_count=1)
            checker.start()
            for _ in range(9):
                checker.update(0.01)
            sample = checker._sample()
            actual = checker._counts
            result = {
                "case": name,
                "expected_inside": expected_inside,
                "expected_spilled": expected_spilled,
                "expected_score": expected_score,
                "actual": actual,
                "actual_score": checker.progress_info["SCORE"],
                "world_centers": sample[0].tolist(),
                "bucket_local_centers": sample[1].tolist(),
            }
            if expected_spilled:
                center = sample[0][0]
                hit = omni.physx.get_physx_scene_query_interface().raycast_closest(
                    tuple(center + [0, 0, 0.03]), (0.0, 0.0, -1.0), 0.08
                )
                # Independent physics query at the reported instance pose:
                # the ray must hit instance 0, above the table surface.
                assert hit["hit"] and hit["protoIndex"] == 0, (name, hit)
                assert abs(hit["position"][2] - center[2] - sample[2][0]) < 0.003, (
                    name,
                    hit,
                )
                result["physics_raycast_surface"] = list(hit["position"])
            results.append(result)
            print(f"POPCORN_PHYSICAL {result}", flush=True)
            assert actual["inside"] == expected_inside, result
            assert actual["spilled"] == expected_spilled, result
            assert checker.progress_info["SCORE"] == expected_score, result
        if os.environ.get("POPCORN_REPORT_PATH"):
            import json

            Path(os.environ["POPCORN_REPORT_PATH"]).write_text(json.dumps(results, indent=2))
        print(f"POPCORN_SMOKE physical_cases={len(results)} PASS", flush=True)
    finally:
        app.close()


if __name__ == "__main__":
    main()

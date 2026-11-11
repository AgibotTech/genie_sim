# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Rule contract and composed-asset checks; no simulator or VLM service needed."""

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest

import numpy as np
from pxr import Gf, Usd, UsdGeom, Vt

from geniesim_benchmark.plugins.ader.action.action_parsing import parse_action
from geniesim_benchmark.rigid_instancer import (
    snapshot_rigid_instancer_prototypes,
)
from geniesim_benchmark.plugins.ader.action.custom.popcorn import ScoopPopcorn
from geniesim_benchmark.plugins.ader.action.custom.popcorn_geometry import (
    BucketCavity,
    score_counts,
    transform,
)
from geniesim_benchmark.plugins.output_system.eval_utils import (
    TaskEvaluation,
    get_task_steps,
)

PACKAGE = Path(__file__).resolve().parents[1] / "src/geniesim_benchmark"
CONFIG = PACKAGE / "benchmark/config/llm_task/scoop_popcorn"


class RuleTests(unittest.TestCase):
    def test_scores_and_dependency_of_second_condition(self):
        for inside, spilled, expected in [
            (0, 1, 0),
            (0, 0, 0),
            (1, 1, 0.5),
            (1, 0, 1),
            (100, 5, 0.5),
        ]:
            self.assertEqual(score_counts(inside, spilled), expected)

    def test_all_twenty_configs_use_rules_and_keep_instruction_and_budget(self):
        paths = list(CONFIG.glob("*/instructions.json"))
        self.assertEqual(len(paths), 20)
        for path in paths:
            instructions = json.loads(path.read_text())
            self.assertNotIn("eval_type", instructions)
            self.assertEqual(
                instructions["instructions"][0]["instruction"],
                "Scoop the popcorn with the right arm and pour it into the popcorn bucket.",
            )
            self.assertEqual(
                instructions["subtask_steps"][0]["checker"],
                {"ScoopPopcorn": {"interval": 10}},
            )
            self.assertEqual(instructions["subtask_steps"][0]["step_out"], 900)
            problem = json.loads(path.with_name("problems.json").read_text())["problem1"]
            rows = []
            tree = parse_action(problem["Acts"][0], rows, SimpleNamespace(api_core=None))
            self.assertEqual(
                [row["class_name"] for row in rows],
                ["ActionSetWaitAny", "ScoopPopcorn", "StepOut"],
            )
            self.assertEqual(tree._actions[1].max_step, 900)

    def test_result_aggregation_keeps_half_credit_and_e2e(self):
        self.assertEqual(get_task_steps("scoop_popcorn"), ["ScoopPopcorn"])
        self.assertEqual(get_task_steps("scoop_popcorn", "subtask"), ["ScoopPopcorn"])
        for score in [0, 0.5, 1]:
            evaluation = TaskEvaluation(sub_task_name="scoop_popcorn")
            evaluation.result["progress"] = [
                {
                    "class_name": "ScoopPopcorn",
                    "id": "grain",
                    "progress": {"SCORE": score},
                }
            ]
            evaluation.summarize_scores()
            self.assertEqual(evaluation.result["scores"]["STEPS"][0]["score"], score)
            self.assertEqual(evaluation.result["scores"]["E2E"], int(score == 1))

    def test_rules_do_not_import_model_client(self):
        code = (
            "import sys; from geniesim_benchmark.plugins.ader.action.action_parsing import parse_action; "
            "from types import SimpleNamespace; "
            "parse_action({'ScoopPopcorn': {}}, [], SimpleNamespace(api_core=None)); "
            "assert 'openai' not in sys.modules; "
            "assert not any(m.endswith('.auto_score') or m.endswith('.custom.vlm') for m in sys.modules)"
        )
        subprocess.run([sys.executable, "-c", code], check=True)


@unittest.skipUnless(
    os.environ.get("GENIESIM_ASSETS_PATH"),
    "Set GENIESIM_ASSETS_PATH for asset geometry tests",
)
class AssetTests(unittest.TestCase):
    def setUp(self):
        self.assets = Path(os.environ["GENIESIM_ASSETS_PATH"])
        self.stage = Usd.Stage.CreateInMemory()
        UsdGeom.Xform.Define(self.stage, "/World").GetPrim().GetReferences().AddReference(
            str(self.assets / "background/popcorn/popcorn_1/background.usda")
        )
        bucket = UsdGeom.Xform.Define(self.stage, "/Workspace/Objects/benchmark_popcorn_bucket_fb1efa0f")
        bucket.GetPrim().GetReferences().AddReference(
            str(self.assets / "objects/benchmark/popcorn_bucket/benchmark_popcorn_bucket_003/Aligned.usda")
        )
        bucket.AddTranslateOp().Set(Gf.Vec3d(0.7, 0.117, 1.092))
        self.api = SimpleNamespace(_stage=self.stage, _env_path=lambda p: p, run_on_physics_loop=lambda f: f())
        self.api._rigid_instancer_prototypes = snapshot_rigid_instancer_prototypes(self.stage)
        self.env = SimpleNamespace(
            api_core=self.api,
            init_task_config={"sub_task_name": "scoop_popcorn"},
            task=SimpleNamespace(update_progress=lambda *args: None),
        )
        self.checker = ScoopPopcorn(self.env)
        self.checker._sample()

    def world_bucket_points(self, local):
        p = self.stage.GetPrimAtPath("/Workspace/Objects/benchmark_popcorn_bucket_fb1efa0f/entity")
        return transform(
            np.asarray(local)[:, [0, 2, 1]],
            UsdGeom.XformCache().GetLocalToWorldTransform(p),
        )

    def set_world_centers(self, world):
        inst = UsdGeom.PointInstancer(self.stage.GetPrimAtPath(self.checker.instancer))
        proto = self.stage.GetPrimAtPath(inst.GetPrototypesRel().GetTargets()[0])
        matrix = self.api._rigid_instancer_prototypes[self.checker.instancer][0]
        # Instance origins include the prototype's authored translation.
        positions = transform(world, matrix.GetInverse())
        inst.GetPositionsAttr().Set(Vt.Vec3fArray.FromNumpy(positions.astype(np.float32)))
        inst.GetProtoIndicesAttr().Set(Vt.IntArray([0] * len(positions)))
        inst.GetOrientationsAttr().Set(Vt.QuathArray([Gf.Quath(1)] * len(positions)))
        self.checker.expected_count = len(positions)

    def tick(self):
        if self.checker.is_init():
            self.checker.start()
        while self.checker._updates % 10:
            self.checker.update(0.01)
        return self.checker.progress_info["SCORE"]

    def test_real_bucket_cavity_rejects_wall_aabb_corner_floor_and_air(self):
        points = [
            [0, 0, 0],
            [0.084, 0, 0],
            [0.09, 0.09, 0],
            [0, 0, -0.09],
            [0, 0, 0.1],
            [0.09, 0, -0.07],
        ]
        inside, shell = self.checker._cavity.classify(points)
        np.testing.assert_array_equal(inside, [True, False, False, False, False, False])
        np.testing.assert_array_equal(shell, [True, True, False, True, False, False])

    def test_cavity_against_independent_mesh_volume_reference(self):
        try:
            import trimesh
            import rtree  # noqa: F401 -- required by trimesh.contains
        except ImportError:
            self.skipTest("Independent mesh reference requires trimesh and rtree")
        bucket = self.stage.GetPrimAtPath("/Workspace/Objects/benchmark_popcorn_bucket_fb1efa0f/entity")
        prim = self.stage.GetPrimAtPath(str(bucket.GetPath()) + "/body/collider_body_00")
        usd_mesh = UsdGeom.Mesh(prim)
        matrix, _ = UsdGeom.XformCache().ComputeRelativeTransform(prim, bucket)
        vertices = transform(usd_mesh.GetPointsAttr().Get(), matrix)[:, [0, 2, 1]]
        self.assertEqual(set(usd_mesh.GetFaceVertexCountsAttr().Get()), {3})
        faces = np.asarray(usd_mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
        self.assertTrue(mesh.is_watertight)
        rng = np.random.default_rng(20261109)
        samples = rng.uniform(vertices.min(0) - 0.02, vertices.max(0) + 0.02, (5000, 3))
        # Closing the mouth with a convex hull and subtracting the closed solid
        # shell is an independent volume-based reference for this round bucket.
        reference = mesh.convex_hull.contains(samples) & ~mesh.contains(samples)
        actual, _ = self.checker._cavity.classify(samples)
        np.testing.assert_array_equal(actual, reference)

    def test_authored_population_in_machine_scores_zero(self):
        self.assertEqual(self.tick(), 0)
        self.assertEqual(self.checker._counts, {"inside": 0, "spilled": 0, "total": 1201})

    def test_one_in_bucket_with_table_spill_scores_half(self):
        self.set_world_centers(np.vstack([self.world_bucket_points([[0, 0, 0]]), [0.65, 0.5, 0.907]]))
        self.assertEqual(self.tick(), 0.5)
        self.assertFalse(self.checker.is_finished())
        self.assertEqual(self.checker._counts["spilled"], 1)

    def test_exposed_table_under_machine_overhang_is_a_spill(self):
        # This lies inside the whole machine's AABB (its roof overhangs),
        # but outside its pedestal on the actual tabletop.
        self.set_world_centers(np.vstack([self.world_bucket_points([[0, 0, 0]]), [1.14, 0.025, 0.907]]))
        self.assertEqual(self.tick(), 0.5)
        self.assertEqual(self.checker._counts["spilled"], 1)

    def test_floor_machine_air_and_off_table_do_not_block_full_score(self):
        centers = np.vstack(
            [
                self.world_bucket_points([[0, 0, 0]]),
                [0.98, -0.23, 1.15],
                [0.65, 0.5, 0.01],
                [0.65, 0.5, 1.3],
                [0.2, 0.5, 0.907],
            ]
        )
        self.set_world_centers(centers)
        self.assertEqual(self.tick(), 1)
        self.assertTrue(self.checker.is_finished())

    def test_table_spill_with_empty_bucket_is_zero(self):
        self.set_world_centers([[0.65, 0.5, 0.907]])
        self.assertEqual(self.tick(), 0)

    def test_rim_wall_and_bucket_aabb_corners_do_not_count_as_entry(self):
        local = [[0, 0, 0.10], [0.084, 0, 0], [0.09, 0.09, 0]]
        self.set_world_centers(self.world_bucket_points(local))
        self.assertEqual(self.tick(), 0)
        self.assertEqual(self.checker._counts["inside"], 0)

    def test_spills_at_all_table_corners_are_detected(self):
        sample = self.checker._sample()
        lower, upper = sample[-2:]
        for x in [lower[0] + 0.015, upper[0] - 0.015]:
            for y in [lower[1] + 0.015, upper[1] - 0.015]:
                self.checker = ScoopPopcorn(self.env)
                self.set_world_centers(
                    np.vstack(
                        [
                            self.world_bucket_points([[0, 0, 0]]),
                            [x, y, upper[2] + 0.011],
                        ]
                    )
                )
                self.assertEqual(self.tick(), 0.5, (x, y))

    def test_visible_standalone_prototype_spill_is_also_counted(self):
        proto = self.stage.GetPrimAtPath("/World/background/benchmark_popcorn_001")
        proto.GetAttribute("xformOp:translate").Set(Gf.Vec3d(0.65, 0.5, 0.907))
        self.set_world_centers(self.world_bucket_points([[0, 0, 0]]))
        self.assertEqual(self.tick(), 0.5)
        self.assertEqual(self.checker._counts["spilled"], 1)

    def test_moving_prototype_does_not_shift_other_instances(self):
        self.set_world_centers(np.vstack([self.world_bucket_points([[0, 0, 0]]), [0.65, 0.5, 0.907]]))
        proto = self.stage.GetPrimAtPath("/World/background/benchmark_popcorn_001")
        proto.GetAttribute("xformOp:translate").Set(Gf.Vec3d(0.98, -0.23, 0.25))
        self.assertEqual(self.tick(), 0.5)
        self.assertEqual(self.checker._counts["inside"], 1)
        self.assertEqual(self.checker._counts["spilled"], 1)

    def test_missing_bind_snapshot_cannot_award_success(self):
        self.api._rigid_instancer_prototypes = {}
        with self.assertRaisesRegex(ValueError, "before physics"):
            self.tick()

    def test_peak_score_survives_loss_of_bucket_contents(self):
        self.set_world_centers(np.vstack([self.world_bucket_points([[0, 0, 0]]), [0.65, 0.5, 0.907]]))
        self.assertEqual(self.tick(), 0.5)
        self.set_world_centers([[0.65, 0.5, 0.907], [0.66, 0.5, 0.907]])
        for _ in range(10):
            self.checker.update(0.01)
        self.assertEqual(self.checker.progress_info["SCORE"], 0.5)
        self.assertEqual(self.checker._counts["inside"], 0)

    def test_timeout_keeps_half_credit_and_cancels_unfinished_checker(self):
        self.set_world_centers(np.vstack([self.world_bucket_points([[0, 0, 0]]), [0.65, 0.5, 0.907]]))
        self.env.current_step = 0
        self.env.has_done = False

        def cancel():
            self.env.has_done = True

        self.env.cancel_eval = cancel
        tree = parse_action(
            {
                "ActionSetWaitAny": [
                    {"ScoopPopcorn": {"expected_count": 2}},
                    {"StepOut": 900},
                ]
            },
            [],
            self.env,
        )
        tree.start()
        for _ in range(10):
            tree.update(0.01)
        checker = tree._actions[0]
        self.assertEqual(checker.progress_info["SCORE"], 0.5)
        self.env.current_step = 901
        tree.update(0.01)
        self.assertTrue(tree.is_finished())
        self.assertTrue(checker.is_canceled())
        self.assertTrue(self.env.has_done)
        self.assertEqual(checker.progress_info["SCORE"], 0.5)

    def test_moving_and_tilting_bucket_uses_current_rigid_frame(self):
        p = self.stage.GetPrimAtPath("/Workspace/Objects/benchmark_popcorn_bucket_fb1efa0f/entity")
        UsdGeom.Xformable(p).AddTranslateOp(opSuffix="test").Set(Gf.Vec3d(0.1, 0.2, 0.05))
        UsdGeom.Xformable(p).AddRotateYOp(opSuffix="test").Set(50)
        self.set_world_centers(self.world_bucket_points([[0, 0, 0]]))
        self.assertEqual(self.tick(), 1)

    def test_missing_or_deleted_population_cannot_receive_credit(self):
        inst = UsdGeom.PointInstancer(self.stage.GetPrimAtPath(self.checker.instancer))
        inst.GetProtoIndicesAttr().Set(Vt.IntArray([]))
        with self.assertRaisesRegex(ValueError, "population"):
            self.tick()

    def test_all_scene_bucket_transforms_are_supported(self):
        root = UsdGeom.Xformable(self.stage.GetPrimAtPath("/Workspace/Objects/benchmark_popcorn_bucket_fb1efa0f"))
        for path in CONFIG.glob("*/scene_info.json"):
            pose = json.loads(path.read_text())["layout"]["benchmark_popcorn_bucket_fb1efa0f"]
            x, y, z, w = pose["xyzw"]
            root.ClearXformOpOrder()
            root.AddTranslateOp().Set(Gf.Vec3d(*pose["xyz"]))
            root.AddOrientOp().Set(Gf.Quatf(w, x, y, z))
            centers = self.world_bucket_points([[0, 0, 0]])
            self.set_world_centers(centers)
            sample = self.checker._sample()
            inside, _ = self.checker._cavity.classify(sample[1])
            self.assertTrue(inside[0], str(path))


if __name__ == "__main__":
    unittest.main()

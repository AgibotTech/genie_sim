# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Runtime planners and generated layout bundles agree on the assembly cache."""

import json
from pathlib import Path
import tempfile
import unittest

import yaml

from geniesim_benchmark.newton.layout_batch import load_plans
from geniesim_benchmark.newton.layouts import generate_layouts
from geniesim_benchmark.rt_engine.session import plan_launch


class RuntimeManifestPathTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.assets = self.root / "assets"
        self.assets.mkdir()
        self.scene = self.root / "scene_chef.yaml"
        self.scene.write_text(
            "robot:\n"
            "  robot_name: G2_crsB_wuji\n"
            "  init_base_pose:\n"
            "    x: 0\n    y: 0\n    z: 0\n    theta: 0\n"
            "  init_joint_pos:\n"
            "    idx21_arm_l_joint1: 0\n"
            "newton:\n  scene_plugin: chef.py\n"
        )
        (self.root / "chef.py").write_text("# Layout plugin fixture.\n")
        self.launcher = self.root / "launcher.yaml"
        self.launcher.write_text(
            yaml.safe_dump(
                {
                    "launcher": {
                        "physics": {
                            "engine": "genie_sim_engine_newton",
                            "engines": {"genie_sim_engine_newton": {"package": "genie_sim_engine_newton"}},
                        }
                    },
                    "genie_sim_engine": {
                        "ros__parameters": {
                            "physics_engine_visualizer": "isaacsim",
                            "step_mode": "lockstep",
                            "step_lockstep_ticks": 1,
                            "wok_evaluation_json": "scoring.json",
                        }
                    },
                }
            )
        )
        self.config = {
            "runtime": "newton",
            "task_name": "wok_flip",
            "episodes": 1,
            "scene": str(self.scene),
            "launcher": str(self.launcher),
            "ready_timeout_s": 10,
            "task_progress_topic": "/task_progress",
        }
        self.evaluation = self.root / "evaluation.yaml"
        self.evaluation.write_text(yaml.safe_dump(self.config))

    def test_managed_launch_uses_scene_cache(self):
        plan = plan_launch(self.evaluation, self.config, self.root / "run", self.assets)
        self.assertEqual(plan.manifest, self.assets / "scenes/scene_chef/manifest.json")

    def test_bundle_generation_and_reader_use_scene_cache(self):
        rules = self.root / "rules.sh"
        rules.write_text('BASE_XY_RANGE = 0.05\nrandom.uniform(-BASE_XY_RANGE, BASE_XY_RANGE)\nf"{d:.4f}"\n')
        bundle = self.root / "bundle"
        index = generate_layouts(
            output_dir=bundle,
            source_scene=self.scene,
            source_launcher=self.launcher,
            source_evaluation=self.evaluation,
            rules_path=rules,
            count=2,
        )
        for legacy_index in (False, True):
            with self.subTest(legacy_index=legacy_index):
                if legacy_index:
                    for item in index["layouts"]:
                        item["manifest_subpath"] = item["manifest_subpath"].replace(
                            "scenes/", "background/common/runtime/", 1
                        )
                    (bundle / "layouts.json").write_text(json.dumps(index))
                plans = load_plans(bundle, self.assets, self.root / "runs", "localhost:8999")
                self.assertEqual(len(plans), 2)
                for plan in plans:
                    expected = self.assets / "scenes" / plan.scene.stem / "manifest.json"
                    self.assertEqual(plan.manifest, expected)
                    self.assertEqual(plan.runner_command[plan.runner_command.index("--manifest") + 1], str(expected))
                    if not legacy_index:
                        self.assertEqual(
                            index["layouts"][plan.layout_id]["manifest_subpath"],
                            expected.relative_to(self.assets).as_posix(),
                        )


if __name__ == "__main__":
    unittest.main()

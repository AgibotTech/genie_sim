# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""HDR replacement must be reproducible and preserve physical task geometry."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

MODULE = Path(__file__).parents[1] / "src/geniesim_benchmark/hdr_background.py"
spec = importlib.util.spec_from_file_location("hdr_background", MODULE)
hdr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hdr)


class HdrSelectionTests(unittest.TestCase):
    def test_pool_selection_replay_and_missing_pool(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(FileNotFoundError):
                hdr.select_hdr(root, "scene")
            pool = root / hdr.HDR_DIRECTORY
            pool.mkdir(parents=True)
            for name in ["0010_office.hdr", "0006_home.HDR", "ignored.exr"]:
                (pool / name).write_text(name)
            for name in ["rooftop.hdr", "rostock_laage_airport_8k.hdr", "unreviewed.hdr"]:
                (pool / name).write_text(name)
            first = hdr.select_hdr(root, "scene", 3)
            self.assertEqual(first, hdr.select_hdr(root, "scene", 3))
            selected = {hdr.select_hdr(root, f"scene-{i}", 3)["relative_path"] for i in range(20)}
            self.assertEqual(
                selected,
                {str(Path(hdr.HDR_DIRECTORY) / "0010_office.hdr"), str(Path(hdr.HDR_DIRECTORY) / "0006_home.HDR")},
            )

    def test_new_unreviewed_asset_does_not_change_the_sample_or_pool_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pool = root / hdr.HDR_DIRECTORY
            pool.mkdir(parents=True)
            for name in ["0010_office.hdr", "0006_home.hdr"]:
                (pool / name).write_text(name)
            before = hdr.select_hdr(root, "same-episode", 11)
            (pool / "new_city_rooftop.hdr").write_text("unreviewed")
            self.assertEqual(before, hdr.select_hdr(root, "same-episode", 11))

    def test_unreviewed_backgrounds_cannot_be_used_as_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pool = root / hdr.HDR_DIRECTORY
            pool.mkdir(parents=True)
            (pool / "rooftop.hdr").write_text("outdoor")
            with self.assertRaisesRegex(FileNotFoundError, "approved indoor"):
                hdr.select_hdr(root, "scene", 3)

    def test_usd_replacement_keeps_task_and_scopes_changes(self):
        from pxr import Usd, UsdGeom, UsdLux

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pool = root / hdr.HDR_DIRECTORY
            pool.mkdir(parents=True)
            (pool / "0010_office.hdr").write_text("HDR test content")
            selection = hdr.select_hdr(root, "phone-scene-1", 8)
            stage = Usd.Stage.CreateInMemory()
            world = "/World/envs/env_0"
            UsdGeom.Xform.Define(stage, world)
            old = UsdGeom.Xform.Define(stage, world + "/hdr_backdrop")
            table = UsdGeom.Cube.Define(stage, world + "/background/Table")
            table.CreateSizeAttr(1.2)
            other = UsdLux.DomeLight.Define(stage, "/World/envs/env_1/Dome")
            other.CreateTextureFileAttr("unchanged.hdr")
            hdr.apply_hdr(stage, selection, world_root=world)
            self.assertEqual(old.GetVisibilityAttr().Get(), "invisible")
            self.assertEqual(table.GetSizeAttr().Get(), 1.2)
            self.assertEqual(other.GetTextureFileAttr().Get().path, "unchanged.hdr")
            panorama = stage.GetPrimAtPath(world + "/HDRBackground/Panorama")
            self.assertFalse(any("Collision" in str(x) or "RigidBody" in str(x) for x in panorama.GetAppliedSchemas()))
            self.assertEqual(
                UsdLux.DomeLight(stage.GetPrimAtPath(world + "/HDRBackground/Dome")).GetTextureFileAttr().Get().path,
                selection["texture_file"],
            )
            count = len(list(stage.Traverse()))
            hdr.apply_hdr(stage, selection, world_root=world)
            self.assertEqual(len(list(stage.Traverse())), count)
            (pool / "0010_office.hdr").write_text("changed")
            with self.assertRaises(ValueError):
                hdr.apply_hdr(stage, selection, world_root=world)


class HdrTaskIntegrationTests(unittest.TestCase):
    def test_phone_configuration_and_seeded_generalization(self):
        import json
        from types import SimpleNamespace
        import numpy as np
        from geniesim_benchmark.utils.generalization_utils import apply_hdr_texture_generalization

        self.addCleanup(np.random.set_state, np.random.get_state())
        for name in ["phone_packaging_g2_90d", "phone_packaging_g2_crsb90d"]:
            config_path = MODULE.parent / "benchmark/config/eval_tasks" / (name + ".json")
            config = json.loads(config_path.read_text())
            config["scene"]["scene_instance_id"] = 7
            received = []
            api = SimpleNamespace(set_hdr_background=lambda *args: received.append(args))
            apply_hdr_texture_generalization(api, config)
            self.assertEqual(received, [])  # Sampling belongs to the episode boundary.
            np.random.seed(21)
            expected_draw = np.random.random()
            np.random.seed(21)
            apply_hdr_texture_generalization(api, config, seed=21, episode_id=0, file_idx=2)
            self.assertEqual(np.random.random(), expected_draw)
            apply_hdr_texture_generalization(api, config, seed=21, episode_id=0, file_idx=2)
            self.assertEqual(received[0], received[1])
            self.assertEqual(received[0][0], hdr.HDR_DIRECTORY)
            self.assertIn(":7:file:2:episode:0", received[0][2])
            apply_hdr_texture_generalization(api, config, seed=21, episode_id=1, file_idx=2)
            self.assertNotEqual(received[0][2], received[-1][2])
            apply_hdr_texture_generalization(api, config, seed=22, episode_id=0, file_idx=2)
            self.assertNotEqual(received[0][1], received[-1][1])

    def test_rubik_background_tracks_layout_identity(self):
        from types import SimpleNamespace
        from geniesim_benchmark.backends.mujoco.rubik.isaac_render import scene_background

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pool = root / hdr.HDR_DIRECTORY
            pool.mkdir(parents=True)
            for name in ["0006_home.hdr", "0007_home.hdr", "0010_office.hdr", "015_office.hdr"]:
                (pool / name).write_text(name)
            (pool / "rooftop.hdr").write_text("outdoor")
            env = SimpleNamespace(
                current_scene=SimpleNamespace(identifier="scene-0", sha256="first"), hdr_seed=21, hdr_episode=0
            )
            first = scene_background(env, root)
            self.assertEqual(first, scene_background(env, root))
            draws = set()
            for seed in range(20):
                env.hdr_seed = seed
                receipt = scene_background(env, root)
                self.assertEqual(receipt["seed"], seed)
                draws.add(receipt["relative_path"])
            self.assertGreater(len(draws), 1)
            env.hdr_seed = 21
            env.hdr_episode = 1
            repeated_layout = scene_background(env, root)
            self.assertNotEqual(first["identity"], repeated_layout["identity"])
            env.current_scene = SimpleNamespace(identifier="scene-1", sha256="next")
            second = scene_background(env, root)
            self.assertNotEqual(first["identity"], second["identity"])
            self.assertTrue(Path(second["texture_file"]).is_file())


if __name__ == "__main__":
    unittest.main()

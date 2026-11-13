# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Offline routing checks for Isaac 6.2 benchmark interpreters."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from geniesim_cli.commands import benchmark

CONFIGS = Path(__file__).resolve().parents[2] / "geniesim_benchmark/src/geniesim_benchmark/config"
RT_TASKS = {
    "g2crsbwuji_dexmanip_wok_flip": "newton",
    "g2crsbwuji_dexmanip_cloth_hanger": "libuipc",
    "g2crsbwuji_dexmanip_fold_cloth": "libuipc",
    "g2crsbwuji_dexmanip_drawstring_bag": "libuipc",
}


def combo_command(name):
    return {"omni_python": "/usr/local/bin/omni_python", "python3.12": "/usr/bin/python3.12"}.get(name)


class BenchmarkRuntimeTests(unittest.TestCase):
    def test_task_inventory_ignores_both_container_defaults(self):
        from geniesim_benchmark.runtime import runtime_for_config

        for default in ("python3", "omni_python"):
            selected_rt = {}
            with patch.dict(os.environ, {"GENIESIM_PY_CMD": default}), patch.object(
                benchmark.shutil, "which", side_effect=combo_command
            ):
                for config in sorted(CONFIGS.glob("*.yaml")):
                    if "_" not in config.stem or config.stem.startswith("_"):
                        continue
                    with self.subTest(default=default, task=config.stem):
                        runtime = runtime_for_config(config)
                        cmd = benchmark._build_app_cmd(config, [], "localhost:8999")
                        if runtime in ("newton", "libuipc"):
                            selected_rt[config.stem] = runtime
                            self.assertEqual(
                                cmd[:3], ["/usr/bin/python3.12", "-m", f"geniesim_benchmark.{runtime}.runner"]
                            )
                        elif runtime == "isaac":
                            self.assertEqual(cmd[0], "/usr/bin/python3.12")
                        else:
                            self.assertEqual(runtime, "mujoco")
                            self.assertTrue(cmd[1].endswith("scripts/run_mujoco.py"))
                        self.assertIn("--benchmark.infer_host=localhost:8999", cmd)
            self.assertEqual(selected_rt, RT_TASKS)

    def test_child_environment_tracks_task_choice(self):
        for task, expected in (
            ("g2op_crsemb_pour_50ml_into_graduated_cylinder", "/usr/bin/python3.12"),
            *((task, "/usr/bin/python3.12") for task in RT_TASKS),
        ):
            for default in ("python3", "omni_python"):
                with self.subTest(task=task, default=default), patch.dict(
                    os.environ, {"GENIESIM_PY_CMD": default}
                ), patch.object(benchmark.shutil, "which", side_effect=combo_command):
                    cmd, env = benchmark._prepared_app_cmd(CONFIGS / f"{task}.yaml", ["--dry-run"], None)
                    self.assertEqual(cmd[0], expected)
                    self.assertEqual(env["GENIESIM_PY_CMD"], expected)
                    self.assertEqual(os.environ["GENIESIM_PY_CMD"], default)

    def test_isaac_62_does_not_require_bundled_51_wrapper(self):
        with patch.dict(os.environ, {"GENIESIM_PY_CMD": "omni_python"}), patch.object(
            benchmark.shutil, "which", side_effect=lambda name: "/usr/bin/python3.12" if name == "python3.12" else None
        ):
            cmd = benchmark._build_app_cmd(CONFIGS / "g2op_crsemb_pour_50ml_into_graduated_cylinder.yaml", [], None)
            self.assertEqual(cmd[0], "/usr/bin/python3.12")

    def test_missing_system_runtime_for_isaac_does_not_fall_back_to_51(self):
        with patch.dict(os.environ, {"GENIESIM_PY_CMD": "python3"}), patch.object(
            benchmark.shutil,
            "which",
            side_effect=lambda name: "/usr/local/bin/omni_python" if name == "omni_python" else None,
        ), self.assertRaisesRegex(RuntimeError, "Python 3.12"):
            benchmark._build_app_cmd(CONFIGS / "g2op_crsemb_pour_50ml_into_graduated_cylinder.yaml", [], None)

    def test_missing_system_runtime_does_not_fall_back_to_51(self):
        with patch.dict(os.environ, {"GENIESIM_PY_CMD": "omni_python"}), patch.object(
            benchmark.shutil, "which", return_value=None
        ), self.assertRaisesRegex(RuntimeError, "Python 3.12"):
            benchmark._build_app_cmd(CONFIGS / "g2crsbwuji_dexmanip_wok_flip.yaml", [], None)

    def test_newton_prepared_overlay_controls_render_children(self):
        with tempfile.TemporaryDirectory() as root:
            virtualenv = Path(root) / "rt_python"
            python = virtualenv / "bin/python"
            python.parent.mkdir(parents=True)
            python.touch()
            with patch.dict(os.environ, {"GENIESIM_PY_CMD": "omni_python"}), patch.object(
                benchmark.shutil, "which", side_effect=combo_command
            ), patch(
                "geniesim_benchmark.rt_engine.environment.runtime_environment",
                return_value={"VIRTUAL_ENV": str(virtualenv)},
            ):
                cmd, env = benchmark._prepared_app_cmd(CONFIGS / "g2crsbwuji_dexmanip_wok_flip.yaml", [], None)
                self.assertEqual(cmd[0], str(python))
                self.assertEqual(env["GENIESIM_PY_CMD"], str(python))

    def test_unrelated_virtualenv_does_not_replace_system_python(self):
        with tempfile.TemporaryDirectory() as root:
            python = Path(root) / "bin/python"
            python.parent.mkdir()
            python.touch()
            with patch.dict(os.environ, {"GENIESIM_PY_CMD": "omni_python"}), patch.object(
                benchmark.shutil, "which", side_effect=combo_command
            ), patch(
                "geniesim_benchmark.rt_engine.environment.runtime_environment", return_value={"VIRTUAL_ENV": root}
            ):
                cmd, env = benchmark._prepared_app_cmd(CONFIGS / "g2crsbwuji_dexmanip_wok_flip.yaml", [], None)
                self.assertEqual(cmd[0], "/usr/bin/python3.12")
                self.assertEqual(env["GENIESIM_PY_CMD"], cmd[0])

    def test_custom_prepared_python_is_retained(self):
        with patch.dict(os.environ, {"GENIESIM_PY_CMD": "/prepared/bin/python"}):
            for task in (*RT_TASKS, "g2op_crsemb_pour_50ml_into_graduated_cylinder"):
                cmd, env = benchmark._prepared_app_cmd(CONFIGS / f"{task}.yaml", ["--dry-run"], None)
                self.assertEqual(cmd[0], "/prepared/bin/python")
                self.assertEqual(env["GENIESIM_PY_CMD"], "/prepared/bin/python")

    def test_native_mujoco_backend_override_keeps_its_environment(self):
        with patch.dict(os.environ, {"GENIESIM_PY_CMD": "omni_python"}):
            cmd, env = benchmark._prepared_app_cmd(
                CONFIGS / "g2op_crsemb_pour_50ml_into_graduated_cylinder.yaml",
                ["--benchmark.backend=mujoco"],
                None,
            )
            self.assertTrue(cmd[1].endswith("scripts/run_mujoco.py"))
            self.assertEqual(env["GENIESIM_PY_CMD"], cmd[0])


if __name__ == "__main__":
    unittest.main()

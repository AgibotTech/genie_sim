# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Check declared ROS SDK requirements before compiling against an older image."""

import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "install_ros_build_deps.py"


class TestROSBuildDependencies(unittest.TestCase):
    def setUp(self):
        self.assertTrue(SCRIPT.is_file(), "ROS build dependency check is missing")
        spec = importlib.util.spec_from_file_location("install_ros_build_deps", SCRIPT)
        self.installer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.installer)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.site = Path(directory.name)
        self.installer.PYPROJECT = self.site / "pyproject.toml"
        self.installer.PYPROJECT.write_text(
            '[build-system]\nrequires = ["setuptools>=68.0", "wheel", "ovrtx>=0.4.0,<0.5.0"]\n'
        )
        paths = patch.object(self.installer.sysconfig, "get_path", return_value=str(self.site))
        paths.start()
        self.addCleanup(paths.stop)

    def test_old_ovrtx_is_rejected_and_current_version_is_accepted(self):
        for version, expected in (("0.3.0.312915", True), ("0.4.0", False), ("0.5.0", True)):
            versions = {"setuptools": "80.0.0", "wheel": "0.45.0", "ovrtx": version}
            with (
                self.subTest(version=version),
                patch.object(self.installer.metadata, "version", side_effect=versions.__getitem__),
            ):
                missing = self.installer.unsatisfied_requirements()
                self.assertEqual(bool(missing), expected)
                if expected:
                    self.assertEqual(missing, ["ovrtx>=0.4.0,<0.5.0"])

    def test_missing_package_is_selected_for_installation(self):
        with patch.object(self.installer.metadata, "version", side_effect=self.installer.metadata.PackageNotFoundError):
            self.assertIn("wheel", self.installer.unsatisfied_requirements())

    def test_install_uses_declared_constraints_and_rechecks(self):
        with (
            patch.object(self.installer, "unsatisfied_requirements", side_effect=[["ovrtx>=0.4.0,<0.5.0"], []]),
            patch.object(self.installer.os, "geteuid", return_value=0),
            patch.object(self.installer.subprocess, "run") as run,
        ):
            self.installer.ensure_dependencies(check_only=False)
        command = run.call_args.args[0]
        self.assertIn("ovrtx>=0.4.0,<0.5.0", command)
        self.assertIn("--no-deps", command)
        self.assertIn("--break-system-packages", command)

    def test_ovrtx_upgrade_keeps_cache_and_removes_only_outdated_metadata(self):
        old = self.site / "ovrtx-0.3.0.dist-info"
        old.mkdir()
        (old / "METADATA").write_text("Name: ovrtx\nVersion: 0.3.0\n")
        cache = self.site / "ovrtx/bin/cache"
        cache.mkdir(parents=True)
        (cache / "shader").write_bytes(b"cached")
        with (
            patch.object(self.installer, "unsatisfied_requirements", side_effect=[["ovrtx>=0.4.0,<0.5.0"], []]),
            patch.object(self.installer.os, "geteuid", return_value=0),
            patch.object(self.installer.subprocess, "run") as run,
        ):
            self.installer.ensure_dependencies(check_only=False)
        self.assertIn("--ignore-installed", run.call_args.args[0])
        self.assertFalse(old.exists())
        self.assertEqual((cache / "shader").read_bytes(), b"cached")

    def test_failed_upgrade_retains_old_metadata(self):
        old = self.site / "ovrtx-0.3.0.dist-info"
        old.mkdir()
        (old / "METADATA").write_text("Name: ovrtx\nVersion: 0.3.0\n")
        with (
            patch.object(self.installer, "unsatisfied_requirements", return_value=["ovrtx>=0.4.0,<0.5.0"]),
            patch.object(self.installer.os, "geteuid", return_value=0),
            patch.object(self.installer.subprocess, "run", side_effect=subprocess.CalledProcessError(1, ["pip"])),
            self.assertRaises(subprocess.CalledProcessError),
        ):
            self.installer.ensure_dependencies(check_only=False)
        self.assertTrue(old.exists())

    def test_check_only_never_installs(self):
        with (
            patch.object(self.installer, "unsatisfied_requirements", return_value=["ovrtx>=0.4.0,<0.5.0"]),
            patch.object(self.installer.subprocess, "run") as run,
        ):
            with self.assertRaisesRegex(RuntimeError, "ovrtx"):
                self.installer.ensure_dependencies(check_only=True)
        run.assert_not_called()

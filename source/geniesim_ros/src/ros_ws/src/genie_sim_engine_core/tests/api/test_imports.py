# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Core imports must succeed without a physics SDK or a backend package."""

import subprocess
import sys
from pathlib import Path


def test_all_core_modules_import_without_backends():
    script = """
import importlib
import importlib.abc
import pkgutil
import sys

class BlockBackends(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {
            'pxr', 'omni', 'newton', 'uipc',
            'genie_sim_engine_isaacsim', 'genie_sim_engine_newton',
            'genie_sim_engine_libuipc',
        }:
            raise ImportError(f'core tried importing backend {fullname}')

sys.meta_path.insert(0, BlockBackends())
sys.path.insert(0, sys.argv[1])
import genie_sim_engine_core as core
for _, name, _ in pkgutil.walk_packages(core.__path__, core.__name__ + '.'):
    importlib.import_module(name)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(Path(__file__).resolve().parents[2])],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr

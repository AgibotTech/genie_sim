# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Newton's backend boundary must hold without libuipc installed."""

import ast
import os
import subprocess
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[2]
WORKSPACE = PACKAGE.parent


def test_newton_has_no_ipc_runtime_or_reverse_dependency():
    newton = PACKAGE
    for path in newton.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            assert not any("libuipc" in name or name == "uipc" for name in names), path
    assert not (newton / "genie_sim_engine_newton/engine/newton/adapters/ipc.py").exists()
    assert not (newton / "genie_sim_engine_newton/engine/newton/solvers/ipc").exists()
    for rel in (
        "engine_base.py",
        "plugin.py",
        "setup/model.py",
        "setup/runtime.py",
        "setup/solver.py",
        "setup/init_pose.py",
    ):
        assert '"ipc"' not in (newton / "genie_sim_engine_newton/engine/newton" / rel).read_text()


def test_newton_factory_runs_without_libuipc_and_rejects_ipc():
    code = """
import importlib.abc, sys
class BlockIPC(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname == 'uipc' or fullname.startswith('genie_sim_engine_libuipc'):
            raise AssertionError(fullname)
sys.meta_path.insert(0, BlockIPC())
from genie_sim_engine_newton.engine.newton.adapters import make_adapter
try:
    make_adapter('ipc', physics_hz=60)
except ValueError:
    pass
else:
    raise AssertionError('Newton still accepts IPC')
"""
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join(
            str(p)
            for p in (
                WORKSPACE / "genie_sim_engine_newton",
                WORKSPACE / "genie_sim_engine_core",
            )
        ),
    )
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr

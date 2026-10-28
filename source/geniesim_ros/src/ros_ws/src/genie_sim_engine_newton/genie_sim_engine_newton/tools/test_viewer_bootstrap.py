#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Check GL backend selection before importing or warming the Newton runtime."""

from __future__ import annotations

import ast
import builtins
import os
from pathlib import Path
import platform
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

_NODE = Path(__file__).resolve().parents[2] / "scripts/genie_sim_engine_newton_node.py"


def _bootstrap_prefix():
    tree = ast.parse(_NODE.read_text(), filename=str(_NODE))
    prefix = []
    for statement in tree.body:
        prefix.append(statement)
        if (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Name)
            and statement.value.func.id == "warmup_newton_kernels"
        ):
            return compile(ast.Module(body=prefix, type_ignores=[]), str(_NODE), "exec")
    raise AssertionError("Newton node no longer has the expected warmup boundary")


def _run_bootstrap(*, visualizer, gui, environment, operating_system="linux"):
    observations = []

    def early_param(_argv, name, default):
        return visualizer if name == "physics_engine_visualizer" else default

    def warmup(_solver):
        observations.append(("warmup-call", os.environ.get("PYOPENGL_PLATFORM")))

    def import_module(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "genie_sim_engine_core.common.early_params":
            return SimpleNamespace(_early_param=early_param)
        if name == "genie_sim_engine_newton.warmup":
            # Capture at import as well as invocation: MuJoCo can select and
            # cache PyOpenGL's platform while this module imports its solvers.
            observations.append(("warmup-import", os.environ.get("PYOPENGL_PLATFORM")))
            return SimpleNamespace(warmup_newton_kernels=warmup)
        return builtins.__import__(name, globals, locals, fromlist, level)

    execution_builtins = dict(vars(builtins), __import__=import_module)
    namespace = {"__name__": "_viewer_bootstrap_under_test", "__builtins__": execution_builtins}
    argv = [str(_NODE)] + (["--gui"] if gui else [])
    system_name = {"linux": "Linux", "win32": "Windows", "darwin": "Darwin"}[operating_system]
    with (
        patch.dict(os.environ, environment, clear=True),
        patch.object(sys, "argv", argv),
        patch.object(sys, "platform", operating_system),
        patch.object(platform, "system", return_value=system_name),
    ):
        exec(_bootstrap_prefix(), namespace)
        final_environment = dict(os.environ)
    return observations, final_environment


def test_linux_windowed_newton_selects_glx_before_runtime_import_and_warmup():
    observations, environment = _run_bootstrap(
        visualizer="newton", gui=True, environment={"DISPLAY": ":0", "MUJOCO_GL": "egl", "UNRELATED": "preserved"}
    )

    assert observations == [("warmup-import", "glx"), ("warmup-call", "glx")]
    assert environment["PYOPENGL_PLATFORM"] == "glx"
    assert environment["MUJOCO_GL"] == "egl"
    assert environment["UNRELATED"] == "preserved"


@pytest.mark.parametrize(
    "visualizer,gui,environment,operating_system",
    [
        ("newton", False, {"DISPLAY": ":0"}, "linux"),
        ("none", True, {"DISPLAY": ":0"}, "linux"),
        ("ovrtx", True, {"DISPLAY": ":0"}, "linux"),
        ("newton", True, {}, "linux"),
        ("newton", True, {"WAYLAND_DISPLAY": "wayland-0"}, "linux"),
        ("newton", True, {"DISPLAY": ":0"}, "win32"),
        ("newton", True, {"DISPLAY": ":0"}, "darwin"),
    ],
)
def test_other_runtime_modes_leave_gl_backend_unchanged(visualizer, gui, environment, operating_system):
    observations, final_environment = _run_bootstrap(
        visualizer=visualizer, gui=gui, environment=environment, operating_system=operating_system
    )

    assert observations == [("warmup-import", None), ("warmup-call", None)]
    assert "PYOPENGL_PLATFORM" not in final_environment


@pytest.mark.parametrize("backend", ["egl", "glx"])
def test_explicit_gl_backend_is_preserved(backend):
    observations, environment = _run_bootstrap(
        visualizer="newton", gui=True, environment={"DISPLAY": ":0", "PYOPENGL_PLATFORM": backend}
    )

    assert observations == [("warmup-import", backend), ("warmup-call", backend)]
    assert environment["PYOPENGL_PLATFORM"] == backend

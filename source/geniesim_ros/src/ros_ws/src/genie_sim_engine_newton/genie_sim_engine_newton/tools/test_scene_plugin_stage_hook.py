#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Regression test for scene-plugin hook discovery."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_NEWTON = _HERE.parent / "engine" / "newton"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


plugin_module = _load("scene_plugin_stage_hook", _NEWTON / "plugin.py")


def test_on_stage_is_a_discoverable_plugin_hook():
    assert plugin_module._PLUGIN_HOOKS[0] == "on_stage"

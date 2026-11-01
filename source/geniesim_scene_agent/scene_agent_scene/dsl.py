# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Execute GenieSim-style DSL recipes with only an external assets directory."""

from __future__ import annotations

import importlib.util
import inspect
import random
import sys
from pathlib import Path

import numpy as np

from .asset_index import load_asset_index
from .dsl_runtime import add_dsl_objects, create_helper_module


def load_python_file(path: str | Path, module_name: str):
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_from_dsl(builder, script: str | Path, assets_root: str | Path,
                   seed: int, params: dict | None = None) -> None:
    index = load_asset_index(assets_root)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    local_helper = create_helper_module(index)
    previous_helper = sys.modules.get("helper")
    sys.modules["helper"] = local_helper
    try:
        recipe = load_python_file(script, f"scene_dsl_{seed}")
        root_scene = getattr(recipe, "root_scene", None)
        if not callable(root_scene):
            raise ValueError("DSL recipe must define root_scene()")
        parameters = params or {}
        if "params" in inspect.signature(root_scene).parameters:
            shape = root_scene(params=parameters)
        elif parameters:
            raise ValueError("DSL root_scene() must accept params when generalization dimensions are set")
        else:
            shape = root_scene()
    finally:
        if previous_helper is None:
            sys.modules.pop("helper", None)
        else:
            sys.modules["helper"] = previous_helper
    add_dsl_objects(builder, shape, index)

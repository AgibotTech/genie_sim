# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Expose a mounted asset tree to its USD-attached Python behavior scripts."""

import importlib.util
import os
from pathlib import Path
import sys


def register_asset_package(root=None):
    """Register SIM_ASSETS as geniesim_assets before loading scripted scenes."""
    root = root or os.environ.get("SIM_ASSETS")
    if not root:
        return
    directory = Path(root).resolve()
    initializer = directory / "__init__.py"
    if not initializer.is_file():
        raise FileNotFoundError(f"Asset package initializer not found: {initializer}")
    existing = sys.modules.get("geniesim_assets")
    if existing is not None:
        if Path(existing.__file__).resolve() != initializer:
            raise RuntimeError("Loaded geniesim_assets differs from the configured asset root")
        return
    spec = importlib.util.spec_from_file_location(
        "geniesim_assets", initializer, submodule_search_locations=[str(directory)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[spec.name]
        raise

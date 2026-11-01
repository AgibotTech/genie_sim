# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""GenieSim scene generation tools."""

from importlib import import_module

__all__ = ["SceneBuilder", "generate_instances"]


def __getattr__(name):
    """Load USD generation tools only when their public symbols are requested."""
    modules = {"SceneBuilder": ".builder", "generate_instances": ".generate"}
    if name not in modules:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(modules[name], __name__), name)
    globals()[name] = value
    return value

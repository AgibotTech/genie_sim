# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Lightweight package identity and bundled resource discovery."""

from pathlib import Path

from ._version import _resolve_version


def resource_root() -> Path:
    """Locate resources in either an editable checkout or an installed wheel."""
    bundled = Path(__file__).resolve().parent / "resources"
    if bundled.is_dir():
        return bundled
    return Path(__file__).resolve().parents[2]


__version__ = _resolve_version("geniesim_scene_agent")

__all__ = ["__version__", "resource_root"]

# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Lightweight package identity and bundled resource discovery."""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def resource_root() -> Path:
    """Locate resources in either an editable checkout or an installed wheel."""
    bundled = Path(__file__).resolve().parent / "resources"
    if bundled.is_dir():
        return bundled
    return Path(__file__).resolve().parents[2]


try:
    __version__ = version("geniesim_scene_agent")
except PackageNotFoundError:
    __version__ = (resource_root() / "VERSION").read_text(encoding="utf-8").strip()

__all__ = ["__version__", "resource_root"]

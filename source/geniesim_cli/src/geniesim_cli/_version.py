# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Single source of truth for the GenieSim version.

The repo-root ``VERSION`` file is the authoritative source for all eight
first-class distributions. Each build reads its local VERSION symlink.
``geniesim version --bump`` / ``--sync`` keep the links and owned ROS manifests
in sync; ``--check`` validates release metadata. Runtime imports use the source
release in a checkout and installed distribution metadata in a wheel.

To bump: ``geniesim version --bump <NEW_VERSION>`` (or edit ``VERSION``
and run ``geniesim version --sync``). Never edit a sibling
``pyproject.toml`` or a ``package.xml`` ``<version>`` by hand.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _pkg_version
from pathlib import Path


def _read_repo_version() -> str | None:
    """Read root VERSION only when this module is in its source checkout.

    Wheel installs use distribution metadata even if the surrounding SDK or
    another ancestor directory has an unrelated VERSION file.
    """
    here = Path(__file__).resolve()
    for ancestor in here.parents:
        package = ancestor / "source" / here.parent.name
        # Installed SDKs may have their own VERSION. Only the canonical
        # source layout identifies this repository's release file.
        if here != package / "src" / here.parent.name / "_version.py" or not (package / "pyproject.toml").is_file():
            continue
        candidate = ancestor / "VERSION"
        if candidate.is_file():
            text = candidate.read_text().strip()
            if text:
                return text
    return None


def _resolve_version() -> str:
    repo = _read_repo_version()
    if repo:
        return repo
    try:
        return _pkg_version("geniesim_cli")
    except PackageNotFoundError:
        return "0.0.0+unknown"


__version__: str = _resolve_version()

__all__ = ["__version__"]

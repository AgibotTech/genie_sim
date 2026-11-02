# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Resolve the package version from the repo-root ``VERSION`` file.

The repo-root ``VERSION`` is the source of truth for all eight first-class
Genie Sim distributions. Each build reads a local VERSION symlink, and every
package re-exports that release value via a self-contained resolver. Edits go through
``geniesim version --bump <NEW>`` (or hand-edit ``VERSION`` then
``geniesim version --sync``); never hardcode a literal here.

Resolution order:

1. Recognize the canonical source checkout layout and read its root VERSION.
2. Fall back to ``importlib.metadata.version`` of the installed
   distribution — hits in wheel installs where ``source/`` isn't
   present and build metadata records the release version.
3. Return ``"0.0.0+unknown"`` as a last resort.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _pkg_version
from pathlib import Path


def _read_repo_version() -> str | None:
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


def _resolve_version(dist_name: str) -> str:
    repo = _read_repo_version()
    if repo:
        return repo
    try:
        return _pkg_version(dist_name)
    except PackageNotFoundError:
        return "0.0.0+unknown"

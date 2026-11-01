# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Discover external assets without importing their code or requiring a symlink."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path


def _checked(value: str | Path, source: str) -> Path:
    root = Path(value).expanduser().resolve()
    if not (root / "objects").is_dir():
        raise FileNotFoundError(f"{source} must point to an assets directory containing objects/: {root}")
    return root


def _nearby(anchor: Path):
    for parent in (anchor, *anchor.parents):
        for candidate in (
            parent,
            parent / "assets",
            parent / "geniesim" / "assets",
            parent / "source" / "geniesim" / "assets",
        ):
            if (candidate / "objects").is_dir():
                yield candidate.resolve()


def resolve_assets_root(explicit: str | Path | None = None, *, workspace: str | Path | None = None) -> Path:
    """Resolve explicit configuration, optional CLI/package discovery, then relative layouts.

    Explicit or configured invalid paths fail instead of selecting another library.
    Relative discovery examines the workspace/module and working-directory ancestors.
    ``geniesim_cli`` and ``geniesim_assets`` are optional and are never dependencies.
    """
    if explicit is not None:
        return _checked(explicit, "--assets-root")
    configured = os.environ.get("GENIESIM_ASSETS")
    if configured:
        return _checked(configured, "GENIESIM_ASSETS")
    try:
        from geniesim_cli._env import assets_path
    except ImportError:
        configured = os.environ.get("GENIESIM_ASSETS_PATH") or os.environ.get("SIM_ASSETS")
    else:
        configured = assets_path()
    if configured:
        return _checked(configured, "CLI assets configuration")
    try:
        spec = importlib.util.find_spec("geniesim_assets")
    except (ImportError, ValueError):
        spec = None
    if spec:
        locations = list(spec.submodule_search_locations or [])
        if spec.origin:
            locations.append(str(Path(spec.origin).parent))
        for location in locations:
            found = next(_nearby(Path(location)), None)
            if found:
                return found
    anchors = (
        [Path(workspace).expanduser().resolve()] if workspace else [Path(__file__).resolve().parent.parent, Path.cwd()]
    )
    for anchor in anchors:
        found = next(_nearby(anchor), None)
        if found:
            return found
    raise FileNotFoundError(
        "Cannot locate assets containing objects/. Pass --assets-root, set GENIESIM_ASSETS "
        "or GENIESIM_ASSETS_PATH, install the external asset package, or place assets/ "
        "beside the main repository."
    )

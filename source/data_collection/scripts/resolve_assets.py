# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# License: Mozilla Public License Version 2.0

"""Locate external assets for host launchers using only the standard library."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

try:
    from .collection_settings import load_settings
except ImportError:
    from collection_settings import load_settings


def _is_assets(root: Path) -> bool:
    return (root / "robot").is_dir() and (root / "objects").is_dir()


def _layouts(root: Path):
    yield root
    yield root / "assets"
    yield root / "geniesim_assets"
    yield root / "geniesim_assets" / "assets"
    yield root / "geniesim" / "assets"
    yield root / "source" / "geniesim" / "assets"
    yield root / "source" / "geniesim_assets"
    yield root / "source" / "geniesim_assets" / "assets"


def resolve_assets(module_dir: Path | None = None) -> Path:
    """Prefer configured paths, then the installed package, then nearby assets.

    Relative fallback is anchored to the module, independent of the launch cwd.
    An invalid explicit setting fails rather than selecting a different library.
    """
    for name in ("GENIESIM_ASSETS_PATH", "SIM_ASSETS", "GENIESIM_ASSETS_SRC"):
        value = os.environ.get(name)
        if value:
            root = Path(value).expanduser().resolve()
            candidates = _layouts(root) if name == "GENIESIM_ASSETS_SRC" else (root,)
            for candidate in candidates:
                if _is_assets(candidate):
                    return candidate.resolve()
            raise FileNotFoundError(f"{name} must locate assets containing robot/ and objects/: {root}")

    saved = load_settings().get("assets_root")
    if saved:
        root = Path(saved).expanduser().resolve()
        if not _is_assets(root):
            raise FileNotFoundError(f"Saved assets_root is unavailable: {root}; override it or rerun data-collection install")
        return root

    try:
        spec = importlib.util.find_spec("geniesim_assets")
    except (ImportError, ValueError):
        spec = None
    if spec:
        locations = list(spec.submodule_search_locations or [])
        if spec.origin:
            locations.append(str(Path(spec.origin).parent))
        for location in locations:
            anchor = Path(location).resolve()
            for parent in (anchor, *anchor.parents):
                for candidate in _layouts(parent):
                    if _is_assets(candidate):
                        return candidate.resolve()
                if (parent / "pyproject.toml").is_file():
                    break

    module_dir = (module_dir or Path(__file__).resolve().parent.parent).resolve()
    # Cover the module, source/, repository root, and its parent directory.
    for parent in (module_dir, *list(module_dir.parents)[:3]):
        for candidate in _layouts(parent):
            if _is_assets(candidate):
                return candidate.resolve()
    raise FileNotFoundError(
        "Cannot locate assets containing robot/ and objects/. Set GENIESIM_ASSETS_PATH "
        "or SIM_ASSETS, install geniesim_assets, or place assets/ under the module, "
        "the main repository, or beside the repository."
    )


if __name__ == "__main__":
    try:
        print(resolve_assets())
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

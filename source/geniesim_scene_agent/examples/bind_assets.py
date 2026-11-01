# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Bind the external GenieSim assets tree used by the collected USD examples."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def bind(link: Path, root: Path) -> None:
    if not (root / "objects").is_dir():
        raise ValueError(f"expected an assets directory containing objects/: {root}")
    if link.is_symlink():
        if link.resolve() == root:
            print(f"Assets already bound: {link} -> {root}")
            return
        link.unlink()
    elif link.exists():
        raise ValueError(f"refusing to replace a non-symlink: {link}")
    link.symlink_to(root, target_is_directory=True)
    print(f"Assets bound: {link} -> {root}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-root", help="External assets root; auto-discovered when omitted")
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scene_agent_assets.location import resolve_assets_root

    try:
        root = resolve_assets_root(args.assets_root)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    base = Path(__file__).resolve().parent
    if not (root / "objects").is_dir() or ((base / "assets").exists() and not (base / "assets").is_symlink()):
        parser.error(f"invalid assets directory or non-symlink destination: {root}, {base / 'assets'}")
    bind(base / "assets", root)


if __name__ == "__main__":
    main()

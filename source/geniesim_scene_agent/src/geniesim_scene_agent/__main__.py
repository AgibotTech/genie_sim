# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Expose scene tools and the installed skill library through one module."""

import sys


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args == ["assets-root"]:
        from scene_agent_assets.location import resolve_assets_root

        try:
            print(resolve_assets_root())
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        return 0
    if args == ["resources"]:
        from . import resource_root

        print(resource_root())
        return 0
    from scene_agent_scene.__main__ import main as scene_main

    return scene_main(args)


if __name__ == "__main__":
    raise SystemExit(main())

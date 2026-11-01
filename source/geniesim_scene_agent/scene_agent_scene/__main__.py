# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Command-line entry point for GenieSim scene generation and validation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .generate import generate_instances
from .render import render_scene
from .supports import inspect_supports, write_support_markers
from .validate import validate_scene


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Generate and inspect GenieSim scene instances")
    subcommands = parser.add_subparsers(dest="command", required=True)
    build = subcommands.add_parser("build", help="Build one scene or a deterministic variant batch")
    build.add_argument("--mode", choices=("dsl", "direct"), required=True)
    build.add_argument("--recipe", required=True)
    build.add_argument("--scene-id", required=True)
    build.add_argument("--output-root", required=True)
    build.add_argument("--assets-root", help="External assets root; auto-discovered when omitted")
    build.add_argument("--patch", help="Python patch_scene(builder, rng, params) hook")
    build.add_argument("--generalization", help="Reviewed JSON dimension specification")
    build.add_argument("--count", type=int, default=1)
    build.add_argument("--seed", type=int, default=0)
    build.add_argument("--isaac-python", default=os.environ.get("ISAACSIM_PYTHON"))
    build.add_argument("--no-render", action="store_true", help="Create USD-only drafts without Isaac Sim images")

    render = subcommands.add_parser("render", help="Render an existing scene in Isaac Sim")
    render.add_argument("scene")
    render.add_argument("--output-dir")
    render.add_argument("--isaac-python", default=os.environ.get("ISAACSIM_PYTHON"))

    check = subcommands.add_parser("validate", help="Check final USDA and scene metadata")
    check.add_argument("scene_dir")
    check.add_argument("--require-renders", action="store_true")

    supports = subcommands.add_parser("supports", help="Find shelf or cabinet placement surfaces")
    supports.add_argument("usd")
    supports.add_argument("--prim")
    supports.add_argument("--layers-json")
    supports.add_argument("--no-auto-layers", action="store_true")
    supports.add_argument("--markers", help="Write a separate marker overlay USDA")

    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            spec = json.loads(Path(args.generalization).read_text(encoding="utf-8")) if args.generalization else None
            result = generate_instances(
                args.mode,
                args.recipe,
                args.output_root,
                args.scene_id,
                count=args.count,
                seed=args.seed,
                generalization=spec,
                patch=args.patch,
                assets_root=args.assets_root,
                render=not args.no_render,
                isaac_python=args.isaac_python,
            )
        elif args.command == "render":
            scene = Path(args.scene).resolve()
            folder = scene.parent
            output = Path(args.output_dir).resolve() if args.output_dir else folder / "renders"
            render_scene(scene, output, isaac_python=args.isaac_python)
            if scene.name == "scene.usda" and (folder / "scene_info.json").is_file() and output == folder / "renders":
                report = validate_scene(folder, require_renders=True)
                (folder / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
                result = report
            else:
                result = {
                    "ok": True,
                    "renders": {view: str(output / f"{view}.png") for view in ("front", "side", "top")},
                }
        elif args.command == "validate":
            result = validate_scene(args.scene_dir, require_renders=args.require_renders)
            (Path(args.scene_dir) / "validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        else:
            result = inspect_supports(
                args.usd, args.prim, layers_json=args.layers_json, auto_layers=not args.no_auto_layers
            )
            if args.markers:
                result["markers_usda"] = str(write_support_markers(args.usd, result["surfaces"], args.markers))
        print(json.dumps(result, indent=2))
        return 0 if result.get("ok", True) else 2
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError, ImportError, OSError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

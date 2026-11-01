# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Command-line entry point for local GenieSim asset discovery."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .catalog import build_catalog, load_catalog, search_catalog, in_folder
from .vl_backend import QwenVLBackend
from .vl_index import VLIndex
from .location import resolve_assets_root


def _emit(value):
    print(json.dumps(value, ensure_ascii=False, indent=2))


def _backend(args):
    selected = args.model_path or os.environ.get("QWEN3_VL_MODEL") or ".models/Qwen3-VL-Embedding-2B"
    return QwenVLBackend(Path(selected).expanduser().resolve())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Local GenieSim asset discovery tools")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("assets-root", help="Print the discovered external asset directory")

    build = commands.add_parser("catalog-build", help="Generate hierarchical filename tables")
    build.add_argument("--assets-root", help="External assets root; auto-discovered when omitted")
    build.add_argument("--catalog", required=True)
    build.add_argument("--scope", default=".", help="Folder beneath assets/objects")

    lookup = commands.add_parser("catalog-search", help="Search filename and description tables")
    lookup.add_argument("query")
    lookup.add_argument("--catalog", required=True)
    lookup.add_argument("--folder")
    lookup.add_argument("--limit", type=int, default=50)

    inspect = commands.add_parser("inspect-usd", help="Inspect composed mesh bounds")
    inspect.add_argument("usd")

    render = commands.add_parser("render", help="Render folder candidates and create a contact sheet")
    render.add_argument("--catalog", required=True)
    render.add_argument("--folder", required=True)
    render.add_argument("--output-dir", required=True)
    render.add_argument("--query", default="")
    render.add_argument("--limit", type=int, default=30)
    render.add_argument("--force", action="store_true")
    render.add_argument("--isaac-python", default=sys.executable)

    for name, help_text in (
        ("vl-build", "Build Qwen3-VL video vectors"),
        ("vl-search", "Search Qwen3-VL video vectors"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--catalog", required=True)
        command.add_argument("--db", required=True)
        command.add_argument("--folder")
        command.add_argument("--model-path")
        if name == "vl-search":
            command.add_argument("query")
            command.add_argument("--top-k", type=int, default=20)

    args = parser.parse_args(argv)
    try:
        if args.command == "assets-root":
            _emit({"assets_root": str(resolve_assets_root())})
        elif args.command == "catalog-build":
            manifest = build_catalog(args.assets_root, args.catalog, args.scope)
            _emit(
                {
                    "catalog": str(Path(args.catalog).resolve() / "index.json"),
                    "scope": manifest["scope"],
                    "asset_count": manifest["asset_count"],
                    "table_count": len(manifest["tables"]),
                }
            )
        elif args.command == "catalog-search":
            _emit({"results": search_catalog(args.catalog, args.query, args.folder, args.limit)})
        elif args.command == "inspect-usd":
            from .geometry import inspect_usd

            _emit(inspect_usd(args.usd))
        elif args.command == "render":
            from .render import render_records

            records = [record for record in load_catalog(args.catalog) if in_folder(record, args.folder)]
            if args.query:
                matches = {
                    record["asset_id"] for record in search_catalog(args.catalog, args.query, args.folder, args.limit)
                }
                records = [record for record in records if record["asset_id"] in matches]
            _emit(render_records(records[: args.limit], args.output_dir, args.isaac_python, args.force))
        else:
            records = load_catalog(args.catalog)
            backend = _backend(args)
            index = VLIndex(args.db)
            if args.command == "vl-build":
                _emit(index.build(records, backend, args.folder))
            else:
                _emit(index.search(args.query, records, backend, args.top_k, args.folder))
    except (ValueError, FileNotFoundError, RuntimeError, ImportError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

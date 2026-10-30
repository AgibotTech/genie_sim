#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Set teleop_result on unmarked recordings in the configured sub-task."""

import argparse
import json
import sys
from pathlib import Path

import yaml


def get_sub_task_name_from_yaml(config_path: Path) -> str | None:
    """Read the selected sub-task; malformed YAML must fail before marking data."""
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return (data or {}).get("benchmark", {}).get("sub_task_name")


def patch_recording_info(
    base_dir: Path,
    sub_task_name: str,
    teleop_result: bool = True,
    dry_run: bool = False,
    existing: list[str] | None = None,
) -> int:
    """Mark unlabelled numeric episodes and return the number of updated files."""
    task_dir = base_dir / sub_task_name
    if not task_dir.is_dir():
        print(f"Info: directory does not exist: {task_dir}")
        return 0

    patched = 0
    unfinished = []
    existing = set(existing or [])
    subdirs = sorted(
        [d for d in task_dir.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda d: int(d.name),
    )
    for subdir in subdirs:
        if subdir.name in existing:
            continue
        json_path = subdir / "recording_info.json"
        if not json_path.is_file():
            continue
        data = json.loads(json_path.read_text(encoding="utf-8"))
        if "teleop_result" in data:
            continue
        if not (subdir / "metadata.yaml").is_file():
            unfinished.append(str(subdir))
            continue
        data["teleop_result"] = teleop_result
        if not dry_run:
            json_path.write_text(
                json.dumps(data, indent=4, ensure_ascii=False),
                encoding="utf-8",
            )
        print(f"Patched: {json_path}")
        patched += 1
    if unfinished:
        raise RuntimeError(f"Recordings did not finalize; leaving them unmarked: {', '.join(unfinished)}")
    return patched


def main():
    parser = argparse.ArgumentParser(
        description="Add teleop_result to recording_info.json under recording_data/<sub_task_name>/.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("/workspace/source/geniesim_benchmark/src/geniesim_benchmark/config/teleop.yaml"),
        help="Path to teleop config YAML (benchmark.sub_task_name).",
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=Path("/workspace/output/recording_data"),
        help="Base directory containing <sub_task_name> subdirs.",
    )
    parser.add_argument(
        "--teleop-result",
        type=lambda x: x.lower() in ("1", "true", "yes"),
        default=True,
        metavar="true|false",
        help="Value for teleop_result when missing (default: true).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only print what would be patched.")
    parser.add_argument("--snapshot", action="store_true", help="Print existing episode directory names as JSON.")
    parser.add_argument(
        "--existing", type=json.loads, default=[], help="JSON list of episode directories to leave unchanged."
    )
    args = parser.parse_args()

    sub_task_name = get_sub_task_name_from_yaml(args.config)
    if not sub_task_name:
        print(f"Error: could not read sub_task_name from {args.config}", file=sys.stderr)
        return 1
    if args.snapshot:
        print(
            json.dumps(sorted(p.name for p in (args.base / sub_task_name).glob("*") if p.is_dir() and p.name.isdigit()))
        )
        return 0
    n = patch_recording_info(
        args.base, sub_task_name, teleop_result=args.teleop_result, dry_run=args.dry_run, existing=args.existing
    )
    if n:
        print(f"Patched {n} file(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""
Patch recording_info.json under recording_data/<sub_task_name>/: add "teleop_result": true
if the field is missing. sub_task_name is read from geniesim config (e.g. teleop.yaml). Use --teleop-result true|false.
"""

import argparse
import json
import re
import sys
from pathlib import Path


def get_sub_task_name_from_yaml(config_path: Path) -> str | None:
    """Read benchmark.sub_task_name from YAML config. Fallback to regex if yaml not available."""
    text = config_path.read_text(encoding="utf-8")
    try:
        import yaml

        data = yaml.safe_load(text)
        return (data or {}).get("benchmark", {}).get("sub_task_name")
    except Exception:
        pass
    # Fallback: match "sub_task_name: value" (with optional quotes)
    m = re.search(r"sub_task_name\s*:\s*['\"]?([^'\"]+)['\"]?\s*$", text, re.MULTILINE)
    return m.group(1).strip() if m else None


def patch_recording_info(
    base_dir: Path,
    sub_task_name: str,
    teleop_result: bool = True,
    dry_run: bool = False,
) -> int:
    """
    For each numeric subdir under base_dir/sub_task_name, if recording_info.json exists
    and has no "teleop_result", add "teleop_result" with the given value. Returns number of patched files.
    """
    task_dir = base_dir / sub_task_name
    if not task_dir.is_dir():
        print(f"Info: directory does not exist: {task_dir}")
        return 0

    patched = 0
    # List numeric subdirs and sort numerically (1, 2, 10, ...)
    subdirs = sorted(
        [d for d in task_dir.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda d: int(d.name),
    )
    for subdir in subdirs:
        json_path = subdir / "recording_info.json"
        if not json_path.is_file():
            continue
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: skip {json_path}: {e}", file=sys.stderr)
            continue
        if "teleop_result" in data:
            continue
        data["teleop_result"] = teleop_result
        if not dry_run:
            json_path.write_text(
                json.dumps(data, indent=4, ensure_ascii=False),
                encoding="utf-8",
            )
        print(f"Patched: {json_path}")
        patched += 1
    return patched


def main():
    parser = argparse.ArgumentParser(
        description="Add teleop_result to recording_info.json under recording_data/<sub_task_name>/.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("/geniesim/main/source/geniesim/config/teleop.yaml"),
        help="Path to teleop config YAML (benchmark.sub_task_name).",
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=Path("/geniesim/main/output/recording_data"),
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
    args = parser.parse_args()

    sub_task_name = get_sub_task_name_from_yaml(args.config)
    if not sub_task_name:
        print(f"Warning: could not read sub_task_name from {args.config}", file=sys.stderr)
        return 0
    n = patch_recording_info(args.base, sub_task_name, teleop_result=args.teleop_result, dry_run=args.dry_run)
    if n:
        print(f"Patched {n} file(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""
Compute the mean of statistics.average across all JSON files under a given directory.
Usage: python scripts/stat_average.py [dir]
Default dir: output/benchmark (relative to project root).
"""

import argparse
import json
import os
import sys

# Project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


def find_jsons(root_dir: str):
    root_dir = os.path.abspath(root_dir)
    if not os.path.isdir(root_dir):
        return []
    paths = []
    for dirpath, _dirnames, filenames in os.walk(root_dir):
        for name in filenames:
            if name.endswith(".json"):
                paths.append(os.path.join(dirpath, name))
    return paths


def extract_average(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    stats = data.get("statistics")
    if not isinstance(stats, dict):
        return None
    if "average" not in stats:
        return None
    val = stats["average"]
    if isinstance(val, (int, float)):
        return float(val)
    return None


def main():
    parser = argparse.ArgumentParser(description="Compute mean of statistics.average in JSON files under dir")
    default_dir = os.path.join(PROJECT_ROOT, "output", "benchmark")
    parser.add_argument(
        "dir",
        nargs="?",
        default=default_dir,
        help=f"Directory to scan (default: {default_dir})",
    )
    args = parser.parse_args()

    root = os.path.abspath(args.dir)
    if not os.path.isdir(root):
        print(f"Error: not a directory: {root}", file=sys.stderr)
        sys.exit(1)

    paths = find_jsons(root)
    if not paths:
        print(f"No json files under: {root}")
        print("Average: N/A (no data)")
        return

    # (path, score) for each json that has statistics.average
    items = []
    for path in paths:
        v = extract_average(path)
        if v is not None:
            items.append((path, v))

    if not items:
        print(f"Found {len(paths)} json file(s), none contain statistics.average")
        print("Overall average: N/A (no data)")
        return

    average_values = [v for _p, v in items]
    avg = sum(average_values) / len(average_values)
    print(f"Directory: {root}")
    print(f"JSON files with statistics.average: {len(average_values)} / {len(paths)}")
    print()
    print("Per-file scores:")
    for path, score in items:
        parent_name = os.path.basename(os.path.dirname(path))
        print(f"  {parent_name}: {score:.4f}")
    print()
    print(f"Overall average: {avg:.4f}")
    print(f"min: {min(average_values):.4f}, max: {max(average_values):.4f}")


if __name__ == "__main__":
    main()

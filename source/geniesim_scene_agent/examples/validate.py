# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Check example metadata, file integrity, Python syntax, and USD dependencies."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path


def validate(root: Path) -> dict:
    from pxr import Sdf

    errors = []
    counts = {"tasks": 0, "scenes": 0, "writing_scenes": 0,
              "python_files": 0, "usd_layers": 0}
    directories = sorted(p for p in (root / "tasks").iterdir() if p.is_dir())
    if not directories:
        errors.append("no task directories found")
    for directory in directories:
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            errors.append(f"{directory}: missing task manifest")
            continue
        manifest = json.loads(manifest_path.read_text())
        task = json.loads((directory / "task_info.json").read_text())
        if manifest.get("task_name") != directory.name or task.get("task_name") != directory.name:
            errors.append(f"{directory}: inconsistent task manifest identity")
        title = manifest.get("english_task_name")
        if task.get("task_description", {}).get("english_task_name") != title:
            errors.append(f"{directory}: inconsistent English task title")
        counts["tasks"] += 1
        scene_ids = [s["scene_id"] for s in manifest["scenes"]]
        if len(set(scene_ids)) != len(scene_ids):
            errors.append(f"{directory}: duplicate scene IDs")
        if len(scene_ids) != manifest["scene_count"]:
            errors.append(f"{directory}: scene count mismatch or shortfall")
        actual_ids = {p.name for p in directory.iterdir() if p.is_dir() and (p / "scene.usda").is_file()}
        if actual_ids != set(scene_ids):
            errors.append(f"{directory}: scene folders do not match its manifest")
        for source in manifest["files"]:
            path = directory / source["path"]
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]:
                errors.append(f"{path}: task file hash mismatch")
        for scene in manifest["scenes"]:
            sd = directory / scene["scene_id"]
            counts["scenes"] += 1
            info = json.loads((sd / "scene_info.json").read_text())
            if info.get("task_name") != directory.name or info.get("english_task_name") != title:
                errors.append(f"{sd}: inconsistent public task identity")
            if manifest.get("writing_contract"):
                counts["writing_scenes"] += 1
                letter = info.get("target_letter")
                if letter != scene.get("target_letter") or not isinstance(letter, str) or letter not in list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
                    errors.append(f"{sd}: invalid writing target_letter")
            if not isinstance(info.get("layout"), dict) or not info["layout"]:
                errors.append(f"{sd}: missing scene layout")
            for relative, expected in scene["files"].items():
                path = sd / relative
                if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                    errors.append(f"{path}: normalized scene file hash mismatch")
            for path in sd.rglob("*"):
                if path.suffix not in {".usd", ".usda", ".usdc"}:
                    continue
                counts["usd_layers"] += 1
                try:
                    layer = Sdf.Layer.FindOrOpen(str(path.resolve()))
                    if layer is None:
                        raise ValueError("cannot open USD layer")
                    for ref in layer.GetExternalReferences():
                        if Path(ref).is_absolute():
                            errors.append(f"{path}: absolute dependency {ref}")
                        if not (path.parent / ref).is_file():
                            errors.append(f"{path}: missing dependency {ref}")
                    if path.name == "scene.usda" and not layer.GetPrimAtPath("/World/Objects"):
                        errors.append(f"{path}: missing /World/Objects")
                except Exception as exc:
                    errors.append(f"{path}: {exc}")
    for path in (root / "tasks").rglob("*.py"):
        counts["python_files"] += 1
        try:
            ast.parse(path.read_text(), filename=str(path))
        except (SyntaxError, UnicodeError) as exc:
            errors.append(f"{path}: {exc}")
    return {"ok": not errors, "counts": counts, "errors": errors,
            "scope": "Task/scene inventory, file hashes, JSON metadata, Python syntax, USD layer parsing and direct dependencies. Robot trajectories, rendering and recursive external asset contents are not executed."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate(Path(__file__).resolve().parent)
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded)
    print(encoded, end="")
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()

# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Compile a portable download index from version-pinned dependency audits.

Requires PyYAML only for this maintainer operation; the installed CLI uses the
compiled JSON and the standard library. No asset bytes are copied into the wheel.
"""

import argparse
from collections import deque
import gzip
import hashlib
import json
from pathlib import Path
import re

import yaml


def compile_index(manifests, benchmark):
    """Require exact agreement between each audited group and its task closures."""
    files, tasks, groups, revisions, commits = {}, {}, {}, set(), set()
    for path in manifests:
        audit = json.loads(path.read_text())
        if audit["scan_errors"] or audit["source_content_mismatches"]:
            raise ValueError(f"Unverified asset audit: {path}")
        revisions.add(audit["revision"])
        commits.add(audit["benchmark_commit"])
        inventory = {r["path"]: {"size": r["size"], "sha256": r["sha256"]} for r in audit["files"]}
        for name, row in inventory.items():
            if name in files and files[name] != row:
                raise ValueError(f"Conflicting asset bytes: {name}")
            files[name] = row
        common = {
            n
            for n in inventory
            if "/" not in n or (n.startswith("background/") and Path(n).suffix.lower() in {".hdr", ".exr"})
        }
        edges = audit["edges"]
        union = set()
        group = []
        for task in audit["tasks"]:
            name = Path(task["config"]).stem
            roots = (
                set(common) | {p.lstrip("/") for p in task.get("background_entries", [])} | set(task["scene_entries"])
            )
            if task.get("robot_entry"):
                roots.add(task["robot_entry"])
            config = benchmark / "config" / (name + ".yaml")
            cfg = yaml.safe_load(config.read_text())
            runtime = task.get("runtime", "isaac")
            roots.add("code:config/" + name + ".yaml")
            if runtime == "isaac":
                settings = cfg["benchmark"]
                ep = "benchmark/config/eval_tasks/" + settings["task_name"] + ".json"
                evaluation = json.loads((benchmark / ep).read_text())
                rp = "app/robot_cfg/" + evaluation["robot"]["robot_cfg"]
                robot = json.loads((benchmark / rp).read_text())
                roots.update(["code:" + ep, "code:" + rp, robot["robot"]["robot_usd"].lstrip("/")])
                backgrounds = evaluation["scene"]["scene_usd"]
                roots.update(s.lstrip("/") for s in (backgrounds if isinstance(backgrounds, list) else [backgrounds]))
                prefix = "benchmark/config/llm_task/" + settings["sub_task_name"] + "/"
                roots.update("code:" + p for p in audit["code_files"] if p.startswith(prefix))
            if runtime == "mujoco":
                roots.update(n for n in inventory if n.startswith("mujoco/rubik_cube/"))
                roots.update(audit.get("runtime_extra_roots", []))
            seen, pending = set(), deque(roots)
            while pending:
                node = pending.popleft()
                if node in seen:
                    continue
                seen.add(node)
                pending.extend(edges.get(node, []))
            selected = seen & inventory.keys()
            absent = {n for n in seen if not n.startswith("code:")} - inventory.keys()
            if absent:
                raise ValueError(f"Closure absent from audited file set: {name}: {absent}")
            if name in tasks:
                raise ValueError(f"Duplicate task: {name}")
            tasks[name] = {
                "category": name.split("_")[1],
                "runtime": runtime,
                "instances": task["instances"],
                "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
                "files": sorted(selected),
            }
            group.append(name)
            union.update(selected)
        if union != inventory.keys():
            raise ValueError(
                f'Task union differs from audited group {audit["selector"]}: missing={sorted(inventory.keys()-union)} extra={sorted(union-inventory.keys())}'
            )
        groups[audit["selector"]] = group
    if len(revisions) != 1 or len(commits) != 1:
        raise ValueError("Audit versions must match")
    paths = sorted(files)
    ids = {name: i for i, name in enumerate(paths)}
    for task in tasks.values():
        task["files"] = [ids[p] for p in task["files"]]
    revision = revisions.pop()
    return {
        "schema_version": 1,
        "dataset": "agibot_world/GenieSimAssets",
        "endpoint": "https://www.modelscope.cn",
        "revision": revision,
        "benchmark_commit": commits.pop(),
        "files": [[p, files[p]["size"], files[p]["sha256"]] for p in paths],
        "tasks": tasks,
        "groups": groups,
        "known_issues": [
            {
                "tasks": ["g2crsbwuji_dexmanip_rubik_cube_solve"],
                "message": "Published MuJoCo package MANIFEST.json requires absent README.md; native initialization fails until the release is repaired.",
            }
        ],
    }


def attach_sources(index, huggingface_revision, default):
    """Record identical-content hosting mirrors as switchable sources.

    Each mirror is a separate repository with its own commit history, so each
    pins its own revision; the shared SHA256 file table guarantees the bytes
    are identical across sources. Only emitted when a mirror revision is given.
    """
    sources = {
        "modelscope": {
            "kind": "modelscope",
            "endpoint": index["endpoint"],
            "dataset": index["dataset"],
            "revision": index["revision"],
        }
    }
    if huggingface_revision:
        if not re.fullmatch(r"[a-f0-9]{40}", huggingface_revision):
            raise ValueError("HuggingFace revision must be a pinned commit, not a moving branch")
        sources["huggingface"] = {
            "kind": "huggingface",
            "endpoint": "https://huggingface.co",
            "dataset": "agibot-world/GenieSimAssets",
            "revision": huggingface_revision,
        }
    if default not in sources:
        raise ValueError(f"default source {default!r} is not among {sorted(sources)}")
    index["sources"] = sources
    index["default_source"] = default
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--huggingface-revision",
        help="Pinned HuggingFace mirror commit to record as an alternate source (content must match)",
    )
    parser.add_argument(
        "--default-source", default="modelscope", help="Source used when the CLI is not given --source"
    )
    parser.add_argument("manifests", nargs="+", type=Path)
    args = parser.parse_args()
    result = compile_index(args.manifests, args.benchmark_source)
    result = attach_sources(result, args.huggingface_revision, args.default_source)
    raw = (json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    if args.output.suffix == ".gz":
        args.output.write_bytes(gzip.compress(raw, mtime=0))
    else:
        args.output.write_bytes(raw)
    mirrors = ", ".join(f'{name}@{src["revision"][:12]}' for name, src in sorted(result["sources"].items()))
    print(f'{len(result["tasks"])} tasks, {len(result["files"])} files, sources: {mirrors}')


if __name__ == "__main__":
    main()

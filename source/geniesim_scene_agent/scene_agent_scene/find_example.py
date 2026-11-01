# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Build and search the Markdown task tables used by the find-example skill."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re

from .similar import _representative_scene, _terms


FAMILIES = {
    "stacking-and-assembly": "Stacking and assembly",
    "numbers-and-reasoning": "Numbers and reasoning",
    "letters-and-sequences": "Letters and sequences",
    "chess": "Chess",
    "spatial-relations": "Spatial relations",
    "drawers-and-shelves": "Drawers and shelves",
    "bimanual-and-held-containers": "Bimanual tasks and held containers",
    "storage-and-sorting": "Storage and sorting",
    "articulations": "Articulations and pressing",
    "insertion": "Insertion and fine assembly",
    "everyday-manipulation": "Everyday manipulation",
    "deformable-and-contact": "Deformable objects and contact trajectories",
}
CJK = re.compile(r"[\u3400-\u9fff]")
DEFAULT_INDEX = Path(__file__).resolve().parents[1] / "skills" / "find-example" / "references"
if not DEFAULT_INDEX.is_dir():
    from geniesim_scene_agent import resource_root

    DEFAULT_INDEX = resource_root() / "skills" / "find-example" / "references"
STOP_WORDS = {"a", "an", "and", "at", "by", "for", "from", "in", "into", "of",
              "on", "the", "then", "to", "using", "with", "each", "all", "both", "arms", "arm"}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def describe_task(directory: Path) -> dict:
    """Extract actual calls and scene keywords without importing task code."""
    manifest = _load(directory / "manifest.json")
    info = _load(directory / "task_info.json")
    source = (directory / "data_gen.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    methods, configs, client_methods = set(), set(), set()
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "client.llm_gen.robot_api":
            aliases.update({entry.asname or entry.name: entry.name for entry in node.names})
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "robot":
            methods.add(func.attr)
        elif (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Attribute)
              and isinstance(func.value.value, ast.Name) and func.value.value.id == "robot"
              and func.value.attr == "client"):
            client_methods.add(f"client.{func.attr}")
        elif (isinstance(func, ast.Name) and func.id == "getattr" and len(node.args) >= 2
              and isinstance(node.args[0], ast.Name) and node.args[0].id == "robot"
              and isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str)):
            methods.add(node.args[1].value)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = aliases.get(node.func.id, node.func.id)
            if name.endswith("Config") or name.startswith("Dual"):
                configs.add(name)

    frequency = Counter()
    assets, metadata_fields = set(), set()
    for path in sorted(directory.glob("*/scene_info.json")):
        scene = _load(path)
        if "target_letter" in scene:
            metadata_fields.add("target_letter")
        scene_words = set()
        for entry in scene.get("layout", {}).values():
            if not isinstance(entry, dict):
                continue
            if isinstance(entry.get("usd"), str):
                assets.add(entry["usd"])
            scene_words.update(word for word in entry.get("keywords", [])
                               if isinstance(word, str) and not CJK.search(word)
                               and re.search(r"[A-Za-z]", word) and len(word) <= 55
                               and "|" not in word and "`" not in word)
        frequency.update(scene_words)
    literals = {node.value for node in ast.walk(tree)
                if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    keywords = sorted(frequency, key=lambda word: (word not in literals, -frequency[word], word))[:20]
    tags = {"keywords"} if frequency else set()
    asset_text = " ".join(assets).lower()
    name = manifest["task_name"]
    if "shelf" in asset_text or "shelf" in name:
        tags.add("shelf_scene")
    elif "drawer" in name or "drawer" in asset_text:
        tags.add("drawer_scene")
    elif any(word in asset_text for word in ("bowl", "box", "cup", "trash_can", "holder", "bag")):
        tags.add("table_with_containers")
    elif "table" in asset_text:
        tags.add("table_only")
    if any(word in name for word in ("microwave", "trash_can", "drawer", "button")):
        tags.add("table_with_articulation")
    for word, tag in (
        ("stack", "stack"), ("sort", "sort"), ("swap", "swap"),
        ("open", "open_action"), ("press", "press"), ("pour", "pour"),
        ("shake", "shake"), ("fold", "fold"), ("insert", "insertion"),
        ("plug", "insertion"), ("assemble", "assembly"), ("disassemble", "disassembly"),
        ("capture", "capture"), ("demonstration", "demo_reference"),
        ("hidden", "sequential_inspection"), ("relative", "relative_position"),
        ("held", "held_container"), ("write", "contact_trajectory"),
    ):
        if word in name.split("_"):
            tags.add(tag)
    if any("pick_and_place" in method for method in methods) or (
        "pick_actor" in methods and "place_actor_to_target_pose" in methods
    ):
        tags.add("pick_and_place")
    elif "pick_actor" in methods:
        tags.add("pick")
    if {"dual_act", "bimanual_rigid_move"} & methods:
        tags.add("bimanual")
    if "move_actor_to_target_pose" in methods:
        tags.add("intermediate_motion")
    if "follow_trajectory" in methods:
        tags.add("follow_trajectory")
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg in {"extra_checkers", "unmovable_objects", "path_constraint"}:
            tags.add(node.arg)
    tags.update(f"scene_info_{field}" for field in metadata_fields)
    if "observe" in name:
        tags.update({"observation_event", "sequential_press"})
    if "turn_to" in methods:
        tags.add("multi_table")
    if "get_cloth_nodal_positions" in methods:
        tags.add("deformable_cloth")
    family = manifest.get("task_family", "other")
    if family not in FAMILIES:
        family = "other"
    description = manifest["english_task_name"]
    if CJK.search(description):
        raise ValueError(f"expected an English description: {directory}")
    return {
        "task_name": name, "description": description,
        "robot": info.get("robot", {}).get("robot_id", "unknown"),
        "family": family, "tags": sorted(tags), "interfaces": sorted(methods),
        "configs": sorted(configs), "keywords": keywords,
        "client_interfaces": sorted(client_methods),
        "data_gen": str((directory / "data_gen.py").resolve()),
    }


def _codes(values: list[str]) -> str:
    return " ".join(f"`{value.replace('|', '/').replace('`', '')}`" for value in values) or "-"


def write_task_interface_docs(task_root: str | Path) -> int:
    """Refresh each task's interface tags and generated local reference page."""
    count = 0
    overview_rows = []
    reference = DEFAULT_INDEX.parents[1] / "geniesim-task-orchestration/references/robot_api_reference.md"
    for directory in sorted(Path(task_root).resolve().iterdir()):
        if not directory.is_dir() or not (directory / "manifest.json").is_file():
            continue
        task = describe_task(directory)
        manifest = _load(directory / "manifest.json")
        interfaces = [f"robot.{method}" for method in task["interfaces"]]
        clients = [f"robot.{method}" for method in task["client_interfaces"]]
        manifest["interface_tags"] = interfaces + clients
        manifest["configuration_tags"] = task["configs"]
        manifest["scene_action_tags"] = task["tags"]
        overview_rows.append((directory, task, interfaces + clients))
        api_link = Path(os.path.relpath(reference, directory)).as_posix()
        lines = [f"# {task['description']}", "", f"Task: `{task['task_name']}`. Robot: `{task['robot']}`.", "",
                 "## Task and scene references", "",
                 "- [Orchestration code](data_gen.py)", "- [Robot and recording configuration](task_info.json)",
                 "- [Scene inventory and interface tags](manifest.json)",
                 f"- [Complete API parameter reference]({api_link})", "",
                 "## Interface tags", "",
                 "These tags are extracted from data_gen.py without executing it. They describe static calls",
                 "and literal capability lookups, not physical success or coverage of every branch.", "",
                 "### RobotAPI", "", _codes(interfaces), "", "### Client RPC", "", _codes(clients), "",
                 "### Configuration classes", "", _codes(task["configs"]), "",
                 "## Scene and action tags", "", _codes(task["tags"]), "",
                 "## Scene keyword sample", "", _codes(task["keywords"]), "", "## Scene instances", "",
                 "All instances below share the task code and interface tags above. Inspect their actual",
                 "metadata and USD before adapting target selection, poses, or checkers.", "",
                 "| Instance | Metadata | USD |", "| --- | --- | --- |"]
        for scene in manifest.get("scenes", []):
            sid = scene["scene_id"]
            lines.append(f"| `{sid}` | [scene_info.json]({sid}/scene_info.json) | [scene.usda]({sid}/scene.usda) |")
        if manifest.get("writing_contract"):
            lines += ["", "The writing instances specify `scene_info.target_letter` (A-Z).",
                      "The task reads this field for both the instruction and the contact strokes."]
        (directory / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        files = [item for item in manifest.get("files", []) if item["path"] != "README.md"]
        files.append({"path": "README.md", "sha256": hashlib.sha256((directory / "README.md").read_bytes()).hexdigest()})
        manifest["files"] = sorted(files, key=lambda item: item["path"])
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        count += 1
    overview = Path(task_root).resolve().parent / "README.md"
    if overview.is_file():
        start, end = "<!-- BEGIN TASK INTERFACE INDEX -->", "<!-- END TASK INTERFACE INDEX -->"
        lines = [start, "", "## Task interface index", "",
                 "Each task links to its local guide and scene instances. Interface and configuration tags",
                 "are static code observations. Use the complete API reference for parameter semantics.", "",
                 "| Task | Description | Family | Robot | Interfaces | Configurations |",
                 "| --- | --- | --- | --- | --- | --- |"]
        for directory, task, interfaces in overview_rows:
            link = Path(os.path.relpath(directory / "README.md", overview.parent)).as_posix()
            cells = [f"[{task['task_name']}]({link})", task["description"].replace("|", "/"),
                     task["family"], task["robot"], _codes(interfaces), _codes(task["configs"])]
            lines.append("| " + " | ".join(cells) + " |")
        lines.extend(["", end])
        block = "\n".join(lines)
        content = overview.read_text(encoding="utf-8")
        if start in content and end in content:
            before, rest = content.split(start, 1)
            _, after = rest.split(end, 1)
            content = before + block + after
        else:
            content = content.rstrip() + "\n\n" + block + "\n"
        overview.write_text(content, encoding="utf-8")
    return count


def build_index(task_root: str | Path, index_root: str | Path = DEFAULT_INDEX) -> int:
    """Refresh English Markdown tables; no JSON catalog is generated."""
    root, output = Path(task_root).resolve(), Path(index_root).resolve()
    tasks = [describe_task(path) for path in sorted(root.iterdir()) if path.is_dir()]
    output.mkdir(parents=True, exist_ok=True)
    categories = dict(FAMILIES)
    categories["other"] = "Other tasks"
    index = ["# Task reference tables", "", f"The tables cover **{len(tasks)} current tasks**.", "",
             "Read the relevant task family first, then compare several examples.", "",
             "| Task family | Tasks | Table |", "| --- | ---: | --- |"]
    for category, title in categories.items():
        members = [task for task in tasks if task["family"] == category]
        if not members:
            continue
        filename = f"{category}.md"
        index.append(f"| {title} | {len(members)} | [{category}]({filename}) |")
        lines = [f"# {title}", "",
                 "API columns list RobotAPI and direct client calls found in each example.",
                 "Keywords are a compact sample of literal scene keywords, prioritizing those used by task code. Read scene_info.json for the complete object contract.",
                 "", "| Task / code | Brief description | Robot | Scene and action tags | Interfaces | Configs | Scene keywords |",
                 "| --- | --- | --- | --- | --- | --- | --- |"]
        for task in members:
            relative = Path(os.path.relpath(task["data_gen"], output)).as_posix()
            lines.append("| " + " | ".join([
                f"[{task['task_name']}]({relative})", task["description"].replace("|", "/"),
                task["robot"], _codes(task["tags"]),
                _codes(task["interfaces"] + task["client_interfaces"]),
                _codes(task["configs"]), _codes(task["keywords"]),
            ]) + " |")
        (output / filename).write_text("\n".join(lines) + "\n", encoding="utf-8")
    index.extend(["", "## Field notes", "",
                  "- Each task link opens its actual data_gen.py; task_info.json and scene folders are alongside it.",
                  "- Similar robot profiles are separate rows; search results diversify equivalent implementations.",
                  "- scene_info_target_letter marks the scene-defined A-Z writing contract.",
                  "- Configs and calls are extracted from code without executing it. Keyword samples span the stored scenes.",
                  "- Each task also has a README and manifest interface tags; refresh them and the library overview with find_example tag.",
                  "- Validate adapted task code in the configured robot runtime."])
    (output / "index.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    return len(tasks)


def read_index(index_root: str | Path = DEFAULT_INDEX) -> list[dict]:
    root = Path(index_root).resolve()
    entries = []
    for filename in re.findall(r"\]\(([a-z0-9-]+\.md)\)", (root / "index.md").read_text()):
        path = root / filename
        for line in path.read_text().splitlines():
            if not line.startswith("| ["):
                continue
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if len(cells) != 7:
                raise ValueError(f"invalid task table row in {path}")
            link = re.fullmatch(r"\[([a-z0-9_]+)\]\(([^)]+)\)", cells[0])
            if link is None:
                raise ValueError(f"invalid task link in {path}")
            code = (path.parent / link[2]).resolve()
            scene = _representative_scene(code.parent)
            if not code.is_file() or scene is None:
                raise ValueError(f"task reference is missing code or a scene: {code.parent}")
            interfaces = re.findall(r"`([^`]+)`", cells[4])
            entries.append({
                "task_name": link[1], "description": cells[1], "robot": cells[2],
                "task_readme": str(code.parent / "README.md") if (code.parent / "README.md").is_file() else None,
                "family": path.stem, "tags": re.findall(r"`([^`]+)`", cells[3]),
                "interfaces": [item for item in interfaces if not item.startswith("client.")],
                "client_interfaces": [item for item in interfaces if item.startswith("client.")],
                "configs": re.findall(r"`([^`]+)`", cells[5]),
                "keywords": re.findall(r"`([^`]+)`", cells[6]),
                "reference_table": str(path), "data_gen": str(code),
                "task_info": str(code.parent / "task_info.json"),
                "manifest": str(code.parent / "manifest.json"),
                "scene_info": str(scene / "scene_info.json"),
                "scene_usda": str(scene / "scene.usda"),
                "scene_recipes": [str(recipe) for recipe in sorted(code.parent.glob("LLM_RESULT*.py"))],
                "task_documents": [str(document) for document in sorted(code.parent.glob("*.md"))],
            })
    return entries


def find_examples(query: str, index_root: str | Path = DEFAULT_INDEX, *, limit: int = 5,
                  robot: str | None = None, apis: list[str] | None = None,
                  tags: list[str] | None = None) -> list[dict]:
    if limit < 1:
        raise ValueError("limit must be positive")
    terms = _terms(query) - STOP_WORDS
    if "hold" in terms:
        terms.add("held")
    if re.search(r"both\s+arms|dual[-_ ]arm|two\s+hands", query, re.IGNORECASE):
        terms.add("bimanual")
    if not terms:
        raise ValueError("provide action, scene, object, or API search terms")
    matches = []
    for row in read_index(index_root):
        if robot and row["robot"].lower() != robot.lower():
            continue
        if not set(apis or []).issubset(row["interfaces"] + row["client_interfaces"]):
            continue
        if not set(tags or []).issubset(row["tags"]):
            continue
        fields = [(row["task_name"], 3), (row["description"], 4),
                  (" ".join(row["tags"]), 4), (" ".join(row["interfaces"] + row["configs"]), 2),
                  (" ".join(row["keywords"]), 2)]
        hit_terms = set()
        score = 0
        for text, weight in fields:
            hits = terms & _terms(text)
            hit_terms.update(hits)
            score += weight * len(hits)
        if hit_terms:
            matches.append({**row, "score": score, "matched_terms": sorted(hit_terms)})
    matches.sort(key=lambda row: (-row["score"], row["robot"] != "G2", row["task_name"]))
    diverse, seen = [], set()
    minimum_score = matches[0]["score"] * 0.35 if matches else 0
    for row in matches:
        if row["score"] < minimum_score:
            continue
        description = re.sub(r" \(scene set \d+\)$", "", row["description"])
        signature = (description, tuple(row["interfaces"]), tuple(row["configs"]))
        if signature in seen:
            continue
        seen.add(signature)
        diverse.append(row)
        if len(diverse) >= limit:
            break
    return diverse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    search = commands.add_parser("search", help="Find several reference tasks from Markdown tables")
    search.add_argument("query")
    search.add_argument("--index-root", type=Path, default=DEFAULT_INDEX)
    search.add_argument("--limit", type=int, default=5)
    search.add_argument("--robot")
    search.add_argument("--api", action="append", dest="apis")
    search.add_argument("--tag", action="append", dest="tags")
    rebuild = commands.add_parser("rebuild", help="Refresh tables from task code and scene metadata")
    rebuild.add_argument("--task-root", type=Path, default=Path("examples/tasks"))
    rebuild.add_argument("--index-root", type=Path, default=DEFAULT_INDEX)
    tags = commands.add_parser("tag", help="Refresh task-local README files and manifest interface tags")
    tags.add_argument("--task-root", type=Path, default=Path("examples/tasks"))
    args = parser.parse_args(argv)
    try:
        if args.command == "rebuild":
            count = build_index(args.task_root, args.index_root)
            print(f"Indexed {count} tasks in Markdown tables: {args.index_root}")
        elif args.command == "tag":
            count = write_task_interface_docs(args.task_root)
            print(f"Updated interface tags and local documentation for {count} tasks")
        else:
            matches = find_examples(args.query, args.index_root, limit=args.limit,
                                    robot=args.robot, apis=args.apis, tags=args.tags)
            print(json.dumps({"query": args.query, "match_count": len(matches), "matches": matches},
                             ensure_ascii=False, indent=2))
    except (OSError, ValueError, KeyError, SyntaxError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

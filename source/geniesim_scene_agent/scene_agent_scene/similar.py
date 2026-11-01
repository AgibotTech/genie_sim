# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Find comparable GenieSim tasks without importing Isaac Sim or USD."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path


def _terms(value: str) -> set[str]:
    text = value.lower()
    terms = set()
    for word in re.findall(r"[a-z0-9]+", text):
        if len(word) >= 2:
            terms.add(word)
    for span in re.findall(r"[\u3400-\u9fff]+", text):
        for size in (2, 3):
            terms.update(span[index:index + size] for index in range(len(span) - size + 1))
    return terms


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _representative_scene(task_dir: Path) -> Path | None:
    preferred = task_dir / "0"
    if (preferred / "scene.usda").is_file() and (preferred / "scene_info.json").is_file():
        return preferred
    for child in sorted(task_dir.iterdir()):
        if child.is_dir() and (child / "scene.usda").is_file() and (child / "scene_info.json").is_file():
            return child
    return None


def _description(task_info: dict) -> str:
    value = task_info.get("task_description", "")
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(str(value.get(key, "")) for key in
                        ("task_name", "english_task_name", "init_scene_text") if value.get(key))
    return ""


def find_similar_scenes(query: str, task_roots: list[str | Path], limit: int = 5) -> list[dict]:
    """Rank task-level scene examples by name, description, asset IDs, and keywords."""
    if limit < 1:
        raise ValueError("limit must be positive")
    query_terms = _terms(query)
    if not query_terms:
        raise ValueError("query needs English words or at least two Chinese characters")
    matches = []
    for root_value in task_roots:
        root = Path(root_value).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(root)
        for task_dir in sorted(root.iterdir()):
            if not task_dir.is_dir():
                continue
            scene_dir = _representative_scene(task_dir)
            if scene_dir is None:
                continue
            info = _read_json(task_dir / "task_info.json")
            scene_info = _read_json(scene_dir / "scene_info.json")
            layout = scene_info.get("layout", {})
            if not isinstance(layout, dict):
                continue
            assets = sorted({entry.get("usd") for entry in layout.values()
                             if isinstance(entry, dict) and isinstance(entry.get("usd"), str)})
            keywords = sorted({str(keyword) for entry in layout.values() if isinstance(entry, dict)
                               for keyword in entry.get("keywords", []) if isinstance(keyword, str)})
            description = _description(info)
            fields = ((task_dir.name, 3), (description, 3), (" ".join(assets), 2),
                      (" ".join(keywords), 2))
            field_terms = [(_terms(text), weight) for text, weight in fields]
            hits = query_terms & set().union(*(terms for terms, _ in field_terms))
            if not hits:
                continue
            score = sum(weight * len(query_terms & terms) for terms, weight in field_terms)
            preview = next((str(path) for path in (
                scene_dir / "renders" / "front.png", task_dir / "snapshot" / "0.png")
                if path.is_file()), None)
            matches.append({
                "task_name": info.get("task_name") or task_dir.name,
                "task_dir": str(task_dir), "scene_dir": str(scene_dir),
                "scene_usda": str(scene_dir / "scene.usda"),
                "scene_info": str(scene_dir / "scene_info.json"),
                "preview": preview, "description": description,
                "assets": assets, "keywords": keywords,
                "matched_terms": sorted(hits), "score": score,
            })
    matches.sort(key=lambda match: (-match["score"], match["task_dir"]))
    diverse = []
    seen = set()
    for match in matches:
        signature = (match["description"], tuple(match["assets"]), tuple(match["keywords"]))
        if signature in seen:
            continue
        seen.add(signature)
        diverse.append(match)
        if len(diverse) == limit:
            break
    return diverse


def _markdown_field(text: str, name: str) -> str:
    match = re.search(r"^\*\*" + re.escape(name) + r":\*\*\s*(.+)$", text, flags=re.MULTILINE | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def find_similar_examples(query: str, example_roots: list[str | Path], limit: int = 5) -> list[dict]:
    """Search tagged INPUT/OUTPUT example documents without loading their Python code."""
    if limit < 1:
        raise ValueError("limit must be positive")
    query_terms = _terms(query)
    if not query_terms:
        raise ValueError("query needs English words or at least two Chinese characters")
    matches = []
    for root_value in example_roots:
        root = Path(root_value).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(root)
        for path in sorted(root.glob("*.md")):
            if path.name.lower() == "readme.md":
                continue
            try:
                document = path.read_text(encoding="utf-8")
            except OSError:
                continue
            tags = re.findall(r"`([^`]+)`", _markdown_field(document, "Tags"))
            scene_description = _markdown_field(document, "scene_description")
            task_description = _markdown_field(document, "task description")
            if not tags and not scene_description:
                continue
            title = document.splitlines()[0].lstrip("# ").strip() if document else path.stem
            fields = ((title, 2), (" ".join(tags), 3), (scene_description, 3), (task_description, 1))
            field_terms = [(_terms(value), weight) for value, weight in fields]
            hits = query_terms & set().union(*(terms for terms, _ in field_terms))
            if not hits:
                continue
            score = sum(weight * len(query_terms & terms) for terms, weight in field_terms)
            matches.append({"title": title, "example_file": str(path), "tags": tags,
                            "scene_description": scene_description,
                            "task_description": task_description,
                            "matched_terms": sorted(hits), "score": score})
    matches.sort(key=lambda match: (-match["score"], match["example_file"]))
    return matches[:limit]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Find similar existing GenieSim scenes")
    parser.add_argument("query", help="Scene description; search Chinese and English terms separately if useful")
    parser.add_argument("--task-root", action="append", dest="task_roots",
                        help="Directory containing existing task directories; repeat for multiple libraries")
    parser.add_argument("--example-root", action="append", dest="example_roots",
                        help="Directory of tagged task example Markdown files; repeat if needed")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args(argv)
    roots = args.task_roots or [part for part in os.environ.get("GENIESIM_SCENE_LIBRARY", "").split(os.pathsep) if part]
    example_roots = args.example_roots
    if example_roots is None:
        example_roots = [str(Path(root) / "example") for root in roots if (Path(root) / "example").is_dir()]
    if not roots and not example_roots:
        parser.error("provide --task-root, --example-root, or GENIESIM_SCENE_LIBRARY")
    try:
        scenes = find_similar_scenes(args.query, roots, args.limit)
        examples = find_similar_examples(args.query, example_roots, args.limit)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    print(json.dumps({"query": args.query, "task_roots": roots, "example_roots": example_roots,
                      "matches": scenes, "examples": examples}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

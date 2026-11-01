# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Build small, browsable tables from GenieSim's on-disk asset tree."""

from __future__ import annotations

import ast
import json
from collections import defaultdict
from pathlib import Path

from .location import resolve_assets_root


def _literal_dict(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        value = ast.literal_eval(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (SyntaxError, ValueError, UnicodeError):
        return {}


def _text(value) -> str:
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value or "")


def resolve_folder(objects_root: Path, folder: str | Path | None) -> Path:
    root = objects_root.resolve()
    selected = (root / (folder or ".")).resolve()
    if not selected.is_relative_to(root):
        raise ValueError(f"Folder must stay under {root}: {folder}")
    return selected


def in_folder(record: dict, folder: str | None) -> bool:
    if not folder or folder == ".":
        return True
    category = Path(record["category"])
    scope = Path(folder)
    return category == scope or scope in category.parents


def build_catalog(assets_root: str | Path | None, index_dir: str | Path, scope: str = ".") -> dict:
    assets_root = resolve_assets_root(assets_root)
    objects_root = assets_root / "objects"
    selected = resolve_folder(objects_root, scope)
    if not selected.is_dir():
        raise FileNotFoundError(selected)
    tables: dict[str, list[dict]] = defaultdict(list)
    for item_path in sorted(selected.rglob("item.py")):
        folder = item_path.parent
        item = _literal_dict(item_path)
        asset_id = item.get("id", folder.name)
        if not isinstance(asset_id, str) or not asset_id:
            continue
        usd_path = folder / "Aligned.usda"
        if not usd_path.is_file():
            continue
        description = _literal_dict(folder / "description.py")
        category = folder.parent.relative_to(objects_root).as_posix()
        visualization = assets_root / "extra" / "visualization" / asset_id
        video = visualization / "video" / "merged.mp4"
        snapshots = sorted((visualization / "snapshot").glob("*.png"))
        record = {
            "asset_id": asset_id,
            "category": category,
            "usd": str(usd_path),
            "item": str(item_path),
            "video": str(video) if video.is_file() else None,
            "snapshots": [str(path) for path in snapshots],
            "semantic_name": _text(description.get("semantic_name")),
            "description": _text(description.get("full_description")),
            "shapes": item.get("shapes", []),
        }
        tables[category].append(record)
    output = Path(index_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"assets_root": str(assets_root), "scope": scope, "asset_count": 0, "tables": {}}
    for category, records in sorted(tables.items()):
        table_path = Path("tables") / f"{category}.json"
        target = output / table_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest["tables"][category] = {"file": table_path.as_posix(), "count": len(records)}
        manifest["asset_count"] += len(records)
    (output / "index.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def load_catalog(index_dir: str | Path) -> list[dict]:
    root = Path(index_dir)
    manifest = json.loads((root / "index.json").read_text(encoding="utf-8"))
    records = []
    for entry in manifest["tables"].values():
        records.extend(json.loads((root / entry["file"]).read_text(encoding="utf-8")))
    return records


def search_catalog(index_dir: str | Path, query: str, folder: str | None = None, limit: int = 50) -> list[dict]:
    tokens = query.casefold().split()
    results = []
    for record in load_catalog(index_dir):
        if not in_folder(record, folder):
            continue
        haystack = " ".join(
            str(record.get(key) or "") for key in ("asset_id", "category", "semantic_name", "description")
        ).casefold()
        if all(token in haystack for token in tokens):
            results.append(record)
    return results[:limit]

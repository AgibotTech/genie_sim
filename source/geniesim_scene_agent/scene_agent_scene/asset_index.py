# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Read scene DSL asset metadata directly from an external assets tree."""

from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path


def _literal_dict(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        value = ast.literal_eval(path.read_text(encoding="utf-8"))
    except (SyntaxError, ValueError, UnicodeError):
        return {}
    return value if isinstance(value, dict) else {}


@lru_cache(maxsize=8)
def load_asset_index(assets_root: str | Path) -> dict[str, dict]:
    """Index object assets without importing code from the GenieSim repository."""
    root = Path(assets_root).expanduser().resolve()
    objects = root / "objects"
    if not objects.is_dir():
        raise FileNotFoundError(f"Expected an assets root containing objects/: {root}")
    records: dict[str, dict] = {}
    duplicates: set[str] = set()
    for item_path in sorted(objects.rglob("item.py")):
        asset_path = item_path.parent / "Aligned.usda"
        if not asset_path.is_file():
            continue
        item = _literal_dict(item_path)
        asset_id = item.get("id", item_path.parent.name)
        if not isinstance(asset_id, str) or not asset_id:
            continue
        record = {
            "id": asset_id,
            "path": asset_path,
            "url": asset_path.relative_to(root).as_posix(),
            "shapes": item.get("shapes", []),
            "description": _literal_dict(item_path.parent / "description.py"),
        }
        if asset_id in records:
            current = records[asset_id]
            current_exact = current["path"].parent.name == asset_id
            new_exact = item_path.parent.name == asset_id
            if new_exact and not current_exact:
                records[asset_id] = record
                duplicates.discard(asset_id)
            elif new_exact == current_exact:
                duplicates.add(asset_id)
            continue
        records[asset_id] = record
    if not records:
        raise ValueError(f"No assets with item.py and Aligned.usda under {objects}")
    for asset_id in duplicates:
        records[asset_id]["ambiguous"] = True
    return records


def resolve_asset(index: dict[str, dict], asset_id: str) -> dict:
    try:
        record = index[asset_id]
    except KeyError as exc:
        raise KeyError(f"Asset ID not found in the external assets tree: {asset_id}") from exc
    if record.get("ambiguous"):
        raise ValueError(f"Asset ID occurs in multiple folders: {asset_id}")
    return record

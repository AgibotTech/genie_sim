# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# License: Mozilla Public License Version 2.0

"""Locate an external LLM task library without importing the simulation SDK."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

try:
    from .collection_settings import load_settings
except ImportError:
    from collection_settings import load_settings


def resolve_llm_task_root(module_dir: Path | None = None) -> Path | None:
    """Resolve configured, monorepo, installed example, or standalone task roots.

    A missing optional library returns None so normal JSON tasks still work.
    An invalid explicit setting fails before Docker creates a container.
    """
    for name in ("GENIESIM_LLM_TASK_ROOT", "LLM_TASK_ROOT"):
        value = os.environ.get(name)
        if value:
            root = Path(value).expanduser().resolve()
            if not root.is_dir():
                raise FileNotFoundError(f"{name} must point to an existing task library: {root}")
            return root

    saved = load_settings().get("llm_task_root")
    if saved:
        root = Path(saved).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Saved llm_task_root is unavailable: {root}; override it or rerun data-collection install")
        return root

    module_dir = (module_dir or Path(__file__).resolve().parent.parent).resolve()
    candidates = [module_dir.parent / "geniesim_scene_agent" / "examples" / "tasks"]
    try:
        spec = importlib.util.find_spec("geniesim_scene_agent")
    except (ImportError, ValueError):
        spec = None
    if spec:
        for location in spec.submodule_search_locations or []:
            package = Path(location)
            candidates.extend([
                package / "resources" / "examples" / "tasks",
                package.parent.parent / "examples" / "tasks",
            ])
    candidates.extend([module_dir / "llm_tasks_data_collection", module_dir / "llm_tasks"])
    assets_root = os.environ.get("GENIESIM_ASSETS_PATH") or os.environ.get("SIM_ASSETS")
    if assets_root:
        candidates.append(Path(assets_root).expanduser() / "llm_tasks")
    return next((root.resolve() for root in candidates if root.is_dir()), None)


if __name__ == "__main__":
    try:
        root = resolve_llm_task_root()
        if root:
            print(root)
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Pick a task's official background/robot pairing from the server's released suites."""

from __future__ import annotations

from typing import Optional


def official_task_name(task_metadata: dict, sub_task_name: str) -> Optional[str]:
    """Return the released ``task_name`` for ``sub_task_name``, or ``None`` if it has none.

    ``task_metadata`` is a ``task_metadata()`` reply; its ``suites`` hold the
    released pairings. The instruction-following suite is preferred, then the
    suite config name."""
    hits = [
        entry
        for entries in (task_metadata.get("suites") or {}).values()
        for entry in entries
        if entry.get("sub_task_name") == sub_task_name
    ]
    if not hits:
        return None
    hits.sort(key=lambda entry: (entry.get("category") != "if", entry.get("config", "")))
    return str(hits[0]["task_name"])

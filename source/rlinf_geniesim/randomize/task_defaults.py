# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Per-task default generalization, from ``config/task_generalization.yaml``.

The eval tasks' ``generalization`` blocks describe the benchmark's evaluation
protocol, which mostly leaves the robot's start pose alone. This file holds
the defaults rlinf_geniesim applies on top of that for every user: the
``defaults`` section for every task, a task's entry under ``tasks`` over it.
Values use the task-init ``randomization`` keys (see
``RandomizationSpec.resolve`` for the priority order); ``null`` turns a
dimension off for that task. A reset with seed -1 applies none of it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "config" / "task_generalization.yaml"
_SECTIONS = {"defaults", "tasks"}


def _load(path: Path) -> dict:
    doc = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(doc, dict) or set(doc) - _SECTIONS:
        raise ValueError(
            f"{path}: top-level keys must be {sorted(_SECTIONS)}, "
            f"got {sorted(doc) if isinstance(doc, dict) else type(doc).__name__}"
        )
    for name in ("defaults",):
        if not isinstance(doc.get(name) or {}, dict):
            raise ValueError(f"{path}: '{name}' must be a mapping")
    tasks = doc.get("tasks") or {}
    if not isinstance(tasks, dict) or any(not isinstance(v or {}, dict) for v in tasks.values()):
        raise ValueError(f"{path}: 'tasks' must map sub task names to mappings")
    return doc


def for_task(sub_task_name: str, *, path: Optional[Path] = None, instance_switch: Optional[dict] = None) -> dict:
    """The default ``randomization`` keys for ``sub_task_name``.

    A task that places the robot base per instance (``robot_init.json``, so
    ``instance_switch.robot_base_xy``) keeps the engine's one base channel for
    that: its default base perturbation is dropped rather than tripping the
    base-channel conflict (``check_base_xy_channel_conflict``).
    """
    doc = _load(path or DEFAULT_PATH)
    out = dict(doc.get("defaults") or {})
    out.update((doc.get("tasks") or {}).get(sub_task_name) or {})
    if (instance_switch or {}).get("robot_base_xy") and out.get("init_base_xy_thresh") is not None:
        out.pop("init_base_xy_thresh")
        print(
            f"[randomize.task_defaults] {sub_task_name}: per-instance robot poses place the "
            f"base, so the default base perturbation is skipped",
            flush=True,
        )
    return out

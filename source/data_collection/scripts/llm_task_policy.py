# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# License: Mozilla Public License Version 2.0
"""Collection limits shared by host CLI and container entrypoints."""

import json
from pathlib import Path


def validate_run_times(task_dir, scene, run_times):
    """Reject repeated collection when task or candidate-scene metadata enables deformables."""
    if type(run_times) is not int or run_times < 1:
        raise ValueError("run_times must be a positive integer")
    if run_times == 1:
        return
    task_dir = Path(task_dir)
    metadata = [task_dir / "task_info.json"]
    if scene == "random":
        metadata.extend(
            path / "task_info.json"
            for path in sorted(task_dir.iterdir())
            if path.is_dir() and (path / "scene.usda").is_file() and (path / "scene_info.json").is_file()
        )
    else:
        metadata.append(task_dir / scene / "task_info.json")
    for path in metadata:
        if path.is_file() and json.loads(path.read_text(encoding="utf-8")).get("enable_deformable", False):
            raise ValueError(
                f"{task_dir.name}: deformable tasks do not support run_times > 1 "
                f"(enabled in {path}); use --run-times 1"
            )

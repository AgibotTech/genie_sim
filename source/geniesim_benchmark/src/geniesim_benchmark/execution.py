# Copyright (c) 2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0
"""Simulator-independent public launch and artifact locations."""

import argparse
import os
from pathlib import Path


def boolean(value):
    if isinstance(value, bool):
        return value
    if value.lower() not in ("true", "false"):
        raise argparse.ArgumentTypeError("expected true or false")
    return value.lower() == "true"


def task_output_directory(config, root):
    """Use EvaluationSummary's root/task/sub-task layout for native tasks too."""
    benchmark = config.get("benchmark", {})
    task = benchmark.get("task_name") or config.get("task_name")
    sub_task = benchmark.get("sub_task_name") or task
    if any(
        not isinstance(name, str) or not name or Path(name).name != name or name in (".", "..")
        for name in (task, sub_task)
    ):
        raise ValueError("task and sub-task names must be local directory names")
    return Path(root).expanduser().resolve() / task / sub_task


def output_root(config_path, configured=None):
    """Explicit root, shared platform environment, then the checkout default."""
    value = configured or os.environ.get("GENIESIM_BENCHMARK_OUTPUT_ROOT") or os.environ.get("OUTPUT_DIR")
    if value:
        return Path(value).expanduser().resolve()
    root = Path(os.environ["SIM_REPO_ROOT"]) if os.environ.get("SIM_REPO_ROOT") else None
    if root is None:
        root = next(
            (
                p
                for p in Path(config_path).resolve().parents
                if (p / "source/geniesim_benchmark/pyproject.toml").is_file()
            ),
            Path.cwd(),
        )
    return (root / "output/benchmark").resolve()

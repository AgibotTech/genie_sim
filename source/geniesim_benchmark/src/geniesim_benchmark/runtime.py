# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Inspect benchmark-owned configuration before loading a simulator runtime."""

from pathlib import Path

import yaml


def runtime_for_config(path: Path, overrides=()) -> str:
    config = yaml.safe_load(path.read_text()) or {}
    if not isinstance(config, dict):
        raise ValueError("benchmark configuration must be a mapping")
    runtime = config.get("runtime")
    if "runtime" not in config:
        benchmark = config.get("benchmark", {})
        if not isinstance(benchmark, dict):
            raise ValueError("benchmark configuration must contain a benchmark mapping")
        backend = benchmark.get("backend", "isaacsim")
        for argument in overrides:
            if argument.startswith("--benchmark.backend="):
                backend = argument.split("=", 1)[1]
        if backend not in ("isaacsim", "mujoco"):
            raise ValueError(f"Unsupported benchmark backend: {backend!r}")
        runtime = "mujoco" if backend == "mujoco" else "isaac"
    if runtime not in ("isaac", "newton", "mujoco"):
        raise ValueError(f"Unsupported benchmark runtime: {runtime!r}")
    return runtime

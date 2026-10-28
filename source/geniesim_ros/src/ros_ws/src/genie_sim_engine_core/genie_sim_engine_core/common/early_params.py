# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Pre-bootstrap ROS-arg parsing utilities.

These helpers read individual ROS 2 parameters from ``sys.argv`` **before**
any omni / isaacsim import, so the result is available to gate which
bootstrap path (Kit vs Kit-free) to take.

Both entry points (``genie_sim_engine_isaacsim`` and
``genie_sim_engine_newton``) use the same parsing logic, so it lives here
rather than being copy-pasted.
"""

from __future__ import annotations


def _early_param_from_file(path: str, name: str):
    """Read a single param from a ROS 2 params YAML. Returns str or None."""
    try:
        import yaml

        with open(path, "r") as f:
            doc = yaml.safe_load(f) or {}
        for node_block in doc.values():
            if not isinstance(node_block, dict):
                continue
            ros_params = node_block.get("ros__parameters")
            if isinstance(ros_params, dict) and name in ros_params:
                v = ros_params[name]
                return "true" if v is True else "false" if v is False else str(v)
    except Exception:
        pass
    return None


def _early_param(argv, name: str, default: str) -> str:
    """Extract a single ROS param from --ros-args before SimulationApp is created."""
    prefix = f"{name}:="
    file_value = None
    it = iter(argv)
    for a in it:
        if a == "-p":
            kv = next(it, "")
            if kv.startswith(prefix):
                return kv.split(":=", 1)[1].strip()
        elif a.startswith("-p="):
            kv = a[3:]
            if kv.startswith(prefix):
                return kv.split(":=", 1)[1].strip()
        elif a == "--params-file":
            path = next(it, "")
            if path:
                v = _early_param_from_file(path, name)
                if v is not None:
                    file_value = v
        elif a.startswith("--params-file="):
            path = a[len("--params-file=") :]
            if path:
                v = _early_param_from_file(path, name)
                if v is not None:
                    file_value = v
    return file_value if file_value is not None else default

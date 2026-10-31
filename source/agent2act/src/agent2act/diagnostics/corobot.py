# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Native-parser compatibility checks, usable from the installed CLI or Python API."""

import ast
import logging
from pathlib import Path
from typing import Any

from agent2act.adapters.simulators.corobot import encode_result
from agent2act.domain.errors import ContractError
from agent2act.domain.models import ActionChunk


def check_compatibility(source: Path) -> list[str]:
    try:
        import numpy as np
    except ImportError as exc:
        raise ContractError("compat requires the pi05 extra (uv sync --extra pi05)") from exc
    try:
        tree = ast.parse(source.read_text(encoding="utf-8"))
    except SyntaxError as exc:
        raise ContractError("source is not valid Python") from exc
    candidates = [
        method
        for cls in tree.body
        if isinstance(cls, ast.ClassDef) and cls.name == "CoRobotPolicy"
        for method in cls.body
        if isinstance(method, ast.FunctionDef) and method.name == "_parse_result"
    ]
    if len(candidates) != 1:
        raise ContractError("source must define CoRobotPolicy._parse_result exactly once")
    method = candidates[0]
    method.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    namespace: dict[str, Any] = {"np": np, "logger": logging.getLogger(__name__)}
    # Explicitly requested local Python source, not network content or model output.
    exec(compile(module, str(source), "exec"), namespace)
    messages = []
    for kind, row in (("JOINT_ABS", [0.0] * 7), ("EEF_ABS", [0.4, 0.2, 0.5, 0.0, 0.0, 0.0, 1.0])):
        chunk = ActionChunk(kind, "base_link", 0.05, [row], [row], [[0.5, 0.5]])
        result = encode_result(chunk, history={}, need_depth=False)["result"]
        parsed = namespace["_parse_result"](result)
        if len(parsed) != 1 or parsed[0]["kind"] != kind or parsed[0]["eef_frame"] != "base_link":
            raise ContractError(f"native parser returned unexpected {kind} interpretation")
        if parsed[0]["gripper"].tolist() != [0.5, 0.5]:
            raise ContractError("native parser changed gripper order or values")
        messages.append(f"{kind}: native CoRobot parser accepted 1 step")
    return messages

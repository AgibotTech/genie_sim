# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Protect dependency direction as features and adapters are added."""

import ast
import sys
from importlib.util import resolve_name
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[2] / "src" / "agent2act"


@pytest.mark.parametrize(
    "layer,allowed",
    [
        ("domain", {"domain"}),
        ("application", {"application", "domain"}),
        ("execution", {"execution", "domain"}),
        ("adapters", {"adapters", "domain"}),
    ],
)
def test_layers_do_not_depend_on_outer_layers(layer, allowed):
    violations = []
    for file in (SOURCE / layer).rglob("*.py"):
        package = ".".join(("agent2act", *file.relative_to(SOURCE).parts[:-1]))
        for node in ast.walk(ast.parse(file.read_text())):
            imports = []
            if isinstance(node, ast.Import):
                imports = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                name = "." * node.level + (node.module or "")
                imports = [resolve_name(name, package) if node.level else name]
            for name in imports:
                if name.startswith("agent2act.") and name.split(".")[1] not in allowed:
                    violations.append(f"{file.name}:{node.lineno} imports {name}")
                if layer != "adapters" and name.split(".")[0] not in sys.stdlib_module_names | {"agent2act"}:
                    violations.append(f"{file.name}:{node.lineno} imports external dependency {name}")
    assert violations == []

# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""The shared client and its examples must not import either application."""

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]


def test_shared_package_has_no_application_or_engine_imports():
    forbidden = {"rlinf_geniesim", "geniesim_benchmark", "geniesim_ros", "isaacsim", "warp", "newton"}
    for file in (PACKAGE / "genie_sim_client").rglob("*.py"):
        for node in ast.walk(ast.parse(file.read_text())):
            names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else ([node.module or ""] if isinstance(node, ast.ImportFrom) and not node.level else [])
            )
            assert all(name.split(".")[0] not in forbidden for name in names), file

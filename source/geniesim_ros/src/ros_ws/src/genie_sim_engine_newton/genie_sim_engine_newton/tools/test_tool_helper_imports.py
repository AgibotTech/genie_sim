#!/usr/bin/env python3

# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Helpers the ``tools/`` harnesses borrow from other packages must exist.

The tools in this directory are deliberately thin: each one drives the
real pipeline's own functions rather than re-implementing them, so a
harness and the thing it harnesses cannot disagree.  The cost of that
choice is a cross-package import for every borrowed helper, and those
imports are all **lazy** -- inside the stage function that needs them,
because the modules they come from are Isaac-coupled and must not be
paid for by a harness run that does not reach that stage.

Lazy plus cross-package is precisely the combination that rots in
silence.  ``test_assemble_robot.py`` sat broken for an unknown length of
time for exactly this reason: it imported six helpers from a top-level
``assemble_robot`` module, reached by inserting the parent directory on
``sys.path``.  The orchestrator had moved into
``genie_sim_engine_isaacsim.assemble._robot_impl``, so every one of
those imports raised ``ModuleNotFoundError`` -- but only at the moment a
stage actually ran, which no test and no collector ever does.  Nothing
in the tree noticed, while ``tools/AGENTS.md`` kept documenting the tool
across 130 lines and two other docs kept pointing readers at it.

The check is static for the usual reason: importing ``_robot_impl`` to
find out whether its names are importable needs Isaac, which is the
thing a host running this suite does not have.  So both sides are read
out of the AST -- the import in the harness, the ``def``/``class``/
assignment in the target -- and the test runs anywhere.

A name is accepted if the target module binds it (``def`` / ``class`` /
assignment / its own re-export) or -- for a package -- if it names a
submodule on disk, since ``from pkg import submodule`` binds nothing in
``pkg/__init__.py``.

What it cannot catch: a helper that still exists under the same name but
whose signature changed.  Nothing static can.  Keep the call sites in
the harness close enough to the real pipeline's own call sites that a
reviewer can compare them.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

_HERE = pathlib.Path(__file__).resolve()
_TOOLS = _HERE.parent
_SRC = _HERE.parents[3]  # ros_ws/src/, holding every genie_sim_engine_* package

# Cross-package prefixes worth checking.  A module path is resolved as
# ``<prefix>/<dotted path>.py``, i.e. the ROS convention where the package
# directory and its inner Python package share a name.
_PACKAGE_PREFIXES = (
    "genie_sim_engine_core",
    "genie_sim_engine_isaacsim",
    "genie_sim_engine_newton",
)


def _module_to_path(dotted: str) -> pathlib.Path | None:
    """``a.b.c`` -> ``<src>/a/a/b/c.py``, or the package's ``__init__.py``."""
    parts = dotted.split(".")
    if not parts or parts[0] not in _PACKAGE_PREFIXES:
        return None
    base = _SRC / parts[0]
    if not base.is_dir():
        return None
    candidate = base.joinpath(*parts)
    for path in (candidate.with_suffix(".py"), candidate / "__init__.py"):
        if path.is_file():
            return path
    return None


def _top_level_names(path: pathlib.Path) -> set[str]:
    """Names a module binds at top level: defs, classes, and assignments.

    Includes ``from x import y`` re-exports, since importing a name a
    module itself imported is legal and common.
    """
    names: set[str] = set()
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(a.asname or a.name.split(".")[0] for a in node.names)
        elif isinstance(node, (ast.If, ast.Try)):
            # ``try: import x except ImportError: x = None`` and
            # ``if TYPE_CHECKING:`` both bind names a caller may import.
            for sub in ast.walk(node):
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    names.add(sub.name)
                elif isinstance(sub, (ast.Import, ast.ImportFrom)):
                    names.update(a.asname or a.name.split(".")[0] for a in sub.names)
                elif isinstance(sub, ast.Assign):
                    names.update(t.id for t in sub.targets if isinstance(t, ast.Name))
    return names


def _cross_package_imports(tool: pathlib.Path) -> list[tuple[int, str, str]]:
    """``(lineno, module, name)`` for every cross-package ``from`` import."""
    out: list[tuple[int, str, str]] = []
    for node in ast.walk(ast.parse(tool.read_text())):
        if not isinstance(node, ast.ImportFrom) or node.level != 0 or not node.module:
            continue
        if node.module.split(".")[0] not in _PACKAGE_PREFIXES:
            continue
        for alias in node.names:
            out.append((node.lineno, node.module, alias.name))
    return out


_TOOL_FILES = sorted(p for p in _TOOLS.glob("*.py") if p.name != "__init__.py")


def test_there_are_tools_to_check() -> None:
    """Guard against the glob silently matching nothing."""
    assert len(_TOOL_FILES) > 5, f"only found {len(_TOOL_FILES)} tool modules; glob is wrong"


@pytest.mark.parametrize("tool", _TOOL_FILES, ids=lambda p: p.name)
def test_borrowed_helpers_still_exist(tool: pathlib.Path) -> None:
    missing: list[str] = []
    for lineno, module, name in _cross_package_imports(tool):
        target = _module_to_path(module)
        if target is None:
            missing.append(f"{tool.name}:{lineno} imports from '{module}', which resolves to no module on disk")
            continue
        if name == "*":
            continue
        # ``from pkg import submodule`` is legal and binds nothing in
        # ``pkg/__init__.py``, so check the package directory too.
        if target.name == "__init__.py":
            sibling = target.parent / name
            if sibling.with_suffix(".py").is_file() or (sibling / "__init__.py").is_file():
                continue
        if name not in _top_level_names(target):
            rel = target.relative_to(_SRC)
            missing.append(f"{tool.name}:{lineno} imports '{name}' from '{module}' ({rel}), which does not define it")

    assert not missing, (
        "A harness borrows a helper that no longer exists. These imports are lazy "
        "(inside the stage that needs them), so this raises only when someone runs "
        "that stage -- it will not show up in any test run.\n  " + "\n  ".join(missing)
    )

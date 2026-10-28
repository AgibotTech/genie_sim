#!/usr/bin/env python3

# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""No package ``__init__`` may import a hard-dependency module eagerly.

This guards a failure mode that looks nothing like a test failure.  ROS's
``launch_testing`` pytest plugin resolves its entrypoint by actually
importing every module it walks::

    launch_testing/pytest/hooks.py:193   find_launch_test_entrypoint
    launch_testing/pytest/hooks.py:186   module = path.pyimport()

So a module-level import that cannot be satisfied does not fail one test
-- it aborts the whole collection, and pytest reports ``1 error during
collection`` with zero tests run.  Every unrelated test in the tree
disappears at once, which reads as "the branch is broken" rather than
"one import is in the wrong place".

That is exactly what ``visualizers/__init__.py`` did: it re-exported
``InlineOvrtxVisualizer`` from ``ovrtx``, which imports the compiled
``genie_sim_engine_py`` at module scope.  Any host without the core built
-- i.e. any checkout before ``colcon build`` -- could not collect this
package at all, even though no test here touches OVRtx.  The fix was
:pep:`562` lazy re-exports; this test keeps it fixed.

Why the check is static: importing the subject to find out whether it
imports cleanly is the thing we cannot do on a host that is missing the
dependency.  So the names are read out of the AST instead, and the test
runs everywhere.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

_HERE = pathlib.Path(__file__).resolve()
_INNER_PKG = _HERE.parents[1]  # genie_sim_engine_newton/

# Modules that import a dependency which is NOT guaranteed present at
# collection time, mapped to the dependency that makes them unsafe.
# Keyed by path relative to the inner package.
_HARD_DEP_MODULES = {
    "engine/newton/visualizers/ovrtx.py": "genie_sim_engine_py (compiled core)",
}

# ``__init__.py`` files that must not import any of the above eagerly.
_GUARDED_INITS = [
    "engine/newton/visualizers/__init__.py",
]


def _module_level_imports(path: pathlib.Path) -> list[str]:
    """Dotted names imported at module scope, excluding ``TYPE_CHECKING``.

    A ``if TYPE_CHECKING:`` block is skipped deliberately -- it never runs
    at import time, so it cannot abort collection, and it is how the
    re-exports keep working for type checkers.
    """
    tree = ast.parse(path.read_text())
    out: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.If):
            test = node.test
            name = getattr(test, "id", None) or getattr(test, "attr", None)
            if name == "TYPE_CHECKING":
                continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Import):
                out.extend(a.name for a in sub.names)
            elif isinstance(sub, ast.ImportFrom) and sub.level == 0 and sub.module:
                out.append(sub.module)
    return out


@pytest.mark.parametrize("rel", _HARD_DEP_MODULES)
def test_the_hard_dep_inventory_is_still_accurate(rel: str) -> None:
    """Each listed module must still exist and still import its dependency.

    If a module gets its import made lazy upstream, drop it from the map --
    otherwise this file claims a constraint that no longer applies and the
    next person cannot tell which entries are real.
    """
    path = _INNER_PKG / rel
    assert path.is_file(), f"{rel} no longer exists; drop it from _HARD_DEP_MODULES"
    assert "genie_sim_engine_py" in _module_level_imports(path), (
        f"{rel} no longer imports genie_sim_engine_py at module scope -- it is "
        f"collection-safe now, so remove it from _HARD_DEP_MODULES"
    )


@pytest.mark.parametrize("rel", _GUARDED_INITS)
def test_package_init_does_not_eagerly_import_a_hard_dep(rel: str) -> None:
    init = _INNER_PKG / rel
    assert init.is_file(), rel
    pkg_dir = init.parent
    imported = _module_level_imports(init)

    for dep_rel, why in _HARD_DEP_MODULES.items():
        dep = _INNER_PKG / dep_rel
        try:
            inside = dep.relative_to(pkg_dir)
        except ValueError:
            continue  # not a submodule of this package
        submodule = inside.with_suffix("").as_posix().replace("/", ".")
        offenders = [name for name in imported if name.split(".")[-1] == submodule.split(".")[-1]]
        assert not offenders, (
            f"{rel} imports '{submodule}' at module scope, which pulls in "
            f"{why}.  ROS's launch_testing collector imports every module it "
            f"walks, so this aborts pytest collection for the ENTIRE package "
            f"on any host without that dependency.  Re-export it lazily with "
            f"a PEP 562 module __getattr__ instead."
        )


def test_the_lazy_reexport_contract_is_complete() -> None:
    """Every name in ``__all__`` must be resolvable by the lazy mapping.

    A PEP 562 ``__getattr__`` trades an import error at module scope for an
    ``AttributeError`` at first access, so a name present in ``__all__`` but
    missing from the lookup table fails only when someone touches it.  This
    pairs the two lists statically, with no import.
    """
    init = _INNER_PKG / "engine" / "newton" / "visualizers" / "__init__.py"
    tree = ast.parse(init.read_text())

    declared: set[str] = set()
    lazy: set[str] = set()
    for node in tree.body:
        # Accept both ``X = ...`` and the annotated ``X: dict = ...``.
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        for target in targets:
            if not isinstance(target, ast.Name):
                continue
            if target.id == "__all__" and isinstance(value, (ast.List, ast.Tuple)):
                declared = {e.value for e in value.elts if isinstance(e, ast.Constant)}
            elif target.id == "_LAZY" and isinstance(value, ast.Dict):
                lazy = {k.value for k in value.keys if isinstance(k, ast.Constant)}

    assert declared, "could not read __all__ out of visualizers/__init__.py"
    assert lazy, "could not read _LAZY out of visualizers/__init__.py"
    assert declared == lazy, (
        f"__all__ and _LAZY disagree: only in __all__ {sorted(declared - lazy)}, "
        f"only in _LAZY {sorted(lazy - declared)}.  A name in __all__ with no "
        f"_LAZY entry raises AttributeError on first access."
    )

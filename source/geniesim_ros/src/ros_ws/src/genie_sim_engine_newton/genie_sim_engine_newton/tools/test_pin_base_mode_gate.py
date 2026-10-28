#!/usr/bin/env python3

# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""``pin_base_mode: mocap`` must never reach an adapter that cannot hold the base.

What goes wrong without the gate
--------------------------------
``mocap`` keeps the robot root FREE joint — ``setup/stage.py`` deactivates the
URDF ``root_joint`` weld exactly as it does for ``free`` — and then relies on
the adapter clamping that joint every substep.  Only
``MuJoCoWarpMultiEnvAdapter`` implements the clamp (``_build_base_pin`` unions
the root free joint's qpos/qvel slots into its KINEMATIC-body pin).  On any
other solver the weld came off and nothing replaced it: the base falls under
full gravity while the startup log still reads

    [engine] pin_base_mode=mocap (explicit)

which reads as a mysterious physics problem rather than a configuration error.

So the capability is declared (``SolverAdapter.supports_base_pin``) and
``engine_base`` raises on it, the same shape as the cloth gate in
``setup/model.py``.  This file guards that declaration.

Why source-level rather than importing the adapters
---------------------------------------------------
The adapter modules import ``newton`` / ``warp`` at module scope, so importing
them needs the GPU stack.  The declaration being guarded is a plain class
attribute, so ``ast`` reads it out of the file — and the inventory check means
a NEW adapter file fails this test instead of silently defaulting to whichever
answer its class body happens to give.  Same tactic, same reason, as
``test_mjw_adapter_gate.py``.
"""

from __future__ import annotations

import ast
import pathlib
import re

import pytest

_HERE = pathlib.Path(__file__).resolve()
_INNER_PKG = _HERE.parents[1]  # genie_sim_engine_newton/
_ADAPTER_DIR = _INNER_PKG / "engine" / "newton" / "adapters"
_BASE = _ADAPTER_DIR / "base.py"
_ENGINE_BASE = _INNER_PKG / "engine" / "newton" / "engine_base.py"

#: adapter module -> whether its class may declare the per-substep base clamp.
#: Adding an adapter means adding a row, which is the point: somebody decides
#: whether that solver can hold the base instead of inheriting a default and
#: finding out at runtime.
_EXPECTED = {
    "mujoco_warp_multi_env.py": True,
    "mujoco_warp.py": False,
    "featherstone.py": False,
    "avbd.py": False,
}

#: not adapters — ``base.py`` is the ABC that declares the default
_NOT_ADAPTERS = {"__init__.py", "base.py"}


def _class_level_bools(path: pathlib.Path, attr: str) -> dict[str, bool]:
    """``{class name: value}`` for every class-body ``attr = <bool>``.

    Class level only: an instance attribute set in ``__init__`` would not be
    visible to ``engine_base``'s gate, which reads the capability off the
    adapter right after ``make_adapter`` and before anything is built.
    """
    tree = ast.parse(path.read_text())
    out: dict[str, bool] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for stmt in node.body:
            targets = (
                [stmt.target]
                if isinstance(stmt, ast.AnnAssign)
                else stmt.targets if isinstance(stmt, ast.Assign) else []
            )
            for t in targets:
                if isinstance(t, ast.Name) and t.id == attr and isinstance(stmt.value, ast.Constant):
                    out[node.name] = bool(stmt.value.value)
    return out


def test_the_adapter_inventory_is_unchanged():
    """Guard the guard.  A new adapter must not slip past by not being listed:
    it would inherit ``supports_base_pin = False`` from the ABC, which is the
    SAFE answer — but inheriting it silently is how nobody notices that a
    solver which could hold the base doesn't say so."""
    found = {p.name for p in _ADAPTER_DIR.glob("*.py")} - _NOT_ADAPTERS
    assert found == set(_EXPECTED), (
        f"adapter inventory changed: {sorted(found ^ set(_EXPECTED))}. Add the new "
        f"adapter to _EXPECTED with its base-clamp decision — ``pin_base_mode: "
        f"mocap`` on a solver that cannot clamp the root free joint drops the weld "
        f"and leaves the base under gravity with nothing holding it."
    )


def test_the_abc_default_is_no():
    """``supports_base_pin`` is non-abstract and defaults to False on purpose:
    unlike cloth, the base clamp is a single-adapter extension, so abstract
    would mean four edits to say "no" four times.  The default must therefore
    be the safe answer."""
    assert _class_level_bools(_BASE, "supports_base_pin") == {"SolverAdapter": False}


def test_exactly_one_adapter_claims_the_base_clamp():
    """★ The real check.  ``mocap`` is a mjwarp-multi-env-only capability; if a
    second adapter ever claims it, that adapter needs its own
    ``_build_base_pin`` and this test is where that gets noticed."""
    claimed = {}
    for fname, want in _EXPECTED.items():
        decls = _class_level_bools(_ADAPTER_DIR / fname, "supports_base_pin")
        got = any(decls.values())
        assert got == want, (
            f"{fname}: supports_base_pin declarations {decls} do not match the "
            f"expected {want}. Either the adapter gained/lost a per-substep base "
            f"clamp (update _EXPECTED) or the declaration is wrong."
        )
        if got:
            claimed[fname] = decls
    assert len(claimed) == 1, f"expected exactly one adapter with the base clamp, got {sorted(claimed)}"


def test_the_claiming_adapter_actually_implements_the_clamp():
    """A ``True`` with no ``_build_base_pin`` behind it is worse than a False:
    the config-time gate passes and the base still falls."""
    src = (_ADAPTER_DIR / "mujoco_warp_multi_env.py").read_text()
    assert "def _build_base_pin(" in src
    assert "def base_pin_registered(" in src, (
        "the post-build half of the gate is missing: ``supports_base_pin`` says the "
        "adapter CAN clamp the base, ``base_pin_registered()`` says it DID — only "
        "the second one knows whether a root FREE joint was actually found."
    )
    # ``_build_base_pin``'s body is wrapped in ``except Exception -> warn``, so
    # it cannot raise on "no base found" — it records the flag instead and the
    # engine owns the refusal.
    assert "self._base_pin_registered = True" in src


@pytest.mark.parametrize("needle", ["supports_base_pin", "base_pin_registered"])
def test_both_halves_of_the_gate_are_wired_in_the_engine(needle):
    """Capability and realization are knowable at different times — the class
    attribute right after ``make_adapter``, the registration only once the
    solver is built — so the engine checks twice.  Dropping either one brings
    back the silent fall."""
    src = _ENGINE_BASE.read_text()
    assert needle in src, f"engine_base.py no longer reads {needle}"
    # ...and the check must be a refusal, not a warning.
    assert re.search(r"needs_base_clamp", src), "engine_base.py no longer gates on needs_base_clamp"
    assert src.count("raise RuntimeError(") >= 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

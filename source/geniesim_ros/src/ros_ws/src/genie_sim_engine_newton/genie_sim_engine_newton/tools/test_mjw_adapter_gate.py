#!/usr/bin/env python3

# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""``mjc_contact._is_mjw`` must recognise EVERY MuJoCo-Warp-routed adapter.

This gate decides whether the per-class contact injection runs at all —
solref, solimp, the friction triplet, and condim.  When it says "no", the
module returns after one info line and the model keeps Newton's defaults --
including any ``condim`` the scene's ``newton.mjc_contact:`` block asked for,
which is then simply never written, so contacts stay on MuJoCo's stock 3 and
the torsional / rolling friction slots go unread.  Nothing raises; grasping
just gets worse.

It has already failed exactly this way.  The gate used to be an EXACT
match against ``{"mujoco-warp", "mujoco_warp"}``, so the day the solver
was pointed at ``mujoco-warp-multi-env`` the whole module started
skipping — found 2026-08-19, in a live log, by reading the one line

    [mjc_contact] adapter='mujoco-warp-multi-env': skipping (non-MJW path)

Every other adapter gate in the engine (``setup/model.py``,
``setup/init_pose.py``, ``setup/normalize.py``, ``engine_base.py``)
already used ``startswith``; this one was the odd one out.

What the gate actually receives
-------------------------------
``apply_pre_finalize(..., adapter_name=getattr(self._adapter, "name", ""))``
— the adapter INSTANCE's ``name`` property, **not** the factory key.
Those differ: ``mjvbd`` and ``mjxpbd`` are factory keys that construct
``MuJoCoWarpAdapter``, whose ``name`` is plain ``"mujoco-warp"``.  So the
value set to test against is the set of ``name`` properties, which this
file reads out of the adapter sources.

Deliberately NOT a second hard-coded name list — a second list is what
created the bug.  Instead the adapter FILES are enumerated and their
``name`` return values extracted; a new adapter file makes
``test_the_adapter_inventory_is_unchanged`` fail, forcing whoever adds it
to state which side of this gate it belongs on.

Loaded by path (no colcon install, no GPU): ``mjc_contact`` imports
``newton`` only at call time.
"""

from __future__ import annotations

import importlib.util
import pathlib
import re
import sys

import pytest

_HERE = pathlib.Path(__file__).resolve()
_INNER_PKG = _HERE.parents[1]  # genie_sim_engine_newton/
_SRC_ROOT = _HERE.parents[2]  # dir that CONTAINS the package
_ADAPTER_DIR = _INNER_PKG / "engine" / "newton" / "adapters"

if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

#: adapter module -> the ``name`` property it must report, and whether the
#: contact-injection gate must accept it.  Adding an adapter means adding a
#: row here, which is the point: the decision gets made, not defaulted.
_EXPECTED = {
    "mujoco_warp.py": ("mujoco-warp", True),
    "mujoco_warp_multi_env.py": ("mujoco-warp-multi-env", True),
    "featherstone.py": ("featherstone", False),
    "avbd.py": ("avbd", False),
}

#: not adapters — no ``name`` property of their own
_NOT_ADAPTERS = {"__init__.py", "base.py"}


def _declared_name(path: pathlib.Path) -> str | None:
    """The string a module's ``name`` property returns, or None.

    Match non-greedily from the signature so a property docstring before
    its return does not hide the declared adapter name from this inventory.
    """
    m = re.search(r"def name\(self\)\s*->\s*str:.*?return\s+\"([^\"]+)\"", path.read_text(), re.S)
    return m.group(1) if m else None


@pytest.fixture(scope="module")
def gate():
    mod_path = _INNER_PKG / "engine" / "newton" / "mjc_contact.py"
    assert mod_path.is_file(), f"mjc_contact.py not found at {mod_path}"
    spec = importlib.util.spec_from_file_location("_mjcgate_probe", mod_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def test_the_adapter_inventory_is_unchanged():
    """Guard the guard.  A new adapter file must not slip past this test
    by simply not being listed — that would let it default to whichever
    side of the gate its name happens to land on."""
    found = {p.name for p in _ADAPTER_DIR.glob("*.py")} - _NOT_ADAPTERS
    assert found == set(_EXPECTED), (
        f"adapter inventory changed: {sorted(found ^ set(_EXPECTED))}. "
        f"Add the new adapter to _EXPECTED with the contact-gate decision "
        f"it needs — running with the per-class contact injection off is a "
        f"silent physics change, not a startup error."
    )


def test_each_adapter_reports_the_name_we_think_it_does():
    """The gate is fed ``adapter.name``; if a rename lands without this
    file noticing, every assertion below tests the wrong string."""
    for fname, (want, _) in _EXPECTED.items():
        got = _declared_name(_ADAPTER_DIR / fname)
        assert got == want, f"{fname}: name property is {got!r}, expected {want!r}"


def test_every_mjw_adapter_passes_the_contact_gate(gate):
    """★ The real check: no MuJoCo-Warp adapter may silently bypass the
    per-class contact injection."""
    rejected = sorted(name for fname, (name, want_pass) in _EXPECTED.items() if want_pass and not gate._is_mjw(name))
    assert not rejected, (
        f"mjc_contact._is_mjw rejects these MuJoCo-Warp adapters: {rejected}. "
        f"They would run with NO per-class solref / solimp / friction, and the "
        f"scene's condim override silently unwritten — contacts stay on MuJoCo's "
        f"stock 3 and the torsional + rolling friction slots go unread. The only "
        f"symptom is worse grasping and one info line."
    )


def test_the_gate_still_rejects_the_other_solvers(gate):
    """The prefix must not widen so far that Featherstone / AVBD get
    contact writes meant for MuJoCo-Warp's custom-attribute store."""
    for fname, (name, want_pass) in _EXPECTED.items():
        if not want_pass:
            assert not gate._is_mjw(name), f"{name!r} must not pass the mjwarp gate"
    for name in ("fsvbd", "", None):
        assert not gate._is_mjw(name), f"{name!r} must not pass the mjwarp gate"


def test_the_gate_is_case_and_whitespace_tolerant(gate):
    """``adapter.name`` reaches the gate straight off a duck-typed
    ``getattr``; a stray case or space must not turn contact params off."""
    for name in ("MuJoCo-Warp", "  mujoco-warp-multi-env  ", "MUJOCO_WARP"):
        assert gate._is_mjw(name), f"{name!r} should pass the mjwarp gate"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

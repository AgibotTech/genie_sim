#!/usr/bin/env python3

# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Selecting a non-MuJoCo solver must not silently ignore the MuJoCo profile.

``physics_knobs.md`` says it plainly: setting a value the active adapter never
reads is silent — no warning, no error, the yaml just has no effect.  Within the
mjwarp family that silence is covered by a "read by" column and by
``test_knob_reader_scope.py``, which holds the column against the code.  Across
families it was covered by one sentence of prose ("Featherstone and AVBD read
none of these"), and by nothing at runtime: ``FeatherstoneAdapter`` and
``AVBDAdapter`` take no constructor arguments at all, so they never even receive
``physics_params`` and cannot report what they are ignoring.

That is the case ``warn_inert_params`` covers, and the reason it lives in
``adapters/base.py`` and is called from the factory rather than from an adapter.

The tests below use plain dataclasses rather than the real ``PhysicsParams``:
``base.py`` duck-types the params object on purpose (the adapters do not import
from ``genie_sim_engine_core``), so the duck is what should be tested.
"""

from __future__ import annotations

import importlib.util
import pathlib
from dataclasses import dataclass

import pytest

_ADAPTERS = pathlib.Path(__file__).resolve().parents[1] / "engine" / "newton" / "adapters"


def _load_base():
    """Load ``adapters/base.py`` by path — see ``test_mjc_iterations.py``."""
    path = _ADAPTERS / "base.py"
    spec = importlib.util.spec_from_file_location("_adapters_base_inert_under_test", path)
    assert spec and spec.loader, f"cannot load {path}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_BASE = _load_base()
warn_inert_params = _BASE.warn_inert_params


@dataclass
class _Solver:
    mjc_iterations: int = 0
    impratio: float = 0.0


@dataclass
class _GripperContact:
    gripper_master_force_cap: float = 0.0


@dataclass
class _ArmContact:
    arm_max_torque: float = 0.0


@dataclass
class _DriveGripper:
    # Mirrors ``GripperDriveParams``: no shipped default, so every profile
    # authors it.
    master_stiffness: float


@dataclass
class _Params:
    solver: object = None
    gripper_contact: object = None
    arm_contact: object = None
    drive_gripper: object = None


def _params(**kw):
    return _Params(
        solver=_Solver(**{k: v for k, v in kw.items() if k in ("mjc_iterations", "impratio")}),
        gripper_contact=_GripperContact(kw.get("gripper_master_force_cap", 0.0)),
        arm_contact=_ArmContact(kw.get("arm_max_torque", 0.0)),
        drive_gripper=_DriveGripper(kw["master_stiffness"]) if "master_stiffness" in kw else None,
    )


class _Logger:
    def __init__(self):
        self.warns: list = []

    def warn(self, msg):
        self.warns.append(str(msg))

    def info(self, msg):  # pragma: no cover - not used here
        pass


@pytest.mark.parametrize(
    "solver",
    ["mujoco-warp", "mujoco_warp", "mujoco-warp-multi-env", "mujoco_warp_multi_env", "mjvbd", "mjxpbd"],
)
def test_no_warning_on_any_mjwarp_solver(solver):
    """★ The costly direction.  A false "this has no effect" on an adapter
    that DOES read the entry sends an operator to delete a line that was
    working — worse than the silence this replaces.  ``mjvbd``/``mjxpbd`` are
    included because they route to the stock mjwarp adapter; a key list that
    only knew the two canonical spellings would warn on every combo run."""
    log = _Logger()
    assert warn_inert_params(solver, _params(impratio=30.0, arm_max_torque=100.0, master_stiffness=100.0), log) == []
    assert log.warns == []


def test_warns_once_naming_every_authored_entry():
    log = _Logger()
    inert = warn_inert_params(
        "featherstone",
        _params(mjc_iterations=100, impratio=30.0, gripper_master_force_cap=30.0, arm_max_torque=100.0),
        log,
    )
    assert sorted(inert) == [
        "arm_contact.arm_max_torque=100.0",
        "gripper_contact.gripper_master_force_cap=30.0",
        "solver.impratio=30.0",
        "solver.mjc_iterations=100",
    ]
    assert len(log.warns) == 1, "one warning per build, not one per entry"
    msg = log.warns[0]
    assert "featherstone" in msg
    # The operator needs the yaml key, not the attribute path the code reads.
    assert "arm_contact.arm_max_torque" in msg
    assert "physics_knobs.md" in msg, "the warning must point at the table that explains itself"


def test_shipped_defaults_are_not_reported():
    """Every profile carries every key, so "is it present" would warn on all
    of them.  The question is whether anyone moved it off the shipped value —
    and the shipped value comes from the dataclass, not a second copy of the
    numbers in ``base.py``."""
    log = _Logger()
    assert warn_inert_params("avbd", _params(), log) == []
    assert log.warns == []


def test_a_field_with_no_shipped_default_counts_as_authored():
    """``GripperDriveParams.master_stiffness`` has no default: there is no
    value to compare against, and the duck-typed path has no other evidence.
    Report it — over-reporting is recoverable, and on a non-mjwarp adapter the
    claim is true either way."""
    inert = warn_inert_params("avbd", _params(master_stiffness=100.0), None)
    assert inert == ["usd_drive_api.gripper.master_stiffness=100.0"]


def test_no_logger_and_no_params_are_both_fine():
    """The factory is callable from a tool or a test without a logger, and
    ``make_adapter``'s ``physics_params`` is optional."""
    assert warn_inert_params("featherstone", _params(impratio=30.0), None) == ["solver.impratio=30.0"]
    assert warn_inert_params("featherstone", None, _Logger()) == []


def test_a_params_object_missing_sections_entirely_is_tolerated():
    """The duck-typed contract: ``base.py`` must survive a params object with
    no ``arm_contact`` at all, which is what ``test_newton_solver.py``-style
    invocations hand it."""

    @dataclass
    class _Bare:
        solver: object = None

    assert warn_inert_params("featherstone", _Bare(), None) == []
    assert warn_inert_params("featherstone", _Bare(solver=_Solver(impratio=30.0)), None) == ["solver.impratio=30.0"]


def test_the_factory_calls_the_warning_before_it_dispatches():
    """Source-shape check: the adapters that need warning about are the ones
    constructed with no arguments, so the call has to happen in the factory
    and ahead of every ``return``.  Moving it into an adapter would skip
    exactly the cases it exists for."""
    src = (_ADAPTERS / "__init__.py").read_text()
    assert "warn_inert_params" in src, "make_adapter no longer calls the warning"
    first_return = src.index("    if key in (")
    assert src.index("warn_inert_params(physics_solver") < first_return, (
        "warn_inert_params must run before the dispatch chain — a non-mjwarp "
        "adapter returns before any adapter code could report the profile"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

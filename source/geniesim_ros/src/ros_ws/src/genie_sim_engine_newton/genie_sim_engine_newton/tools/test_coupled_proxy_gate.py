#!/usr/bin/env python3

# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""``newton.coupling`` must be inert by default and loud when misconfigured.

``newton:`` is an opaque passthrough dict in the scene schema
(``genie_sim_engine_core/schema/scene.py``): unknown keys under it are carried
and ignored.  That is the right default for a passthrough, and exactly the
wrong one for a coupling request — a scene asking for two-way MuJoCo+XPBD
coupling that quietly ran one-way would look like a physics problem, not a
config problem, and nothing on the outside distinguishes the two.  So
``_coupled_proxy_options`` raises on every misconfiguration instead.

The other half matters as much: the gate has to stay invisible to every scene
that does not opt in.  ``enabled: false``, an empty ``coupling`` block, no
``newton`` key, and ``scene_cfg = None`` must all return ``None``, which is
what keeps the sequential rigid-then-cloth path byte-identical.

Constructing the real coupler needs a finalized Newton model and a GPU, so
these tests exercise the gate on a duck-typed stand-in; the construction it
guards is covered by the live smoke test, not here.
"""

from __future__ import annotations

import importlib.util
import pathlib
import types

import pytest

_SETUP = pathlib.Path(__file__).resolve().parents[1] / "engine" / "newton" / "setup"


def _mixin():
    """Load ``setup/solver.py`` by path.

    Importing it through the package drags in the whole ``__init__`` chain,
    which reaches ``genie_sim_engine_core`` and fails when this file runs on
    its own.
    """
    path = _SETUP / "solver.py"
    spec = importlib.util.spec_from_file_location("_solver_setup_under_test", path)
    assert spec and spec.loader, f"cannot load {path}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._SolverMixin


def _engine(coupling=None, *, adapter="mujoco-warp", cloth_pref="xpbd", particles=512, newton_key=True):
    """A stand-in carrying only what ``_coupled_proxy_options`` reads."""
    obj = _mixin()()
    newton_cfg = {}
    if coupling is not None:
        newton_cfg["coupling"] = coupling
    obj._scene_cfg = {"newton": newton_cfg} if newton_key else {}
    obj._adapter = types.SimpleNamespace(name=adapter)
    obj._cloth_solver_pref = cloth_pref
    obj._model = types.SimpleNamespace(particle_count=particles)
    return obj


# ----- inert unless opted in --------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"coupling": None},  # no coupling block
        {"coupling": {}},  # empty block
        {"coupling": {"enabled": False}},  # explicitly off
        {"coupling": None, "newton_key": False},  # no newton: section at all
    ],
    ids=["absent", "empty", "disabled", "no-newton-key"],
)
def test_returns_none_without_opt_in(kwargs):
    """★ Every shipped scene is one of these. None means the sequential
    rigid-then-cloth path runs exactly as it did before coupling existed."""
    assert _engine(**kwargs)._coupled_proxy_options() is None


def test_scene_cfg_none_is_tolerated():
    obj = _engine()
    obj._scene_cfg = None
    assert obj._coupled_proxy_options() is None


# ----- opted in ----------------------------------------------------------


def test_defaults_when_only_enabled_is_set():
    opts = _engine(coupling={"enabled": True})._coupled_proxy_options()
    assert opts == {"iterations": 1, "mass_scale": 1.0, "mode": "lagged", "proxy_relaxation": 1.0}


def test_every_field_is_read():
    opts = _engine(
        coupling={
            "enabled": True,
            "iterations": 4,
            "mass_scale": 0.25,
            "mode": "SIMULTANEOUS",
            "proxy_relaxation": 0.5,
        }
    )._coupled_proxy_options()
    assert opts == {"iterations": 4, "mass_scale": 0.25, "mode": "simultaneous", "proxy_relaxation": 0.5}


# ----- loud, not silent, when it cannot be honoured ----------------------


def test_unsupported_type_raises():
    with pytest.raises(ValueError, match="unsupported"):
        _engine(coupling={"enabled": True, "type": "hard"})._coupled_proxy_options()


def test_wrong_rigid_adapter_raises():
    with pytest.raises(ValueError, match="mujoco-warp"):
        _engine(coupling={"enabled": True}, adapter="featherstone")._coupled_proxy_options()


def test_wrong_cloth_solver_raises():
    with pytest.raises(ValueError, match="xpbd"):
        _engine(coupling={"enabled": True}, cloth_pref="vbd")._coupled_proxy_options()


def test_no_particles_raises():
    """A rigid-only scene has nothing to couple TO. Without this the coupler
    would build an XPBD entry over an empty particle set and step nothing."""
    with pytest.raises(ValueError, match="no particles"):
        _engine(coupling={"enabled": True}, particles=0)._coupled_proxy_options()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

#!/usr/bin/env python3

# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""``solver.impratio`` and ``newton.mujoco_enable_multiccd`` reach the solver.

Both knobs existed on this branch without a reader on the adapter that most
scenes actually run.

``solver.impratio``
    Declared in ``params.py``, shipped in ``physics_params_rlinf.yaml``, and
    applied *only* by ``mujoco_warp_multi_env.py``.  The stock adapter — which
    is what ``mujoco-warp``, ``mjvbd`` and ``mjxpbd`` all resolve to — never
    read it, so setting it under those solvers changed nothing and the
    pyramidal cone's regularised friction let grasped objects creep.

``newton.mujoco_enable_multiccd``
    Set to ``true`` by the retired Swiftpicker chef configuration with **no reader
    anywhere in the tree**.  ``newton:`` is an opaque passthrough dict in the
    scene schema, so the key was carried and ignored: the scene believed it
    had multi-point convex manifolds and had single-point contacts.

The tri-state matters and is asserted here.  The two adapters disagree on the
default — multi-env has had MULTICCD unconditionally on since a measured
pinch-decay fix, the stock adapter has never had it — so ``None`` must mean
"keep your own default" rather than resolving to a shared bool that would
silently undo one side or enable the other.

Modules are loaded by path: importing them normally pulls in
``engine/newton/__init__.py`` -> ``engine_base.py`` -> a
``genie_sim_engine_core`` import that is not resolvable from a bare shell.
Same pattern as ``test_body_aabb_latch.py``.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_INNER_PKG = _HERE.parent
_ADAPTERS = _INNER_PKG / "engine" / "newton" / "adapters"


def _load(mod_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


_base = _load("mjc_contact_base", _ADAPTERS / "base.py")
resolve = _base.resolve_mjc_impratio_cone


class _Solver:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class _Params:
    def __init__(self, solver=None):
        if solver is not None:
            self.solver = solver


# ----------------------------------------------------------------------
# The resolver
# ----------------------------------------------------------------------


def test_zero_leaves_mujocos_defaults_alone():
    """0 is the shipped value; it must not assert an opinion on either field."""
    assert resolve(_Params(_Solver(impratio=0.0))) == (None, None)


def test_a_positive_value_brings_the_elliptic_cone_with_it():
    """impratio is only evaluated on the elliptic path — passing it with the
    pyramidal cone is a no-op that reads like a working knob."""
    assert resolve(_Params(_Solver(impratio=30.0))) == (30.0, "elliptic")


def test_no_params_at_all():
    assert resolve(None) == (None, None)


def test_params_without_a_solver_section():
    assert resolve(_Params()) == (None, None)


@pytest.mark.parametrize("bad", [-5.0, "not-a-number", None])
def test_junk_is_inert_rather_than_fatal(bad):
    assert resolve(_Params(_Solver(impratio=bad))) == (None, None)


def test_an_int_is_accepted():
    assert resolve(_Params(_Solver(impratio=15))) == (15.0, "elliptic")


# ----------------------------------------------------------------------
# The stock adapter actually passes them
# ----------------------------------------------------------------------


def _build_solver_body(path: Path) -> str:
    src = path.read_text()
    m = re.search(r"\n    def build_solver\(.*?\n    \) -> Any:\n", src, re.S)
    assert m, f"build_solver not found in {path.name}"
    return src[m.end() :]


def test_stock_adapter_forwards_impratio_and_cone():
    """★ The gap: this adapter serves mujoco-warp / mjvbd / mjxpbd and had no
    impratio route at all."""
    body = _build_solver_body(_ADAPTERS / "mujoco_warp.py")
    assert "resolve_mjc_impratio_cone(self._physics_params)" in body
    assert "impratio=mj_impratio" in body
    assert "cone=mj_cone" in body


def test_stock_adapter_forwards_multiccd():
    body = _build_solver_body(_ADAPTERS / "mujoco_warp.py")
    assert "enable_multiccd=" in body


def test_stock_adapter_does_not_hardcode_a_contact_default():
    """A module-scope ``_DEFAULT_MJC_IMPRATIO = 15.0`` would be a second route
    to physics that ``solver.impratio`` already owns — AGENTS.md, "one knob,
    one route"."""
    src = (_ADAPTERS / "mujoco_warp.py").read_text()
    assert "_DEFAULT_MJC_IMPRATIO" not in src
    assert "_DEFAULT_MJC_CONE" not in src


# ----------------------------------------------------------------------
# Tri-state defaults
# ----------------------------------------------------------------------


def test_absent_keeps_each_adapters_own_default():
    stock = (_ADAPTERS / "mujoco_warp.py").read_text()
    multi = (_ADAPTERS / "mujoco_warp_multi_env.py").read_text()
    # stock: None -> off (SolverMuJoCo's own default)
    assert "self._enable_multiccd = None if enable_multiccd is None else bool(enable_multiccd)" in stock
    # multi-env: None -> on (the measured pinch-decay fix stays)
    assert "self._enable_multiccd = True if enable_multiccd is None else bool(enable_multiccd)" in multi


def test_multi_env_honours_an_explicit_false():
    """Before this it enabled MULTICCD unconditionally, so a scene saying
    ``false`` was overruled without a word."""
    src = (_ADAPTERS / "mujoco_warp_multi_env.py").read_text()
    assert "if not self._enable_multiccd:" in src
    assert "MULTICCD disabled by scene" in src


@pytest.mark.parametrize("name", ["mujoco_warp.py", "mujoco_warp_multi_env.py"])
def test_both_adapters_accept_the_tri_state_kwarg(name):
    src = (_ADAPTERS / name).read_text()
    assert "enable_multiccd: Optional[bool] = None," in src


def test_the_factory_routes_the_scene_key_to_both_adapters():
    src = (_ADAPTERS / "__init__.py").read_text()
    assert "mujoco_enable_multiccd: Optional[bool] = None," in src
    assert src.count("enable_multiccd=mujoco_enable_multiccd,") == 2


# ----------------------------------------------------------------------
# The scene that asked for it
# ----------------------------------------------------------------------


def test_the_chef_scene_key_now_has_a_reader():
    """This is the regression in one assertion: the key was live in a shipped
    scene and read by nothing."""
    eb = (_INNER_PKG / "engine" / "newton" / "engine_base.py").read_text()
    assert 'get("mujoco_enable_multiccd")' in eb
    assert "mujoco_enable_multiccd=_multiccd," in eb


def test_a_yaml_string_is_not_taken_as_bool():
    """``bool("false")`` is True — the exact way this key would betray an
    author who quoted it."""
    eb = (_INNER_PKG / "engine" / "newton" / "engine_base.py").read_text()
    assert '"false", "0", "no", "off"' in eb

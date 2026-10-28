#!/usr/bin/env python3

# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""The rigid solver must not inherit the particle solver's iteration count.

``physics_solver_iterations`` in a launcher is the PARTICLE count.  FEM tet
scenes set it to 4 (``launcher_newton_mjxpbd.yaml``) or 5
(``launcher_newton_mjvbd.yaml``) because ``substeps: 24`` carries the stiffness
there.  Both of those map to ``MuJoCoWarpAdapter``, which drives *both* solvers,
and ``build_solver`` forwarded that same number into
``SolverMuJoCo(iterations=...)`` — running MuJoCo's rigid constraint solve at 4
against its default of 100.

A frictional contact block cannot converge in 4 iterations.  What you see is a
grasped object creeping out of the jaws, which reads as missing friction and
costs an afternoon of tuning coefficients that were never wrong.  Nothing warns:
both numbers are legal, and ``iterations=4`` even shows up in the bring-up log
looking deliberate.

The regression test below is deliberately a source-text check.  Constructing a
real ``SolverMuJoCo`` needs a built Newton model and a GPU, so the assertion
that survives in CI is the structural one: neither ``build_solver`` may derive
its MuJoCo iteration count from ``sim_iterations``.
"""

from __future__ import annotations

import importlib.util
import pathlib
import re
from dataclasses import dataclass

import pytest

_ADAPTERS = pathlib.Path(__file__).resolve().parents[1] / "engine" / "newton" / "adapters"
_BOTH = ("mujoco_warp.py", "mujoco_warp_multi_env.py")


def _load_base():
    """Load ``adapters/base.py`` by path.

    Importing it as ``genie_sim_engine_newton.engine.newton.adapters.base``
    drags in the whole package ``__init__`` chain, which reaches
    ``genie_sim_engine_core`` and fails when this file is run on its own.
    ``base.py`` itself only needs ``abc``/``typing``/``warp``.
    """
    path = _ADAPTERS / "base.py"
    spec = importlib.util.spec_from_file_location("_adapters_base_under_test", path)
    assert spec and spec.loader, f"cannot load {path}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


resolve_mjc_iterations = _load_base().resolve_mjc_iterations


@dataclass
class _Solver:
    mjc_iterations: int = 0


@dataclass
class _Params:
    solver: object = None


# ----- resolve_mjc_iterations -------------------------------------------


def test_zero_means_solvermujoco_keeps_its_own_default():
    """0 must become None, not 0.  ``iterations=0`` would be a solver that
    never iterates; None is "argument not supplied"."""
    assert resolve_mjc_iterations(_Params(_Solver(0))) is None


def test_positive_value_passes_through():
    assert resolve_mjc_iterations(_Params(_Solver(150))) == 150


def test_no_params_at_all_is_the_default():
    """``build_solver`` runs before physics params exist in some test paths."""
    assert resolve_mjc_iterations(None) is None


def test_params_without_a_solver_section_is_the_default():
    assert resolve_mjc_iterations(_Params(None)) is None


def test_negative_is_treated_as_unset():
    assert resolve_mjc_iterations(_Params(_Solver(-5))) is None


def test_garbage_degrades_to_the_default_rather_than_raising():
    """A hand-edited profile yaml can put a string here.  Killing bring-up over
    it would be worse than falling back to MuJoCo's 100."""
    assert resolve_mjc_iterations(_Params(_Solver("many"))) is None  # type: ignore[arg-type]


def test_float_from_yaml_is_accepted():
    """``mjc_iterations: 100.0`` parses as a float; the solver wants an int."""
    assert resolve_mjc_iterations(_Params(_Solver(100.0))) == 100  # type: ignore[arg-type]


# ----- the regression, structurally -------------------------------------


@pytest.mark.parametrize("name", _BOTH)
def test_build_solver_does_not_derive_mj_iter_from_sim_iterations(name):
    """★ The bug itself.  ``mj_iter = sim_iterations if ...`` is what shipped."""
    src = (_ADAPTERS / name).read_text()
    body = _build_solver_body(src, name)
    offenders = [
        line.strip() for line in body.splitlines() if "sim_iterations" in line and not line.lstrip().startswith("#")
    ]
    assert not offenders, (
        f"{name}::build_solver uses sim_iterations in code: {offenders}. That is "
        f"the PARTICLE-solver count; feeding it to SolverMuJoCo runs the rigid "
        f"constraint solve at a FEM scene's 4 and frictional grasps creep. Read "
        f"solver.mjc_iterations via resolve_mjc_iterations() instead."
    )


@pytest.mark.parametrize("name", _BOTH)
def test_build_solver_reads_the_config_key(name):
    src = (_ADAPTERS / name).read_text()
    body = _build_solver_body(src, name)
    assert "resolve_mjc_iterations(" in body, (
        f"{name}::build_solver no longer calls resolve_mjc_iterations() — either "
        f"the rigid iteration count went back to a hardcoded default (which "
        f"AGENTS.md 'one knob, one route' forbids) or it is unset by accident."
    )


def _build_solver_body(src: str, name: str) -> str:
    """Source text of ``build_solver``, signature excluded.

    The signature has to be excluded: ``sim_iterations`` legitimately stays in
    the parameter list for ``SolverAdapter`` ABC compatibility.
    """
    m = re.search(r"\n    def build_solver\(.*?\n    \) -> Any:\n", src, re.S)
    if m is None:
        pytest.fail(f"could not locate build_solver in {name}", pytrace=False)
    rest = src[m.end() :]
    end = re.search(r"\n    def \w", rest)
    return rest[: end.start()] if end else rest


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

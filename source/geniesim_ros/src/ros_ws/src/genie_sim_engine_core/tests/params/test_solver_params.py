# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""``solver.mjc_iterations`` must survive the yaml round trip.

The rigid constraint solver used to inherit the launcher's
``physics_solver_iterations``, which is the PARTICLE count.  A FEM tet scene
runs that at 4, so MuJoCo's rigid solve ran at 4 against its default of 100 and
frictional grasps crept.  The fix gives the rigid solver its own key here.

The key is only useful if it actually reaches the adapter, and the way it stops
reaching is a silent one: an unknown key in a profile yaml is dropped without a
warning, so a typo in ``_solver()`` would leave every profile back on the
default with nothing to see.  These tests pin the whole path — dataclass
default, yaml parse, and the two shipped profiles carrying the key.
"""

from __future__ import annotations

import pathlib

import pytest
import yaml

from genie_sim_engine_core.common.params import SolverParams, load_physics_params

_CONFIG = pathlib.Path(__file__).resolve().parents[2] / "config"


class _Log:
    def info(self, *_a, **_k):
        pass

    warn = error = debug = info


def test_shipped_default_leaves_solvermujoco_alone():
    """0 is the shipped value and means "do not pass iterations= at all"."""
    assert SolverParams().mjc_iterations == 0


def test_loader_reads_the_key(tmp_path):
    p = tmp_path / "profile.yaml"
    p.write_text("solver:\n  mjc_iterations: 150\n  impratio: 30.0\n")
    phys = load_physics_params(str(p), _Log())
    assert phys.solver.mjc_iterations == 150
    assert phys.solver.impratio == pytest.approx(30.0)


def test_absent_key_falls_back_to_the_default(tmp_path):
    """A profile that only sets impratio must not lose the other fields."""
    p = tmp_path / "profile.yaml"
    p.write_text("solver:\n  impratio: 30.0\n")
    phys = load_physics_params(str(p), _Log())
    assert phys.solver.mjc_iterations == SolverParams().mjc_iterations
    assert phys.solver.impratio == pytest.approx(30.0)


@pytest.mark.parametrize("name", ["physics_params.yaml", "physics_params_rlinf.yaml"])
def test_shipped_profiles_carry_the_key(name):
    """Every launcher but the multi-env one selects ``physics_params.yaml``
    (see ``test_profile_selection.py``), so if the base profile does not mention
    the key, the knob is undiscoverable for exactly the launchers the bug bit.

    Until 2026-08-21 this said "no launcher sets ``physics_params_file``" — true
    at the time, and the reason the key had to be in the base profile to reach
    anything at all."""
    doc = yaml.safe_load((_CONFIG / name).read_text())
    assert "mjc_iterations" in doc.get("solver", {}), (
        f"{name} has no solver.mjc_iterations — the rigid solver's iteration "
        f"count is back to being invisible to whoever reads the profile."
    )


@pytest.mark.parametrize("name", ["physics_params.yaml", "physics_params_rlinf.yaml"])
def test_shipped_profiles_load(name):
    phys = load_physics_params(str(_CONFIG / name), _Log())
    assert phys.solver.mjc_iterations >= 0

#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Pure host tests (no GPU) for resolve_constraint_capacity + the
EngineNodeParams capacity fields."""

from __future__ import annotations

import pathlib
import sys

import pytest
import yaml

_HERE = pathlib.Path(__file__).resolve()
# .../genie_sim_engine_newton/genie_sim_engine_newton/tools/<this>
_NEWTON_SRC_ROOT = _HERE.parents[2]  # dir containing the newton package
_SRC = _HERE.parents[3]  # ros_ws/.../src (holds both packages)
_CORE_SRC_ROOT = _SRC / "genie_sim_engine_core"  # dir containing the core package
_MJXPBD_LAUNCHER = _SRC / "genie_sim_bringup" / "config" / "launcher_newton_mjxpbd.yaml"

for _p in (str(_NEWTON_SRC_ROOT), str(_CORE_SRC_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from genie_sim_engine_newton.engine.newton.adapters.mujoco_warp_multi_env import (
    resolve_constraint_capacity,
)


def test_auto_at_calibration_point_61_bodies_keeps_floor():
    # 61-body supermarket scene (measured peak njmax ~750): auto must stay
    # >= the old default, never below it
    njmax, nconmax = resolve_constraint_capacity(61)
    assert njmax == 1220  # 20*61, above the old default of 1024
    assert nconmax == 512  # 7*61=427 < 512 -> the floor holds


def test_auto_at_calibration_point_156_bodies_exceeds_observed_demand():
    # measured demand from the 20-superset probe: njmax>=2096 / nconmax>=700
    njmax, nconmax = resolve_constraint_capacity(156)
    assert njmax == 3120 and njmax > 2096
    assert nconmax == 1092 and nconmax > 700


def test_explicit_wins_verbatim():
    assert resolve_constraint_capacity(156, njmax_explicit=4096) == (4096, 1092)
    assert resolve_constraint_capacity(156, nconmax_explicit=2000) == (3120, 2000)
    assert resolve_constraint_capacity(1, 777, 888) == (777, 888)


def test_zero_and_negative_body_count_fall_back_to_floors():
    assert resolve_constraint_capacity(0) == (1024, 512)
    assert resolve_constraint_capacity(-5) == (1024, 512)


def test_engine_node_params_parse_capacity_fields():
    from genie_sim_engine_core.common.params import EngineNodeParams

    ep = EngineNodeParams.from_dict({"mujoco_njmax": "4096", "mujoco_nconmax": "1500"})
    assert ep.mujoco_njmax == 4096 and ep.mujoco_nconmax == 1500
    ep0 = EngineNodeParams.from_dict({})
    assert ep0.mujoco_njmax == 0 and ep0.mujoco_nconmax == 0
    epneg = EngineNodeParams.from_dict({"mujoco_njmax": "-3"})
    assert epneg.mujoco_njmax == 0  # _int_nonneg semantics


# ---------------------------------------------------------------------------
# ABI: the launcher-documented capacity knobs must reach the backend
# ---------------------------------------------------------------------------
#
# ``launcher_newton_mjwarp.yaml`` documents ``mujoco_njmax`` /
# ``mujoco_nconmax`` as node params.  ``EngineNodeParams`` parses them and
# ``common/session.py`` writes them into ``engine_kwargs``, but
# ``PhysicsEngine.create`` validates those kwargs against the registered
# backend's ``__init__`` signature (``engine/base.py::_validate_create_kwargs``)
# and the newton backend takes no ``**kwargs`` — so a name missing from
# ``_NewtonStandaloneBase.__init__`` is a hard startup ``TypeError``, not a
# silently-ignored param.  These tests pin the whole chain.


def test_newton_backend_abi_accepts_capacity_kwargs():
    from genie_sim_engine_newton.engine.newton import engine as _engine  # noqa: F401 -- registers "newton"
    from genie_sim_engine_core.engine.base import PhysicsEngine

    abi = PhysicsEngine.backend_abi("newton")
    # No ``**kwargs`` escape hatch: every launcher param must be named.
    assert abi.accepts_extra_kwargs is False
    accepted = set(abi.accepted_names)
    assert "mujoco_njmax" in accepted
    assert "mujoco_nconmax" in accepted


def test_capacity_kwargs_reach_the_adapters():
    from genie_sim_engine_newton.engine.newton.adapters import make_adapter

    # Stock adapter: positive values win, zeros must NOT clobber its own
    # 1024/512 defaults (make_adapter drops them rather than forwarding 0).
    explicit = make_adapter("mujoco-warp", mujoco_njmax=4096, mujoco_nconmax=1500)
    assert (explicit._njmax, explicit._nconmax) == (4096, 1500)
    unset = make_adapter("mujoco-warp", mujoco_njmax=0, mujoco_nconmax=0)
    assert (unset._njmax, unset._nconmax) == (1024, 512)

    # Multi-env adapter takes the zeros verbatim -- it resolves them later via
    # ``resolve_constraint_capacity`` against the per-env body count.
    auto = make_adapter("mujoco-warp-multi-env", mujoco_njmax=0, mujoco_nconmax=0)
    assert (auto._njmax, auto._nconmax) == (0, 0)


def test_mjxpbd_launcher_has_deformable_scene_capacity():
    """Keep the MJW pair buffer sized for dense deformable scenes."""
    params = yaml.safe_load(_MJXPBD_LAUNCHER.read_text())["genie_sim_engine"]["ros__parameters"]
    assert params["mujoco_nconmax"] >= 16384

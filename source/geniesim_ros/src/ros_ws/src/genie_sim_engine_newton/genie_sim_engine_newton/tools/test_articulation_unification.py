#!/usr/bin/env python3

# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""The mjwarp articulation merge has to leave BOTH representations valid.

newton 1.4.0 does not pick one scheme.  Measured on the pinned version, a
three-articulation model finalizes to::

    articulation_start == [0, 2, 4, 6]   # CSR, N+1, trailing joint_count
    articulation_end   == [1, 3, 5]      # split, N, excludes loop joints

and different consumers read different arrays, unconditionally:

* ``sim/articulation.py`` (``eval_articulation_fk``) -> ``articulation_end[id]``
* ``solvers/coupled/solver_coupled.py``              -> ``articulation_start[id + 1]``
* ``utils/selection.py``                             -> both

The bug this pins: an earlier revision detected "split repr" by the mere
presence of ``articulation_end`` and then wrote a ONE-element
``articulation_start``.  FK kept working, so nothing complained — but every
reader of the CSR sentinel was left indexing past the end of the array.  It
was invisible until a scene enabled ``newton.coupling``, whose ModelView
compaction is exactly such a reader.

Same shape for ``joint_articulation``: FK only needs owner >= 0, so leaving
stale owner ids 1..N-1 looked harmless, while anything that INDEXES the
articulation arrays by that owner runs off a now-1-entry array.

These are source-level assertions.  Exercising the real merge needs a
finalized Model, and the mixin's import chain reaches
``genie_sim_engine_core``; the arithmetic is verified against live newton in
``test_newton_reports_both_representations`` below, which is skipped when
newton is not importable.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_NORMALIZE = _HERE.parent / "engine" / "newton" / "setup" / "normalize.py"


@pytest.fixture(scope="module")
def block() -> str:
    src = _NORMALIZE.read_text()
    i = src.index("# Fix on mjwarp: merge all articulations into one")
    j = src.index("articulation unification: skipped on")
    return src[i:j]


# ----------------------------------------------------------------------
# The regression
# ----------------------------------------------------------------------


def test_the_csr_sentinel_is_written(block):
    """★ ``[0]`` alone is what shipped; it strands every sentinel reader."""
    assert "np.array([0, joint_count], dtype=np.int32)" in block
    assert "np.array([0], dtype=np.int32)" not in block


def test_the_split_end_is_written_too(block):
    assert "np.array([joint_count], dtype=np.int32)" in block


def test_neither_write_is_conditional_on_the_other(block):
    """The two arrays are not alternatives on 1.4.0, so there must be no
    branch choosing between them."""
    assert "split_repr" not in block


def test_joint_articulation_owners_are_remapped(block):
    assert "unified_owners[unified_owners >= 0] = 0" in block


def test_loop_closure_joints_keep_their_minus_one(block):
    """``>= 0`` is load-bearing: a bare ``[:] = 0`` would hand loop-closure
    joints a real owner and FK would start walking them."""
    assert "unified_owners[unified_owners >= 0] = 0" in block
    assert "unified_owners[:] = 0" not in block


def test_the_already_unified_check_covers_every_representation(block):
    """Checking only ``articulation_start`` would call a model with stale
    owners or a stale end 'already unified' and skip the fix."""
    m = re.search(r"spans_all = \((.*?)\n                    \)", block, re.S)
    assert m, "spans_all not found"
    cond = m.group(1)
    assert "len(start_arr) == 2" in cond
    assert "end_arr" in cond
    assert "owner" in cond


# ----------------------------------------------------------------------
# Scope guards
# ----------------------------------------------------------------------


def test_featherstone_is_still_excluded(block):
    """Unifying there collapses self-collision filtering and NaNs substep 1."""
    assert 'adapter_name.startswith(("mujoco-warp", "mujoco_warp"))' in block


def test_multi_env_still_gets_the_merge(block):
    """``startswith`` covers ``mujoco-warp-multi-env``.  Narrowing to an exact
    match would freeze passive FREE-joint scene bodies on that adapter."""
    assert "adapter_name in (" not in block


def test_world_arrays_are_left_alone_when_there_is_more_than_one(block):
    """Naming one world as 'the' world is false for the other 63."""
    assert "world_count <= 1" in block


# ----------------------------------------------------------------------
# The measurement the comment cites
# ----------------------------------------------------------------------


def test_newton_reports_both_representations():
    newton = pytest.importorskip("newton")
    b = newton.ModelBuilder()
    for _ in range(3):
        body = b.add_body()
        b.add_joint_revolute(parent=-1, child=body, axis=(0, 0, 1))
    m = b.finalize()
    start = m.articulation_start.numpy()
    assert len(start) == 4, "articulation_start is not CSR N+1 on this newton"
    assert int(start[-1]) == int(m.joint_count), "no trailing joint_count sentinel"
    assert m.articulation_end is not None, "articulation_end absent on this newton"
    assert len(m.articulation_end.numpy()) == 3, "articulation_end is not split N"

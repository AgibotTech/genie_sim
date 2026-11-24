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

And for the CUDA FK schedule: ``finalize`` caches level tables per articulation
id, and ``eval_fk`` prefers them over ``articulation_start`` / ``_end``.  Left in
place they describe the pre-merge articulations, so only the first one is walked
and a free body keeps its pre-reset ``body_q``.

Most of these are source-level assertions; the mixin's import chain reaches
``genie_sim_engine_core``.  ``test_newton_reports_both_representations`` checks
the arithmetic against live newton, and
``test_fk_after_the_merge_moves_every_body`` loads ``normalize.py`` alone and
runs the real merge on a finalized Model.  Both are skipped when newton is not
importable.
"""

from __future__ import annotations

import importlib.util
import logging
import re
from pathlib import Path
from types import SimpleNamespace

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


def test_the_cached_fk_schedule_is_dropped(block):
    """``eval_fk`` prefers these tables to the arrays the merge rewrites."""
    for name in (
        "_fk_articulation_level_start",
        "_fk_level_joint_start",
        "_fk_level_joints",
        "_fk_level_parent_pos",
        "_fk_level_capacity",
    ):
        assert name in block


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


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_fk_after_the_merge_moves_every_body(device):
    """A reset writes ``joint_q`` and reads ``body_q`` back through ``eval_fk``
    before any physics step, so FK has to reach the bodies of every articulation
    that was merged away -- on CUDA too, where it has a cached schedule."""
    newton = pytest.importorskip("newton")
    wp = pytest.importorskip("warp")
    if device not in [d.alias for d in wp.get_devices()]:
        pytest.skip(f"no {device}")
    spec = importlib.util.spec_from_file_location("_normalize_under_test", _NORMALIZE)
    normalize = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(normalize)

    inertia = wp.mat33(0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01)
    b = newton.ModelBuilder()
    links, joints = [], []
    for i in range(3):
        links.append(b.add_link(mass=1.0, inertia=inertia))
        joints.append(
            b.add_joint_revolute(
                parent=links[i - 1] if i else -1,
                child=links[i],
                axis=(0.0, 1.0, 0.0),
                parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()),
            )
        )
    b.add_articulation(joints)
    free = b.add_body(mass=1.0, inertia=inertia)
    m = b.finalize(device=device)
    assert m.articulation_count == 2

    engine = SimpleNamespace(
        _model=m, _adapter=SimpleNamespace(name="mujoco-warp"), _logger=logging.getLogger(__name__)
    )
    normalize._NormalizeMixin._phase_normalize_model(engine)
    assert m.articulation_count == 1

    state = m.state()
    free_joint = m.joint_child.numpy().tolist().index(free)
    q = state.joint_q.numpy()
    q[int(m.joint_q_start.numpy()[free_joint]) + 2] = 1.25
    q[0] = 0.5
    state.joint_q.assign(q)
    before = state.body_q.numpy().copy()
    newton.eval_fk(m, state.joint_q, state.joint_qd, state)
    after = state.body_q.numpy()
    assert after[free][2] == pytest.approx(1.25)
    assert abs(after[links[-1]][0] - before[links[-1]][0]) > 0.01

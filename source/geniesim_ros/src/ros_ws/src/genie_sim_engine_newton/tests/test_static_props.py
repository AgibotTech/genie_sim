# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""World AABBs for static scene props (``engine/newton/static_props.py``).

Built on a synthetic in-memory stage rather than a real scene so the
resolution rules — which decide whether an eval checker gets a box at all —
are pinned exactly, including the ones that must REFUSE to answer.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pxr")

from pxr import Gf, Usd, UsdGeom  # noqa: E402

# Loaded straight off disk rather than as
# ``genie_sim_engine_newton.engine.newton.static_props``: that package's
# __init__ pulls in the whole Newton engine (warp, the solver adapters), while
# this module deliberately depends on nothing but pxr. Importing by path keeps
# the test runnable wherever USD is available — no GPU, no warp, no ROS.
import importlib.util as _ilu  # noqa: E402
from pathlib import Path as _Path  # noqa: E402

_MOD_PATH = (_Path(__file__).resolve().parent.parent
             / "genie_sim_engine_newton" / "engine" / "newton" / "static_props.py")
_spec = _ilu.spec_from_file_location("_static_props_under_test", _MOD_PATH)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
compute_static_prop_aabbs = _mod.compute_static_prop_aabbs


def _cube(stage, path, translate, size=1.0):
    """A unit cube of ``size``, centred on ``translate``."""
    cube = UsdGeom.Cube.Define(stage, path)
    cube.GetSizeAttr().Set(size)
    UsdGeom.Xformable(cube).AddTranslateOp().Set(Gf.Vec3d(*translate))
    return cube


@pytest.fixture
def stage():
    """/Background/storage_bin + /Background/background/scanner, plus a
    /World object — the shape a composed market scene actually has."""
    st = Usd.Stage.CreateInMemory()
    _cube(st, "/Background/storage_bin", (2.0, 0.0, 0.5), size=1.0)
    _cube(st, "/Background/background/scanner", (-3.0, 1.0, 0.0), size=2.0)
    _cube(st, "/World/Objects/bottle", (0.0, 0.0, 0.0), size=0.1)
    return st


def test_resolves_checker_dialect_onto_background_path(stage):
    """Checkers say /World/storage_bin; the composed stage says
    /Background/storage_bin. Without this mapping every Inside checker in
    the market scene silently scores zero."""
    names, boxes = compute_static_prop_aabbs(stage, ["/World/storage_bin"])
    assert names == ["/World/storage_bin"]        # keyed as REQUESTED
    assert boxes[0] == pytest.approx([1.5, -0.5, 0.0, 2.5, 0.5, 1.0])


def test_exact_path_is_used_as_is(stage):
    names, boxes = compute_static_prop_aabbs(stage, ["/Background/storage_bin"])
    assert names == ["/Background/storage_bin"]
    assert boxes[0] == pytest.approx([1.5, -0.5, 0.0, 2.5, 0.5, 1.0])


def test_nested_prop_resolves_by_trailing_name(stage):
    """/World/background/scanner -> /Background/background/scanner."""
    names, boxes = compute_static_prop_aabbs(stage, ["/World/background/scanner"])
    assert names == ["/World/background/scanner"]
    assert boxes[0] == pytest.approx([-4.0, 0.0, -1.0, -2.0, 2.0, 1.0])


def test_box_is_world_frame_min_max(stage):
    """Same convention as get_body_aabbs, so both reach the shim in one
    shape: [min_xyz, max_xyz], not centre+half-extent."""
    _names, boxes = compute_static_prop_aabbs(stage, ["/World/storage_bin"])
    lo, hi = boxes[0][:3], boxes[0][3:]
    assert all(l < h for l, h in zip(lo, hi))


def test_unknown_prop_is_skipped_not_zero_boxed(stage):
    """A skipped prop leaves the shim's own "unresolved" warning intact; a
    zero box would masquerade as a real answer."""
    names, boxes = compute_static_prop_aabbs(stage, ["/World/no_such_prop"])
    assert names == [] and boxes == []


def test_ambiguous_name_is_refused(stage):
    """Two props sharing a name at the same depth: guessing would hand the
    checker the wrong box and produce plausible but wrong scores."""
    _cube(stage, "/Background/tray", (0.0, 5.0, 0.0))
    _cube(stage, "/Elsewhere/tray", (0.0, 9.0, 0.0))
    names, boxes = compute_static_prop_aabbs(stage, ["/World/tray"])
    assert names == [] and boxes == []


def test_shallowest_match_wins_over_nested_geometry(stage):
    """A prop's own mesh children share its name space; the prop prim itself
    is the intended answer."""
    _cube(stage, "/Background/crate", (7.0, 0.0, 0.0), size=2.0)
    _cube(stage, "/Background/crate/inner/crate", (7.0, 0.0, 0.0), size=0.2)
    names, boxes = compute_static_prop_aabbs(stage, ["/World/crate"])
    assert names == ["/World/crate"]
    # the outer 2.0 cube, not the 0.2 one nested inside it
    assert boxes[0] == pytest.approx([6.0, -1.0, -1.0, 8.0, 1.0, 1.0])


def test_empty_request_is_a_no_op(stage):
    assert compute_static_prop_aabbs(stage, []) == ([], [])
    assert compute_static_prop_aabbs(stage, ["", "  "]) == ([], [])


def test_multiple_props_keep_request_order(stage):
    names, boxes = compute_static_prop_aabbs(
        stage, ["/World/background/scanner", "/World/storage_bin"])
    assert names == ["/World/background/scanner", "/World/storage_bin"]
    assert len(boxes) == 2

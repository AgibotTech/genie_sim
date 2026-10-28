#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Pure host tests (no GPU) for the base-pin mode: ``PinBaseMode`` and
``resolve_pin_base_mode``."""

from __future__ import annotations

import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve()
# .../genie_sim_engine_newton/genie_sim_engine_newton/tools/<this>
_NEWTON_SRC_ROOT = _HERE.parents[2]  # dir containing the newton package
_SRC = _HERE.parents[3]  # ros_ws/.../src (holds both packages)
_CORE_SRC_ROOT = _SRC / "genie_sim_engine_core"  # dir containing the core package

for _p in (str(_NEWTON_SRC_ROOT), str(_CORE_SRC_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from genie_sim_engine_core.schema.scene import PIN_BASE_MODE_VALUES, PinBaseMode, RobotSource, SchemaError
from genie_sim_engine_newton.engine.newton.engine_base import resolve_pin_base_mode
from genie_sim_engine_newton.engine.newton.adapters.mujoco_warp_multi_env import (
    select_base_mocap_index,
)


def test_pinned_base_defaults_to_weld():
    """Asking for nothing must give the pre-mocap behaviour."""
    assert resolve_pin_base_mode(True, "") == "weld"
    for junk in ("  ", "MOCAPP", "auto", "yes"):
        assert resolve_pin_base_mode(True, junk) == "weld"


def test_explicit_mode_is_the_only_way_to_get_mocap():
    assert resolve_pin_base_mode(True, "mocap") == "mocap"
    assert resolve_pin_base_mode(True, "MoCap") == "mocap"
    assert resolve_pin_base_mode(True, "weld") == "weld"


def test_unpinned_build_stays_free():
    """pin_base_to_world=false outranks any mode request."""
    for mode in ("", "weld", "mocap"):
        assert resolve_pin_base_mode(False, mode) == "free"


# --- PinBaseMode: the value object the routing consumers ask ---------------
# ``resolve_pin_base_mode`` is now a one-line alias for ``PinBaseMode.resolve``
# and the three laws above are the same three laws; what follows tests the
# parts that only exist on the enum.


def test_the_mode_is_still_a_string_everywhere_it_is_printed():
    """★ The str-subclass trap, and the reason ``__str__ = str.__str__`` is in
    ``scene.py``.

    Without that one line, ``f"{PinBaseMode.WELD}"`` renders
    ``PinBaseMode.WELD`` on every Python before 3.12 — which would silently
    corrupt ``engine_base``'s ``pin_base_mode=...`` log line and any dict
    payload carrying the mode.  ``enum.StrEnum`` would have handled this but
    needs 3.11, and this package still builds on 3.10.
    """
    import json

    for mode in PinBaseMode:
        assert f"{mode}" == mode.value, f"f-string of {mode!r} is not its value"
        assert str(mode) == mode.value
        assert "{}".format(mode) == mode.value  # noqa: UP032 - exercising the path on purpose
        assert json.dumps({"m": mode}) == '{"m": "%s"}' % mode.value
        assert mode == mode.value  # the `== "weld"` comparisons still in the tree


def test_the_three_properties_answer_for_every_mode():
    """The whole point of the enum: each consumer asks one named question
    instead of re-deriving it from a string comparison."""
    table = {
        #                      keeps_root_weld, needs_base_clamp, base_replaceable_at_reset
        PinBaseMode.WELD: (True, False, False),
        PinBaseMode.MOCAP: (False, True, True),
        PinBaseMode.FREE: (False, False, True),
    }
    assert set(table) == set(PinBaseMode), "a mode was added without deciding its properties"
    for mode, (weld, clamp, replaceable) in table.items():
        assert mode.keeps_root_weld is weld, mode
        assert mode.needs_base_clamp is clamp, mode
        assert mode.base_replaceable_at_reset is replaceable, mode


def test_only_mocap_needs_the_adapter_clamp():
    """``engine_base``'s capability gate keys off exactly this, so it is worth
    stating separately: nothing but an explicit ``mocap`` can trip the gate
    that refuses a non-mjwarp solver."""
    tripped = [m for m in PinBaseMode if m.needs_base_clamp]
    assert tripped == [PinBaseMode.MOCAP]
    assert not PinBaseMode.resolve(True, "").needs_base_clamp
    assert not PinBaseMode.resolve(False, "mocap").needs_base_clamp


class _FakeEngine:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def test_of_is_the_single_home_of_the_defensive_fallback():
    """``PinBaseMode.of`` replaced four copies of
    ``"weld" if getattr(self, "_pin_base_to_world", False) else "free"``.
    The fallback matters because engines can be constructed by paths that
    predate ``_pin_base_mode``."""
    assert PinBaseMode.of(_FakeEngine(_pin_base_mode="mocap")) is PinBaseMode.MOCAP
    assert PinBaseMode.of(_FakeEngine(_pin_base_mode=PinBaseMode.FREE)) is PinBaseMode.FREE
    # no _pin_base_mode at all -> derive it from the boolean
    assert PinBaseMode.of(_FakeEngine(_pin_base_to_world=True)) is PinBaseMode.WELD
    assert PinBaseMode.of(_FakeEngine(_pin_base_to_world=False)) is PinBaseMode.FREE
    # neither attribute -> the pre-pin default
    assert PinBaseMode.of(_FakeEngine()) is PinBaseMode.FREE
    # junk in the attribute must not raise mid-substep
    assert PinBaseMode.of(_FakeEngine(_pin_base_mode="auto", _pin_base_to_world=True)) is PinBaseMode.WELD


def test_the_schema_refuses_junk_that_resolve_tolerates():
    """Two layers, two jobs.  The config boundary REJECTS an unknown mode so a
    typo is a startup error; ``resolve`` stays TOLERANT because engines also
    get built from direct kwargs that never pass through the schema, and
    falling back to the pre-mocap behaviour is the safe landing."""
    assert PinBaseMode.resolve(True, "auto") is PinBaseMode.WELD  # tolerant
    try:
        RobotSource.from_dict({"pin_base_mode": "auto"})
    except SchemaError as exc:
        assert "pin_base_mode" in str(exc) and "auto" in str(exc)
    else:
        raise AssertionError("schema accepted pin_base_mode='auto'")


def test_the_schema_normalizes_the_mode_once():
    """Downstream consumers never see case or whitespace, so none of them has
    to ``.strip().lower()`` again."""
    assert RobotSource.from_dict({"pin_base_mode": "  MoCap "}).pin_base_mode == "mocap"
    assert RobotSource.from_dict({}).pin_base_mode == ""  # omitted -> take the default
    for value in PIN_BASE_MODE_VALUES:
        RobotSource.from_dict({"pin_base_mode": value})  # must not raise


def test_the_accepted_spellings_match_the_enum():
    """``PIN_BASE_MODE_VALUES`` is what the schema validates against; ``free``
    is deliberately NOT in it — it is a resolved outcome of
    ``pin_base_to_world: false``, not something a yaml asks for."""
    assert PIN_BASE_MODE_VALUES == {"", PinBaseMode.WELD.value, PinBaseMode.MOCAP.value}
    assert PinBaseMode.FREE.value not in PIN_BASE_MODE_VALUES


def test_the_dropped_base_pose_fields_round_trip_through_extra():
    """``init_base_pose2d`` / ``init_base_z`` were typed and had zero
    consumers.  Dropping them is lossless: a yaml that still sets them lands
    them in ``extra`` and emits them back verbatim, so nothing breaks and
    nothing pretends they are honoured."""
    from genie_sim_engine_core.schema.scene import RobotSection

    raw = {"init_base_pose2d": {"x": 1.0, "y": 2.0, "theta": 0.5}, "init_base_z": 0.25}
    sec = RobotSection.from_dict(raw)
    assert not hasattr(sec, "init_base_pose2d")
    assert sec.extra == raw
    out = sec.to_dict()
    assert out["init_base_pose2d"] == raw["init_base_pose2d"]
    assert out["init_base_z"] == raw["init_base_z"]


# --- select_base_mocap_index (pin_base_to_world:false native base) ---------
# Pure host test of the base-mocap picker used by the mjwarp adapter's
# _snapshot_base_mocap. The offset WRITE (mocap_pos + joint_X_p) needs a live
# GPU solver and is covered by the controller's live gate, not here.


def test_base_mocap_single_base():
    # The observed G2 tabletop build: exactly one mocap body, the base.
    base_idx, others = select_base_mocap_index([(0, "_genie_base_link")])
    assert base_idx == 0
    assert others == []


def test_base_mocap_base_plus_object_mocaps():
    # If future assets also export a free-scene root as mocap, the base is
    # still the one whose name lacks "Objects".
    named = [
        (0, "_World_Objects_benchmark_building_blocks_378069fe_entity"),
        (1, "_genie_base_link"),
        (2, "_World_Objects_something_else_entity"),
    ]
    base_idx, others = select_base_mocap_index(named)
    assert base_idx == 1
    assert set(m for m, _ in others) == {0, 2}


def test_base_mocap_all_objects_gives_none():
    named = [(0, "_World_Objects_a_entity"), (1, "_World_Objects_b_entity")]
    base_idx, others = select_base_mocap_index(named)
    assert base_idx is None
    assert len(others) == 2


def test_base_mocap_empty_gives_none():
    base_idx, others = select_base_mocap_index([])
    assert base_idx is None
    assert others == []


def test_base_mocap_first_non_object_wins():
    # Two non-object mocaps → first is the base, the rest go to `others`.
    named = [(3, "_genie_base_link"), (4, "_genie_other_root")]
    base_idx, others = select_base_mocap_index(named)
    assert base_idx == 3
    assert others == [(4, "_genie_other_root")]

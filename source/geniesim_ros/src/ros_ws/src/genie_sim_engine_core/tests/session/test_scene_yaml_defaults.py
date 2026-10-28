# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""``scene_yaml_defaults:`` — a launcher-level base beneath every scene yaml.

The block shipped in ``launcher_newton_mjwarp_multi_env.yaml`` for months
carrying a comment that said it "is merged into every scene yaml loaded with
this launcher".  Nothing read it.  A tree-wide grep for the name found exactly
one hit — the comment describing itself — so an operator editing those contact
numbers was editing a file the engine never opened, and the sim they got was
the one they would have got with the block deleted.

Two things had to be true for the comment to become true, and both are pinned
here because both are silent when they break:

  * ``_merge_params_file`` must hand a nested MAPPING through intact.  It used
    to ``str()`` every non-bool value, so the block arrived as the text
    ``"{'newton': {...}}"`` — present, wrong type, and unusable without a
    consumer noticing.
  * the merge must run UNDER the scene yaml, per key, recursing into mappings
    and treating lists as leaves.  Launcher-wins would make a scene unable to
    disagree with the launcher, which is backwards for a per-fleet default;
    a shallow merge would make ``newton:`` in the scene delete the launcher's
    whole ``newton:`` subtree; an element-wise list merge would splice two
    friction triplets into one that neither file names.
"""

from __future__ import annotations

import textwrap

import pytest

from genie_sim_engine_core.common.loop import _merge_params_file
from genie_sim_engine_core.common.session import _deep_merge_under, _read_scene_yaml_defaults


class _Log:
    def __init__(self):
        self.infos: list[str] = []
        self.warns: list[str] = []

    def info(self, m=""):
        self.infos.append(str(m))

    def warn(self, m=""):
        self.warns.append(str(m))

    warning = warn


# ----------------------------------------------------------------------
# the params-file half — a mapping must survive as a mapping
# ----------------------------------------------------------------------

_LAUNCHER = textwrap.dedent("""
    genie_sim_engine:
      ros__parameters:
        physics_hz: 100.0
        fake_slam: true
        scene_yaml_defaults:
          newton:
            mjc_contact:
              robot:
                friction: [1.5, 0.5, 0.2]
                condim: 3
              gripper_pad:
                condim: 6
    """)


@pytest.fixture
def params(tmp_path):
    p = tmp_path / "launcher.yaml"
    p.write_text(_LAUNCHER)
    out: dict = {}
    _merge_params_file(out, str(p))
    return out


def test_a_nested_block_arrives_as_a_mapping(params):
    """The whole reason the block was inert."""
    block = params["scene_yaml_defaults"]
    assert isinstance(block, dict), f"stringified again: {block!r}"
    assert block["newton"]["mjc_contact"]["gripper_pad"]["condim"] == 6


def test_scalars_are_still_strings(params):
    """``EngineNodeParams.from_dict`` coerces every field from ``str``.

    Widening the mapping case must not widen the scalar case with it — the
    float/int/bool coercions all accept ``str`` and several would change
    meaning if handed a native type.
    """
    assert params["physics_hz"] == "100.0"
    assert params["fake_slam"] == "true"


# ----------------------------------------------------------------------
# the merge half — direction, depth, and what counts as a leaf
# ----------------------------------------------------------------------


def test_the_scene_wins_key_by_key():
    base = {"newton": {"mjc_contact": {"robot": {"condim": 3, "friction": [1.5, 0.5, 0.2]}}}}
    scene = {"newton": {"mjc_contact": {"robot": {"condim": 6}}}}
    got = _deep_merge_under(base, scene)
    robot = got["newton"]["mjc_contact"]["robot"]
    assert robot["condim"] == 6, "the scene must be able to disagree"
    assert robot["friction"] == [1.5, 0.5, 0.2], "and must not have to restate the rest"


def test_a_sibling_the_scene_never_mentions_survives():
    """The failure a shallow merge produces, which looks like nothing at all.

    A scene naming only ``newton.mjc_contact.robot`` would drop the launcher's
    ``gripper_pad`` — and a missing ``gripper_pad:`` block is not an error, it
    is "the pads stay on robot".  Silent, and worth ~27% of real-time.
    """
    base = {"newton": {"mjc_contact": {"robot": {"condim": 3}, "gripper_pad": {"condim": 6}}}}
    scene = {"newton": {"mjc_contact": {"robot": {"condim": 4}}}}
    got = _deep_merge_under(base, scene)
    assert got["newton"]["mjc_contact"]["gripper_pad"] == {"condim": 6}
    assert got["newton"]["mjc_contact"]["robot"] == {"condim": 4}


def test_a_list_is_a_leaf():
    """``friction: [1.0, 1.0, 0.0001]`` means that triplet, not a splice."""
    base = {"newton": {"mjc_contact": {"floor": {"friction": [1.5, 0.5, 0.2]}}}}
    scene = {"newton": {"mjc_contact": {"floor": {"friction": [1.0, 1.0, 0.0001]}}}}
    got = _deep_merge_under(base, scene)
    assert got["newton"]["mjc_contact"]["floor"]["friction"] == [1.0, 1.0, 0.0001]


def test_unrelated_top_level_blocks_both_survive():
    got = _deep_merge_under({"newton": {"a": 1}}, {"robot": {"b": 2}})
    assert got == {"newton": {"a": 1}, "robot": {"b": 2}}


def test_the_inputs_are_not_mutated():
    """``scene_cfg`` is handed to the engine and the params dict outlives it."""
    base = {"newton": {"mjc_contact": {"robot": {"condim": 3}}}}
    scene = {"newton": {"mjc_contact": {"robot": {"condim": 6}}}}
    _deep_merge_under(base, scene)
    assert base["newton"]["mjc_contact"]["robot"]["condim"] == 3
    assert scene["newton"]["mjc_contact"]["robot"] == {"condim": 6}


# ----------------------------------------------------------------------
# reading it off the params dict
# ----------------------------------------------------------------------


def test_a_mapping_is_taken_verbatim():
    block = {"newton": {"mjc_contact": {"robot": {"condim": 6}}}}
    assert _read_scene_yaml_defaults({"scene_yaml_defaults": block}, _Log()) is block


def test_a_cli_override_arrives_as_yaml_text():
    """``-p scene_yaml_defaults:={newton: {...}}`` — the flat-string route.

    A ``-p`` override never goes through the params file, so it is text.  The
    text a launcher yaml would have held is YAML, so that is what it is parsed
    as; refusing it would mean the CLI cannot reach a knob the file can.
    """
    got = _read_scene_yaml_defaults({"scene_yaml_defaults": "{newton: {mjc_contact: {robot: {condim: 6}}}}"}, _Log())
    assert got["newton"]["mjc_contact"]["robot"]["condim"] == 6


@pytest.mark.parametrize("absent", [{}, {"scene_yaml_defaults": None}, {"scene_yaml_defaults": ""}])
def test_absent_is_absent_and_silent(absent):
    log = _Log()
    assert _read_scene_yaml_defaults(absent, log) == {}
    assert not log.warns, "not setting the knob is not a mistake"


@pytest.mark.parametrize("bad", ["[1, 2, 3]", "just a string", 7])
def test_something_that_is_not_a_mapping_warns(bad):
    log = _Log()
    assert _read_scene_yaml_defaults({"scene_yaml_defaults": bad}, log) == {}
    assert log.warns, f"{bad!r} was ignored silently"


# ----------------------------------------------------------------------
# the shipped launcher, end to end through the params path
# ----------------------------------------------------------------------


def _shipped_launcher():
    from pathlib import Path

    here = Path(__file__).resolve()
    # .../src/genie_sim_engine_core/tests/session/<this>
    src = here.parents[3]
    return src / "genie_sim_bringup" / "config" / "launcher_newton_mjwarp_multi_env.yaml"


def test_the_shipped_launchers_block_reaches_the_reader_as_a_tree():
    """The claim in that file's own comment, asserted against that file.

    Reading a sibling package is deliberate: the comment is a promise made to
    whoever edits those contact numbers, and a unit test on synthetic yaml
    cannot keep it.  Skips rather than fails if the layout moves — this test
    is a guard on the promise, not on the path.
    """
    path = _shipped_launcher()
    if not path.is_file():
        pytest.skip(f"launcher yaml not at {path}")
    out: dict = {}
    _merge_params_file(out, str(path))
    block = _read_scene_yaml_defaults(out, _Log())
    assert block, "the launcher's scene_yaml_defaults block did not survive the params path"
    contact = block["newton"]["mjc_contact"]
    assert contact["gripper_pad"]["condim"] == 6
    assert contact["robot"]["condim"] == 3, "condim 6 on the whole robot is the ~27% path"

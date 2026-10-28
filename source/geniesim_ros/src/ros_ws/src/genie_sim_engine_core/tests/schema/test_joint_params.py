# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Tests for :mod:`genie_sim_engine_core.schema.joint_params`.

Each test that enforces a numbered invariant from
``source/geniesim_ros/docs/JOINT_PARAMS.md`` names it in the test name, so
a failure points at the design statement it violates rather than at an
implementation detail.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from genie_sim_engine_core.schema import SchemaError, SchemaVersionError
from genie_sim_engine_core.schema.joint_params import (
    BASELINE,
    OVERLAY_ACTUATOR_PLANES,
    PLANE_BOUND_FIELDS,
    PLANE_ISAACSIM_PHYSX,
    PLANE_MUJOCO_PD,
    PLANE_UNRECORDED,
    CONVERT_BLOCKED,
    CONVERT_CLAMP_RISK,
    CONVERT_EXACT,
    CONVERT_LOSSY,
    CHANNEL_A,
    CHANNEL_AB,
    CHANNEL_B,
    ENGINE_SCOPED_FIELDS,
    ENGINES,
    JOINT_PARAMS_SCHEMA_ID,
    FieldValue,
    JointParamOverlay,
    OverlayStack,
    Scalar,
    convert_to_physx,
    json_schema,
    load_joint_overlays,
    main,
    read_urdf_baseline,
    read_urdf_mimic_followers,
    read_urdf_names,
    resolve_all,
    validate,
    validate_against_urdf,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

URDF_TEXT = """<?xml version="1.0"?>
<!-- a provenance comment that must survive -->
<robot name="t">
  <link name="arm_l_link1">
    <inertial>
      <origin xyz="-0.000184301 0.001568154 0.172803911" rpy="0 0 0"/>
      <mass value="2.17"/>
      <inertia ixx="0.0045684858" ixy="0" ixz="0" iyy="0.0038299629" iyz="0" izz="0.0036068582"/>
    </inertial>
  </link>
  <link name="arm_r_link1"/>
  <joint name="idx21_arm_l_joint1" type="revolute">
    <limit effort="108" lower="-3.0718" upper="3.0718" velocity="3.05"/>
    <dynamics friction="1.0"/>
    <drive stiffness="40000" damping="600"/>
  </joint>
  <joint name="idx61_arm_r_joint1" type="revolute">
    <limit effort="108" lower="-3.0718" upper="3.0718" velocity="3.05"/>
    <dynamics friction="1.0"/>
  </joint>
</robot>
"""


def _overlay_dict(**over):
    base = {
        "schema": JOINT_PARAMS_SCHEMA_ID,
        "id": "test.overlay-1",
        "applies_to": {"robot_model": "genie"},
        # ``identified_on_engine`` is mandatory on any sysid overlay that
        # carries a servo gain, and most fixtures below add a ``stiffness``.
        # The plane is part of a valid overlay, not part of what each test
        # is probing, so it lives in the shared base.
        "meta": {"kind": "sysid", "identified_on_engine": "mujoco_pd"},
        "joints": {"idx21_arm_l_joint1": {"damping": 4.1596658, "frictionloss": 5.2468284}},
        "links": {"arm_l_link1": {"mass": 2.17000000002}},
    }
    base.update(over)
    return base


@pytest.fixture
def urdf(tmp_path: Path) -> Path:
    p = tmp_path / "robot.urdf"
    p.write_text(URDF_TEXT)
    return p


def _write(tmp_path: Path, name: str, data: dict) -> Path:
    import yaml

    p = tmp_path / name
    p.write_text(yaml.safe_dump(data, sort_keys=False))
    return p


# ---------------------------------------------------------------------------
# I9 — per-engine values on the four control-law fields
# ---------------------------------------------------------------------------
#
# mujoco and newton drive rigid bodies through the same mujoco-warp
# adapter, which keeps drive gains (ke, kd, effort) and passive
# (dof_damping, dof_frictionloss, armature) as two independent channels.
# PhysX supports only {kp, kd, max_effort, armature, damping,
# frictionloss}, folds passive damping into drive damping (kd + damping),
# and maps frictionloss onto physxJoint:jointFriction with approximate
# semantics. So the four control-law fields may differ by backend and the
# other six may not.


def _split(field: str = "stiffness", **branches) -> dict:
    """An overlay whose one joint carries ``field`` split across engines."""
    return _overlay_dict(joints={"idx21_arm_l_joint1": {field: branches}}, links={})


def test_flat_form_is_unchanged_by_the_engine_dimension(tmp_path: Path):
    """The ordinary scalar still means "every engine", with or without one.

    This is the compatibility guarantee: no existing overlay changes
    meaning, and no existing overlay has to be migrated.
    """
    path = _write(tmp_path, "flat.yaml", _overlay_dict())
    stack = load_joint_overlays([path])
    plain = stack.resolve_joint("idx21_arm_l_joint1")
    assert plain["damping"].value == 4.1596658
    assert plain["damping"].engine == "" and not plain["damping"].needs_engine
    for engine in ENGINES:
        bound = stack.for_engine(engine).resolve_joint("idx21_arm_l_joint1")
        assert bound["damping"].value == 4.1596658
        # A shared value is not attributed to an engine: it was not
        # measured per engine, and the report must not imply it was.
        assert bound["damping"].engine == ""
    assert not stack.has_per_engine_fields()


def test_per_engine_form_resolves_each_branch(tmp_path: Path):
    path = _write(
        tmp_path,
        "split.yaml",
        _split(mujoco=0.65, newton=0.65, physx={"value": 0.71, "note": "refit on physx"}),
    )
    stack = load_joint_overlays([path])
    assert stack.has_per_engine_fields()
    got = {e: stack.for_engine(e).resolve_joint("idx21_arm_l_joint1")["stiffness"] for e in ENGINES}
    assert got["mujoco"].value == 0.65 and got["mujoco"].engine == "mujoco"
    assert got["newton"].value == 0.65
    assert got["physx"].value == 0.71 and got["physx"].engine == "physx"


def test_no_engine_selected_yields_no_value_not_a_guess(tmp_path: Path):
    """A caller that does not know its backend must not receive one.

    Picking a branch would hand a PhysX number to a Newton run and look
    exactly like a successful resolution.
    """
    path = _write(tmp_path, "split.yaml", _split(mujoco=0.65, newton=0.65, physx=0.71))
    r = load_joint_overlays([path]).resolve_joint("idx21_arm_l_joint1")["stiffness"]
    assert r.value is None and r.needs_engine and r.engine == ""
    assert r.from_overlay, "the overlay does carry the field; only the branch is undecided"


def test_engine_key_on_an_unscoped_field_is_an_error(tmp_path: Path):
    """armature is rotor inertia on every backend — one value, or a bug.

    This is the guardrail that makes enumerating ENGINE_SCOPED_FIELDS
    worth it rather than allowing the form everywhere.
    """
    path = _write(tmp_path, "bad.yaml", _split("armature", physx=0.0005))
    with pytest.raises(SchemaError) as err:
        JointParamOverlay.read(path)
    msg = str(err.value)
    assert "armature" in msg and "same physical quantity" in msg
    for f in ENGINE_SCOPED_FIELDS:
        assert f in msg, "the error must name the fields that WOULD be accepted"


def test_mixing_the_scalar_and_per_engine_forms_is_an_error(tmp_path: Path):
    path = _write(tmp_path, "bad.yaml", _split(**{"value": 1.0, "physx": 2.0}))
    with pytest.raises(SchemaError, match="cannot mix"):
        JointParamOverlay.read(path)


def test_unknown_key_inside_a_per_engine_block_is_rejected(tmp_path: Path):
    """`isaacsim` is a package in this repo but not an engine key here."""
    path = _write(tmp_path, "bad.yaml", _split(mujoco=0.65, isaacsim=0.65))
    with pytest.raises(SchemaError, match="isaacsim"):
        JointParamOverlay.read(path)


def test_per_engine_value_demotes_an_ab_field_to_channel_b(tmp_path: Path):
    """One URDF is shared by every backend, so Channel A cannot carry three.

    stiffness and damping_drive are already B; only damping and
    frictionloss can hit this.
    """
    flat = _write(tmp_path, "flat.yaml", _overlay_dict())
    assert load_joint_overlays([flat]).resolve_joint("idx21_arm_l_joint1")["damping"].channel == CHANNEL_AB

    split = _write(tmp_path, "split.yaml", _split("damping", mujoco=4.15, newton=4.15, physx=3.9))
    stack = load_joint_overlays([split])
    assert stack.for_engine("physx").resolve_joint("idx21_arm_l_joint1")["damping"].channel == CHANNEL_B
    assert any("demotes" in w and "Channel A cannot" in w for w in validate(stack))


def test_i9_a_branch_must_say_what_identified_it(tmp_path: Path):
    """A per-engine value claims a separate measurement; unannotated, it
    is indistinguishable from someone pre-compensating a backend quirk,
    which I7 forbids outright."""
    path = _write(tmp_path, "split.yaml", _split(mujoco=0.65, newton=0.66, physx=0.71))
    warns = validate(load_joint_overlays([path]))
    assert any("carry no note" in w for w in warns)
    assert any("I7, I9" in w for w in warns)


def test_i9_identical_branches_are_a_flat_value_written_twice(tmp_path: Path):
    path = _write(
        tmp_path,
        "split.yaml",
        _split(
            mujoco={"value": 0.65, "note": "fitted on mujoco"},
            newton={"value": 0.65, "note": "fitted on newton"},
            physx={"value": 0.65, "note": "fitted on physx"},
        ),
    )
    assert any("flat value written 3 times" in w for w in validate(load_joint_overlays([path])))


def test_i9_an_unnamed_engine_gets_nothing_at_all(tmp_path: Path):
    """The sharp one: naming only physx leaves mujoco and newton with NO
    value, not with a sibling's. Easy to write when one backend was
    refitted and the author thinks of the rest as unchanged."""
    path = _write(tmp_path, "split.yaml", _split(physx={"value": 0.71, "note": "refit"}))
    warns = validate(load_joint_overlays([path]))
    assert any("resolve to no value at all" in w and "mujoco" in w and "newton" in w for w in warns)


def test_per_engine_roundtrip_keeps_each_branch_terse(tmp_path: Path):
    """Emitting must not inflate a bare number into {value: ..}."""
    data = _split(mujoco=0.65, physx={"value": 0.71, "note": "refit on physx"})
    ov = JointParamOverlay.read(_write(tmp_path, "split.yaml", data))
    out = ov.to_dict()["joints"]["idx21_arm_l_joint1"]["stiffness"]
    assert out == {"mujoco": 0.65, "physx": {"value": 0.71, "note": "refit on physx"}}
    again = JointParamOverlay.from_dict(ov.to_dict())
    assert again.joints["idx21_arm_l_joint1"].values["stiffness"].to_dict() == out


def test_field_count_counts_scalars_not_fields(tmp_path: Path):
    """Three branches are three identified numbers, and the honest size
    of the file says so."""
    ov = JointParamOverlay.read(_write(tmp_path, "split.yaml", _split(mujoco=1.0, newton=2.0, physx=3.0)))
    assert ov.field_count() == 3


def test_a_later_overlay_replaces_a_shared_field_with_a_split_one(tmp_path: Path):
    """I6 is per field: the two forms do not merge into one another."""
    a = _write(tmp_path, "a.yaml", _overlay_dict(id="a", joints={"idx21_arm_l_joint1": {"stiffness": 1.0}}, links={}))
    b = _write(tmp_path, "b.yaml", _split(mujoco=2.0, newton=2.0, physx=3.0) | {"id": "b"})
    stack = load_joint_overlays([a, b])
    assert stack.for_engine("physx").resolve_joint("idx21_arm_l_joint1")["stiffness"].value == 3.0
    assert stack.for_engine("mujoco").resolve_joint("idx21_arm_l_joint1")["stiffness"].value == 2.0


def test_json_schema_agrees_with_the_parser_on_the_per_engine_form():
    jsonschema = pytest.importorskip("jsonschema")
    v = jsonschema.Draft202012Validator(json_schema())
    assert v.is_valid(_split(mujoco=0.65, physx=0.71)), "per-engine stiffness must validate"
    assert v.is_valid(_overlay_dict()), "the flat form must still validate"
    assert not v.is_valid(_split("armature", physx=0.0005)), "armature is not engine-scoped"
    assert not v.is_valid(_split(**{"value": 1.0, "physx": 2.0})), "the two forms must not mix"
    assert not v.is_valid(_split(mujoco=0.65, isaacsim=0.65)), "isaacsim is not an engine key"


def test_for_engine_rejects_an_unknown_engine():
    with pytest.raises(SchemaError, match="expected one of"):
        OverlayStack().for_engine("bullet")


def test_field_value_shared_applies_to_every_engine():
    fv = FieldValue(shared=Scalar(value=7.0))
    assert fv.for_engine("") is not None and not fv.is_per_engine
    assert all(fv.for_engine(e).value == 7.0 for e in ENGINES)



def test_resolve_labels_a_missing_branch_differently_from_a_missing_engine(
    tmp_path: Path, capsys
):
    """The two no-value cases need different fixes, so they read differently.

    ``needs_engine`` is one flag for both: nothing was selected, or
    something was and this overlay carries no branch for it. One label for
    both would send an author hunting for a CLI flag they already passed.
    """
    ov = _write(
        tmp_path,
        "partial.yaml",
        _split(mujoco={"value": 1.0, "note": "only mujoco was refit"}),
    )
    assert main(["resolve", "--overlay", str(ov)]) == 0
    out = capsys.readouterr().out
    assert "[mujoco]" in out and "NEEDS --engine" not in out  # the branch prints

    assert main(["resolve", "--overlay", str(ov), "--engine", "physx"]) == 0
    out = capsys.readouterr().out
    assert "[NO physx BRANCH]" in out
    assert "[NEEDS --engine]" not in out

    # ...and it must never borrow the mujoco number (I9).
    assert "1" not in out.split("stiffness")[1].split("<-")[0]

# ---------------------------------------------------------------------------
# I1 — an empty overlay stack changes nothing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("empty", [None, [], ()])
def test_i1_empty_stack_is_a_noop(empty):
    stack = load_joint_overlays(empty)
    assert isinstance(stack, OverlayStack)
    assert not stack
    assert stack.ids == ()
    assert stack.resolve_joint("idx21_arm_l_joint1") == {}
    assert stack.resolve_link("arm_l_link1") == {}


def test_i1_empty_stack_resolves_to_baseline_only(urdf: Path):
    """Every field reports the baseline, and none claims an overlay source."""
    stack = load_joint_overlays([])
    baseline = read_urdf_baseline(urdf)
    fields = resolve_all(stack, "joint", "idx21_arm_l_joint1", baseline[("joint", "idx21_arm_l_joint1")])
    assert fields, "baseline extraction produced nothing"
    assert all(r.source == BASELINE for r in fields.values())
    assert not any(r.from_overlay for r in fields.values())


def test_i1_phase1_surface_is_read_only(tmp_path: Path, urdf: Path, capsys):
    """validate/resolve/diff/bilateral must not touch the URDF or the overlay.

    Phase 1 ships no mutation path at all; this pins that down so a later
    Channel A implementation cannot leak writes into the query commands.
    """
    ov = _write(tmp_path, "o.yaml", _overlay_dict())
    before = {p: p.read_bytes() for p in (urdf, ov)}
    for argv in (
        ["validate", str(ov), "--urdf", str(urdf)],
        ["resolve", "--overlay", str(ov), "--urdf", str(urdf)],
        ["diff", "--overlay", str(ov), "--urdf", str(urdf)],
        ["bilateral", "--overlay", str(ov)],
    ):
        assert main(argv) == 0, argv
    capsys.readouterr()
    for p, raw in before.items():
        assert p.read_bytes() == raw, f"{p.name} was modified by a read-only command"


def test_elementtree_roundtrip_is_not_byte_identical(urdf: Path):
    """Why Channel A must be a surgical text patch, not an XML round-trip.

    A round-trip through ElementTree drops comments and rewrites the XML
    declaration. On the real generated ``genie_g2_crsB_wuji.urdf`` that is
    8.5% of the file and 121 comments, including the autogenerated
    "EDITING THIS FILE BY HAND IS NOT RECOMMENDED" header. I1 therefore
    rules the approach out. This test is the guard that keeps someone from
    reintroducing it.
    """
    import xml.etree.ElementTree as ET

    original = urdf.read_bytes()
    roundtripped = ET.tostring(ET.parse(str(urdf)).getroot(), encoding="utf-8", xml_declaration=True)
    assert roundtripped != original
    assert b"provenance comment" in original
    assert b"provenance comment" not in roundtripped


# ---------------------------------------------------------------------------
# I2 — an unknown name is a hard error
# ---------------------------------------------------------------------------


def test_i2_unknown_joint_name_raises(tmp_path: Path, urdf: Path):
    ov = _write(tmp_path, "o.yaml", _overlay_dict(joints={"nope_joint9": {"damping": 1.0}}))
    stack = load_joint_overlays([ov])
    with pytest.raises(SchemaError) as err:
        validate_against_urdf(stack, urdf)
    assert "nope_joint9" in str(err.value)
    assert "no such joint" in str(err.value)


def test_i2_unknown_link_name_raises(tmp_path: Path, urdf: Path):
    ov = _write(tmp_path, "o.yaml", _overlay_dict(links={"arm_l_link99": {"mass": 1.0}}))
    with pytest.raises(SchemaError, match="no such link"):
        validate_against_urdf(load_joint_overlays([ov]), urdf)


def test_i2_offers_a_did_you_mean_hint(tmp_path: Path, urdf: Path):
    """The idx prefix asymmetry makes this the likeliest real-world typo.

    URDF joint names carry ``idxNN_``; link names do not. An author who
    writes the joint without its prefix must be told the exact name.
    """
    ov = _write(tmp_path, "o.yaml", _overlay_dict(joints={"arm_l_joint1": {"damping": 1.0}}))
    with pytest.raises(SchemaError) as err:
        validate_against_urdf(load_joint_overlays([ov]), urdf)
    assert "idx21_arm_l_joint1" in str(err.value)


def test_i2_valid_names_pass(tmp_path: Path, urdf: Path):
    ov = _write(tmp_path, "o.yaml", _overlay_dict())
    validate_against_urdf(load_joint_overlays([ov]), urdf)  # must not raise


# ---------------------------------------------------------------------------
# I5 / I6 — overlay beats baseline; presence decides, not magnitude
# ---------------------------------------------------------------------------


def test_i6_zero_is_a_value(tmp_path: Path, urdf: Path):
    """A later overlay setting 0.0 must override an earlier nonzero value.

    The current class table treats ``armature = 0`` as "leave alone" but
    ``dof_damping = 0`` as "write 0" — two zero semantics in one loop.
    Overlays have exactly one: presence decides.
    """
    a = _write(tmp_path, "a.yaml", _overlay_dict(id="a", joints={"idx21_arm_l_joint1": {"damping": 4.2}}))
    b = _write(tmp_path, "b.yaml", _overlay_dict(id="b", joints={"idx21_arm_l_joint1": {"damping": 0.0}}))
    stack = load_joint_overlays([a, b])
    res = stack.resolve_joint("idx21_arm_l_joint1")
    assert res["damping"].value == 0.0
    assert res["damping"].source == "b"


def test_i5_overlay_beats_baseline(tmp_path: Path, urdf: Path):
    ov = _write(tmp_path, "o.yaml", _overlay_dict())
    stack = load_joint_overlays([ov])
    baseline = read_urdf_baseline(urdf)
    fields = resolve_all(stack, "joint", "idx21_arm_l_joint1", baseline[("joint", "idx21_arm_l_joint1")])
    assert fields["frictionloss"].value == pytest.approx(5.2468284)
    assert fields["frictionloss"].from_overlay
    # untouched by the overlay -> still baseline
    assert fields["effort"].value == pytest.approx(108.0)
    assert fields["effort"].source == BASELINE


def test_stacking_is_per_field_not_per_entry(tmp_path: Path):
    """``[older, newer]`` must keep the older fields the newer one omits.

    This is what lets a stack express "the newer inertials over the older
    friction" — which is what the G2 crsB set actually is, currently
    achieved by hand-carrying numbers between two files.
    """
    a = _write(
        tmp_path,
        "a.yaml",
        _overlay_dict(id="old", links={"arm_l_link1": {"mass": 1.0, "inertia": {"ixx": 9.0}}}),
    )
    b = _write(tmp_path, "b.yaml", _overlay_dict(id="new", links={"arm_l_link1": {"inertia": {"ixx": 2.0}}}))
    res = load_joint_overlays([a, b]).resolve_link("arm_l_link1")
    assert res["inertia.ixx"].value == 2.0 and res["inertia.ixx"].source == "new"
    assert res["mass"].value == 1.0 and res["mass"].source == "old"


# ---------------------------------------------------------------------------
# Channel assignment (design doc section 4)
# ---------------------------------------------------------------------------


def test_channel_assignment(tmp_path: Path):
    ov = _write(
        tmp_path,
        "o.yaml",
        _overlay_dict(
            joints={"idx21_arm_l_joint1": {"damping": 1.0, "armature": 2.0}},
            links={"arm_l_link1": {"mass": 3.0}},
        ),
    )
    stack = load_joint_overlays([ov])
    j = stack.resolve_joint("idx21_arm_l_joint1")
    # damping IS urdf-expressible but the class table clobbers it -> both channels
    assert j["damping"].channel == CHANNEL_AB
    # armature has no URDF representation at all -> resolver only
    assert j["armature"].channel == CHANNEL_B
    # inertials are importer-owned end to end -> patch only
    assert stack.resolve_link("arm_l_link1")["mass"].channel == CHANNEL_A


def test_unconsumed_field_is_flagged(tmp_path: Path):
    """command_delay has no mechanism anywhere; recording it must not imply effect."""
    ov = _write(tmp_path, "o.yaml", _overlay_dict(joints={"idx21_arm_l_joint1": {"command_delay": 0.085}}))
    stack = load_joint_overlays([ov])
    assert stack.resolve_joint("idx21_arm_l_joint1")["command_delay"].unconsumed
    assert any("NOTHING CONSUMES IT" in w for w in validate(stack))


# ---------------------------------------------------------------------------
# Schema strictness
# ---------------------------------------------------------------------------


def test_wrong_schema_id_raises_version_error(tmp_path: Path):
    ov = _write(tmp_path, "o.yaml", _overlay_dict(schema="genie_sim.joint_params/99"))
    with pytest.raises(SchemaVersionError):
        JointParamOverlay.read(ov)


def test_unknown_key_is_rejected(tmp_path: Path):
    ov = _write(tmp_path, "o.yaml", _overlay_dict(joints={"idx21_arm_l_joint1": {"dampingg": 1.0}}))
    with pytest.raises(SchemaError, match="unknown key"):
        JointParamOverlay.read(ov)


def test_missing_applies_to_is_rejected(tmp_path: Path):
    ov = _write(tmp_path, "o.yaml", _overlay_dict(applies_to={}))
    with pytest.raises(SchemaError, match="applies_to"):
        JointParamOverlay.read(ov)


def test_bad_meta_kind_is_rejected(tmp_path: Path):
    ov = _write(tmp_path, "o.yaml", _overlay_dict(meta={"kind": "guess"}))
    with pytest.raises(SchemaError, match="meta.kind"):
        JointParamOverlay.read(ov)


def test_bool_is_not_a_number(tmp_path: Path):
    """YAML ``true`` would coerce to 1.0 — never what an author meant."""
    ov = _write(tmp_path, "o.yaml", _overlay_dict(joints={"idx21_arm_l_joint1": {"damping": True}}))
    with pytest.raises(SchemaError, match="expected a number"):
        JointParamOverlay.read(ov)


def test_scalar_mapping_form_requires_value(tmp_path: Path):
    ov = _write(tmp_path, "o.yaml", _overlay_dict(links={"arm_l_link1": {"mass": {"frozen_to_cad": True}}}))
    with pytest.raises(SchemaError, match="requires a 'value' key"):
        JointParamOverlay.read(ov)


def test_com_must_have_three_components(tmp_path: Path):
    ov = _write(tmp_path, "o.yaml", _overlay_dict(links={"arm_l_link1": {"com": [1.0, 2.0]}}))
    with pytest.raises(SchemaError, match="expected 3 numbers"):
        JointParamOverlay.read(ov)


def test_missing_file_raises_schema_error(tmp_path: Path):
    with pytest.raises(SchemaError, match="overlay not found"):
        JointParamOverlay.read(tmp_path / "absent.yaml")


def test_contradiction_with_not_identified_warns(tmp_path: Path):
    ov = _write(
        tmp_path,
        "o.yaml",
        _overlay_dict(
            meta={"kind": "sysid", "not_identified": ["damping"]},
            joints={"idx21_arm_l_joint1": {"damping": 1.0}},
        ),
    )
    assert any("contradictory" in w for w in validate(load_joint_overlays([ov])))


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


def test_roundtrip_preserves_values_and_terseness(tmp_path: Path):
    src = _overlay_dict(
        links={"arm_l_link1": {"mass": {"value": 2.17000000002, "frozen_to_cad": True}, "com": [1.0, 2.0, 3.0]}}
    )
    ov = JointParamOverlay.from_dict(src)
    out = ov.to_dict()
    # annotated scalar keeps its mapping form
    assert out["links"]["arm_l_link1"]["mass"] == {"value": 2.17000000002, "frozen_to_cad": True}
    # a bare scalar does not grow a wrapper
    assert out["joints"]["idx21_arm_l_joint1"]["damping"] == 4.1596658
    assert JointParamOverlay.from_dict(out).to_dict() == out


def test_scalar_terse_roundtrip():
    assert Scalar(value=1.5).to_dict() == 1.5
    assert Scalar(value=1.5, note="x").to_dict() == {"value": 1.5, "note": "x"}


def test_field_count_is_the_honest_size(tmp_path: Path):
    ov = JointParamOverlay.from_dict(_overlay_dict())
    assert ov.field_count() == 3  # damping, frictionloss, mass


# ---------------------------------------------------------------------------
# Baseline extraction
# ---------------------------------------------------------------------------


def test_baseline_reads_urdf_values(urdf: Path):
    base = read_urdf_baseline(urdf)
    link = base[("link", "arm_l_link1")]
    assert link["mass"] == pytest.approx(2.17)
    assert link["com.z"] == pytest.approx(0.172803911)
    assert link["inertia.ixx"] == pytest.approx(0.0045684858)
    joint = base[("joint", "idx21_arm_l_joint1")]
    assert joint["frictionloss"] == pytest.approx(1.0)
    assert joint["effort"] == pytest.approx(108.0)
    # <dynamics> carries no damping attribute, so the baseline does not
    # express it at all -> Channel A must be able to INSERT an attribute.
    assert "damping" not in joint
    # drive@damping is a different quantity from dynamics@damping
    assert joint["damping_drive"] == pytest.approx(600.0)
    assert joint["stiffness"] == pytest.approx(40000.0)


def test_urdf_names_round(urdf: Path):
    names = read_urdf_names(urdf)
    assert "arm_l_link1" in names.links
    assert "idx21_arm_l_joint1" in names.joints
    assert "arm_l_joint1" not in names.joints  # the prefix is in the URDF


def test_bad_urdf_raises(tmp_path: Path):
    p = tmp_path / "bad.urdf"
    p.write_text("<robot><link name='a'>")
    with pytest.raises(SchemaError, match="not valid XML"):
        read_urdf_names(p)


# ---------------------------------------------------------------------------
# JSON Schema
# ---------------------------------------------------------------------------


def test_json_schema_is_serializable_and_strict():
    s = json_schema()
    json.dumps(s)
    assert s["properties"]["schema"]["const"] == JOINT_PARAMS_SCHEMA_ID
    assert s["additionalProperties"] is False
    assert s["properties"]["joints"]["additionalProperties"]["additionalProperties"] is False


def test_json_schema_accepts_a_valid_overlay_and_rejects_a_typo():
    jsonschema = pytest.importorskip("jsonschema")
    validator = jsonschema.Draft202012Validator(json_schema())
    validator.validate(_overlay_dict())
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(_overlay_dict(joints={"idx21_arm_l_joint1": {"dampingg": 1.0}}))


# ---------------------------------------------------------------------------
# The real migrated overlay, when the asset package is present
# ---------------------------------------------------------------------------

_RM = Path(__file__).resolve().parents[3] / "genie_sim_robot_model"
_REAL_OVERLAY = _RM / "robots/genie/g2/overlays/g2_crsB.sysid-2026-09-22.yaml"
_REAL_URDF = _RM / "urdf/genie_g2_crsB_wuji.urdf"


@pytest.mark.skipif(not _REAL_OVERLAY.is_file(), reason="robot_model overlay not present")
def test_real_g2_overlay_loads_and_has_expected_shape():
    ov = JointParamOverlay.read(_REAL_OVERLAY)
    assert ov.meta.kind == "sysid"
    assert len(ov.links) == 14 and len(ov.joints) == 14
    # every joint carries exactly the three identified passive families
    for name, j in ov.joints.items():
        assert set(j.values) == {"damping", "frictionloss", "armature"}, name
    # masses frozen to CAD are annotated, and it is links 1-3 on both arms
    frozen = {n for n, lk in ov.links.items() if lk.mass is not None and lk.mass.frozen_to_cad}
    assert frozen == {f"arm_{s}_link{i}" for s in "lr" for i in (1, 2, 3)}
    # no limit field was measured by the run and none may be carried: it
    # fits on the joint-torque plane, so a limit is an input to it at most
    assert {"effort", "velocity", "max_effort"} <= set(ov.meta.not_identified)
    for f in ("effort", "velocity", "max_effort"):
        assert not any(f in j.values for j in ov.joints.values()), f


@pytest.mark.skipif(
    not (_REAL_OVERLAY.is_file() and _REAL_URDF.is_file()),
    reason="robot_model overlay or built urdf not present",
)
def test_real_g2_overlay_names_all_resolve():
    """I2 against the real asset: every key must exist in the real URDF."""
    validate_against_urdf(load_joint_overlays([_REAL_OVERLAY]), _REAL_URDF)


# ---------------------------------------------------------------------------
# The real arx overlays, when the asset package is present
# ---------------------------------------------------------------------------

_AC1_SYSID = _RM / "robots/arx/acone/overlays/ac1.sysid-2026-09-30.yaml"
_X5A_SYSID = _RM / "robots/arx/x5/overlays/x5a.sysid-2026-09-30.yaml"
_AC1_URDF = _RM / "urdf/arx_acone.urdf"
_X5A_URDF = _RM / "urdf/arx_x5.urdf"

#: ``max_effort`` is deliberately absent: it is the runtime drive clamp, a
#: Channel B field with no URDF counterpart, so no run identifies it and no
#: set publishes it. The clamp each run applied lives in ``meta.caveats``.
_AC1_FAMILIES = {
    "damping",
    "frictionloss",
    "armature",
    "stiffness",
    "damping_drive",
    "command_delay",
}


@pytest.mark.skipif(not _AC1_SYSID.is_file(), reason="robot_model overlay not present")
def test_real_ac1_carries_the_left_arm_on_both_sides():
    """One arm's numbers on both, with the delay as the single exception.

    The right arm's own fit was not accepted: its damping_drive layer
    collapsed, three passive families were never swept there, and R6 is
    provably unidentified. The delays were fitted independently per side
    and agree to 4.7%, so that split is a measurement and must survive.
    """
    ov = JointParamOverlay.read(_AC1_SYSID)
    assert ov.meta.kind == "sysid"
    assert len(ov.joints) == 12 and not ov.links
    asymmetric = set()
    for i in range(1, 7):
        left = ov.joints[f"arm_l_joint{i}"].values
        right = ov.joints[f"arm_r_joint{i}"].values
        assert set(left) == set(right) == _AC1_FAMILIES, i
        asymmetric |= {f for f in left if left[f].shared.value != right[f].shared.value}
    assert asymmetric == {"command_delay"}
    assert ov.joints["arm_l_joint1"].values["command_delay"].shared.value == 0.085
    assert ov.joints["arm_r_joint1"].values["command_delay"].shared.value == 0.089


@pytest.mark.skipif(not _AC1_SYSID.is_file(), reason="robot_model overlay not present")
def test_real_ac1_keeps_the_substitution_checkable():
    """Every substituted right-arm value says it is the left arm's."""
    ov = JointParamOverlay.read(_AC1_SYSID)
    for i in range(1, 7):
        note = ov.joints[f"arm_r_joint{i}"].values["stiffness"].shared.note or ""
        assert "left-arm value" in note, i
    # and the right arm's own rejected numbers are recorded in those notes
    notes = " ".join(
        (sc.shared.note or "") for i in range(1, 7) for sc in ov.joints[f"arm_r_joint{i}"].values.values()
    )
    for rejected in ("587.303", "560.671", "488.049", "339.837", "142.58", "630.295"):
        assert rejected in notes, rejected


@pytest.mark.skipif(not _X5A_SYSID.is_file(), reason="robot_model overlay not present")
def test_real_x5a_overlay_is_the_ac1_left_arm():
    """X5A was never swept; it carries the AC one left arm, value for value."""
    x5a = JointParamOverlay.read(_X5A_SYSID)
    ac1 = JointParamOverlay.read(_AC1_SYSID)
    assert x5a.meta.kind == "sysid"
    assert len(x5a.joints) == 6 and not x5a.links
    for i in range(1, 7):
        got = {k: v.shared.value for k, v in x5a.joints[f"arm_joint{i}"].values.items()}
        want = {k: v.shared.value for k, v in ac1.joints[f"arm_l_joint{i}"].values.items()}
        assert got == want, i


@pytest.mark.skipif(
    not (_AC1_SYSID.is_file() and _AC1_URDF.is_file()),
    reason="robot_model overlay or built urdf not present",
)
def test_real_ac1_overlay_names_all_resolve():
    validate_against_urdf(load_joint_overlays([_AC1_SYSID]), _AC1_URDF)


@pytest.mark.skipif(
    not (_X5A_SYSID.is_file() and _X5A_URDF.is_file()),
    reason="robot_model overlay or built urdf not present",
)
def test_real_x5a_overlay_names_all_resolve():
    validate_against_urdf(load_joint_overlays([_X5A_SYSID]), _X5A_URDF)


# ---------------------------------------------------------------------------
# The real G2 end-effector overlays, when the asset package is present
# ---------------------------------------------------------------------------

_WUJI_SYSID = _RM / "robots/genie/g2/overlays/g2_wuji.sysid-2026-09-24.yaml"
_SP_SYSID = _RM / "robots/genie/g2/overlays/g2_swiftpicker.sysid-2026-09-18.yaml"
_WUJI_URDF = _RM / "urdf/genie_g2_crsB_wuji.urdf"
#: Two swiftpicker URDFs, and the difference matters: the baseline is what an
#: overlay's joint names are checked against (I2), the sysid build is what the
#: converter takes its torque denominator and mimic topology from. Their joint
#: name sets happen to be identical today; one name for both would hide the
#: day they diverge.
_SP_URDF = _RM / "urdf/genie_g2_crsB_swiftpicker.urdf"
_SP_SYSID_URDF = _RM / "urdf/genie_g2_crsB_swiftpicker_sysid.urdf"

#: no ``max_effort`` here either -- see _AC1_FAMILIES.
_EE_FAMILIES = {
    "damping",
    "frictionloss",
    "armature",
    "stiffness",
    "damping_drive",
}

_WUJI_FINGERS = ("thumb", "index", "middle", "ring", "pinky")
#: the five IDENTIFIED fields, uniform on all 40 joints.
_WUJI_SET = {
    "damping": 0.0,
    "frictionloss": 0.0,
    "armature": 0.0005,
    "stiffness": 0.65,
    "damping_drive": 0.05,
}


@pytest.mark.skipif(not _WUJI_SYSID.is_file(), reason="robot_model overlay not present")
def test_real_wuji_is_one_uniform_set_on_all_forty_joints():
    """One class value per field, repeated 40 times — and that is the point.

    The vendor reference it derives from is per-actuator; collapsing it to
    a single class was deliberate, so the file must not read as if the
    spread had been measured away. If a future run ever splits thumb from
    pinky this test is the thing that should fail.
    """
    ov = JointParamOverlay.read(_WUJI_SYSID)
    assert ov.meta.kind == "sysid"
    assert len(ov.joints) == 40 and not ov.links
    for side in "lr":
        for finger in _WUJI_FINGERS:
            for i in (1, 2, 3, 4):
                j = ov.joints[f"hand_{side}_{finger}_joint{i}"].values
                assert set(j) == _EE_FAMILIES, (side, finger, i)
                assert {k: v.shared.value for k, v in j.items()} == _WUJI_SET, (side, finger, i)


@pytest.mark.skipif(not _WUJI_SYSID.is_file(), reason="robot_model overlay not present")
def test_real_wuji_publishes_no_effort_limit_in_either_field():
    """No run identifies a limit, so the overlay publishes none.

    Two different fields are why. ``effort`` is the vendor rating and
    round-trips through URDF ``<limit effort>``, which is where the hand's
    per-joint 0.1468..0.6494 N*m already live. ``max_effort`` is the runtime
    drive clamp -- no URDF counterpart at all -- so it is a condition of the
    run, not a property of the hardware. Both are declared absent, and the
    clamp the gains were selected under stays findable in the caveats.

    That clamp matters: a uniform 0.5 sits ABOVE the joint's own URDF limit on
    32 of the 40, and caps only ever rise, so a run that applies it grants
    torque authority the hand does not have. If a future run ever publishes
    either field here, this fails -- and it should, because one would be a
    hardware claim and the other a run condition masquerading as one.
    """
    ov = JointParamOverlay.read(_WUJI_SYSID)
    for side in "lr":
        note = ov.joints[f"hand_{side}_thumb_joint4"].values["stiffness"].shared.note or ""
        assert "selected, not fitted" in note, side
    for name, jp in ov.joints.items():
        assert "max_effort" not in jp.values, name
        assert "effort" not in jp.values, name
    declared = set(ov.meta.not_identified)
    assert {"effort", "max_effort", "velocity"} <= declared
    caveats = ov.meta.caveats or ""
    assert "MAX_EFFORT IS NOT PUBLISHED, AND THE RUN CONDITION WAS 0.5 N*M" in caveats
    assert "ABOVE the joint's own" in caveats and "32 of the 40" in caveats
    # the hardware figures are the URDF's, and the caveat points at them there
    assert "THE HARDWARE LIMIT IS THE URDF'S, AND IS LEFT THERE" in caveats
    if _WUJI_URDF.is_file():
        base = read_urdf_baseline(_WUJI_URDF)
        limits = {
            n: (base.get(("joint", n)) or {}).get("effort")
            for n in ov.joints
        }
        assert all(v is not None for v in limits.values())
        assert limits["hand_l_thumb_joint4"] == pytest.approx(0.1468)
        assert limits["hand_l_middle_joint1"] == pytest.approx(0.6494)
        # and the 0.5 clamp really is above most of them
        assert sum(1 for v in limits.values() if v < 0.5) == 32


@pytest.mark.skipif(not _SP_SYSID.is_file(), reason="robot_model overlay not present")
def test_real_swiftpicker_splits_master_from_mimic_followers():
    """Six coupled joints per side: one driven master, five followers.

    The passive triple is shared because the run cannot separate the
    chain; the drive gains are the master's alone and the followers carry
    explicit zeros so nothing fights the mimic equality (I6).
    """
    ov = JointParamOverlay.read(_SP_SYSID)
    assert ov.meta.kind == "sysid"
    assert len(ov.joints) == 12 and not ov.links
    passive = {
        "damping": 1.2533456087112427,
        "frictionloss": 0.07205305756313116,
        "armature": 0.00801871022251229,
    }
    followers = (
        "outer_joint1",
        "left_support_joint",
        "inner_joint2",
        "outer_joint2",
        "right_support_joint",
    )
    for side, base in (("l", 30), ("r", 70)):
        master = ov.joints[f"idx{base + 1}_gripper_{side}_inner_joint1"].values
        assert set(master) == _EE_FAMILIES, side
        assert {k: master[k].shared.value for k in passive} == passive, side
        assert master["stiffness"].shared.value == 202.026337
        assert master["damping_drive"].shared.value == 13.345311
        for offset, name in enumerate(followers, start=2):
            f = ov.joints[f"idx{base + offset}_gripper_{side}_{name}"].values
            assert set(f) == _EE_FAMILIES, (side, name)
            assert {k: f[k].shared.value for k in passive} == passive, (side, name)
            assert f["stiffness"].shared.value == 0.0 and f["damping_drive"].shared.value == 0.0


@pytest.mark.skipif(not _SP_SYSID.is_file(), reason="robot_model overlay not present")
def test_real_swiftpicker_records_what_the_schema_cannot_carry():
    """No vendor torque figure exists for this gripper, so none is published.

    18 is the drive clamp the capture ran under — a condition of the fit,
    not hardware data — and ``max_effort`` has no URDF counterpart, so it
    could not be a rating even if the number were one. Four figures are in
    circulation for the master (0, none, 50, 18) and the caveats have to
    enumerate them, because the absence is the finding: until a datasheet
    exists the URDF''s 50 is the effective ceiling and is itself a
    simulation setting.

    The equality-constraint and actuator-filter values frozen by the same
    run have no field here at all, and the caveats must say so too.
    """
    ov = JointParamOverlay.read(_SP_SYSID)
    for name, jp in ov.joints.items():
        assert "max_effort" not in jp.values, name
        assert "effort" not in jp.values, name
    assert {"effort", "max_effort", "velocity"} <= set(ov.meta.not_identified)
    caveats = ov.meta.caveats or ""
    assert "NO VENDOR TORQUE LIMIT EXISTS FOR THIS GRIPPER" in caveats
    assert "MAX_EFFORT IS NOT\n    PUBLISHED" in caveats or "MAX_EFFORT IS NOT PUBLISHED" in caveats
    # all four circulating numbers, and what each one actually is
    assert "none of them\n    is a rating" in caveats or "none of them is a rating" in caveats
    assert "authors effort 0" in caveats and "no force range at all" in caveats
    assert "raised from 5 in" in caveats and "per-module torque of 50" in caveats
    # 18 stays findable as the condition of the fit
    assert "clamped the master drive at 18 N*m" in caveats
    assert "NO FIELD IN THIS SCHEMA" in caveats
    assert "0.35 s low-pass" in caveats
    # nothing claims a per-side measurement
    assert "not a replication result" in caveats


@pytest.mark.skipif(
    not (_WUJI_SYSID.is_file() and _WUJI_URDF.is_file()),
    reason="robot_model overlay or built urdf not present",
)
def test_real_wuji_overlay_names_all_resolve():
    validate_against_urdf(load_joint_overlays([_WUJI_SYSID]), _WUJI_URDF)


@pytest.mark.skipif(
    not (_SP_SYSID.is_file() and _SP_URDF.is_file()),
    reason="robot_model overlay or built urdf not present",
)
def test_real_swiftpicker_overlay_names_all_resolve():
    validate_against_urdf(load_joint_overlays([_SP_SYSID]), _SP_URDF)


# ---------------------------------------------------------------------------
# Package export (lazy, PEP 562)
# ---------------------------------------------------------------------------


def test_every_real_overlay_is_flat_today():
    """No shipped overlay splits a field by engine yet, and that is a fact
    about the DATA, not the schema.

    wuji's 0.65/0.05 was used identically on all three backends,
    swiftpicker's 202.026337/13.345311 came from a pipeline spanning
    them, crsB publishes no gains and arx was fitted in one harness. So
    every controller number currently in the tree is claimed for all
    three engines, and the `.shared` accesses in the tests above are
    reading what is actually there. When the first per-engine refit
    lands, this test is the one that should fail and be updated
    deliberately -- alongside the caveats that say which engine each
    number was fitted on.
    """
    present = [p for p in (_REAL_OVERLAY, _AC1_SYSID, _X5A_SYSID, _WUJI_SYSID, _SP_SYSID) if p.is_file()]
    if not present:
        pytest.skip("robot_model overlays not present")
    for path in present:
        stack = OverlayStack(overlays=(JointParamOverlay.read(path),))
        assert not stack.has_per_engine_fields(), path.name


def test_exported_from_the_schema_package():
    from genie_sim_engine_core import schema

    assert schema.JointParamOverlay is JointParamOverlay
    assert schema.load_joint_overlays is load_joint_overlays
    assert "JointParamOverlay" in dir(schema)
    assert sorted(schema.__all__) == list(schema.__all__), "__all__ drifted out of ASCII order"
    assert not [n for n in schema.__all__ if not hasattr(schema, n)]


def test_schema_package_import_does_not_pull_in_joint_params():
    """The export is lazy, and two things depend on that staying true.

    Every launcher imports this package for SceneConfig/Manifest and
    none of them touch an overlay unless one is selected; and an eager
    import makes ``python -m ...schema.joint_params`` warn (see below).
    Needs a subprocess for a clean ``sys.modules``.
    """
    import subprocess
    import sys

    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "import genie_sim_engine_core.schema, sys;"
            "print('genie_sim_engine_core.schema.joint_params' in sys.modules)",
        ],
        cwd=str(Path(__file__).resolve().parents[2]),
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "False", out.stdout


def test_module_cli_runs_without_a_runpy_double_import_warning():
    """``python -m`` on the documented CLI must be warning-free.

    An eager ``from .joint_params import ...`` in the package __init__
    puts the module in sys.modules before runpy executes it as
    __main__, and runpy then emits a RuntimeWarning about unpredictable
    behaviour on every single invocation of a documented command.
    ``-W error::RuntimeWarning`` turns that back into a test failure.
    """
    import subprocess
    import sys

    out = subprocess.run(
        [
            sys.executable,
            "-W",
            "error::RuntimeWarning",
            "-m",
            "genie_sim_engine_core.schema.joint_params",
            "emit-schema",
        ],
        cwd=str(Path(__file__).resolve().parents[2]),
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stderr
    assert "RuntimeWarning" not in out.stderr, out.stderr
    assert json.loads(out.stdout)["$id"] == JOINT_PARAMS_SCHEMA_ID


# ---------------------------------------------------------------------------
# Converting a mujoco/newton set onto PhysX
# ---------------------------------------------------------------------------
#
# The sysid runs happen on mujoco and newton, so every shipped overlay holds
# numbers identified against the mujoco-warp control law:
#
#     tau = ke*(q* - q) - kd*qd - c*qd - stickslip(tau_f)
#
# PhysX drives a joint as ``tau = kp*(q* - q) + d*(qd* - qd)`` under a
# maxForce clamp. Four fields therefore cross by identity, ``damping`` folds
# into ``kd`` as an exact sum, and ``frictionloss`` does not cross at all.
# These tests pin each of those three outcomes, because the dangerous one is
# silent: a dropped Coulomb term reads as a working conversion.


def _cv(tmp_path: Path, joints: dict, name: str = "cv.yaml"):
    """Convert a one-off overlay, bypassing the CLI."""
    path = _write(tmp_path, name, _overlay_dict(joints=joints, links={}))
    return convert_to_physx(load_joint_overlays([path]))


def test_identity_fields_cross_to_physx_unchanged(tmp_path: Path):
    """No unit factor touches an identified gain.

    The USD DriveAPI is authored per *degree* (``stiffness = 5 * pi/180``),
    which invites a pi/180 somewhere in here. It does not belong: that
    convention lives in the model-build seed, and the runtime gain write
    overwrites it with the SI number at solve time on both backends. So
    kp/kd/max_effort are N*m/rad and N*m, and armature is kg*m^2, exactly
    as identified.
    """
    conv = _cv(
        tmp_path,
        {"idx21_arm_l_joint1": {"stiffness": 202.026337, "damping_drive": 13.345311, "max_effort": 18.0, "armature": 0.008}},
    )
    (row,) = conv.rows
    assert row.kp == 202.026337
    assert row.kd == 13.345311
    assert row.max_effort == 18.0
    assert row.armature == 0.008
    assert row.status == CONVERT_EXACT
    assert row.frictionloss_dropped == 0.0


def test_damping_folds_into_kd_and_both_terms_are_reported(tmp_path: Path):
    """``d = damping_drive + damping``, with the summands kept.

    Exact rather than approximate, because the velocity target is pinned to
    zero (see ``tests/api/test_velocity_target_is_zero.py``): PhysX damps
    velocity *error*, so ``d*(0 - qd)`` is precisely the two terms mujoco
    applies through separate channels.

    Both summands stay in the row. A reader who sees only ``14.5987`` cannot
    tell which part was a drive gain and which was passive, and that is the
    difference between a value the adapter may fold again and one it may not.
    """
    conv = _cv(
        tmp_path,
        {"idx21_arm_l_joint1": {"stiffness": 202.026337, "damping_drive": 13.345311, "damping": 1.2533456087112427}},
    )
    (row,) = conv.rows
    assert row.kd == pytest.approx(14.598656608711243)
    assert row.kd_terms == (("damping_drive", 13.345311), ("damping", 1.2533456087112427))
    # Folding puts a passive term under the maxForce clamp that mujoco leaves
    # unclamped -- true but narrow, so it is a note, not a loss.
    assert row.status == CONVERT_CLAMP_RISK
    assert [t for k, t in row.notes if k == "clamp"]


def test_damping_with_no_stiffness_is_blocked_and_the_cli_exits_nonzero(tmp_path: Path, capsys):
    """The crsB shape: passive damping identified against no drive gain.

    mujoco can apply ``-c*qd`` on a joint with no position gain at all --
    ``dof_damping`` is an independent channel. PhysX cannot: the only sink
    for damping is the drive, so the folded term would act alongside a kp
    this set never measured. There is no number to transfer, and reporting
    one would be an invention. Exit nonzero so CI can gate on it.
    """
    joints = {"idx21_arm_l_joint1": {"damping": 4.1596658, "frictionloss": 5.2468284}}
    conv = _cv(tmp_path, joints)
    (row,) = conv.rows
    assert row.status == CONVERT_BLOCKED
    assert row.kp is None
    assert [t for k, t in row.notes if k == "blocked"]

    path = _write(tmp_path, "blocked.yaml", _overlay_dict(joints=joints, links={}))
    assert main(["convert", "--overlay", str(path), "--to", "physx"]) == 1
    out = capsys.readouterr().out
    assert "BLOCKED" in out and "PhysX-side identification run" in out


def test_frictionloss_is_reported_as_a_loss_and_never_folded_into_kd(tmp_path: Path):
    """The one field with no sink, and the one way to get it wrong.

    Coulomb friction is constant in magnitude; a viscous gain is
    proportional to speed. ``d_eq = tau_f / qd_ref`` would make them agree
    at one speed and disagree everywhere else, so the converter reports the
    dropped torque and leaves kd alone. A PhysX value is a refit on the
    PhysX plane (I9), not arithmetic on this number.
    """
    conv = _cv(
        tmp_path,
        {"idx21_arm_l_joint1": {"stiffness": 100.0, "damping_drive": 2.0, "frictionloss": 0.525452, "max_effort": 7.0}},
    )
    (row,) = conv.rows
    assert row.status == CONVERT_LOSSY
    assert row.frictionloss_dropped == 0.525452
    assert row.kd == 2.0  # untouched by the dropped Coulomb term
    assert row.dropped_fraction == pytest.approx(0.525452 / 7.0)
    assert [t for k, t in row.notes if k == "frictionloss"]


def test_zero_frictionloss_is_exact_not_lossy(tmp_path: Path):
    """The wuji shape -- and why it ran on all three backends unchanged.

    ``frictionloss: 0`` is a measurement that there is nothing to lose, not
    a missing field. Reporting it as a loss would turn the one genuinely
    engine-independent set in the tree into a false alarm.
    """
    conv = _cv(
        tmp_path,
        {"hand_l_index_joint1": {"stiffness": 0.65, "damping_drive": 0.05, "damping": 0.0, "frictionloss": 0.0, "armature": 0.0005}},
    )
    (row,) = conv.rows
    assert row.status == CONVERT_EXACT
    assert row.frictionloss_dropped == 0.0
    # damping measured as zero folds nothing, so the clamp caveat does not apply.
    assert not [t for k, t in row.notes if k == "clamp"]


def test_urdf_supplies_the_denominator_when_the_overlay_has_no_max_effort(tmp_path: Path, urdf: Path, capsys):
    """"0.07 N*m" means nothing without a torque cap to scale it against.

    No overlay in the tree publishes ``max_effort``, so without the URDF
    limit the report cannot say whether the dropped Coulomb is 0.4% of the
    actuator or 40% of it -- and when ``--urdf`` does supply it, the number
    is a vendor rating and the note says so rather than calling it a cap.
    """
    joints = {"idx21_arm_l_joint1": {"stiffness": 100.0, "frictionloss": 5.4}}
    path = _write(tmp_path, "nocap.yaml", _overlay_dict(joints=joints, links={}))

    assert main(["convert", "--overlay", str(path), "--to", "physx"]) == 0
    out = capsys.readouterr().out
    assert "no max_effort or URDF effort limit" in out

    assert main(["convert", "--overlay", str(path), "--to", "physx", "--urdf", str(urdf)]) == 0
    out = capsys.readouterr().out
    assert "5.0% of the URDF effort limit" in out  # 5.4 N*m against the URDF's effort="108"


def test_a_per_engine_field_without_from_is_a_hard_error(tmp_path: Path):
    """I9 again: an unselected branch must not be guessed.

    A per-engine ``stiffness`` means somebody measured the backends to be
    different. Converting from "whichever branch happens to be first" would
    launder that measurement into the wrong one, so the flag is required
    exactly when the data makes it ambiguous -- and only then.
    """
    joints = {"idx21_arm_l_joint1": {"stiffness": {"mujoco": 10.0, "newton": 10.0, "physx": 12.0}, "damping": 1.0}}
    path = _write(tmp_path, "split.yaml", _overlay_dict(joints=joints, links={}))
    stack = load_joint_overlays([path])
    with pytest.raises(SchemaError, match="--from"):
        convert_to_physx(stack)


@pytest.mark.parametrize(("engine", "expected"), [("mujoco", 10.0), ("newton", 10.0), ("physx", 12.0)])
def test_from_selects_the_named_branch(tmp_path: Path, engine: str, expected: float):
    """And having asked, the conversion reads that branch and no other."""
    joints = {"idx21_arm_l_joint1": {"stiffness": {"mujoco": 10.0, "newton": 10.0, "physx": 12.0}, "damping": 1.0}}
    path = _write(tmp_path, "split.yaml", _overlay_dict(joints=joints, links={}))
    conv = convert_to_physx(load_joint_overlays([path]), from_engine=engine)
    assert conv.from_engine == engine
    (row,) = conv.rows
    assert row.kp == expected


def test_json_output_carries_the_provenance_the_table_shows(tmp_path: Path, capsys):
    """``--json`` is the machine-readable form, so it must not lose the terms.

    Channel B does not exist yet, so applying a conversion is a hand edit to
    a physics profile today. The JSON is what that edit is made from, which
    means it has to carry the summands, the dropped torque and the status --
    not just the final kd.
    """
    joints = {
        "idx21_arm_l_joint1": {
            "stiffness": 202.026337,
            "damping_drive": 13.345311,
            "damping": 1.2533456087112427,
            "frictionloss": 0.0720530575631,
            "max_effort": 18.0,
        }
    }
    path = _write(tmp_path, "j.yaml", _overlay_dict(joints=joints, links={}))
    assert main(["convert", "--overlay", str(path), "--to", "physx", "--json"]) == 0
    doc = json.loads(capsys.readouterr().out)

    assert doc["target"] == "physx"
    assert doc["counts"]["lossy"] == 1
    assert doc["total_frictionloss_dropped"] == pytest.approx(0.0720530575631)
    (row,) = doc["joints"]
    assert row["kd"] == pytest.approx(14.598656608711243)
    assert [t["field"] for t in row["kd_terms"]] == ["damping_drive", "damping"]
    assert row["frictionloss_dropped"] == pytest.approx(0.0720530575631)
    assert row["status"] == "lossy"
    assert {n["kind"] for n in row["notes"]} == {"frictionloss", "clamp"}


# ---------------------------------------------------------------------------
# The five shipped sets, converted
# ---------------------------------------------------------------------------
#
# The per-set exposure, measured rather than assumed. This is the table a
# colleague re-running the matrix needs, and pinning it here is what keeps
# the documented figures and the tool from drifting apart.

_SETS = (
    # (path,                  joints, status,              dropped N*m)
    (_WUJI_SYSID, 40, CONVERT_EXACT, 0.0),
    (_SP_SYSID, 12, CONVERT_LOSSY, 0.864636690758),
    (_AC1_SYSID, 12, CONVERT_LOSSY, 2.22720496),
    (_X5A_SYSID, 6, CONVERT_LOSSY, 1.11360248),
    (_REAL_OVERLAY, 14, CONVERT_BLOCKED, 37.1281962354),
)


@pytest.mark.skipif(not _WUJI_SYSID.is_file(), reason="robot_model overlays not present")
@pytest.mark.parametrize(("path", "n_joints", "status", "dropped"), _SETS, ids=lambda v: getattr(v, "stem", ""))
def test_real_sets_convert_with_the_documented_exposure(path: Path, n_joints: int, status: str, dropped: float):
    conv = convert_to_physx(load_joint_overlays([path]), from_engine="newton")
    assert len(conv.rows) == n_joints
    # one status per set: these sets are homogeneous, and a mixed result would
    # mean a joint was identified differently from its neighbours.
    assert {r.status for r in conv.rows} == {status}
    assert conv.total_dropped == pytest.approx(dropped)


@pytest.mark.skipif(not _WUJI_SYSID.is_file(), reason="robot_model overlays not present")
def test_wuji_is_the_one_set_that_needs_no_conversion():
    """Which is why it ran on all three backends unchanged.

    ``damping`` is identically zero, so nothing folds; ``frictionloss`` is
    identically zero, so nothing is lost. The flat numbers in that set are
    genuinely engine-independent -- not by luck, but because the quantities
    that differ between the control laws were both measured as zero.
    """
    conv = convert_to_physx(load_joint_overlays([_WUJI_SYSID]), from_engine="newton")
    assert conv.total_dropped == 0.0
    assert not any(k == "clamp" for r in conv.rows for k, _ in r.notes)
    assert conv.counts()[CONVERT_EXACT] == 40


@pytest.mark.skipif(not _REAL_OVERLAY.is_file(), reason="robot_model overlay not present")
def test_crsB_cannot_be_transferred_to_physx_at_all():
    """The honest answer for the one set PhysX cannot reproduce.

    It carries ``damping`` with no ``stiffness`` and no ``damping_drive``, so
    the fold has no drive gain to land beside, and 1.26..5.37 N*m per joint of
    Coulomb has nowhere to go -- up to 11% of the joint's torque cap. Any
    number this converter printed for crsB would be fabricated, so it prints
    none and says why.
    """
    conv = convert_to_physx(load_joint_overlays([_REAL_OVERLAY]), from_engine="newton")
    assert len(conv.blocked) == 14
    assert all(r.kp is None and r.max_effort is None for r in conv.rows)
    assert min(r.frictionloss_dropped for r in conv.rows) == pytest.approx(1.255381573)
    assert max(r.frictionloss_dropped for r in conv.rows) == pytest.approx(5.36698311178)


@pytest.mark.skipif(
    not (_SP_SYSID.is_file() and _SP_SYSID_URDF.is_file()),
    reason="robot_model overlay or built urdf not present",
)
def test_urdf_resolves_the_torque_cap_swiftpicker_does_not_carry(capsys):
    """swiftpicker publishes no max_effort at all, so the URDF is the only cap.

    The overlay declares both limit fields absent, which leaves the dropped
    Coulomb term unscaled -- 0.07 N*m means nothing on its own. ``--urdf``
    is what turns it into a fraction, and the fractions it produces are
    against ``<limit effort>`` figures that are themselves simulation
    settings (master 50, followers 10), so the percentages are a magnitude
    check and not a hardware headroom claim.
    """
    args = ["convert", "--overlay", str(_SP_SYSID), "--to", "physx", "--from", "newton"]
    assert main(args) == 0
    assert "no max_effort or URDF effort limit to scale it against" in capsys.readouterr().out

    # Exit 1 here is the mimic verdict, not the cap: --urdf is also what makes
    # the ten followers readable as having no drive at all.
    assert main(args + ["--urdf", str(_SP_SYSID_URDF)]) == 1
    out = capsys.readouterr().out
    assert "no max_effort or URDF effort limit" not in out
    assert "0.1%..0.7% of the URDF effort limit" in out


# ---------------------------------------------------------------------------
# Mimic followers: the joints with no PhysX drive at all
# ---------------------------------------------------------------------------
#
# A kinematic follower is constrained by PhysxMimicJointAPI on the PhysX path
# and is given no DriveAPI, deliberately -- a force drive in parallel with a
# mimic constraint fights it. isaac_newton does the opposite and seeds a
# follower DriveAPI, because Newton's importer allocates no POSITION actuator
# for a drive left at (0, 0). Same overlay, same USD, two different sinks, so
# this cannot be read off the numbers: it comes from the URDF.

def test_mimic_follower_reader_finds_exactly_the_followers(tmp_path: Path):
    p = tmp_path / "m.urdf"
    p.write_text(
        '<robot name="t">'
        '<joint name="master" type="revolute"/>'
        '<joint name="f1" type="revolute"><mimic joint="master" multiplier="1"/></joint>'
        '<joint name="f2" type="revolute"><mimic joint="master" multiplier="-1"/></joint>'
        "</robot>"
    )
    assert read_urdf_mimic_followers(p) == {"f1", "f2"}


def test_a_mimic_follower_is_blocked_even_with_a_stiffness(tmp_path: Path):
    """The block is about the *sink*, not about what was identified.

    A follower with gains is the worse case, not the better one: the numbers
    look transferable and there is nowhere to put them.
    """
    joints = {"idx21_arm_l_joint1": {"stiffness": 100.0, "damping_drive": 2.0, "damping": 0.5}}
    path = _write(tmp_path, "f.yaml", _overlay_dict(joints=joints, links={}))
    conv = convert_to_physx(load_joint_overlays([path]), mimic_followers=frozenset({"idx21_arm_l_joint1"}))
    (row,) = conv.rows
    assert row.status == CONVERT_BLOCKED
    # The kind is distinct from the missing-stiffness block, because the report
    # has to blank the kp/kd/maxForce columns for a joint with no drive.
    reason = next(t for k, t in row.notes if k == "blocked-no-drive")
    assert "mimic follower" in reason and "DriveAPI" in reason
    # ...and the fold's clamp caveat is not also reported: there is no drive
    # for the folded term to be clamped by.
    assert not [t for k, t in row.notes if k == "clamp"]


@pytest.mark.skipif(
    not (_SP_SYSID.is_file() and _SP_SYSID_URDF.is_file()),
    reason="robot_model overlay or built urdf not present",
)
def test_swiftpicker_followers_have_no_physx_sink():
    """Ten of twelve swiftpicker joints cannot be transferred at all.

    The set's own documentation reads the hazard the other way round -- that
    the fold turns the followers' explicit ``damping_drive: 0`` into 1.2533
    of drive damping. It does not, on either backend. On PhysX there is no
    follower drive to fold into, so the identified passive damping is simply
    absent; and were a drive authored, ``kp = 0`` with a zero velocity target
    would reproduce mujoco's passive damper exactly rather than add to it.
    """
    followers = read_urdf_mimic_followers(_SP_SYSID_URDF)
    assert len(followers) == 10

    conv = convert_to_physx(
        load_joint_overlays([_SP_SYSID]), from_engine="newton", mimic_followers=followers
    )
    assert conv.counts() == {CONVERT_BLOCKED: 10, CONVERT_LOSSY: 2, CONVERT_CLAMP_RISK: 0, CONVERT_EXACT: 0}
    masters = [r for r in conv.rows if r.status == CONVERT_LOSSY]
    assert all(r.kd == pytest.approx(14.598656608711243) for r in masters)


@pytest.mark.skipif(not _SP_SYSID.is_file(), reason="robot_model overlays not present")
def test_without_a_urdf_the_report_names_its_blind_spot(capsys):
    """An undetectable follower must not read as a clean conversion.

    Without the kinematics the status column is an upper bound, and saying so
    is the difference between a tool that is silent about what it cannot see
    and one that is wrong about it.
    """
    assert main(["convert", "--overlay", str(_SP_SYSID), "--to", "physx", "--from", "newton"]) == 0
    out = capsys.readouterr().out
    assert "without --urdf, kinematic mimic followers cannot be identified" in out
    assert "upper bound" in out


@pytest.mark.skipif(
    not (_REAL_OVERLAY.is_file() and (_RM / "urdf/genie_g2_crsB_omnipicker_sysid.urdf").is_file()),
    reason="robot_model overlay or built urdf not present",
)
def test_crsB_blocks_for_the_other_reason():
    """crsB's URDF has 14 mimic joints and none of them is in the overlay.

    They are the omnipicker's gripper joints; the overlay carries the arm.
    So crsB must block on the missing stiffness, not on the kinematics --
    which is what makes the two reasons worth separating in the report.
    """
    followers = read_urdf_mimic_followers(_RM / "urdf/genie_g2_crsB_omnipicker_sysid.urdf")
    assert len(followers) == 14
    conv = convert_to_physx(
        load_joint_overlays([_REAL_OVERLAY]), from_engine="newton", mimic_followers=followers
    )
    assert not followers & set(r.joint for r in conv.rows)
    assert len(conv.blocked) == 14
    reasons = {t for r in conv.rows for k, t in r.notes if k.startswith("blocked")}
    # One reason per joint, because each quotes its own damping -- but all 14
    # are the missing-stiffness shape, and none is the mimic shape.
    assert len(reasons) == 14
    assert all("no identified stiffness" in r for r in reasons)
    assert not [1 for r in conv.rows for k, _ in r.notes if k == "blocked-no-drive"]


# ---------------------------------------------------------------------------
# Which control law a gain was fitted on
# ---------------------------------------------------------------------------
#
# MuJoCo's position actuator and PhysX's articulation drive are different
# laws, and a kp carries no record of which one it was fitted against. The
# failure is silent -- a gain read onto the wrong plane gives a plausible
# run, not a crash -- so the tag is a load-bearing field rather than a
# comment, and these tests are what keep it mandatory.


def test_the_plane_vocabulary_is_closed(tmp_path: Path):
    """A free-text engine name would re-open exactly what the tag closes."""
    data = _overlay_dict(joints={"idx21_arm_l_joint1": {"stiffness": 1.0}}, links={})
    data["meta"] = {"kind": "sysid", "identified_on_engine": "gazebo"}
    with pytest.raises(SchemaError, match="identified_on_engine"):
        JointParamOverlay.read(_write(tmp_path, "bad.yaml", data))


def test_a_gain_without_its_control_law_is_an_error(tmp_path: Path):
    """The whole point: an untagged kp cannot be converted or even read.

    This is an error and not a warning because nothing downstream can
    recover the missing fact, and every consumer that guesses it guesses
    the same way -- silently, and in favour of whatever it runs on.
    """
    for field in PLANE_BOUND_FIELDS:
        data = _overlay_dict(joints={"idx21_arm_l_joint1": {field: 1.0}}, links={})
        data["meta"] = {"kind": "sysid"}
        path = _write(tmp_path, f"untagged_{field}.yaml", data)
        with pytest.raises(SchemaError, match="identified_on_engine is unset"):
            validate(load_joint_overlays([path]))


def test_a_set_with_no_gain_needs_no_plane(tmp_path: Path):
    """The crsB shape: passive damping and frictionloss, no servo at all.

    Requiring a plane here would force a claim about gains the run never
    produced, so absence has to stay legal -- which is also what makes
    absence unambiguous on a set that *does* carry gains.
    """
    data = _overlay_dict(
        joints={"idx21_arm_l_joint1": {"damping": 4.16, "frictionloss": 5.25}}, links={}
    )
    data["meta"] = {"kind": "sysid"}
    path = _write(tmp_path, "passive.yaml", data)
    validate(load_joint_overlays([path]))  # must not raise
    conv = convert_to_physx(load_joint_overlays([path]))
    assert conv.source_planes == ()
    assert not conv.untransferable


def test_the_plane_round_trips(tmp_path: Path):
    data = _overlay_dict(joints={"idx21_arm_l_joint1": {"stiffness": 1.0}}, links={})
    ov = JointParamOverlay.read(_write(tmp_path, "p.yaml", data))
    assert ov.meta.identified_on_engine == PLANE_MUJOCO_PD
    assert ov.to_dict()["meta"]["identified_on_engine"] == PLANE_MUJOCO_PD
    assert JointParamOverlay.from_dict(ov.to_dict()).meta.identified_on_engine == PLANE_MUJOCO_PD


def test_json_schema_pins_the_plane_enum():
    props = json_schema()["properties"]["meta"]["properties"]
    assert props["identified_on_engine"]["enum"] == list(OVERLAY_ACTUATOR_PLANES)


def test_converting_a_physx_identified_set_is_refused(tmp_path: Path):
    """There is nothing to convert, and converting anyway corrupts it.

    A PhysX-fitted kd already contains the passive damping, because PhysX
    has nowhere else to put it. Folding ``damping`` in a second time would
    double-count the same dissipation.
    """
    data = _overlay_dict(
        joints={"idx21_arm_l_joint1": {"stiffness": 1.0, "damping_drive": 2.0, "damping": 0.5}},
        links={},
    )
    data["meta"] = {"kind": "sysid", "identified_on_engine": PLANE_ISAACSIM_PHYSX}
    stack = load_joint_overlays([_write(tmp_path, "px.yaml", data)])
    with pytest.raises(SchemaError, match="already on the target plane"):
        convert_to_physx(stack)


def test_an_unrecorded_plane_makes_the_whole_table_untransferable(tmp_path: Path, capsys):
    """``unrecorded`` is honest, and honestly unusable.

    The arithmetic still prints -- the numbers may well be MuJoCo gains --
    but the command exits nonzero, because a re-test run must not adopt a
    conversion whose left-hand side nobody can name.
    """
    data = _overlay_dict(joints={"idx21_arm_l_joint1": {"stiffness": 1.0, "damping": 0.5}}, links={})
    data["meta"] = {"kind": "sysid", "identified_on_engine": PLANE_UNRECORDED}
    path = _write(tmp_path, "unrec.yaml", data)

    conv = convert_to_physx(load_joint_overlays([path]))
    assert conv.source_planes == (PLANE_UNRECORDED,)
    assert conv.untransferable
    assert conv.counts()[CONVERT_BLOCKED] == 0  # per-joint it converts fine

    assert main(["convert", "--overlay", str(path), "--to", "physx"]) == 1
    assert "UNTRANSFERABLE" in capsys.readouterr().out


def test_json_carries_the_plane(tmp_path: Path, capsys):
    data = _overlay_dict(joints={"idx21_arm_l_joint1": {"stiffness": 1.0}}, links={})
    assert main(["convert", "--overlay", str(_write(tmp_path, "j.yaml", data)), "--to", "physx", "--json"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["source_planes"] == [PLANE_MUJOCO_PD]
    assert doc["untransferable"] is False


_PLANES_AS_SHIPPED = {
    # Every shipped gain was fitted on the MuJoCo position actuator. No
    # actuator parameter anywhere in this tree was identified on PhysX --
    # the PhysX runs that exist replay these same numbers.
    "robots/arx/acone/overlays/ac1.sysid-2026-09-30.yaml": PLANE_MUJOCO_PD,
    "robots/arx/x5/overlays/x5a.sysid-2026-09-30.yaml": PLANE_MUJOCO_PD,
    "robots/genie/g2/overlays/g2_swiftpicker.sysid-2026-09-18.yaml": PLANE_MUJOCO_PD,
    "robots/genie/g2/overlays/g2_wuji.sysid-2026-09-24.yaml": PLANE_MUJOCO_PD,
    # crsB identified no servo gain at all, so it declares no plane.
    "robots/genie/g2/overlays/g2_crsB.sysid-2026-09-22.yaml": "",
}


@pytest.mark.skipif(not _RM.is_dir(), reason="robot_model package not present")
def test_every_shipped_overlay_declares_its_plane():
    for rel, expected in _PLANES_AS_SHIPPED.items():
        path = _RM / rel
        if not path.is_file():
            continue
        ov = JointParamOverlay.read(path)
        assert ov.meta.identified_on_engine == expected, rel
        # And the invariant agrees with the tag rather than merely allowing
        # it: a declared plane must be backed by a gain, and a blank one by
        # the absence of every gain.
        carries = any(
            f in PLANE_BOUND_FIELDS for entry in ov.joints.values() for f in entry.fields()
        )
        assert carries == bool(expected), rel

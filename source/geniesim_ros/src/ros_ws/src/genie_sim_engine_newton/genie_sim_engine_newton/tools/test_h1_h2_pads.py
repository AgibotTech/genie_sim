#!/usr/bin/env python3

# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Lock-down test for H1 (per-class contact parameters) + H2 (pad CoACD
decomposition) after they are rewired off the G2-specific ``"link4"``
substring / ``_PAD_LEAVES`` literal onto the assemble-time annotation
reader (Task 3 of the engine-hardcoding H1-H4 fix; see
``docs/superpowers/plans/2026-07-21-engine-hardcoding-H1-H4-fix.md``).

Physical truth (the plan's "reconnaissance correction"): the pad set is
EXACTLY the 4 real
fingertip pads ``gripper_[lr]_(inner|outer)_link4`` — via the USD
annotation (re-assembled G2) AND via the reader's gripper-classified
``link4`` fallback (legacy un-annotated G2 USD).  It MUST NOT include
``arm_l/r_link4`` or ``body_link4`` (the old ``"link4" in label``
substring latent bug).

H1 is tested against the contact-parameter API this branch actually ships.
The original H1 cases drove ``apply_pre_finalize(pad_body_indices=...,
physics_params=...)`` — a global ``solver.grasp_contact_fidelity`` boolean plus
a per-BODY pad index set — and were skipped wholesale when that signature was
removed.  Skipping them hid a live break: ``setup/model.py`` was still passing
both kwargs, so the newton backend could not be constructed at all.  The
replacement pins the one design that is left: every override comes from the
active scene's ``newton.mjc_contact:`` yaml block, resolved per CLASS.

Pads-only ``condim=6`` survived that move, but as a REFINEMENT rather than a
kwarg: a ``newton.mjc_contact.gripper_pad:`` block inherits everything ``robot``
resolved to and overrides only the keys it names, and it applies to the robot
shapes whose body carries the ``newton:isGripperPad`` annotation.  So the yaml
supplies the parameters and the asset supplies the membership — the split the
old ``pad_body_indices`` kwarg collapsed into one argument.  Both halves are
pinned below, including what happens when the block is present and the asset
has no pads (warn by name; do not silently pretend it applied).

Without a ``gripper_pad:`` block the pads stay in ``robot``, and
``robot.condim: 6`` still reaches the whole robot at roughly 27% of real-time.
That is asserted too, because it is the trap the refinement exists to let a
scene author avoid.  H2 is unaffected: pad identity is still read from the
annotation, for the CoACD decomposition.

The modules under test genuinely import ``newton`` / ``warp`` (GPU) at
call time; those are unavailable on a plain CI host.  We therefore:

  * load the reader (``common/gripper_annotations.py``) and the two sites
    (``engine/newton/mjc_contact.py``, ``engine/newton/setup/model.py``)
    BY PATH, exactly like Task 2's ``test_gripper_reader.py``;
  * install a minimal fake ``newton`` module so ``apply_pre_finalize``'s
    runtime ``import newton`` resolves (H1);
  * install a minimal fake ``warp`` module so ``setup/model.py`` imports
    at module load (its module-level ``import warp as wp``).  H2's pad
    SELECTION is a pure helper (``_gripper_pad_bodies``) that needs
    neither newton nor warp, so it runs host-side directly.

The annotation-present paths synthesize a tiny in-memory USD stage and
are guarded by ``importorskip("pxr")``.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import types

import pytest

_HERE = pathlib.Path(__file__).resolve()
# .../genie_sim_engine_newton/genie_sim_engine_newton/tools/<this>
_INNER_PKG = _HERE.parents[1]  # the inner package dir
_SRC_ROOT = _HERE.parents[2]  # dir that CONTAINS the package

# The two sites' runtime imports use absolute ``from
# genie_sim_engine_newton.common...`` — make the package importable.
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))


# ----------------------------------------------------------------------
# by-path module loaders
# ----------------------------------------------------------------------
def _load_by_path(mod_name: str, rel_path: str):
    mod_path = _INNER_PKG / rel_path
    assert mod_path.is_file(), f"{rel_path} not found at {mod_path}"
    spec = importlib.util.spec_from_file_location(mod_name, mod_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def _install_fake_newton():
    """Minimal ``newton`` stand-in for ``apply_pre_finalize``'s runtime
    ``import newton`` (GeoType.PLANE + Model.AttributeFrequency.SHAPE)."""
    if "newton" in sys.modules and getattr(sys.modules["newton"], "_FAKE", False):
        return
    m = types.ModuleType("newton")
    m._FAKE = True

    class GeoType:
        PLANE = 0
        MESH = 5
        CONVEX_MESH = 6

    class _Freq:
        SHAPE = "SHAPE"

    class Model:
        AttributeFrequency = _Freq

    m.GeoType = GeoType
    m.Model = Model
    sys.modules["newton"] = m


def _install_fake_warp():
    """Minimal ``warp`` stand-in so ``setup/model.py``'s module-level
    ``import warp as wp`` succeeds on a host without warp."""
    if "warp" in sys.modules:
        return
    sys.modules["warp"] = types.ModuleType("warp")


_GA = _load_by_path("gripper_annotations_h1h2", "common/gripper_annotations.py")


# ----------------------------------------------------------------------
# G2 physical-truth fixture (mirrors test_gripper_reader.py)
# ----------------------------------------------------------------------
_ROBOT_PREFIX = "robot"

# body_label list — one entry per body, prefixed ``/robot/`` so
# ``classify_shape`` buckets every one as OK_ROBOT.
_G2_BODIES = [
    "/robot/gripper_l_inner_link4",  # 0  real pad
    "/robot/gripper_l_outer_link4",  # 1  real pad
    "/robot/gripper_r_inner_link4",  # 2  real pad
    "/robot/gripper_r_outer_link4",  # 3  real pad
    "/robot/arm_l_link4",  # 4  OLD-BUG false hit — NOT a pad
    "/robot/arm_r_link4",  # 5  OLD-BUG false hit — NOT a pad
    "/robot/body_link4",  # 6  OLD-BUG false hit — NOT a pad
    "/robot/arm_l_link3",  # 7  robot, not link4
]
_G2_PAD_LEAVES = {
    "gripper_l_inner_link4",
    "gripper_l_outer_link4",
    "gripper_r_inner_link4",
    "gripper_r_outer_link4",
}
_G2_PAD_BODY_IDX = {0, 1, 2, 3}
_G2_ANTI_BODY_IDX = {4, 5, 6}  # arm/body link4 — must never be pads


def _build_annotated_stage():
    """In-memory USD stage with the 4 pads annotated (models Task 1)."""
    from pxr import Sdf, Usd, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    for pad in sorted(_G2_PAD_LEAVES):
        prim = stage.DefinePrim(f"/World/robot/{pad}", "Xform")
        UsdPhysics.RigidBodyAPI.Apply(prim)
        a = prim.CreateAttribute(_GA.ANNOT_GRIPPER_PAD, Sdf.ValueTypeNames.Bool)
        a.Set(True)
    # arm/body link4 present but UN-annotated — must be ignored.
    for bad in ("arm_l_link4", "arm_r_link4", "body_link4"):
        stage.DefinePrim(f"/World/robot/{bad}", "Xform")
    return stage


def _pads_from_annotation():
    stage = _build_annotated_stage()
    return _GA.read_pad_body_names(stage, labels=_G2_BODIES)


def _pads_from_fallback():
    # No stage -> reader's gripper-classified link4 fallback.
    return _GA.read_pad_body_names(stage=None, labels=_G2_BODIES)


# ======================================================================
# reader sanity — both paths land on exactly the 4 pads
# ======================================================================
def test_fallback_pads_are_the_four_real_pads():
    assert _pads_from_fallback() == _G2_PAD_LEAVES


def test_annotation_pads_are_the_four_real_pads():
    pytest.importorskip("pxr")
    assert _pads_from_annotation() == _G2_PAD_LEAVES


# ======================================================================
# H1 — per-class contact params from the scene yaml
# ======================================================================
class _FakeLogger:
    """Records rather than discards: a refused override is only honest if the
    scene author is told, so the warning text is part of the contract."""

    def __init__(self):
        self.infos: list[str] = []
        self.warns: list[str] = []

    def info(self, msg="", *a, **k):
        self.infos.append(str(msg))

    def warn(self, msg="", *a, **k):
        self.warns.append(str(msg))

    warning = warn


class _FakeBuilder:
    """Just enough of a pre-finalize ModelBuilder for ``apply_pre_finalize``."""

    def __init__(self, body_labels, shape_labels, shape_bodies, shape_types):
        self.body_label = list(body_labels)
        self.shape_label = list(shape_labels)
        self.shape_body = list(shape_bodies)
        self.shape_type = list(shape_types)
        self.shape_count = len(shape_labels)
        n = self.shape_count
        self.shape_material_ke = [0.0] * n
        self.shape_material_kd = [0.0] * n
        self.shape_material_mu = [0.0] * n
        self.shape_material_mu_torsional = [0.0] * n
        self.shape_material_mu_rolling = [0.0] * n
        # presence of these keys enables the condim + solimp write paths
        self.custom_attributes = {"mujoco:condim": {}, "mujoco:geom_solimp": {}}
        self.condim_by_shape: dict[int, int] = {}

    def _process_custom_attributes(self, i, attrs, freq):
        if "mujoco:condim" in attrs:
            self.condim_by_shape[i] = attrs["mujoco:condim"]


_PASSIVE_BODY = "/World/Objects/cup_000/body"

# Shape indices produced by ``_mixed_builder`` below.
_ROBOT_SHAPES = set(range(len(_G2_BODIES)))  # 0..7, every /robot/ body
_PASSIVE_SHAPE = len(_G2_BODIES)  # 8, an OK_PASSIVE scene rigid
_FLOOR_SHAPE = len(_G2_BODIES) + 1  # 9, the OK_FLOOR ground plane


def _mixed_builder():
    """The G2 robot bodies, plus one passive scene body and the ground plane.

    Three of the five ``OK_*`` classes in one builder, which is the minimum
    needed to show that a per-class override lands on ONE class and leaves the
    others alone -- a robot-only builder cannot show that.
    """
    _install_fake_newton()
    newton = sys.modules["newton"]
    bodies = list(_G2_BODIES) + [_PASSIVE_BODY]
    shape_labels = [b + "/collisions/mesh" for b in bodies]
    shape_bodies = list(range(len(bodies)))
    shape_types = [int(newton.GeoType.MESH)] * len(bodies)
    # Ground plane: PLANE geometry bound to body -1 — exactly what
    # ``builder.add_ground_plane()`` emits, and what ``classify_shape`` keys
    # OK_FLOOR off.
    shape_labels.append("ground_plane")
    shape_bodies.append(-1)
    shape_types.append(int(newton.GeoType.PLANE))
    return _FakeBuilder(bodies, shape_labels, shape_bodies, shape_types)


def _scene(**per_class):
    """A scene cfg carrying nothing but a ``newton.mjc_contact:`` block.

    This is the ONLY input ``apply_pre_finalize`` reads overrides from — the
    point of the whole section, so the tests spell the nesting out rather than
    hiding it behind a fixture.
    """
    return {"newton": {"mjc_contact": dict(per_class)}}


_UNSET = object()


def _run_contact(scene_cfg=None, *, builder=None, logger=None, pad_body_names=_UNSET):
    _install_fake_newton()
    mjc = _load_by_path("mjc_contact_h1", "engine/newton/mjc_contact.py")
    builder = _mixed_builder() if builder is None else builder
    kw = {} if pad_body_names is _UNSET else {"pad_body_names": pad_body_names}
    mjc.apply_pre_finalize(
        builder,
        scene_cfg=scene_cfg or {},
        robot_prefix=_ROBOT_PREFIX,
        adapter_name="mujoco-warp",
        physics_hz=100.0,
        sim_substeps=10,
        logger=logger or _FakeLogger(),
        **kw,
    )
    return builder


def test_condim_is_not_written_at_all_without_a_scene_override():
    """Shipped default: no ``condim:`` in the yaml means no condim is written,
    so MuJoCo keeps its stock 3 and the torsional / rolling friction slots stay
    unread.

    This is what makes "not configured" identical to an engine without this
    module rather than merely close to it, and it is the state every scene that
    has not opted in runs in.
    """
    builder = _run_contact()
    assert not builder.condim_by_shape, f"condim was written with no scene override: {builder.condim_by_shape}"


def test_a_per_class_condim_lands_on_that_class_and_no_other():
    builder = _run_contact(_scene(robot={"condim": 6}))
    got = dict(builder.condim_by_shape)
    assert set(got) == _ROBOT_SHAPES, (
        f"condim landed on shapes {sorted(got)}; expected the robot's " f"{sorted(_ROBOT_SHAPES)} and nothing else"
    )
    assert set(got.values()) == {6}
    assert _PASSIVE_SHAPE not in got and _FLOOR_SHAPE not in got


def test_robot_condim_covers_the_whole_robot_when_no_pad_block_is_named():
    """``robot:`` alone is a whole-robot write, and that is the expensive path.

    Six across the robot class costs roughly 27% of real-time (87 Hz against
    120) for a benefit that only lands on the fingertips — which is exactly
    why ``gripper_pad:`` exists.  A scene author who writes ``robot.condim: 6``
    and nothing else should get that behaviour and that cost, so it is
    asserted rather than left as a surprise.
    """
    log = _FakeLogger()
    builder = _run_contact(_scene(robot={"condim": 6}), logger=log)
    got = dict(builder.condim_by_shape)
    for i in _G2_PAD_BODY_IDX:
        assert got.get(i) == 6, "the pads are covered"
    for i in _G2_ANTI_BODY_IDX:
        assert got.get(i) == 6, (
            "arm/body link4 are robot shapes too — with no gripper_pad block " "there is no per-body selector at all"
        )
    assert not any(
        "gripper_pad" in line for line in log.infos
    ), "the pad path ran without being asked for; it must be opt-in"


def test_two_classes_can_carry_different_condim():
    builder = _run_contact(_scene(robot={"condim": 6}, floor={"condim": 1}))
    got = dict(builder.condim_by_shape)
    assert got[_FLOOR_SHAPE] == 1
    assert all(got[i] == 6 for i in _ROBOT_SHAPES)
    assert _PASSIVE_SHAPE not in got, "passive was never named; it stays stock"


@pytest.mark.parametrize("bad", [5, 0, 2, 7, -3, 6.5, "six", None, [6]])
def test_a_condim_outside_1_3_4_6_is_refused_and_warned(bad):
    """A typo must not silently keep the default.

    MuJoCo only defines condim 1/3/4/6, so anything else is a scene-authoring
    mistake.  The resolver refuses it, warns naming the class and the key, and
    writes NO condim — which leaves the shape on MuJoCo's stock 3 rather than
    on a number the solver would reject at compile time.
    """
    log = _FakeLogger()
    builder = _run_contact(_scene(robot={"condim": bad}), logger=log)
    assert not builder.condim_by_shape, f"a bad condim {bad!r} was written"
    assert any("condim" in w and "robot" in w for w in log.warns), log.warns


def test_a_yaml_quoted_condim_is_still_a_condim():
    """``condim: "6"`` — yaml quoting is an authoring accident, not a typo, and
    the value is unambiguous, so it is coerced rather than refused.  ``6.5`` in
    the case above is the opposite: ``int()`` would turn it into a 6 nobody
    wrote."""
    builder = _run_contact(_scene(robot={"condim": "6"}))
    assert set(builder.condim_by_shape.values()) == {6}


def test_a_valid_condim_next_to_a_bad_one_still_applies():
    """The refusal is per class and per key, not a bail-out for the block."""
    log = _FakeLogger()
    builder = _run_contact(_scene(robot={"condim": 9}, floor={"condim": 4}), logger=log)
    assert builder.condim_by_shape == {_FLOOR_SHAPE: 4}
    assert any("robot" in w for w in log.warns)


def test_a_class_override_that_is_not_a_mapping_is_warned_not_crashed():
    """``newton.mjc_contact.robot: 6`` — the shape a hurried edit produces."""
    log = _FakeLogger()
    builder = _run_contact(_scene(robot=6), logger=log)
    assert not builder.condim_by_shape
    assert any("expected mapping" in w for w in log.warns), log.warns


def test_friction_comes_from_the_same_yaml_block_per_class():
    """``condim`` is not a special case — every knob has one source.

    Friction is the one that shows it best: three independent slots, and the
    old API used to swap the whole table from a boolean.  Now the class names
    itself in the yaml or it keeps the default.
    """
    from genie_sim_engine_newton.common.object_classification import MJC_CONTACT_DEFAULTS, OK_PASSIVE

    builder = _run_contact(_scene(robot={"friction": [1.5, 0.02, 0.001]}))
    for i in _ROBOT_SHAPES:
        assert builder.shape_material_mu[i] == 1.5
        assert builder.shape_material_mu_torsional[i] == 0.02
        assert builder.shape_material_mu_rolling[i] == 0.001
    stock = MJC_CONTACT_DEFAULTS[OK_PASSIVE]["friction"]
    assert builder.shape_material_mu[_PASSIVE_SHAPE] == stock[0]


# ----------------------------------------------------------------------
# gripper_pad — the refinement, and where its two halves come from
# ----------------------------------------------------------------------
#
# ``robot`` is resolved from the yaml; ``gripper_pad`` is resolved from
# ``robot`` plus its own block; MEMBERSHIP comes from the asset. Every test
# below turns on keeping those three separable.

_PAD_SHAPES = set(_G2_PAD_BODY_IDX)  # 0..3, one shape each
_NON_PAD_ROBOT_SHAPES = _ROBOT_SHAPES - _PAD_SHAPES  # 4..7


def test_gripper_pad_condim_lands_on_the_pads_and_the_arm_keeps_robots():
    """The whole point: six where it buys something, three where it does not."""
    builder = _run_contact(_scene(robot={"condim": 3}, gripper_pad={"condim": 6}))
    got = dict(builder.condim_by_shape)
    assert {i: got.get(i) for i in _PAD_SHAPES} == dict.fromkeys(_PAD_SHAPES, 6)
    assert {i: got.get(i) for i in _NON_PAD_ROBOT_SHAPES} == dict.fromkeys(_NON_PAD_ROBOT_SHAPES, 3), (
        "arm_l/r_link4 and body_link4 are robot, not pads — the old "
        '``"link4" in label`` substring bug is what put them in the pad set'
    )
    assert got.get(_PASSIVE_SHAPE) is None and got.get(_FLOOR_SHAPE) is None


def test_gripper_pad_inherits_every_robot_value_it_does_not_name():
    """Refinement, not a sixth class.

    A block naming only ``condim`` must not drop the pads back to the stock
    friction — restating the whole robot tuple under ``gripper_pad:`` just to
    add one key is exactly the duplication that goes stale.
    """
    builder = _run_contact(_scene(robot={"friction": [1.5, 0.5, 0.2], "condim": 3}, gripper_pad={"condim": 6}))
    for i in _PAD_SHAPES:
        assert builder.shape_material_mu[i] == 1.5
        assert builder.shape_material_mu_torsional[i] == 0.5
        assert builder.shape_material_mu_rolling[i] == 0.2
        assert builder.condim_by_shape[i] == 6


def test_gripper_pad_can_override_an_inherited_key():
    """Inheritance is per key, so the pads can disagree with the arm."""
    builder = _run_contact(_scene(robot={"friction": [1.5, 0.5, 0.2]}, gripper_pad={"friction": [2.0, 0.9, 0.4]}))
    for i in _PAD_SHAPES:
        assert builder.shape_material_mu[i] == 2.0
        assert builder.shape_material_mu_rolling[i] == 0.4
    for i in _NON_PAD_ROBOT_SHAPES:
        assert builder.shape_material_mu[i] == 1.5
        assert builder.shape_material_mu_rolling[i] == 0.2


def test_gripper_pad_alone_still_inherits_the_robot_defaults():
    """No ``robot:`` block at all — the pads inherit the DEFAULTS robot got."""
    from genie_sim_engine_newton.common.object_classification import MJC_CONTACT_DEFAULTS, OK_ROBOT

    builder = _run_contact(_scene(gripper_pad={"condim": 6}))
    stock = MJC_CONTACT_DEFAULTS[OK_ROBOT]["friction"]
    for i in _PAD_SHAPES:
        assert builder.shape_material_mu[i] == stock[0]
        assert builder.condim_by_shape[i] == 6
    assert not (set(builder.condim_by_shape) - _PAD_SHAPES)


def test_pad_membership_comes_from_the_asset_not_from_the_yaml():
    """The split the old ``pad_body_indices`` kwarg collapsed.

    Feeding a deliberately WRONG name set moves the refinement onto those
    bodies and off the real pads.  That is the assertion: the yaml never names
    a body, so nothing in the yaml could have produced this — the annotation
    reader did, and swapping what it returns swaps what gets refined.
    """
    builder = _run_contact(
        _scene(robot={"condim": 3}, gripper_pad={"condim": 6}),
        pad_body_names={"arm_l_link4"},
    )
    got = dict(builder.condim_by_shape)
    assert got[4] == 6, "the name set decided membership"
    assert all(got[i] == 3 for i in _PAD_SHAPES), "the real pads were not named"


def test_the_annotation_name_set_is_what_the_engine_passes():
    """End to end on the reader both call sites share.

    ``setup/model.py`` reads the pad names once, hands the same set to CoACD
    and to this module, and this module maps names to body indices itself
    (bodies are still being added when the reader runs).
    """
    pytest.importorskip("pxr")
    builder = _run_contact(
        _scene(robot={"condim": 3}, gripper_pad={"condim": 6}),
        pad_body_names=_pads_from_annotation(),
    )
    got = dict(builder.condim_by_shape)
    assert {i: got[i] for i in _PAD_SHAPES} == dict.fromkeys(_PAD_SHAPES, 6)
    assert all(got[i] == 3 for i in _NON_PAD_ROBOT_SHAPES)


def test_a_gripper_pad_block_with_no_identifiable_pad_warns_and_falls_back():
    """The migration trap, met as a log line instead of as a bad training run.

    A robot USD predating the annotation whose links do not match the
    ``link4`` fallback either: the refinement cannot apply.  Falling back to
    ``robot`` silently would give a grasp that behaves like condim 3 with
    nothing in the log to explain it.
    """
    _install_fake_newton()
    newton = sys.modules["newton"]
    bodies = ["/robot/finger_tip_a", "/robot/finger_tip_b"]
    builder = _FakeBuilder(
        bodies,
        [b + "/collisions/mesh" for b in bodies],
        list(range(len(bodies))),
        [int(newton.GeoType.MESH)] * len(bodies),
    )
    log = _FakeLogger()
    _run_contact(
        _scene(robot={"condim": 3}, gripper_pad={"condim": 6}),
        builder=builder,
        logger=log,
        pad_body_names=None,
    )
    assert set(builder.condim_by_shape.values()) == {3}, "everything stayed on robot"
    assert any("gripper_pad" in w and "isGripperPad" in w for w in log.warns), log.warns


def test_the_summary_names_the_pads_as_their_own_line():
    """An operator reading bring-up must see WHICH shapes got the six."""
    log = _FakeLogger()
    _run_contact(_scene(robot={"condim": 3}, gripper_pad={"condim": 6}), logger=log)
    summary = "\n".join(log.infos)
    assert "gripper_pad" in summary
    assert "condim=6" in summary and "condim=3" in summary


def test_an_unknown_class_under_mjc_contact_is_warned():
    """``gripper_pads:`` / ``static-prop:`` — a migration typo is otherwise
    indistinguishable from "I chose not to override that class"."""
    log = _FakeLogger()
    builder = _run_contact({"newton": {"mjc_contact": {"gripper_pads": {"condim": 6}}}}, logger=log)
    assert not builder.condim_by_shape
    assert any("gripper_pads" in w and "unknown class" in w for w in log.warns), log.warns


def test_an_unknown_key_inside_a_class_is_warned():
    log = _FakeLogger()
    builder = _run_contact(_scene(robot={"frictoin": [1.0, 1.0, 1.0]}), logger=log)
    assert not builder.condim_by_shape
    assert any("frictoin" in w and "unknown key" in w for w in log.warns), log.warns


def test_mjc_contact_itself_being_a_non_mapping_is_warned_not_crashed():
    log = _FakeLogger()
    _run_contact({"newton": {"mjc_contact": [{"robot": {"condim": 6}}]}}, logger=log)
    assert any("newton.mjc_contact" in w and "expected mapping" in w for w in log.warns), log.warns


def test_the_removed_kwargs_are_not_quietly_accepted_again():
    """The two arguments whose reappearance broke the build.

    ``apply_pre_finalize`` has no ``**kwargs``, so passing either is a
    ``TypeError`` — which is the good outcome, and is what this asserts.  The
    bad outcome is someone adding ``**kwargs`` "for compatibility": the call
    would then bind and the override would be silently ignored.
    """
    _install_fake_newton()
    mjc = _load_by_path("mjc_contact_h1", "engine/newton/mjc_contact.py")
    for dead in ({"pad_body_indices": {0, 1}}, {"physics_params": object()}):
        with pytest.raises(TypeError):
            mjc.apply_pre_finalize(
                _mixed_builder(),
                scene_cfg={},
                robot_prefix=_ROBOT_PREFIX,
                adapter_name="mujoco-warp",
                physics_hz=100.0,
                sim_substeps=10,
                logger=_FakeLogger(),
                **dead,
            )


def test_the_model_py_call_site_binds_against_this_signature():
    """The regression that motivated the rewrite above, pinned at the seam.

    ``setup/model.py`` called ``apply_pre_finalize`` with ``pad_body_indices=``
    and ``physics_params=`` for as long as this signature had them.  When the
    signature narrowed, the caller did not — and because that call is
    unconditional inside ``_phase_build_model``, which ``__init__`` reaches,
    the newton backend could not be CONSTRUCTED.  Nothing caught it: the two
    files import each other lazily, the H1 tests that exercised the callee were
    skipped, and every test of the caller stops short of a GPU build.

    Read both out of the source and bind them, so the next narrowing fails here
    instead of at a customer's bring-up.
    """
    import ast
    import inspect

    def _src(rel):
        return (_INNER_PKG / rel).read_text(encoding="utf-8")

    call = None
    for node in ast.walk(ast.parse(_src("engine/newton/setup/model.py"))):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "_mjc_apply":
            assert call is None, "more than one call site; this test reads only one"
            call = node
    assert call is not None, "the call site moved or was renamed; update this test"
    assert all(k.arg for k in call.keywords), "**-splat at the call site defeats this check"

    fn = next(
        n
        for n in ast.walk(ast.parse(_src("engine/newton/mjc_contact.py")))
        if isinstance(n, ast.FunctionDef) and n.name == "apply_pre_finalize"
    )
    params = [inspect.Parameter(a.arg, inspect.Parameter.POSITIONAL_OR_KEYWORD) for a in fn.args.args] + [
        inspect.Parameter(a.arg, inspect.Parameter.KEYWORD_ONLY) for a in fn.args.kwonlyargs
    ]
    # Positional args at the call site are the builder; keywords are what drift.
    inspect.Signature(params).bind(*[None] * len(call.args), **{k.arg: None for k in call.keywords})


# ======================================================================
# H2 — CoACD selection: exactly the 4 pad bodies (both paths)
# ======================================================================
def _load_model_module():
    _install_fake_warp()
    return _load_by_path("setup_model_h2", "engine/newton/setup/model.py")


def _selected_body_idx(pairs) -> set[int]:
    return {body for body, _leaf in pairs}


def test_h2_selection_fallback_path_is_four_pads():
    model = _load_model_module()
    pads = _pads_from_fallback()
    pairs = model._gripper_pad_bodies(_G2_BODIES, pads)
    assert _selected_body_idx(pairs) == _G2_PAD_BODY_IDX


def test_h2_selection_annotation_path_is_four_pads():
    pytest.importorskip("pxr")
    model = _load_model_module()
    pads = _pads_from_annotation()
    pairs = model._gripper_pad_bodies(_G2_BODIES, pads)
    assert _selected_body_idx(pairs) == _G2_PAD_BODY_IDX


def test_h2_selection_default_none_uses_reader_fallback():
    """``pad_body_names=None`` -> the helper resolves the reader's
    gripper-classified link4 fallback (== the 4 pads, == old _PAD_LEAVES)."""
    model = _load_model_module()
    pairs = model._gripper_pad_bodies(_G2_BODIES, None)
    assert _selected_body_idx(pairs) == _G2_PAD_BODY_IDX


def test_h2_selection_excludes_arm_and_body_link4():
    model = _load_model_module()
    for pads in (_pads_from_fallback(),):
        pairs = model._gripper_pad_bodies(_G2_BODIES, pads)
        sel = _selected_body_idx(pairs)
        assert not (sel & _G2_ANTI_BODY_IDX)


def test_h2_pad_leaves_literal_removed():
    """The G2-specific ``_PAD_LEAVES`` tuple must be gone (rewired onto the
    annotation reader)."""
    model = _load_model_module()
    assert not hasattr(model, "_PAD_LEAVES")


# ======================================================================
# The migration contract: the shipped profile == grasp_contact_fidelity: true
# ======================================================================
#
# ``solver.grasp_contact_fidelity: true`` was ONE boolean in
# ``physics_params.yaml`` that moved condim AND the friction table together.
# Rolling has no such switch, so migrating off it means writing the state it
# produced as per-class yaml -- and "the numbers came out the same" is the
# whole promise being made to whoever migrates.  Asserting it against the
# profile file that actually ships is the only form of that promise that
# cannot quietly rot: prose in a migration note is not executable, and a
# hand-copied dict in this test would only pin the copy.
#
# The old ON state, from the removed branch:
#
#     condim    6 on (OK_ROBOT and pad), 3 on OK_ROBOT non-pad, 4 elsewhere
#     friction  (1.5, 0.5, 0.2) on the robot, (1.0, 1.0, 0.0001) elsewhere

_SHIPPED_LAUNCHER = _HERE.parents[3] / "genie_sim_bringup" / "config" / "launcher_newton_mjwarp_multi_env.yaml"


def _shipped_profile():
    """``scene_yaml_defaults:`` out of the launcher, as a scene-cfg fragment.

    ``EngineSession`` deep-merges this block beneath the scene yaml, so a
    scene that overrides nothing sees exactly this -- which is what makes it
    usable directly as ``scene_cfg`` here.
    """
    if not _SHIPPED_LAUNCHER.is_file():
        pytest.skip(f"launcher yaml not at {_SHIPPED_LAUNCHER}")
    import yaml

    doc = yaml.safe_load(_SHIPPED_LAUNCHER.read_text()) or {}
    for node, body in doc.items():
        if node == "launcher" or not isinstance(body, dict):
            continue
        block = (body.get("ros__parameters") or {}).get("scene_yaml_defaults")
        if block:
            return block
    pytest.fail(f"no ros__parameters.scene_yaml_defaults in {_SHIPPED_LAUNCHER}")


@pytest.fixture
def profiled():
    """The G2 builder driven by the shipped profile, pads named by the asset."""
    return _run_contact(_shipped_profile(), pad_body_names=_pads_from_fallback())


def test_the_shipped_profile_reproduces_the_old_condim_split(profiled):
    for i in sorted(_PAD_SHAPES):
        assert profiled.condim_by_shape.get(i) == 6, f"pad shape {i} lost its condim 6"
    for i in sorted(_NON_PAD_ROBOT_SHAPES):
        assert profiled.condim_by_shape.get(i) == 3, f"robot shape {i} is not on condim 3"
    for i in (_PASSIVE_SHAPE, _FLOOR_SHAPE):
        assert profiled.condim_by_shape.get(i) == 4, f"shape {i} is not on condim 4"


def test_the_shipped_profile_reproduces_the_old_friction_table(profiled):
    def friction(i):
        return (
            profiled.shape_material_mu[i],
            profiled.shape_material_mu_torsional[i],
            profiled.shape_material_mu_rolling[i],
        )

    for i in sorted(_ROBOT_SHAPES):
        # Pads included: ``gripper_pad`` names only ``condim``, so friction is
        # inherited from ``robot`` -- the refinement must not fork the table.
        assert friction(i) == (1.5, 0.5, 0.2), f"robot shape {i}: {friction(i)}"
    for i in (_PASSIVE_SHAPE, _FLOOR_SHAPE):
        assert friction(i) == (1.0, 1.0, 0.0001), f"shape {i}: {friction(i)}"


def test_the_profile_is_a_no_op_on_a_scene_that_overrides_it():
    """A per-fleet default has to stay overridable per scene, or it is a fork.

    ``EngineSession`` merges the launcher block UNDER the scene, so this is
    what a scene pinning its own ``robot.condim`` gets.  Simulated here with
    the merged result, because ``apply_pre_finalize`` is downstream of the
    merge and never sees the two halves.
    """
    profile = _shipped_profile()
    merged = {"newton": {"mjc_contact": dict(profile["newton"]["mjc_contact"])}}
    merged["newton"]["mjc_contact"]["robot"] = {"friction": [1.5, 0.5, 0.2], "condim": 4}
    b = _run_contact(merged, pad_body_names=_pads_from_fallback())
    for i in sorted(_NON_PAD_ROBOT_SHAPES):
        assert b.condim_by_shape.get(i) == 4, "the scene could not disagree with the launcher"
    for i in sorted(_PAD_SHAPES):
        assert b.condim_by_shape.get(i) == 6, "the pad refinement was collateral damage"


def test_the_profile_names_no_class_the_engine_does_not_know():
    """A typo in the shipped file is a warn nobody reads at 3am on a cluster."""
    from genie_sim_engine_newton.common.object_classification import ALL_OBJECT_KINDS

    log = _FakeLogger()
    _run_contact(_shipped_profile(), logger=log, pad_body_names=_pads_from_fallback())
    assert not log.warns, f"the shipped profile warns at bring-up: {log.warns}"
    known = set(ALL_OBJECT_KINDS) | {"gripper_pad"}
    assert set(_shipped_profile()["newton"]["mjc_contact"]) <= known

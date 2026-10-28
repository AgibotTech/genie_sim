#!/usr/bin/env python3

# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Lock-down test for the ENGINE-SIDE gripper annotation reader + H4.

Task 2 of the engine-hardcoding H1-H4 fix (see
``docs/superpowers/plans/2026-07-21-engine-hardcoding-H1-H4-fix.md`` and
``docs/PORT_ENGINE_HARDCODING_AUDIT.md``).  Task 1 authors two USD
annotations at assemble time (``newton:isGripperPad`` on pad bodies,
``newton:isLoopCoupler`` on coupler joints); Task 2 builds the engine's
reader for them — annotation-first with a legacy literal/regex fallback —
and rewires H4 (the gripper master force-cap) to identify the master via
``gripper_masters(mimic_followers)`` instead of the old
``gripper_[lr]_outer_joint1$`` regex.

Physical truth (the plan's "reconnaissance correction"):

  * master set == the URDF ``<mimic>`` referenceJoint set — the two
    ``inner_joint1`` joints — NOT ``outer_joint1``.  This is exactly
    ``set(mimic_followers.keys())``.
  * pad set == the 4 real fingertip pads
    (``gripper_[lr]_(inner|outer)_link4``); the fallback must NOT hit
    ``arm_l/r_link4`` / ``body_link4`` (the latent substring bug).
  * coupler set == the URDF ``<mimic>``-follower gripper joints whose
    parent link is NOT the gripper base link (12 on G2; the two
    base-parented ``outer_joint1`` followers are driven, not couplers).

The pure set/selection logic needs no ``pxr``.  The two stage-driven
tests synthesize a minimal in-memory USD stage and are guarded by
``importorskip("pxr")``.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

_HERE = pathlib.Path(__file__).resolve()


def _load_reader():
    """Import ``common/gripper_annotations.py`` by path.

    Loading the file directly (via ``importlib.util``) avoids importing
    the ``genie_sim_engine_newton`` package, whose ``engine.newton.*``
    stages pull ``newton`` / ``warp`` (GPU) — unavailable on a plain CI
    host.  The reader module itself imports only stdlib, so a by-path
    load is dependency-free.
    """
    # .../genie_sim_engine_newton/genie_sim_engine_newton/tools/<this>
    #                              ^ parents[1] == the inner package dir
    mod_path = _HERE.parents[1] / "common" / "gripper_annotations.py"
    assert mod_path.is_file(), f"gripper_annotations.py not found at {mod_path}"
    spec = importlib.util.spec_from_file_location("gripper_annotations_under_test", mod_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


_GA = _load_reader()


# ----------------------------------------------------------------------
# G2 physical-truth golden sets (mirror test_gripper_annotations.py)
# ----------------------------------------------------------------------
# The two masters = URDF <mimic> referenceJoint set (inner_joint1 x2).
G2_MASTERS = {
    "idx31_gripper_l_inner_joint1",
    "idx71_gripper_r_inner_joint1",
}

# The old H4 regex would have picked these instead — must NOT be masters.
G2_OLD_OUTER_MASTERS = {
    "idx41_gripper_l_outer_joint1",
    "idx81_gripper_r_outer_joint1",
}

# 12 coupler joints (6 per gripper); all mimic inner_joint1.  This is the
# ANNOTATION set == the URDF ``<mimic>``-follower gripper joints whose
# parent link is NOT the gripper base link (what Task 1 authors).  The two
# base-parented followers (``gripper_[lr]_outer_joint1``) are DRIVEN joints,
# not passive couplers, so they are excluded.
G2_COUPLER_JOINTS = {
    "idx32_gripper_l_inner_joint3",
    "idx33_gripper_l_inner_joint4",
    "idx39_gripper_l_inner_joint0",
    "idx42_gripper_l_outer_joint3",
    "idx43_gripper_l_outer_joint4",
    "idx49_gripper_l_outer_joint0",
    "idx72_gripper_r_inner_joint3",
    "idx73_gripper_r_inner_joint4",
    "idx79_gripper_r_inner_joint0",
    "idx82_gripper_r_outer_joint3",
    "idx83_gripper_r_outer_joint4",
    "idx89_gripper_r_outer_joint0",
}

# The LEGACY FALLBACK regex ``gripper_[lr]_(inner|outer)_joint[034]$`` is
# joint-NUMBER-based (0/3/4 only).  Now that the annotation set correctly
# EXCLUDES the two base-parented driven ``outer_joint1`` followers, the
# annotation path and this legacy fallback AGREE at 12 joints — the
# comprehension below simply re-derives the same 12 by number.  The
# fallback still reproduces pre-Task-1 behaviour byte-for-byte on
# un-annotated USD.
G2_COUPLER_JOINTS_LEGACY = {
    j for j in G2_COUPLER_JOINTS if not j.endswith("joint1")
}  # 12 joints — identical to G2_COUPLER_JOINTS (annotation == fallback)

G2_PAD_LINKS = {
    "gripper_l_inner_link4",
    "gripper_l_outer_link4",
    "gripper_r_inner_link4",
    "gripper_r_outer_link4",
}

# Names that MUST NOT be classified as pads by the fallback (old bug).
G2_PAD_ANTI_SET = {"arm_l_link4", "arm_r_link4", "body_link4"}


def _g2_mimic_followers():
    """A G2-like mimic map: two masters (inner_joint1), each with a few
    followers.  The exact follower coefficients don't matter here — only
    that the KEYS are the two masters."""
    return {
        "idx31_gripper_l_inner_joint1": [
            ("idx32_gripper_l_inner_joint3", -1.0, 0.0),
            ("idx41_gripper_l_outer_joint1", 1.0, 0.0),
        ],
        "idx71_gripper_r_inner_joint1": [
            ("idx72_gripper_r_inner_joint3", -1.0, 0.0),
            ("idx81_gripper_r_outer_joint1", 1.0, 0.0),
        ],
    }


# ======================================================================
# gripper_masters — the H4 master set
# ======================================================================
def test_gripper_masters_are_inner_joint1():
    """``gripper_masters`` == the mimic map keys (the two inner_joint1)."""
    masters = _GA.gripper_masters(_g2_mimic_followers())
    assert masters == G2_MASTERS


def test_gripper_masters_excludes_old_outer_joint1():
    """The physical-truth master set must NOT contain the old H4
    ``outer_joint1`` joints (the whole point of the reconnaissance
    correction)."""
    masters = _GA.gripper_masters(_g2_mimic_followers())
    assert not (masters & G2_OLD_OUTER_MASTERS)


def test_gripper_masters_empty_on_no_mimic():
    assert _GA.gripper_masters(None) == set()
    assert _GA.gripper_masters({}) == set()


# ======================================================================
# H4 selection — which actuators get the force cap
# ======================================================================
# The compiled MJCF prefixes every joint name; H4 iterates actuator joint
# names in this namespace.  Reproduce the four gripper joint1 names.
_MJCF_JOINT1_NAMES = [
    "_genie_joints_idx31_gripper_l_inner_joint1",
    "_genie_joints_idx41_gripper_l_outer_joint1",
    "_genie_joints_idx71_gripper_r_inner_joint1",
    "_genie_joints_idx81_gripper_r_outer_joint1",
    "_genie_joints_idx21_arm_l_joint1",  # non-gripper: never capped
]


def test_h4_caps_inner_not_outer_joint1():
    """H4 force-cap selection: with a live mimic map, the capped set is
    the two INNER joint1 (masters), NOT the outer joint1 (old regex)."""
    masters = _GA.gripper_masters(_g2_mimic_followers())
    capped = _GA.master_actuator_names(_MJCF_JOINT1_NAMES, masters)
    assert capped == {
        "_genie_joints_idx31_gripper_l_inner_joint1",
        "_genie_joints_idx71_gripper_r_inner_joint1",
    }
    # explicitly assert the old outer_joint1 actuators are NOT capped
    assert not any("outer_joint1" in c for c in capped)
    # and the arm joint is untouched
    assert not any("arm_" in c for c in capped)


def test_h4_fallback_regex_when_no_mimic_map():
    """No mimic map -> the legacy ``outer_joint1`` regex still caps
    *something* (backwards-compatible degradation)."""
    capped = _GA.master_actuator_names(_MJCF_JOINT1_NAMES, set())
    assert capped == {
        "_genie_joints_idx41_gripper_l_outer_joint1",
        "_genie_joints_idx81_gripper_r_outer_joint1",
    }


def test_h4_adapter_loop_faithful_simulation():
    """Reproduce the mujoco_warp ``_fix_gripper_loop_constraints`` H4 loop
    with a tiny fake mj_model and assert the CAPPED actuator set equals
    the master set (inner_joint1), not the old outer_joint1.

    Mirrors the adapter's exact per-actuator iteration:
    ``joint_is_master(mj_id2name(...), self._mimic_masters)``.
    """
    masters = _GA.gripper_masters(_g2_mimic_followers())  # inner_joint1 x2
    # one actuator per gripper joint1 (as in the compiled MJCF)
    actuator_joint_names = list(_MJCF_JOINT1_NAMES)

    capped = []
    for jn2 in actuator_joint_names:  # == mj_id2name(...) per actuator
        if _GA.joint_is_master(jn2, masters):  # == the adapter's H4 predicate
            capped.append(jn2.split("_genie_joints_")[-1])

    assert set(capped) == {
        "idx31_gripper_l_inner_joint1",
        "idx71_gripper_r_inner_joint1",
    }
    # the two things the plan's reconnaissance correction demands:
    # outer_joint1 NOT capped,
    # exactly the two masters capped.
    assert not any("outer_joint1" in c for c in capped)
    assert len(capped) == len(masters) == 2


# ======================================================================
# pad reader — fallback (no pxr needed)
# ======================================================================
_G2_BODY_LABELS = [
    "/World/robot/gripper_l_inner_link4",
    "/World/robot/gripper_l_outer_link4",
    "/World/robot/gripper_r_inner_link4",
    "/World/robot/gripper_r_outer_link4",
    "/World/robot/arm_l_link4",  # old-bug false hit — must be excluded
    "/World/robot/arm_r_link4",  # old-bug false hit — must be excluded
    "/World/robot/body_link4",  # old-bug false hit — must be excluded
    "/World/robot/gripper_l_base_link",
]


def test_pad_reader_fallback_matches_true_pads_only():
    """No stage -> fallback lands on the 4 real pads, none of the arm/body
    ``link4`` false hits the old substring test picked up."""
    pads = _GA.read_pad_body_names(stage=None, labels=_G2_BODY_LABELS)
    assert pads == G2_PAD_LINKS
    for bad in G2_PAD_ANTI_SET:
        assert bad not in pads


def test_pad_body_indices_maps_names_to_indices():
    pads = _GA.read_pad_body_names(stage=None, labels=_G2_BODY_LABELS)
    idx = _GA.pad_body_indices(_G2_BODY_LABELS, pads)
    assert idx == {0, 1, 2, 3}


# ======================================================================
# coupler reader — fallback (no pxr needed)
# ======================================================================
_G2_JOINT_LABELS = sorted(
    "_genie_joints_" + n
    for n in (
        G2_COUPLER_JOINTS
        | G2_MASTERS  # inner_joint1 masters -> joint1, NOT a coupler
        | {"idx21_arm_l_joint1", "idx01_body_joint1"}
    )
)


def test_coupler_reader_fallback_matches_legacy_regex_set():
    """No stage -> the fallback regex selects exactly the LEGACY loop
    couplers (joints 0/3/4 — 12 joints), NOT the master joint1 nor the
    two ``outer_joint1`` (which the number-based regex misses).  This
    reproduces pre-Task-1 behaviour byte-for-byte on un-annotated USD."""
    names = _GA.read_loop_coupler_joint_names(stage=None, joint_names=_G2_JOINT_LABELS)
    got_short = {n.split("_genie_joints_")[-1] for n in names}
    assert got_short == G2_COUPLER_JOINTS_LEGACY
    # masters (inner joint1) are not couplers
    assert not (got_short & G2_MASTERS)
    # the fallback regex misses outer_joint1 — and so does the corrected
    # annotation set (outer_joint1 is a driven joint, not a coupler), so
    # annotation and fallback now agree at 12.
    assert "idx41_gripper_l_outer_joint1" not in got_short


def test_coupler_joint_labels_maps_names_to_labels():
    names = _GA.read_loop_coupler_joint_names(stage=None, joint_names=_G2_JOINT_LABELS)
    labels = _GA.coupler_joint_labels(_G2_JOINT_LABELS, names)
    # identity on the fallback path (names ARE full labels there)
    assert labels == names


# ======================================================================
# stage-driven readers — annotation-present path (needs pxr)
# ======================================================================
def _build_annotated_stage():
    """Minimal in-memory USD stage carrying the two annotations on the
    physical-truth prims — models what Task 1's authoring produces."""
    from pxr import Sdf, Usd, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    # pad bodies (annotated) + a decoy same-named-ish body without annot
    for pad in sorted(G2_PAD_LINKS):
        prim = stage.DefinePrim(f"/World/robot/{pad}", "Xform")
        UsdPhysics.RigidBodyAPI.Apply(prim)
        a = prim.CreateAttribute(_GA.ANNOT_GRIPPER_PAD, Sdf.ValueTypeNames.Bool)
        a.Set(True)
    # an un-annotated arm link4 — must be ignored on the annotation path
    stage.DefinePrim("/World/robot/arm_l_link4", "Xform")
    # coupler joints (annotated)
    for cj in sorted(G2_COUPLER_JOINTS):
        prim = stage.DefinePrim(f"/World/robot/joints/{cj}", "PhysicsRevoluteJoint")
        a = prim.CreateAttribute(_GA.ANNOT_LOOP_COUPLER, Sdf.ValueTypeNames.Bool)
        a.Set(True)
    # a master joint1 WITHOUT the coupler annotation — must be excluded
    stage.DefinePrim("/World/robot/joints/idx31_gripper_l_inner_joint1", "PhysicsRevoluteJoint")
    return stage


def test_pad_reader_annotation_path():
    pytest.importorskip("pxr")
    stage = _build_annotated_stage()
    # labels deliberately WRONG so a fallback would give a different set —
    # proves the annotation path is what's used.
    pads = _GA.read_pad_body_names(stage, labels=["/nonsense/link4_gripper"])
    assert pads == G2_PAD_LINKS


def test_coupler_reader_annotation_path():
    pytest.importorskip("pxr")
    stage = _build_annotated_stage()
    couplers = _GA.read_loop_coupler_joint_names(stage, joint_names=["_genie_joints_nonsense"])
    assert couplers == G2_COUPLER_JOINTS


def test_readers_degrade_to_fallback_on_unannotated_stage():
    """A stage with NO annotations -> readers fall back to literals/labels
    rather than returning empty (the zero-risk path for pre-Task-1 USD)."""
    pytest.importorskip("pxr")
    from pxr import Usd

    empty = Usd.Stage.CreateInMemory()
    empty.DefinePrim("/World/robot/gripper_l_inner_link4", "Xform")
    pads = _GA.read_pad_body_names(empty, labels=_G2_BODY_LABELS)
    assert pads == G2_PAD_LINKS  # from the fallback labels, not the stage
    couplers = _GA.read_loop_coupler_joint_names(empty, joint_names=_G2_JOINT_LABELS)
    got_short = {n.split("_genie_joints_")[-1] for n in couplers}
    assert got_short == G2_COUPLER_JOINTS_LEGACY  # fallback = legacy 12-set


def test_reader_does_not_hard_fail_without_stage_or_pxr():
    """Missing stage must degrade, never raise (import-safe on any host)."""
    assert _GA.read_pad_body_names(None) == set()
    assert _GA.read_loop_coupler_joint_names(None) == set()
    # a bogus non-stage object (no Traverse) also degrades to fallback
    assert _GA.read_pad_body_names(object(), labels=_G2_BODY_LABELS) == G2_PAD_LINKS

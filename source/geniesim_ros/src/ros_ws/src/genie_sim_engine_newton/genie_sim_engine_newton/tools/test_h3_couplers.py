#!/usr/bin/env python3

# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Lock-down test for H3 — the gripper loop-closure coupler handling.

Task 4 of the engine-hardcoding H1-H4 fix (see
``docs/superpowers/plans/2026-07-21-engine-hardcoding-H1-H4-fix.md`` and
``docs/PORT_ENGINE_HARDCODING_AUDIT.md``).

``mujoco_warp.py::_fix_gripper_loop_constraints`` used to identify the
Robotiq-style four-bar loop joints with three G2-specific name regexes:

  1. coupler ACTUATOR UNLOCK  — ``gripper_[lr]_(inner|outer)_joint[034]$``
     (free the 12 passive couplers from their PD actuators).
  2. passive damping **8.0**  — ``gripper_[lr]_(inner|outer)_joint1$``
     (the 4 DRIVEN joints: the 2 masters + the 2 ``outer_joint1``).
  3. passive damping **1.0**  — ``gripper_[lr]_(inner|outer)_joint[034]$``
     (the same 12 couplers as site 1).

Task 4 rewires all three off the regexes and onto the annotation reader
(``newton:isLoopCoupler`` truth, legacy ``joint[034]`` regex fallback)
plus the mimic structure, via the pure helper
``gripper_annotations.gripper_loop_constraint_labels`` — so the engine is
a thin CONSUMER of the robot's own declared topology.

Physical truth on G2 (the plan's "reconnaissance correction" — annotation
and fallback AGREE
at 12 now that Task 1 excludes the base-parented ``outer_joint1``):

  * COUPLERS (12) = the ``newton:isLoopCoupler`` joints == the URDF
    ``<mimic>``-follower gripper joints whose parent link is NOT the
    gripper base link.  Sites 1 + 3 act on this set.
  * DRIVEN  (4) = ``masters ∪ (all_mimic_followers − couplers)`` == the
    two ``inner_joint1`` masters + the two ``outer_joint1`` followers
    (which are DRIVEN, not couplers).  Site 2 (damping 8.0) acts on this
    set — computed structurally from the mimic map, NOT a ``joint1$``
    regex.

This must EXACTLY reproduce the old regex split (12/4/12) on G2 — the
"no physics change on G2" gate.  The crux of the T1-fix consistency:
``outer_joint1`` is DRIVEN (damping-8), NOT a coupler/unlock joint.

The pure set logic needs no ``pxr``.  The annotation-path test
synthesizes a minimal in-memory USD stage and is guarded by
``importorskip("pxr")``.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

_HERE = pathlib.Path(__file__).resolve()


def _load_reader():
    """Import ``common/gripper_annotations.py`` by path (dependency-free —
    the reader imports only stdlib, so this avoids pulling the
    ``genie_sim_engine_newton`` package's GPU deps on a plain host)."""
    mod_path = _HERE.parents[1] / "common" / "gripper_annotations.py"
    assert mod_path.is_file(), f"gripper_annotations.py not found at {mod_path}"
    spec = importlib.util.spec_from_file_location("gripper_annotations_h3_under_test", mod_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


_GA = _load_reader()


# ----------------------------------------------------------------------
# G2 physical-truth golden sets (mirror test_gripper_reader.py)
# ----------------------------------------------------------------------
# 12 passive in-loop couplers (6 per gripper) — the isLoopCoupler set.
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

# The 2 drive masters (URDF <mimic> referenceJoints) — inner_joint1.
G2_MASTERS = {
    "idx31_gripper_l_inner_joint1",
    "idx71_gripper_r_inner_joint1",
}

# The 2 base-parented followers — DRIVEN, not couplers.  These are the
# joints the T1 fix moved OUT of the coupler set and into the driven set.
G2_OUTER_JOINT1 = {
    "idx41_gripper_l_outer_joint1",
    "idx81_gripper_r_outer_joint1",
}

# DRIVEN (damping-8) = masters ∪ outer_joint1 = 4 joints.
G2_DRIVEN_JOINTS = G2_MASTERS | G2_OUTER_JOINT1

# Non-gripper joints — never touched by any of the three sites.
G2_OTHER_JOINTS = {"idx21_arm_l_joint1", "idx01_body_joint1"}

_ALL_SHORTS = G2_COUPLER_JOINTS | G2_MASTERS | G2_OUTER_JOINT1 | G2_OTHER_JOINTS

# The compiled MJCF prefixes every joint name; the adapter iterates joint
# names in THIS namespace (``mj_id2name``).  The helper must map the
# short-name coupler/mimic sets onto these prefixed labels.
_MJCF = "_genie_joints_"
_MJM_JOINT_NAMES = sorted(_MJCF + s for s in _ALL_SHORTS)

# Expected label sets (what the three sites should see).
_EXPECT_COUPLER_LABELS = {_MJCF + s for s in G2_COUPLER_JOINTS}
_EXPECT_DRIVEN_LABELS = {_MJCF + s for s in G2_DRIVEN_JOINTS}


def _g2_mimic_followers():
    """The FULL G2 mimic map: two masters (inner_joint1), each with all 7
    followers (3 inner couplers + 3 outer couplers + outer_joint1).  Only
    the KEYS (masters) and the follower NAMES matter for set membership;
    the coefficients are illustrative."""
    return {
        "idx31_gripper_l_inner_joint1": [
            ("idx32_gripper_l_inner_joint3", -1.0, 0.0),
            ("idx33_gripper_l_inner_joint4", 1.0, 0.0),
            ("idx39_gripper_l_inner_joint0", 1.0, 0.0),
            ("idx41_gripper_l_outer_joint1", 1.0, 0.0),
            ("idx42_gripper_l_outer_joint3", -1.0, 0.0),
            ("idx43_gripper_l_outer_joint4", 1.0, 0.0),
            ("idx49_gripper_l_outer_joint0", 1.0, 0.0),
        ],
        "idx71_gripper_r_inner_joint1": [
            ("idx72_gripper_r_inner_joint3", -1.0, 0.0),
            ("idx73_gripper_r_inner_joint4", 1.0, 0.0),
            ("idx79_gripper_r_inner_joint0", 1.0, 0.0),
            ("idx81_gripper_r_outer_joint1", 1.0, 0.0),
            ("idx82_gripper_r_outer_joint3", -1.0, 0.0),
            ("idx83_gripper_r_outer_joint4", 1.0, 0.0),
            ("idx89_gripper_r_outer_joint0", 1.0, 0.0),
        ],
    }


def _mimic_name_sets():
    mf = _g2_mimic_followers()
    masters = set(mf.keys())
    followers = {n for lst in mf.values() for (n, _c, _o) in lst}
    return masters, followers


# ======================================================================
# FALLBACK path (no stage / un-annotated USD) — no pxr needed
# ======================================================================
def test_fallback_coupler_and_driven_labels():
    """Stage absent -> couplers come from the legacy ``joint[034]`` regex
    fallback; the driven set comes structurally from the mimic map.  The
    12/4/12 split must reproduce the old regex behaviour byte-for-byte."""
    masters, followers = _mimic_name_sets()
    coupler_names = _GA.read_loop_coupler_joint_names(stage=None, joint_names=list(_ALL_SHORTS))
    coupler_labels, driven_labels = _GA.gripper_loop_constraint_labels(
        _MJM_JOINT_NAMES, coupler_names, followers, masters
    )
    assert coupler_labels == _EXPECT_COUPLER_LABELS
    assert len(coupler_labels) == 12
    assert driven_labels == _EXPECT_DRIVEN_LABELS
    assert len(driven_labels) == 4


def test_outer_joint1_is_driven_not_coupler():
    """The crux of the T1-fix consistency: ``outer_joint1`` is a DRIVEN
    joint (damping-8), NOT a passive coupler (unlock / damping-1)."""
    masters, followers = _mimic_name_sets()
    coupler_names = _GA.read_loop_coupler_joint_names(stage=None, joint_names=list(_ALL_SHORTS))
    coupler_labels, driven_labels = _GA.gripper_loop_constraint_labels(
        _MJM_JOINT_NAMES, coupler_names, followers, masters
    )
    for oj1 in (_MJCF + s for s in G2_OUTER_JOINT1):
        assert oj1 in driven_labels
        assert oj1 not in coupler_labels
    # and the two inner_joint1 masters are driven, never couplers
    for m in (_MJCF + s for s in G2_MASTERS):
        assert m in driven_labels
        assert m not in coupler_labels


def test_non_gripper_joints_untouched():
    """arm / body joints appear in neither the coupler nor the driven set."""
    masters, followers = _mimic_name_sets()
    coupler_names = _GA.read_loop_coupler_joint_names(stage=None, joint_names=list(_ALL_SHORTS))
    coupler_labels, driven_labels = _GA.gripper_loop_constraint_labels(
        _MJM_JOINT_NAMES, coupler_names, followers, masters
    )
    for other in (_MJCF + s for s in G2_OTHER_JOINTS):
        assert other not in coupler_labels
        assert other not in driven_labels


def test_three_site_semantics_reproduce_old_split():
    """Faithful re-simulation of the three H3 sites over the mjm joint
    list, asserting the exact old-regex split: unlock == 12 couplers,
    damping-8 == 4 driven, damping-1 == 12 couplers."""
    masters, followers = _mimic_name_sets()
    coupler_names = _GA.read_loop_coupler_joint_names(stage=None, joint_names=list(_ALL_SHORTS))
    coupler_labels, driven_labels = _GA.gripper_loop_constraint_labels(
        _MJM_JOINT_NAMES, coupler_names, followers, masters
    )

    # site 1 — coupler actuator unlock
    unlocked = {jn for jn in _MJM_JOINT_NAMES if jn in coupler_labels}
    # site 2 — passive damping 8.0 (driven)
    damp8 = {jn for jn in _MJM_JOINT_NAMES if jn in driven_labels}
    # site 3 — passive damping 1.0 (couplers), else-if after site 2
    damp1 = {jn for jn in _MJM_JOINT_NAMES if jn not in driven_labels and jn in coupler_labels}

    assert unlocked == _EXPECT_COUPLER_LABELS
    assert damp8 == _EXPECT_DRIVEN_LABELS
    assert damp1 == _EXPECT_COUPLER_LABELS
    # damping-8 and damping-1 sets are disjoint (a joint is driven XOR coupler)
    assert not (damp8 & damp1)


def test_no_mimic_map_degrades_to_legacy_driven_regex():
    """When NO mimic map was parsed (masters and followers both empty —
    never on G2), the driven set degrades to the legacy
    ``gripper_[lr]_(inner|outer)_joint1$`` regex so un-mimic'd scenes keep
    pre-fix behaviour, mirroring the coupler reader's stage-absent
    fallback.  On the G2 joint list that regex picks all four joint1
    (inner + outer)."""
    coupler_names = _GA.read_loop_coupler_joint_names(stage=None, joint_names=list(_ALL_SHORTS))
    coupler_labels, driven_labels = _GA.gripper_loop_constraint_labels(_MJM_JOINT_NAMES, coupler_names, set(), set())
    assert coupler_labels == _EXPECT_COUPLER_LABELS  # couplers still 12
    # legacy joint1$ regex = inner_joint1 x2 + outer_joint1 x2 = 4
    assert driven_labels == {_MJCF + s for s in (G2_MASTERS | G2_OUTER_JOINT1)}


# ======================================================================
# ANNOTATION path (newton:isLoopCoupler present) — needs pxr
# ======================================================================
def _build_annotated_stage():
    """Minimal in-memory USD stage carrying ``newton:isLoopCoupler=True``
    on the 12 coupler joints — models what Task 1's authoring produces.
    The two ``inner_joint1`` masters and two ``outer_joint1`` followers
    are present WITHOUT the coupler annotation (they are not couplers)."""
    from pxr import Sdf, Usd

    stage = Usd.Stage.CreateInMemory()
    for cj in sorted(G2_COUPLER_JOINTS):
        prim = stage.DefinePrim(f"/World/robot/joints/{cj}", "PhysicsRevoluteJoint")
        a = prim.CreateAttribute(_GA.ANNOT_LOOP_COUPLER, Sdf.ValueTypeNames.Bool)
        a.Set(True)
    # driven joints present but UN-annotated (must be excluded from couplers)
    for dj in sorted(G2_MASTERS | G2_OUTER_JOINT1):
        stage.DefinePrim(f"/World/robot/joints/{dj}", "PhysicsRevoluteJoint")
    return stage


def test_annotation_path_coupler_and_driven_labels():
    """With the stage annotated, couplers come from ``newton:isLoopCoupler``
    (not the regex); the driven set is still structural.  Same 12/4/12
    split as the fallback path (annotation and fallback AGREE on G2)."""
    pytest.importorskip("pxr")
    stage = _build_annotated_stage()
    masters, followers = _mimic_name_sets()
    # pass a deliberately-wrong joint_names so a fallback would give a
    # DIFFERENT coupler set — proves the annotation path is what's used.
    coupler_names = _GA.read_loop_coupler_joint_names(stage, joint_names=["nonsense_joint0"])
    assert coupler_names == G2_COUPLER_JOINTS  # from the annotation, not the bogus fallback
    coupler_labels, driven_labels = _GA.gripper_loop_constraint_labels(
        _MJM_JOINT_NAMES, coupler_names, followers, masters
    )
    assert coupler_labels == _EXPECT_COUPLER_LABELS
    assert driven_labels == _EXPECT_DRIVEN_LABELS
    # outer_joint1 stays DRIVEN, never a coupler, on the annotation path too
    for oj1 in (_MJCF + s for s in G2_OUTER_JOINT1):
        assert oj1 in driven_labels and oj1 not in coupler_labels


def test_annotation_and_fallback_agree_on_g2():
    """The whole point of the T1 fix: on G2 the annotation coupler set and
    the legacy-regex fallback coupler set are IDENTICAL, so H3 is
    byte-equivalent regardless of which path a given USD takes."""
    pytest.importorskip("pxr")
    stage = _build_annotated_stage()
    annot = _GA.read_loop_coupler_joint_names(stage, joint_names=["nonsense_joint0"])
    fallback = _GA.read_loop_coupler_joint_names(stage=None, joint_names=list(_ALL_SHORTS))
    assert annot == fallback == G2_COUPLER_JOINTS

#!/usr/bin/env python3

# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Lock-down test for the assemble-time gripper semantic annotations.

Task 1 of the engine-hardcoding H1-H4 fix (see
``docs/superpowers/plans/2026-07-21-engine-hardcoding-H1-H4-fix.md`` and
``docs/PORT_ENGINE_HARDCODING_AUDIT.md``).  The physics engine used to
identify fingertip "pad" bodies and 4-bar "coupler" joints with G2
specific link/joint name patterns (``"link4" in label``,
``gripper_[lr]_(inner|outer)_joint[034]$``).  Task 1 moves that
knowledge to ASSEMBLE TIME: when ``robot.usda`` is built from the URDF,
two boolean custom attributes are authored so the engine can read the
gripper topology from the robot's own USD instead of hard-coding it:

  * ``newton:isGripperPad``  (bool) on each fingertip pad body prim
  * ``newton:isLoopCoupler`` (bool) on each 4-bar coupler joint prim

Acceptance is PHYSICAL TRUTH, not the old literals:

  * pad set == the 4 real gripper fingertip pad links
    (``gripper_[lr]_(inner|outer)_link4``) — NOT ``arm_l_link4`` /
    ``arm_r_link4`` / ``body_link4`` (the latent bug being fixed).
  * coupler set == the URDF ``<mimic>``-follower gripper joints whose
    parent link is NOT the gripper base link (12 on G2, 6 per gripper).
    The two base-parented followers (``gripper_[lr]_outer_joint1``) are
    DRIVEN joints, not passive couplers, and are excluded.

The pure derivation (URDF string -> pad/coupler name sets) is the real
thing this test locks down; it needs no pxr.  A second test drives the
full USD authoring against the real G2 ``robot.usda`` and is guarded by
``importorskip("pxr")``.
"""

from __future__ import annotations

import importlib.util
import pathlib
import shutil

import pytest

_HERE = pathlib.Path(__file__).resolve()


def _load_robot_impl():
    """Import the assemble impl module directly from its file.

    We load ``_robot_impl.py`` by path (via ``importlib.util``) rather
    than importing the ``genie_sim_engine_isaacsim.assemble`` package so
    the package ``__init__`` side effects (which pull Isaac / Newton
    pipeline stages) are not triggered — only the pure helpers under
    test are needed here.  ``_robot_impl.py`` keeps all Isaac / pxr
    imports gated inside functions, so importing the module itself is
    dependency-free.
    """
    # .../ros_ws/src/genie_sim_engine_newton/genie_sim_engine_newton/tools/<this>
    #                ^ parents[3] == the ros_ws/src dir holding both packages
    src_root = _HERE.parents[3]
    impl = src_root / "genie_sim_engine_isaacsim" / "genie_sim_engine_isaacsim" / "assemble" / "_robot_impl.py"
    assert impl.is_file(), f"_robot_impl.py not found at {impl}"
    spec = importlib.util.spec_from_file_location("_robot_impl_under_test", impl)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def _find_g2_scene():
    """Locate the committed G2 scene fixture by walking up from here."""
    for parent in _HERE.parents:
        cand = parent / "geniesim" / "assets" / "scenes" / "scene_flat_g2_op"
        if (cand / "robot_raw.urdf").is_file():
            return cand
    return None


_RI = _load_robot_impl()
_G2 = _find_g2_scene()

# ----------------------------------------------------------------------
# G2 physical-truth golden sets
# ----------------------------------------------------------------------
# 4 real fingertip pad links (leaf links of the movable gripper chains,
# excluding the 4-bar loop-closure bodies). Deliberately does NOT
# include arm_l/r_link4 or body_link4 — the old ``"link4" in label``
# substring bug hit those; the annotation must not.
G2_PAD_LINKS = {
    "gripper_l_inner_link4",
    "gripper_l_outer_link4",
    "gripper_r_inner_link4",
    "gripper_r_outer_link4",
}

# 12 coupler joints = the URDF ``<mimic>`` follower gripper joints whose
# parent link is NOT the gripper base link (6 per gripper; all mimic
# ``inner_joint1``).  The two base-parented followers
# (``gripper_[lr]_outer_joint1``) are the second DRIVEN joints, not passive
# 4-bar couplers, so they are excluded.
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

# Names that MUST NOT be classified as pads (the latent-bug false hits).
G2_PAD_ANTI_SET = {
    "arm_l_link4",
    "arm_r_link4",
    "body_link4",
    "gripper_l_camera_link",
    "gripper_r_camera_link",
    "gripper_l_base_link",
    "gripper_r_base_link",
}


@pytest.mark.skipif(_G2 is None, reason="G2 scene fixture (scene_flat_g2_op) not found")
def test_derivation_matches_physical_truth():
    """Pure URDF -> (pads, couplers) derivation == G2 physical truth."""
    urdf = (_G2 / "robot_raw.urdf").read_text()
    ann = _RI._derive_gripper_annotations(urdf)
    assert ann["pads"] == G2_PAD_LINKS, f"pads mismatch: {sorted(ann['pads'])}"
    assert ann["couplers"] == G2_COUPLER_JOINTS, f"couplers mismatch: {sorted(ann['couplers'])}"


@pytest.mark.skipif(_G2 is None, reason="G2 scene fixture (scene_flat_g2_op) not found")
def test_derivation_excludes_arm_and_body_link4():
    """The derivation must not repeat the old ``"link4"`` substring bug."""
    urdf = (_G2 / "robot_raw.urdf").read_text()
    ann = _RI._derive_gripper_annotations(urdf)
    for bad in G2_PAD_ANTI_SET:
        assert bad not in ann["pads"], f"{bad!r} wrongly classified as a gripper pad"


@pytest.mark.skipif(_G2 is None, reason="G2 scene fixture (scene_flat_g2_op) not found")
def test_usd_authoring_on_g2(tmp_path):
    """Author the annotations onto a COPY of the real G2 robot.usda and
    assert the prims carrying each attribute == the physical-truth sets."""
    pytest.importorskip("pxr")
    from pxr import Usd

    # Copy the whole scene dir (robot.usda + payloads/) to a scratch
    # location so we never mutate the committed fixture.  symlinks=True
    # preserves the (possibly dangling) genie_sim_robot_model link
    # without dereferencing it.
    dst = tmp_path / "scene_flat_g2_op"
    shutil.copytree(_G2, dst, symlinks=True)

    usd_path = dst / "robot.usda"
    urdf_path = dst / "robot_raw.urdf"
    assert usd_path.is_file() and urdf_path.is_file()

    _RI._apply_mimic_joint_overlay(str(usd_path), urdf_path=str(urdf_path))

    stage = Usd.Stage.Open(str(usd_path))
    assert stage is not None
    pads: set[str] = set()
    couplers: set[str] = set()
    for prim in stage.Traverse():
        a = prim.GetAttribute("newton:isGripperPad")
        if a and a.IsValid() and a.HasAuthoredValue() and bool(a.Get()):
            pads.add(prim.GetName())
        c = prim.GetAttribute("newton:isLoopCoupler")
        if c and c.IsValid() and c.HasAuthoredValue() and bool(c.Get()):
            couplers.add(prim.GetName())

    assert pads == G2_PAD_LINKS, f"authored pad set mismatch: {sorted(pads)}"
    assert couplers == G2_COUPLER_JOINTS, f"authored coupler set mismatch: {sorted(couplers)}"


@pytest.mark.skipif(_G2 is None, reason="G2 scene fixture (scene_flat_g2_op) not found")
def test_authoring_is_idempotent(tmp_path):
    """Re-running the overlay must not change the annotation sets."""
    pytest.importorskip("pxr")
    from pxr import Usd

    dst = tmp_path / "scene_flat_g2_op"
    shutil.copytree(_G2, dst, symlinks=True)
    usd_path = dst / "robot.usda"
    urdf_path = dst / "robot_raw.urdf"

    _RI._apply_mimic_joint_overlay(str(usd_path), urdf_path=str(urdf_path))
    _RI._apply_mimic_joint_overlay(str(usd_path), urdf_path=str(urdf_path))

    stage = Usd.Stage.Open(str(usd_path))
    pads = {
        p.GetName()
        for p in stage.Traverse()
        if (a := p.GetAttribute("newton:isGripperPad")) and a.IsValid() and a.HasAuthoredValue() and bool(a.Get())
    }
    couplers = {
        p.GetName()
        for p in stage.Traverse()
        if (c := p.GetAttribute("newton:isLoopCoupler")) and c.IsValid() and c.HasAuthoredValue() and bool(c.Get())
    }
    assert pads == G2_PAD_LINKS
    assert couplers == G2_COUPLER_JOINTS

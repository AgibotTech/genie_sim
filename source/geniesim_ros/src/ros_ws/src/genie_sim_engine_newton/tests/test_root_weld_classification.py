# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Root-weld repair must preserve internal joints through fixed USD frames."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("pxr")
pytest.importorskip("warp")
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics  # noqa: E402

_PKG_ROOT = Path(__file__).resolve().parent.parent
_CORE_ROOT = _PKG_ROOT.parent / "genie_sim_engine_core"
if str(_CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CORE_ROOT))
_STAGE_PATH = _PKG_ROOT / "genie_sim_engine_newton/engine/newton/setup/stage.py"
_SPEC = importlib.util.spec_from_file_location("_root_weld_stage_under_test", _STAGE_PATH)
assert _SPEC and _SPEC.loader
stage_module = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(stage_module)


def _rigid_body(stage, path):
    prim = UsdGeom.Xform.Define(stage, path).GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    return prim


def _fixed_joint(stage, path, body0, body1):
    joint = UsdPhysics.FixedJoint.Define(stage, path)
    joint.GetBody0Rel().SetTargets([body0] if body0 else [])
    joint.GetBody1Rel().SetTargets([body1] if body1 else [])
    joint.CreateLocalPos0Attr(Gf.Vec3f(0.02, 0.03, 0.04))
    joint.CreateLocalRot0Attr(Gf.Quatf(0.9238795, 0, 0.3826834, 0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0.01, 0))
    joint.CreateLocalRot1Attr(Gf.Quatf(0.9238795, 0.3826834, 0, 0))
    return joint


@pytest.fixture
def robot():
    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/genie")
    root.AddTranslateOp().Set(Gf.Vec3d(0.3, 0, 0.04))
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    base = _rigid_body(stage, "/genie/base_link")
    arm = _rigid_body(stage, f"{base.GetPath()}/arm_r_link7")
    # The converter retains this fixed frame while collapsing its rigid body.
    # Newton resolves the endpoint through the ancestor rigid arm link.
    end = UsdGeom.Xform.Define(stage, f"{arm.GetPath()}/arm_r_end_link").GetPrim()
    hand = _rigid_body(stage, f"{end.GetPath()}/hand_r_base_link")
    root_joint = _fixed_joint(stage, "/genie/Physics/root_joint", "/genie", str(base.GetPath()))
    hand_joint = _fixed_joint(stage, "/genie/Physics/hand_r_base_joint", str(end.GetPath()), str(hand.GetPath()))
    roots = stage_module.topmost_rigid_body_paths(stage, Sdf.Path("/genie"))
    assert roots == [str(base.GetPath())]
    return SimpleNamespace(stage=stage, roots=roots, base=base, root_joint=root_joint, hand_joint=hand_joint)


def test_internal_fixed_frame_to_hand_joint_is_not_a_world_weld(robot):
    rec = stage_module.classify_world_weld(robot.stage, robot.hand_joint.GetPrim(), Sdf.Path("/genie"), robot.roots)

    assert rec is None


@pytest.mark.parametrize("world_side", ["empty", "dangling", "non-rigid-prim"])
def test_genuine_world_to_topmost_root_weld_still_classifies(robot, world_side):
    if world_side == "empty":
        robot.root_joint.GetBody0Rel().SetTargets([])
    elif world_side == "dangling":
        robot.root_joint.GetBody0Rel().SetTargets(["/missing_world_frame"])
    else:
        robot.root_joint.GetBody0Rel().SetTargets(["/genie"])

    rec = stage_module.classify_world_weld(robot.stage, robot.root_joint.GetPrim(), Sdf.Path("/genie"), robot.roots)

    assert rec is not None
    assert rec["world_side"] == world_side
    assert rec["body1"] == robot.roots
    assert rec["misplaces"] is True


def test_joint_without_a_topmost_child_cannot_be_repaired_as_a_world_weld(robot):
    robot.root_joint.GetBody1Rel().SetTargets([])

    assert (
        stage_module.classify_world_weld(robot.stage, robot.root_joint.GetPrim(), Sdf.Path("/genie"), robot.roots)
        is None
    )


def test_repair_changes_only_real_root_anchor_and_keeps_internal_hand_endpoints_and_poses(robot):
    engine = stage_module._StageMixin()
    engine._stage = robot.stage
    engine._robot_prefix_str = "genie"
    engine._pin_base_to_world = True
    engine._logger = Mock()
    root_layer_before = robot.stage.GetRootLayer().ExportToString()
    hand = robot.hand_joint
    hand_before = (
        hand.GetBody0Rel().GetTargets(),
        hand.GetBody1Rel().GetTargets(),
        hand.GetLocalPos0Attr().Get(),
        hand.GetLocalRot0Attr().Get(),
        hand.GetLocalPos1Attr().Get(),
        hand.GetLocalRot1Attr().Get(),
    )

    assert engine._repair_root_weld_anchor() == 1

    assert (
        hand.GetBody0Rel().GetTargets(),
        hand.GetBody1Rel().GetTargets(),
        hand.GetLocalPos0Attr().Get(),
        hand.GetLocalRot0Attr().Get(),
        hand.GetLocalPos1Attr().Get(),
        hand.GetLocalRot1Attr().Get(),
    ) == hand_before
    assert not hand.GetPrim().GetAttribute(stage_module.WELD_ANCHOR_ANNOT)
    assert robot.root_joint.GetBody0Rel().GetTargets() == []
    assert tuple(robot.root_joint.GetLocalPos0Attr().Get()) == pytest.approx((0.3, 0, 0.04))
    assert robot.root_joint.GetLocalPos1Attr().Get() == Gf.Vec3f(0)
    assert robot.root_joint.GetPrim().GetAttribute(stage_module.WELD_ANCHOR_ANNOT).Get() is True
    assert engine._repair_root_weld_anchor() == 0
    assert robot.stage.GetRootLayer().ExportToString() == root_layer_before
    engine._logger.warn.assert_not_called()

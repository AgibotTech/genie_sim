# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Body-path assembly reads ordinary USD files at the requested path."""

import pytest

pytest.importorskip("pxr")
from pxr import Usd, UsdGeom, UsdPhysics

from genie_sim_engine_core.common.usd_walk import build_body_paths


@pytest.mark.parametrize("suffix", [".usda", ".usdc", ".usd"])
def test_plain_usd_body_paths_preserve_rigid_body_selection(tmp_path, suffix):
    path = tmp_path / ("robot" + suffix)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.Xform.Define(stage, "/robot/base_link")
    body = UsdGeom.Xform.Define(stage, "/robot/base_link/base_link")
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    stage.GetRootLayer().Save()
    assert build_body_paths(path, ["base_link", "missing"]) == {"base_link": "/robot/base_link/base_link"}


def test_missing_usd_does_not_select_an_alternate_file(tmp_path, capsys):
    (tmp_path / "robot.usdc.gs").write_bytes(b"not a USD file")
    assert build_body_paths(tmp_path / "robot.usda", ["base_link"]) == {}
    assert "not found/openable" in capsys.readouterr().err

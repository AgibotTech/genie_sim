# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Baseline robots must initialize with the actual engine-session logger."""

import pytest

from genie_sim_engine_core.common.joint_params import JointParamResolver
from genie_sim_engine_core.common.session import SimpleLogger


@pytest.mark.parametrize("engine", ["physx", "newton"])
def test_baseline_joint_params_with_session_logger(engine, capsys):
    # G2 crs/Omnipicker has no compatible sysid overlay. The empty-stack
    # diagnostic used to crash startup before initial pose or ROS publication.
    resolver = JointParamResolver.load(
        [], engine=engine, robot={"robot_model": "genie", "arm": "crs"}, logger=SimpleLogger("genie_sim_engine")
    )
    assert not resolver
    resolver.validate_names(["idx22_arm_l_joint2"])
    assert resolver.drive("idx22_arm_l_joint2", 50000.0, 5000.0, 5000.0) == (50000.0, 5000.0, 5000.0)
    assert "[genie_sim_engine] DEBUG: [joint_params] no overlays" in capsys.readouterr().out

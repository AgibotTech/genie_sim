# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Newton's wire body table must include plugin bodies absent from USD."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def session_type():
    source = Path(__file__).resolve().parents[1] / "genie_sim_engine_newton/common/session.py"
    tree = ast.parse(source.read_text())
    definition = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "GenieSimEngineSession"
    )

    class Base:
        def startup(self, headless):
            self.events.append(("backend_startup", headless))

    namespace = {"EngineSession": Base, "Any": object}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["GenieSimEngineSession"]


def make_session(paths, poses=None, worlds=1):
    session = session_type()()
    session.events = []
    if poses is None:
        poses = np.arange(len(paths) * 7, dtype=float).reshape(-1, 7)
    session.sim = SimpleNamespace(
        body_paths=paths,
        num_envs=worlds,
        _robot_prefix_str="genie",
        get_body_transforms=lambda: (poses, list(paths)),
    )
    session.robot_joint_names = ["arm_r_joint1"]
    session.core = SimpleNamespace(set_topology=lambda *values: session.events.append(("topology", *values)))
    return session


def test_startup_registers_model_bodies_including_runtime_plugin_geometry():
    paths = ["/genie/Geometry/base_link", "/genie/Geometry/head_link3", "chef_wok", "chef_stove"]
    session = make_session(paths)
    session.startup(headless=False)
    session.bind_newton_body_topology()
    assert session.events == [
        ("backend_startup", False),
        ("topology", ["arm_r_joint1"], paths, [True, True, False, False]),
    ]


def test_incomplete_readback_does_not_register_a_truncated_body_table():
    session = make_session(["/genie/base_link", "chef_wok"], poses=np.zeros((1, 7)))
    session.startup(headless=True)
    with pytest.raises(RuntimeError, match="body"):
        session.bind_newton_body_topology()
    assert session.events == [("backend_startup", True)]


def test_multiple_worlds_preserve_model_order_and_robot_mask():
    session = make_session(["/genie/base_link", "chef_wok"] * 2, worlds=2)
    session.startup(headless=True)
    session.bind_newton_body_topology()
    assert session.events[-1] == (
        "topology",
        ["arm_r_joint1"],
        ["env_0/genie/base_link", "env_0/chef_wok", "env_1/genie/base_link", "env_1/chef_wok"],
        [True, False, True, False],
    )

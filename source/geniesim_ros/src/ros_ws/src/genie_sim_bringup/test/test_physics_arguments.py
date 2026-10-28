# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Exercise the launcher's actual argv stanza without launching any processes."""

import ast
from pathlib import Path

import pytest


LAUNCH = Path(__file__).resolve().parents[1] / "launch/physics_backend.launch.py"


def physics_arguments(headless):
    # The surrounding launch callback resolves assets and builds actions. Only
    # its small, side-effect-free argv stanza is needed to test ROS boundaries.
    tree = ast.parse(LAUNCH.read_text())
    for parent in ast.walk(tree):
        body = getattr(parent, "body", None)
        if not isinstance(body, list):
            continue
        for index, statement in enumerate(body):
            if isinstance(statement, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "physics_args" for target in statement.targets
            ):
                selected = []
                for item in body[index:]:
                    if (
                        isinstance(item, ast.Expr)
                        and isinstance(item.value, ast.Call)
                        and isinstance(item.value.func, ast.Attribute)
                        and isinstance(item.value.func.value, ast.Name)
                        and item.value.func.value.id == "nodes"
                    ):
                        break
                    selected.append(item)
                namespace = {"headless": headless, "ros_log_args": ["--ros-args", "--log-level", "info"]}
                exec(compile(ast.Module(body=selected, type_ignores=[]), str(LAUNCH), "exec"), namespace)
                assert namespace["ros_log_args"] == ["--ros-args", "--log-level", "info"]
                return namespace["physics_args"]
    raise AssertionError("physics argument stanza was not found")


def test_gui_option_precedes_the_first_ros_argument_section():
    args = physics_arguments("false")
    assert args.index("--gui") < args.index("--ros-args")


def test_headless_launch_keeps_ros_logging_without_gui_option():
    assert physics_arguments("true") == ["--ros-args", "--log-level", "info"]


def test_gui_and_launch_ros_remaps_are_accepted_by_real_rclpy_parser():
    utilities = pytest.importorskip("rclpy.utilities")
    argv = ["newton_node", *physics_arguments("false"), "--ros-args", "-r", "__node:=genie_sim_engine"]
    assert utilities.remove_ros_args(args=argv) == ["newton_node", "--gui"]

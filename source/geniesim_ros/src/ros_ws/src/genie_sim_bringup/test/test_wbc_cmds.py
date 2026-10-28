# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Exercise the command driver with the shipped Wuji kinematics, without DDS."""

import importlib.util
import io
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("rclpy")
import xacro
from builtin_interfaces.msg import Time
from sensor_msgs.msg import JointState
from std_msgs.msg import String

BRINGUP = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("wbc_cmds", BRINGUP / "scripts/wbc_cmds.py")
WBC = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WBC)


@pytest.fixture
def wuji_xml():
    source = BRINGUP.parent / "genie_sim_robot_model/robots/genie/g2/xacro/gripper.wuji.urdf.xacro"
    doc = xacro.parse(io.StringIO(f"""<robot name="hands" xmlns:xacro="http://www.ros.org/wiki/xacro">
              <xacro:include filename="{source}"/>
              <xacro:wuji_hand prefix="hand_l" side="left" mesh_path="." attach="false"/>
              <xacro:wuji_hand prefix="hand_r" side="right" mesh_path="." attach="false"/>
            </robot>"""))
    xacro.process_doc(doc)
    return doc.toxml()


@pytest.fixture
def commander(monkeypatch):
    """Keep real driver logic/messages; replace only Node's ROS transport."""
    records, published = [], []
    logger = SimpleNamespace(
        **{level: lambda msg: records.append(json.loads(msg)) for level in ("info", "warn", "error")}
    )
    monkeypatch.setattr(WBC.Node, "__init__", lambda *a, **kw: None)
    monkeypatch.setattr(WBC.WbcCommander, "get_logger", lambda self: logger)
    monkeypatch.setattr(WBC.WbcCommander, "create_subscription", lambda *a, **kw: None)
    monkeypatch.setattr(WBC.WbcCommander, "create_timer", lambda *a, **kw: None)
    monkeypatch.setattr(
        WBC.WbcCommander, "create_publisher", lambda *a, **kw: SimpleNamespace(publish=published.append)
    )
    monkeypatch.setattr(
        WBC.WbcCommander, "get_clock", lambda self: SimpleNamespace(now=lambda: SimpleNamespace(to_msg=Time))
    )

    def make(*args):
        return WBC.WbcCommander(WBC._parse_args(list(args))), records, published

    return make


def _feed_states(node):
    # Zero where allowed; thumb joint1 has a positive lower limit.
    values = {name: min(hi, max(lo, 0.0)) for name, (lo, hi) in node._joints.items()}
    node._on_joint_states(JointState(name=list(values), position=list(values.values())))


def test_both_wuji_hands_get_visible_goals_and_hand_speed(wuji_xml, commander):
    node, records, published = commander(
        "--joints", "hand_l_", "hand_r_", "--cmd-mode", "ramp", "--hand-deg", "30", "--vel-hand", "10"
    )
    node._on_description(String(data=wuji_xml))
    assert node._group_counts() == {"hand": 40}
    _feed_states(node)
    WBC.random.seed(42)
    node._step()
    assert len(published[-1].name) == 40
    assert max(abs(goal - node._init_states[n]) for n, goal in node._goals.items()) > math.radians(10)
    for name, value in zip(published[-1].name, published[-1].position):
        assert abs(value - node._states[name]) <= math.radians(10) / node._rate_hz + 1e-12
    for name, value in node._goals.items():
        lo, hi = node._joints[name]
        assert lo <= value <= hi
        assert abs(value - node._init_states[name]) <= math.radians(30)


def test_parallel_gripper_filter_reports_no_wuji_matches(wuji_xml, commander):
    node, records, published = commander("--joints", "gripper")
    node._on_description(String(data=wuji_xml))
    node._step()
    node._log_status()
    assert not published
    assert any(r.get("event") == "no_matching_joints" for r in records)
    assert records[-1]["status"] == "NO_MATCHING_JOINTS"


def test_missing_finger_feedback_cannot_start_or_complete_a_goal(wuji_xml, commander):
    node, records, published = commander("--joints", "hand_l_", "hand_r_")
    node._on_description(String(data=wuji_xml))
    _feed_states(node)
    missing, value = node._states.popitem()
    node._step()
    assert not published
    node._states[missing] = value
    node._step()
    node._states = dict(node._goals)
    assert node._all_reached()
    node._states.pop(missing)
    assert not node._all_reached()
    node._log_status()
    assert records[-1]["progress"] == "39/40"
    assert records[-1]["missing_feedback"] == [missing]


def test_parallel_gripper_classification_and_mimic_exclusion():
    xml = """<robot name="jaws">
      <joint name="idx31_gripper_l_inner_joint1" type="revolute"><limit lower="-1" upper="0"/></joint>
      <joint name="idx32_gripper_l_outer_joint1" type="revolute"><limit lower="-1" upper="0"/>
        <mimic joint="idx31_gripper_l_inner_joint1"/></joint>
    </robot>"""
    joints = WBC._parse_urdf_joints(xml, [])
    assert list(joints) == ["idx31_gripper_l_inner_joint1"]
    assert WBC._classify(next(iter(joints))) == "gripper"


@pytest.mark.parametrize("mode", ["step", "ramp"])
def test_timeout_reports_thumb_target_command_and_measured_error(wuji_xml, commander, mode):
    node, records, published = commander("--joints", "hand_l_", "hand_r_", "--cmd-mode", mode)
    node._on_description(String(data=wuji_xml))
    _feed_states(node)
    node._goals = dict(node._states)
    thumb = "hand_l_thumb_joint1"
    node._goals[thumb] = 0.25
    node._states[thumb] = 0.10
    node._smoothed[thumb] = 0.20
    node._goal_start = WBC.time.monotonic() - 6
    node._step()
    timeout = next(r for r in records if r.get("event") == "goal_timeout")
    assert timeout["stuck"] == [thumb]
    assert timeout["stuck_detail"][thumb] == pytest.approx(
        {"goal": 0.25, "command": 0.25 if mode == "step" else 0.20, "state": 0.10, "error": 0.15}
    )
    assert timeout["tolerance"] == 0.05
    assert timeout["cmd_mode"] == mode

# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Validate Wuji commands against the selected task's robot model."""

import math
import xml.etree.ElementTree as ET

import yaml

from geniesim_teleop.config.task_config import load_teleop_task, robot_family

HAND_JOINT_NAMES = {
    side: tuple(
        f"hand_{code}_{finger}_joint{joint}"
        for finger in ("thumb", "index", "middle", "ring", "pinky")
        for joint in range(1, 5)
    )
    for side, code in (("left", "l"), ("right", "r"))
}


def load_task(config):
    task = load_teleop_task(config)
    if robot_family(task.robot_cfg) != "G2_crsB_wuji":
        raise ValueError("Wuji launcher requires a task selecting G2_crsB_wuji")
    task.initial_states()
    settings = yaml.safe_load(task.config_path.read_text())
    benchmark = settings["benchmark"]
    if (benchmark.get("model_arc"), benchmark.get("env_class"), benchmark.get("policy_class")) != (
        "pi",
        "DummyEnv",
        "DemoPolicy",
    ):
        raise ValueError("Wuji task must select pi / DummyEnv / DemoPolicy")
    if not settings.get("app", {}).get("enable_ros") or not benchmark.get("enable_ros"):
        raise ValueError("Wuji task must enable ROS in app and benchmark")
    return task


class HandCommands:
    def __init__(self, urdf_path):
        self.sides = HAND_JOINT_NAMES
        self.aliases = {
            f"{side}_finger{finger}_joint{joint}": names[(finger - 1) * 4 + joint - 1]
            for side, names in self.sides.items()
            for finger in range(1, 6)
            for joint in range(1, 5)
        }
        joints = {joint.get("name"): joint for joint in ET.parse(urdf_path).getroot().findall("joint")}
        self.limits = {}
        for names in self.sides.values():
            for name in names:
                joint = joints.get(name)
                if joint is None or joint.get("type") != "revolute" or joint.find("limit") is None:
                    raise ValueError(f"Wuji URDF is missing a revolute joint with limits: {name}")
                limit = joint.find("limit")
                lower, upper = float(limit.get("lower")), float(limit.get("upper"))
                if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
                    raise ValueError(f"Invalid Wuji joint limits: {name}")
                self.limits[name] = lower, upper

    def split(self, names, positions):
        """Accept complete hands only; reject unknown DOFs and clamp to URDF limits."""
        if not names or len(names) != len(positions):
            raise ValueError("Hand names and positions must be nonempty and have equal length")
        values = {}
        for name, position in zip(names, positions):
            name = self.aliases.get(name, name)
            if name not in self.limits or name in values or not math.isfinite(position):
                raise ValueError(f"Invalid, duplicate, or non-finite hand joint: {name}")
            lower, upper = self.limits[name]
            values[name] = max(lower, min(upper, float(position)))
        commands = {}
        for side, joints in self.sides.items():
            present = set(joints).intersection(values)
            if present and len(present) != 20:
                raise ValueError(f"Expected all 20 {side} hand joints, got {len(present)}")
            if present:
                commands[side] = (list(joints), [values[name] for name in joints])
        return commands

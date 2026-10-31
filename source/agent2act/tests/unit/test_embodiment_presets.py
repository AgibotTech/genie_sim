# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import copy
import json

import pytest

from agent2act.adapters.embodiments import bind_reference, planner_hardware, validate_definition
from agent2act.domain.errors import ConfigurationError


def test_builtin_definition_is_frozen_and_describes_only_supported_frames():
    binding = bind_reference({"preset": "g2_omnipicker"})
    assert bind_reference(binding) == binding
    definition = binding["definition"]
    hardware = planner_hardware(definition, True)
    assert hardware["gripper_center_offset_m"]["left"] == [0, 0, 0.14845]
    assert hardware["gripper_open_closed"]["right"] == [-0.785, 0]
    assert "26 mm" in hardware["grasp_geometry_calibration"]["gripper_notes"]
    assert "60.4 mm" in hardware["grasp_geometry_calibration"]["gripper_notes"]
    assert "hand_left" not in hardware["grasp_geometry_calibration"]["gripper_notes"]
    assert "five times" not in json.dumps(hardware)
    assert "pad" not in hardware["pose_reference"]
    assert "pose_reference" not in planner_hardware(definition, False)
    hardware["gripper_center_offset_m"]["left"][0] = 99
    assert definition["tcp"]["offset_m"]["left"][0] == 0


@pytest.mark.parametrize(
    "change",
    ["rotation", "nan", "unknown", "axes", "missing_side", "equal_commands", "manual_text", "unsourced"],
)
def test_bad_hardware_is_rejected_before_model_construction(change):
    data = bind_reference({"preset": "g2_omnipicker"})["definition"]
    if change == "rotation":
        data["tcp"]["rotation"] = [0, 0, 0, 1]
    elif change == "nan":
        data["tcp"]["offset_m"]["left"][0] = float("nan")
    elif change == "unknown":
        data["motion_backend"] = {"type": "another"}
    elif change == "axes":
        data["grasp_geometry_calibration"]["tcp_axes"]["open_close"] = "z"
    elif change == "missing_side":
        del data["gripper_open_closed"]["right"]
    elif change == "equal_commands":
        data["gripper_open_closed"]["left"] = [0, 0]
    elif change == "manual_text":
        data["grasp_geometry_calibration"]["gripper_notes"] = "another gripper"
    else:
        data["observations"] = [{"text": "unverified"}]
    with pytest.raises(ConfigurationError):
        validate_definition(data)


@pytest.mark.parametrize("change", ["hash", "version", "definition"])
def test_frozen_definition_cannot_silently_change(change):
    binding = bind_reference({"preset": "g2_omnipicker"})
    if change == "hash":
        binding["sha256"] = "0" * 64
    elif change == "version":
        binding["description_version"] = "another"
    else:
        binding["definition"]["tcp"]["offset_m"]["left"][2] = 0.185
    with pytest.raises(ConfigurationError):
        bind_reference(binding)


def test_custom_file_and_freeze_survive_source_removal(tmp_path):
    definition = bind_reference({"preset": "g2_omnipicker"})["definition"]
    path = tmp_path / "gripper.json"
    path.write_text(json.dumps(definition))
    frozen = bind_reference({"path": "gripper.json"}, tmp_path)
    path.unlink()
    assert bind_reference(copy.deepcopy(frozen)) == frozen
    with pytest.raises(ConfigurationError, match="cannot load"):
        bind_reference({"path": "gripper.json"}, tmp_path)

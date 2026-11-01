# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0
"""In-code templates for simulated parameters/hardware metadata.

These fixed values follow G2 reference hardware files when simulation has no
firmware or serial source. Repeated motor entries are generated in code.

The crs arm uses legacy crs motor metadata. The crsB arm uses ROKAE
metadata for both G2_crsB_90D and G2_crsB_omnipicker variants.
"""
from __future__ import annotations

# Arm motor metadata by model; crsB versions use numeric YAML values.
_ARM_MOTOR_BY_ARM = {
    "crs":  {"hw_version": "H00", "sw_version": "V3.0.2.16", "producer": "crs"},
    "crsB": {"hw_version": 1.0,   "sw_version": 1.00,        "producer": "rokae"},
}
_ARM_END_PLATE_BY_ARM = {
    "crs":  {"hw_version": "01.04.20230925",
             "sw_version": "00.04.03.0402.20260317",
             "producer": "crs"},
    "crsB": {"hw_version": 1.1, "sw_version": 2.2, "producer": "rokae"},
}
# Arm section of swappable_modules; reducer is omitted as in the 90D reference.
_SWAPPABLE_ARMS_BY_ARM = {
    "crs":  {"name": "crs",  "producer": "crs", "model": "crs"},
    "crsB": {"name": "crsB", "producer": "rokae",  "model": "crs"},
}


def _arm_version(arm_model: str) -> dict:
    motor = _ARM_MOTOR_BY_ARM[arm_model]
    end_plate = _ARM_END_PLATE_BY_ARM[arm_model]
    out: dict = {}
    for i in range(1, 8):  # Seven left-arm motors at positions 2 through 8.
        out[f"left_arm_motor_{i}"] = {"position": i + 1, **motor}
    for i in range(1, 8):  # Seven right-arm motors at positions 10 through 16.
        out[f"right_arm_motor_{i}"] = {"position": i + 9, **motor}
    out["left_arm_end_plate"] = {"position": 9, **end_plate}
    out["right_arm_end_plate"] = {"position": 17, **end_plate}
    return {"arm_version": out}


def _head_version() -> dict:
    return {
        "type": "head",
        "joints": [
            {"type": "motor", "model": "ZY_R48", "node_id": nid, "producer": "JXZN",
             "hardware_version": "V1.20", "software_version": "V1.16.04"}
            for nid in (3, 2, 1)
        ],
    }


def _waist_version() -> dict:
    joints = []
    for nid, model in ((8, "ZY_R120MAX"), (7, "ZY_R120MAX"), (6, "ZY_R120"), (5, "ZY_R120"), (4, "ZY_R120")):
        joints.append({"type": "motor", "model": model, "node_id": nid, "producer": "JXZN",
                       "hardware_version": "V1.30", "software_version": "V1.16.08"})
    return {"type": "waist", "joints": joints}


_CHASSIS_VERSION = {
    "type": "chassis", "producer": "linkhou", "model": "Four-wheel-steering",
    "serialNumber": "g02_chassis", "HardwareVersion": "1.0.1", "softwareVersion": "0.0.17",
    "tractionMotorVersion": "2025.514.300", "SteerMotorVersion": "1.14.06",
}

_POWER_BOARD_VERSION = {
    "type": "power-board", "producer": "power_board", "serial_num": "G02T2",
    "hardware_version": 1.3, "software_version": "1.0.8.2.2",  # Reference YAML stores the hardware version as a number.
}


def _swappable_modules(arm_model: str) -> dict:
    eff = [
        {"slot": slot, "name": "auto", "type": "auto", "producer": "auto",
         "model": "auto", "structure": "auto", "camera": "auto", "interface": "auto"}
        for slot in ("left_tool", "right_tool")
    ]
    return {
        "type": "swappable_modules", "robot_model": "G2", "version": "v0.1.0", "author": "agibot",
        "init_time": "2025-08-27 03:06", "last_update": "2025-08-27 03:06",
        "modules": {
            "body": {"model": "t2"},
            "arms": dict(_SWAPPABLE_ARMS_BY_ARM[arm_model]),
            "end_effectors": eff,
        },
        "checksum": "31a7c6f6d7ff5595a366be4e6b8c7010c963ec64d1fd957ef78128927cdb4175",
    }


# aid_info template; the caller sets type and AID from the robot model.
AID_INFO = {
    "type": "G2A", "AID": "", "customer": "genie", "status": "",
    "Activation time": "", "version": "v0.0.1", "author": "agibot",
}


def hand_version(model: str) -> dict:
    """Return gripper version metadata for omnipicker or ctek90d."""
    return {"type": "gripper", "producer": "zhiyuan", "model": model, "firmware_version": ""}


def fixed_g2_yaml_files(arm_model: str = "crs") -> dict:
    """Return fixed G2 hardware files as a filename-to-content mapping.

    ``crs`` uses legacy crs motors; ``crsB`` uses ROKAE motors for 90D
    and crsB omnipicker. The default crs model preserves older G2 recordings.
    """
    return {
        "arm_version.yaml": _arm_version(arm_model),
        "chassis_version.yaml": _CHASSIS_VERSION,
        "head_version.yaml": _head_version(),
        "power_board_version.yaml": _POWER_BOARD_VERSION,
        "waist_version.yaml": _waist_version(),
        "swappable_modules.yaml": _swappable_modules(arm_model),
    }

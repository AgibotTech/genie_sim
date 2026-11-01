# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

G1_JOINT_NAMES = [
    "idx21_arm_l_joint1",
    "idx22_arm_l_joint2",
    "idx23_arm_l_joint3",
    "idx24_arm_l_joint4",
    "idx25_arm_l_joint5",
    "idx26_arm_l_joint6",
    "idx27_arm_l_joint7",
    "idx61_arm_r_joint1",
    "idx62_arm_r_joint2",
    "idx63_arm_r_joint3",
    "idx64_arm_r_joint4",
    "idx65_arm_r_joint5",
    "idx66_arm_r_joint6",
    "idx67_arm_r_joint7",
    "idx11_head_joint1",
    "idx12_head_joint2",
    "idx02_body_joint2",
    "idx01_body_joint1",
]


G2_JOINT_NAMES = [
    "idx21_arm_l_joint1",
    "idx22_arm_l_joint2",
    "idx23_arm_l_joint3",
    "idx24_arm_l_joint4",
    "idx25_arm_l_joint5",
    "idx26_arm_l_joint6",
    "idx27_arm_l_joint7",
    "idx61_arm_r_joint1",
    "idx62_arm_r_joint2",
    "idx63_arm_r_joint3",
    "idx64_arm_r_joint4",
    "idx65_arm_r_joint5",
    "idx66_arm_r_joint6",
    "idx67_arm_r_joint7",
    "idx01_body_joint1",
    "idx02_body_joint2",
    "idx03_body_joint3",
    "idx04_body_joint4",
    "idx05_body_joint5",
    "idx11_head_joint1",
    "idx12_head_joint2",
    "idx13_head_joint3",
]


OMNIPICKER_AJ_NAMES = ["idx41_gripper_l_outer_joint1", "idx81_gripper_r_outer_joint1"]
# G2 ctek90d (CRS-B 90D) gripper control joints: left idx32, right idx72.
CTEK90D_AJ_NAMES = ["idx32_gripper_l_outer_joint1", "idx72_gripper_r_outer_joint1"]


def is_g2_90d(robot_name) -> bool:
    """Identify the ctek90d gripper using a shared, case-insensitive name check.
    Extraction, conversion, and client control must agree on the model.
    Matches G2_crs_90d, G2_crsB_90D, and G2_90d. Older narrower checks selected
    Omnipicker joints instead of idx32/idx72, shifting recorded joint slots
    and applying the wrong gripper conversion."""
    return "90d" in str(robot_name or "").lower()


def joint_state_order_key(robot_name) -> str:
    """Select the recording joint-order entry in robot_joint_names.json.
    Check 90D first, then G2, ARX, Aloha, and default to G1.
    Keep this selection independent of ROS for direct testing.
    Missing joints are skipped by reorder_joint_state, which can shift values
    relative to the complete name list; use a matching joint configuration."""
    name = str(robot_name or "")
    if is_g2_90d(name):
        return "G2_crsB_90d"
    if "G2" in name:
        return "G2"
    if "ARX" in name:
        return "ARX"
    if "Aloha" in name or "aloha" in name:
        return "Aloha"
    return "G1"

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

G1_LEFT_ARM_JOINT_NAMES = [
    "idx21_arm_l_joint1",
    "idx22_arm_l_joint2",
    "idx23_arm_l_joint3",
    "idx24_arm_l_joint4",
    "idx25_arm_l_joint5",
    "idx26_arm_l_joint6",
    "idx27_arm_l_joint7",
]

G1_RIGHT_ARM_JOINT_NAMES = [
    "idx61_arm_r_joint1",
    "idx62_arm_r_joint2",
    "idx63_arm_r_joint3",
    "idx64_arm_r_joint4",
    "idx65_arm_r_joint5",
    "idx66_arm_r_joint6",
    "idx67_arm_r_joint7",
]
G1_DUAL_ARM_JOINT_NAMES = G1_LEFT_ARM_JOINT_NAMES + G1_RIGHT_ARM_JOINT_NAMES

G1_HEAD_JOINT_NAMES = [
    "idx11_head_joint1",
    "idx12_head_joint2",
]

G1_WAIST_JOINT_NAMES = [
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
]

G2_LEFT_ARM_JOINT_NAMES = [
    "idx21_arm_l_joint1",
    "idx22_arm_l_joint2",
    "idx23_arm_l_joint3",
    "idx24_arm_l_joint4",
    "idx25_arm_l_joint5",
    "idx26_arm_l_joint6",
    "idx27_arm_l_joint7",
]
G2_RIGHT_ARM_JOINT_NAMES = [
    "idx61_arm_r_joint1",
    "idx62_arm_r_joint2",
    "idx63_arm_r_joint3",
    "idx64_arm_r_joint4",
    "idx65_arm_r_joint5",
    "idx66_arm_r_joint6",
    "idx67_arm_r_joint7",
]

G2_DUAL_ARM_JOINT_NAMES = G2_LEFT_ARM_JOINT_NAMES + G2_RIGHT_ARM_JOINT_NAMES

G2_HEAD_JOINT_NAMES = [
    "idx11_head_joint1",
    "idx12_head_joint2",
    "idx13_head_joint3",
]
G2_WAIST_JOINT_NAMES = [
    "idx05_body_joint5",
    "idx04_body_joint4",
    "idx03_body_joint3",
    "idx02_body_joint2",
    "idx01_body_joint1",
]

OMNIPICKER_AJ_NAMES = [
    "idx41_gripper_l_outer_joint1",
    "idx81_gripper_r_outer_joint1",
]
G1_CHASSIS = [
    "base_linear_joint_x",
    "base_linear_joint_y",
    "base_angular_joint",
]


def robot_type_mapping(robot_type):
    if "G1_omnipicker" in robot_type:
        return "G1_omnipicker"
    elif "G2_omnipicker" in robot_type:
        return "G2_omnipicker"
    else:
        raise ValueError(f"Invalid robot type: {robot_type}")

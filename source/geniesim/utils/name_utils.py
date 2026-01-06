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

G1_IDX_MAPPING = {
    "idx01_body_joint1": 0,
    "idx02_body_joint2": 1,
    "idx11_head_joint1": 2,
    "idx12_head_joint2": 3,
    "idx21_arm_l_joint1": 4,
    "idx61_arm_r_joint1": 5,
    "idx22_arm_l_joint2": 6,
    "idx62_arm_r_joint2": 7,
    "idx23_arm_l_joint3": 8,
    "idx63_arm_r_joint3": 9,
    "idx24_arm_l_joint4": 10,
    "idx64_arm_r_joint4": 11,
    "idx25_arm_l_joint5": 12,
    "idx65_arm_r_joint5": 13,
    "idx26_arm_l_joint6": 14,
    "idx66_arm_r_joint6": 15,
    "idx27_arm_l_joint7": 16,
    "idx67_arm_r_joint7": 17,
    "idx31_gripper_l_inner_joint1": 18,
    "idx41_gripper_l_outer_joint1": 19,
    "idx71_gripper_r_inner_joint1": 20,
    "idx81_gripper_r_outer_joint1": 21,
    "idx32_gripper_l_inner_joint3": 22,
    "idx42_gripper_l_outer_joint3": 23,
    "idx72_gripper_r_inner_joint3": 24,
    "idx82_gripper_r_outer_joint3": 25,
    "idx33_gripper_l_inner_joint4": 26,
    "idx43_gripper_l_outer_joint4": 27,
    "idx73_gripper_r_inner_joint4": 28,
    "idx83_gripper_r_outer_joint4": 29,
    "idx39_gripper_l_inner_joint0": 30,
    "idx49_gripper_l_outer_joint0": 31,
    "idx79_gripper_r_inner_joint0": 32,
    "idx89_gripper_r_outer_joint0": 33,
}


G2_IDX_MAPPING = {
    "idx01_body_joint1": 0,
    "idx02_body_joint2": 1,
    "idx111_chassis_lwheel_front_joint1": 2,
    "idx121_chassis_lwheel_rear_joint1": 3,
    "idx131_chassis_rwheel_front_joint1": 4,
    "idx141_chassis_rwheel_rear_joint1": 5,
    "idx03_body_joint3": 6,
    "idx112_chassis_lwheel_front_joint2": 7,
    "idx122_chassis_lwheel_rear_joint2": 8,
    "idx132_chassis_rwheel_front_joint2": 9,
    "idx142_chassis_rwheel_rear_joint2": 10,
    "idx04_body_joint4": 11,
    "idx05_body_joint5": 12,
    "idx11_head_joint1": 13,
    "idx12_head_joint2": 14,
    "idx21_arm_l_joint1": 15,
    "idx61_arm_r_joint1": 16,
    "idx13_head_joint3": 17,
    "idx22_arm_l_joint2": 18,
    "idx62_arm_r_joint2": 19,
    "idx23_arm_l_joint3": 20,
    "idx63_arm_r_joint3": 21,
    "idx24_arm_l_joint4": 22,
    "idx64_arm_r_joint4": 23,
    "idx25_arm_l_joint5": 24,
    "idx65_arm_r_joint5": 25,
    "idx26_arm_l_joint6": 26,
    "idx66_arm_r_joint6": 27,
    "idx27_arm_l_joint7": 28,
    "idx67_arm_r_joint7": 29,
    "idx31_gripper_l_inner_joint1": 30,
    "idx41_gripper_l_outer_joint1": 31,
    "idx71_gripper_r_inner_joint1": 32,
    "idx81_gripper_r_outer_joint1": 33,
    "idx32_gripper_l_inner_joint3": 34,
    "idx42_gripper_l_outer_joint3": 35,
    "idx72_gripper_r_inner_joint3": 36,
    "idx82_gripper_r_outer_joint3": 37,
    "idx33_gripper_l_inner_joint4": 38,
    "idx43_gripper_l_outer_joint4": 39,
    "idx73_gripper_r_inner_joint4": 40,
    "idx83_gripper_r_outer_joint4": 41,
    "idx39_gripper_l_inner_joint0": 42,
    "idx49_gripper_l_outer_joint0": 43,
    "idx79_gripper_r_inner_joint0": 44,
    "idx89_gripper_r_outer_joint0": 45,
}

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

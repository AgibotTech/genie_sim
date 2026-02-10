# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from enum import Enum


class RobotType(Enum):
    G2 = "G2"


# cfg
# G2
g2_arm_joints0 = [0.0, -0.66, 0.0, -1.6, 0.0, -0.8, 0.0]
g2_waist_joints0 = [0.0, 0.0, 0.0, 0.0, 0.0]
G2 = {
    #  limits is tmp
    "lb": [],
    "ub": [],
    "tol": [],
    "ee_frames": ["arm_l_end_link", "arm_r_end_link"],
    "ref_frames": ["arm_l_link3", "arm_r_link3"],
    "waist_frame": "body_link5",
    "home_joints": [g2_waist_joints0, g2_arm_joints0, g2_arm_joints0],
}

robot_desc_map = {
    RobotType.G2: G2,
}
BODY_JOINT_NAMES = [
    "idx01_body_joint1",
    "idx02_body_joint2",
    "idx03_body_joint3",
    "idx04_body_joint4",
    "idx05_body_joint5",
]
HEAD_JOINT_NAMES = ["idx11_head_joint1", "idx12_head_joint2", "idx13_head_joint3"]
LEFT_ARM_JOINT_NAMES = [
    "idx21_arm_l_joint1",
    "idx22_arm_l_joint2",
    "idx23_arm_l_joint3",
    "idx24_arm_l_joint4",
    "idx25_arm_l_joint5",
    "idx26_arm_l_joint6",
    "idx27_arm_l_joint7",
]
RIGHT_ARM_JOINT_NAMES = [
    "idx61_arm_r_joint1",
    "idx62_arm_r_joint2",
    "idx63_arm_r_joint3",
    "idx64_arm_r_joint4",
    "idx65_arm_r_joint5",
    "idx66_arm_r_joint6",
    "idx67_arm_r_joint7",
]

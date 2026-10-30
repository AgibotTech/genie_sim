# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from geniesim_benchmark.utils.infer_post_process import (
    label_state_omnipicker,
    label_state_crsb,
    label_state_passthrough,
    process_gripper_action_relabel,
    process_gripper_action_crsb,
    process_gripper_action_passthrough,
)

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
    "idx01_body_joint1",
    "idx02_body_joint2",
    "idx03_body_joint3",
    "idx04_body_joint4",
    "idx05_body_joint5",
]

OMNIPICKER_AJ_NAMES = [
    "idx41_gripper_l_outer_joint1",
    "idx81_gripper_r_outer_joint1",
]

GP_90D_AJ_NAMES = [
    "idx32_gripper_l_outer_joint1",
    "idx72_gripper_r_outer_joint1",
]

WUJI_FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
WUJI_LEFT_GRIPPER_JOINT_NAMES = [f"hand_l_{f}_joint{j}" for f in WUJI_FINGERS for j in range(1, 5)]
WUJI_RIGHT_GRIPPER_JOINT_NAMES = [f"hand_r_{f}_joint{j}" for f in WUJI_FINGERS for j in range(1, 5)]
WUJI_AJ_NAMES = WUJI_LEFT_GRIPPER_JOINT_NAMES + WUJI_RIGHT_GRIPPER_JOINT_NAMES

G1_CHASSIS = [
    "base_linear_joint_x",
    "base_linear_joint_y",
    "base_angular_joint",
]

ROBOT_CONFIGS = {
    "G1_omnipicker": {
        "arm_joints": G1_DUAL_ARM_JOINT_NAMES,
        "left_arm_joints": G1_LEFT_ARM_JOINT_NAMES,
        "right_arm_joints": G1_RIGHT_ARM_JOINT_NAMES,
        "gripper_joints": OMNIPICKER_AJ_NAMES,
        "waist_joints": G1_WAIST_JOINT_NAMES,
        "head_joints": G1_HEAD_JOINT_NAMES,
        "gripper_offset": 0.0,
        "limit_val": 0.78,
        "label_state": label_state_omnipicker,
        "process_gripper_action": process_gripper_action_relabel,
        "obs_extra_joints": G1_HEAD_JOINT_NAMES,
        "init_gripper_open": [1.0, 1.0],
    },
    "G1_120s": {
        "arm_joints": G1_DUAL_ARM_JOINT_NAMES,
        "left_arm_joints": G1_LEFT_ARM_JOINT_NAMES,
        "right_arm_joints": G1_RIGHT_ARM_JOINT_NAMES,
        "gripper_joints": OMNIPICKER_AJ_NAMES,
        "waist_joints": G1_WAIST_JOINT_NAMES,
        "head_joints": G1_HEAD_JOINT_NAMES,
        "gripper_offset": 0.0,
        "limit_val": 0.78,
        "label_state": label_state_passthrough,
        "process_gripper_action": process_gripper_action_passthrough,
        "obs_extra_joints": G1_HEAD_JOINT_NAMES,
        "init_gripper_open": [1.0, 1.0],
    },
    "G2_omnipicker": {
        "arm_joints": G2_DUAL_ARM_JOINT_NAMES,
        "left_arm_joints": G2_LEFT_ARM_JOINT_NAMES,
        "right_arm_joints": G2_RIGHT_ARM_JOINT_NAMES,
        "gripper_joints": OMNIPICKER_AJ_NAMES,
        "waist_joints": G2_WAIST_JOINT_NAMES,
        "head_joints": G2_HEAD_JOINT_NAMES,
        "gripper_offset": 0.0,
        "gripper_scale": -1.0,
        "limit_val": 0.785,
        "label_state": label_state_crsb,
        "process_gripper_action": process_gripper_action_crsb,
        "obs_extra_joints": [],
        "init_gripper_open": [0.785, 0.785],
    },
    "G2_90d_gp": {
        "arm_joints": G2_DUAL_ARM_JOINT_NAMES,
        "left_arm_joints": G2_LEFT_ARM_JOINT_NAMES,
        "right_arm_joints": G2_RIGHT_ARM_JOINT_NAMES,
        "gripper_joints": GP_90D_AJ_NAMES,
        "waist_joints": G2_WAIST_JOINT_NAMES,
        "head_joints": G2_HEAD_JOINT_NAMES,
        "gripper_offset": -0.8,
        "limit_val": 0.78,
        "label_state": label_state_omnipicker,
        "process_gripper_action": process_gripper_action_relabel,
        "obs_extra_joints": [],
        "init_gripper_open": [1.0, 1.0],
    },
    "G2_90d": {
        "arm_joints": G2_DUAL_ARM_JOINT_NAMES,
        "left_arm_joints": G2_LEFT_ARM_JOINT_NAMES,
        "right_arm_joints": G2_RIGHT_ARM_JOINT_NAMES,
        "gripper_joints": GP_90D_AJ_NAMES,
        "waist_joints": G2_WAIST_JOINT_NAMES,
        "head_joints": G2_HEAD_JOINT_NAMES,
        "gripper_offset": 0.0,
        "limit_val": 0.78,
        "label_state": label_state_passthrough,
        "process_gripper_action": process_gripper_action_passthrough,
        "obs_extra_joints": [],
        "init_gripper_open": [1.0, 1.0],
    },
    "G2_crsB_omnipicker": {
        "arm_joints": G2_DUAL_ARM_JOINT_NAMES,
        "left_arm_joints": G2_LEFT_ARM_JOINT_NAMES,
        "right_arm_joints": G2_RIGHT_ARM_JOINT_NAMES,
        "gripper_joints": OMNIPICKER_AJ_NAMES,
        "waist_joints": G2_WAIST_JOINT_NAMES,
        "head_joints": G2_HEAD_JOINT_NAMES,
        "gripper_offset": 0.0,
        "gripper_scale": -0.785,
        "limit_val": 0.78,
        "label_state": label_state_crsb,
        "process_gripper_action": process_gripper_action_crsb,
        "obs_extra_joints": [],
        "init_gripper_open": [1.0, 1.0],
    },
    "G2_crsB_wuji": {
        "robot_type": "g2a_sim",
        "eef_type": "wuji",
        "arm_joints": G2_DUAL_ARM_JOINT_NAMES,
        "left_arm_joints": G2_LEFT_ARM_JOINT_NAMES,
        "right_arm_joints": G2_RIGHT_ARM_JOINT_NAMES,
        "gripper_joints": WUJI_AJ_NAMES,
        "waist_joints": G2_WAIST_JOINT_NAMES,
        "head_joints": G2_HEAD_JOINT_NAMES,
        "gripper_offset": 0.0,
        "limit_val": 0.78,
        "label_state": label_state_passthrough,
        "process_gripper_action": process_gripper_action_passthrough,
        "init_gripper_open": [0.0] * len(WUJI_AJ_NAMES),
    },
    "G2_crsB_90d": {
        "robot_type": "g2a_sim",
        "eef_type": "zhiyuan_gripper_ctek90d",
        "arm_joints": G2_DUAL_ARM_JOINT_NAMES,
        "left_arm_joints": G2_LEFT_ARM_JOINT_NAMES,
        "right_arm_joints": G2_RIGHT_ARM_JOINT_NAMES,
        "gripper_joints": GP_90D_AJ_NAMES,
        "waist_joints": G2_WAIST_JOINT_NAMES,
        "head_joints": G2_HEAD_JOINT_NAMES,
        "gripper_offset": 0.0,
        "limit_val": 0.78,
        "label_state": label_state_passthrough,
        "process_gripper_action": process_gripper_action_passthrough,
        "obs_extra_joints": [],
        "init_gripper_open": [1.0, 1.0],
    },
}

DEFAULT_ROBOT_CONFIG = {
    "gripper_offset": 0.0,
    "gripper_scale": 1.0,
    "limit_val": 0.78,
    "label_state": label_state_passthrough,
    "process_gripper_action": process_gripper_action_passthrough,
}


def robot_type_mapping(robot_type):
    if "G1_120s" in robot_type:
        return "G1_120s"
    elif "G1_omnipicker" in robot_type:
        return "G1_omnipicker"
    elif "G2_omnipicker" in robot_type:
        return "G2_omnipicker"
    elif "G2_90d_gp" in robot_type:
        return "G2_90d_gp"
    elif "G2_90d" in robot_type:
        return "G2_90d"
    elif "G2_crsB_omnipicker" in robot_type:
        return "G2_crsB_omnipicker"
    elif "G2_crsB_90d" in robot_type:
        return "G2_crsB_90d"
    elif "G2_crsB_wuji" in robot_type:
        return "G2_crsB_wuji"
    else:
        raise ValueError(f"Invalid robot type: {robot_type}")

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

# wuji five-finger dexterous hand: 5 fingers x 4 joints per hand, no mimic
# coupling. Full-dexterous control drives every finger joint directly.
# Names taken from assets/robot/G2_crsB_wuji{,_old}/robot_fix.usda; the finger
# order is thumb->pinky, matching the aimdk hand protocol.
WUJI_FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
WUJI_LEFT_GRIPPER_JOINT_NAMES = [f"hand_l_{f}_joint{j}" for f in WUJI_FINGERS for j in range(1, 5)]
WUJI_RIGHT_GRIPPER_JOINT_NAMES = [f"hand_r_{f}_joint{j}" for f in WUJI_FINGERS for j in range(1, 5)]
WUJI_AJ_NAMES = WUJI_LEFT_GRIPPER_JOINT_NAMES + WUJI_RIGHT_GRIPPER_JOINT_NAMES

G1_CHASSIS = [
    "base_linear_joint_x",
    "base_linear_joint_y",
    "base_angular_joint",
]

# arx / agilex: single-right-arm-driven 6-DOF platforms, registered for
# abs_joint replay only (no Lula/curobo IK configured for either — see
# app/robot_cfg/{arx,agilex}/config.yaml). Both arms exist in the URDF but
# only the right arm is ever commanded; the left is a fixed decorative dual.
ARX_LEFT_ARM_JOINT_NAMES = [
    "arm_l_joint1",
    "arm_l_joint2",
    "arm_l_joint3",
    "arm_l_joint4",
    "arm_l_joint5",
    "arm_l_joint6",
]
ARX_RIGHT_ARM_JOINT_NAMES = [
    "arm_r_joint1",
    "arm_r_joint2",
    "arm_r_joint3",
    "arm_r_joint4",
    "arm_r_joint5",
    "arm_r_joint6",
]
ARX_DUAL_ARM_JOINT_NAMES = ARX_LEFT_ARM_JOINT_NAMES + ARX_RIGHT_ARM_JOINT_NAMES
ARX_GRIPPER_JOINT_NAMES = [
    "gripper_l_active_joint",
    "gripper_r_active_joint",
]

AGILEX_LEFT_ARM_JOINT_NAMES = [
    "arm_fl_joint1",
    "arm_fl_joint2",
    "arm_fl_joint3",
    "arm_fl_joint4",
    "arm_fl_joint5",
    "arm_fl_joint6",
]
AGILEX_RIGHT_ARM_JOINT_NAMES = [
    "arm_fr_joint1",
    "arm_fr_joint2",
    "arm_fr_joint3",
    "arm_fr_joint4",
    "arm_fr_joint5",
    "arm_fr_joint6",
]
AGILEX_DUAL_ARM_JOINT_NAMES = AGILEX_LEFT_ARM_JOINT_NAMES + AGILEX_RIGHT_ARM_JOINT_NAMES
AGILEX_GRIPPER_JOINT_NAMES = [
    "gripper_fl_active_joint",
    "gripper_fr_active_joint",
]

ROBOT_CONFIGS = {
    "G1_omnipicker": {
        "arm_joints": G1_DUAL_ARM_JOINT_NAMES,
        "left_arm_joints": G1_LEFT_ARM_JOINT_NAMES,
        "right_arm_joints": G1_RIGHT_ARM_JOINT_NAMES,
        "gripper_joints": OMNIPICKER_AJ_NAMES,
        "waist_joints": G1_WAIST_JOINT_NAMES,
        "head_joints": G1_HEAD_JOINT_NAMES,
        # G1 and G2 carry the same omnipicker, so they share one gripper
        # convention: the raw joint angle in rad, sign-flipped. idx41/idx81
        # travel 0 rad (closed) .. 0.785 rad (open), so state is sent as
        # -0.785 (open) .. 0.0 (closed) and actions come back in the same
        # units. scale -1.0 makes label_state_crsb/process_gripper_action_crsb
        # a plain sign flip, and the pair is self-inverse.
        "gripper_offset": 0.0,
        "gripper_scale": -1.0,
        "limit_val": 0.785,
        "label_state": label_state_crsb,
        "process_gripper_action": process_gripper_action_crsb,
        "obs_extra_joints": G1_HEAD_JOINT_NAMES,
        "init_gripper_open": [0.785, 0.785],
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
        # Every G2 sends robot_type "g2a_sim", so ee_type is what tells the policy
        # server this is a five-finger hand (40 DOF) and not a 2-DOF gripper.
        "eef_type": "wuji",
        "arm_joints": G2_DUAL_ARM_JOINT_NAMES,
        "left_arm_joints": G2_LEFT_ARM_JOINT_NAMES,
        "right_arm_joints": G2_RIGHT_ARM_JOINT_NAMES,
        "gripper_joints": WUJI_AJ_NAMES,
        "waist_joints": G2_WAIST_JOINT_NAMES,
        "head_joints": G2_HEAD_JOINT_NAMES,
        "gripper_offset": 0.0,
        "limit_val": 0.78,
        # Full-dexterous: policy emits 40 finger-joint targets (20/hand), applied
        # directly. No open/close relabel or scaling — passthrough both ways.
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
    "arx": {
        "robot_type": "arx_sim",
        "arm_joints": ARX_DUAL_ARM_JOINT_NAMES,
        "left_arm_joints": ARX_LEFT_ARM_JOINT_NAMES,
        "right_arm_joints": ARX_RIGHT_ARM_JOINT_NAMES,
        "gripper_joints": ARX_GRIPPER_JOINT_NAMES,
        "waist_joints": [],
        "head_joints": [],
        "obs_extra_joints": [],
        "gripper_offset": 0.0,
        # IsaacSim gripper travel is [0 closed, 0.044 open]; training data uses
        # [0 closed, -3.5 open]. Reuses the crsB scale path: obs *= scale, action /= scale.
        "gripper_scale": -3.5 / 0.044,
        "limit_val": 0.044,
        "label_state": label_state_crsb,
        "process_gripper_action": process_gripper_action_crsb,
        "init_gripper_open": [0.044, 0.044],
    },
    "agilex": {
        "robot_type": "agilex_sim",
        "arm_joints": AGILEX_DUAL_ARM_JOINT_NAMES,
        "left_arm_joints": AGILEX_LEFT_ARM_JOINT_NAMES,
        "right_arm_joints": AGILEX_RIGHT_ARM_JOINT_NAMES,
        "gripper_joints": AGILEX_GRIPPER_JOINT_NAMES,
        "waist_joints": [],
        "head_joints": [],
        "obs_extra_joints": [],
        "gripper_offset": 0.0,
        # Same scale trick as arx, travel is [0 closed, 0.04 open] here.
        "gripper_scale": -3.5 / 0.04,
        "limit_val": 0.04,
        "label_state": label_state_crsb,
        "process_gripper_action": process_gripper_action_crsb,
        "init_gripper_open": [0.04, 0.04],
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
    elif "agilex" in robot_type:
        return "agilex"
    elif "arx" in robot_type:
        return "arx"
    else:
        raise ValueError(f"Invalid robot type: {robot_type}")

# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import json

import numpy as np


class Robot:
    def __init__(self, robot_name):
        self.robot_name = robot_name

    def reset(self, uibuilder):
        pass


class RobotCfg(Robot):
    def __init__(self, cfg_file):
        with open(cfg_file, "r") as f:
            robot_cfg = json.load(f)
        # init robot
        super(RobotCfg, self).__init__(robot_cfg["robot"]["robot_name"])
        self.robot_prim_path = robot_cfg["robot"]["base_prim_path"]
        self.urdf_name = robot_cfg["robot"]["urdf_name"]
        self.arm_type = robot_cfg["robot"]["arm"]
        if robot_cfg["robot"]["arm"] == "single":
            self.is_single = True
        else:
            self.is_single = False
        if "active_arm_joints" in robot_cfg["robot"]:
            self.active_arm_joints = robot_cfg["robot"]["active_arm_joints"]
        else:
            self.active_arm_joints = None
        self.robot_usd = robot_cfg["robot"]["robot_usd"]
        self.robot_description_path = robot_cfg["robot"]["robot_description"]
        self.dof_nums = robot_cfg["robot"]["dof_nums"]
        self.joint_delta_time = robot_cfg["robot"]["joint_delta_time"]
        self.lock_joints = robot_cfg["robot"]["lock_joints"]
        self.init_joint_position = robot_cfg["robot"]["init_joint_position"]
        # init camera
        self.cameras = robot_cfg["camera"]
        # init gripper
        self.gripper_type = robot_cfg["gripper"]["gripper_type"]
        self.gripper_max_force = robot_cfg["gripper"]["max_force"]
        gripper_names = robot_cfg["gripper"].get("gripper_name", {"left": "omnipicker", "right": "omnipicker"})
        self.left_gripper_name = gripper_names["left"]
        self.right_gripper_name = gripper_names["right"]
        if robot_cfg["robot"]["arm"] == "dual":

            self.end_effector_name = robot_cfg["gripper"]["end_effector_name"]
        elif robot_cfg["robot"]["arm"] == "right":
            self.end_effector_name = robot_cfg["gripper"]["end_effector_name"]["right"]
        else:
            self.end_effector_name = robot_cfg["gripper"]["end_effector_name"]["left"]
        self.end_effector_prim_path = robot_cfg["gripper"]["end_effector_prim_path"]
        if "end_effector_center_prim_path" in robot_cfg["gripper"]:
            self.end_effector_center_prim_path = robot_cfg["gripper"]["end_effector_center_prim_path"]
        else:
            self.end_effector_center_prim_path = self.end_effector_prim_path
        if "arm_base_prim_path" in robot_cfg["robot"]:
            self.arm_base_prim_path = robot_cfg["robot"]["arm_base_prim_path"]
        else:
            self.arm_base_prim_path = self.robot_prim_path
        self.finger_names = robot_cfg["gripper"]["finger_names"]
        self.gripper_controll_joint = robot_cfg["gripper"]["gripper_controll_joint"]
        self.opened_positions = robot_cfg["gripper"]["opened_positions"]
        self.closed_velocities = robot_cfg["gripper"]["closed_velocities"]
        self.closed_positions = robot_cfg["closed_positions"]
        self.action_deltas = np.array([-0.1, -0.1])
        # init curobo
        self.curobo_config_file = robot_cfg["curobo"]["curobo_config_file"]
        self.curobo_urdf_path = robot_cfg["curobo"]["curobo_urdf_path"]
        self.curobo_urdf_name = robot_cfg["curobo"]["curobo_urdf_name"]

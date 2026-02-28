# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import os
import sys

import ik_solver
import numpy as np
from scipy.spatial.transform import Rotation as R

current_dir = os.path.dirname(os.path.abspath(__file__))


def xyzquat_to_xyzrpy(xyzquat):
    xyz = xyzquat[:3]
    rpy = R.from_quat(xyzquat[3:], scalar_first=True).as_euler("xyz", degrees=False)
    xyzrpy = np.concatenate([xyz, rpy])
    return xyzrpy


def xyzrpy_to_xyzquat(xyzrpy):
    xyz = xyzrpy[:3]
    quat = R.from_euler("xyz", xyzrpy[3:]).as_quat(scalar_first=False)
    xyzquat = np.concatenate([xyz, quat])
    return xyzquat


def xyzrpy2mat(xyzrpy):
    rot = R.from_euler("xyz", xyzrpy[3:6]).as_matrix()
    mat = np.eye(4)
    mat[0:3, 0:3] = rot
    mat[0:3, 3] = xyzrpy[0:3]
    return mat


def mat2xyzrpy(mat):
    rpy = R.from_matrix(mat[0:3, 0:3]).as_euler("xyz", degrees=False)
    xyz = mat[0:3, 3]
    xyzrpy = np.concatenate([xyz, rpy])
    return xyzrpy


class IKFKSolver:
    def __init__(self, arm_init_joint_position, head_init_position, waist_init_position, robot_cfg="G1_omnipicker"):
        if "G2" in robot_cfg:
            urdf_name, config_name = "G2_NO_GRIPPER.urdf", "g2_solver.yaml"
        else:
            urdf_name, config_name = "G1_NO_GRIPPER.urdf", "g1_solver.yaml"

        sdk_dir = os.path.join(current_dir, "IK-SDK")
        urdf_path = os.path.join(sdk_dir, urdf_name)
        config_path = os.path.join(sdk_dir, config_name)

        self.left_solver = ik_solver.Solver(
            part=ik_solver.RobotPart.LEFT_ARM,
            urdf_path=urdf_path,
            config_path=config_path,
        )
        self.right_solver = ik_solver.Solver(
            part=ik_solver.RobotPart.RIGHT_ARM,
            urdf_path=urdf_path,
            config_path=config_path,
        )

        self.left_solver.set_debug_mode(False)
        self.right_solver.set_debug_mode(False)
        self.left_solver.sync_target_with_joints(arm_init_joint_position[:7])
        self.right_solver.sync_target_with_joints(arm_init_joint_position[7:14])

    def eef_actions_to_joint(self, eef_actions, arm_joint_states, head_init_position):
        joint_actions = []
        self.left_solver.sync_target_with_joints(arm_joint_states[:7])
        self.right_solver.sync_target_with_joints(arm_joint_states[7:14])

        for _, action in enumerate(eef_actions):
            eefrot_left_cur = np.array(action[:6], dtype=np.float32)
            eefrot_right_cur = np.array(action[6:12], dtype=np.float32)

            target_pos_left = xyzrpy_to_xyzquat(eefrot_left_cur)
            target_pos_right = xyzrpy_to_xyzquat(eefrot_right_cur)
            self.left_solver.update_target_quat(
                target_pos=target_pos_left[:3],
                target_quat=target_pos_left[3:],
            )
            self.right_solver.update_target_quat(
                target_pos=target_pos_right[:3],
                target_quat=target_pos_right[3:],
            )

            left_joints = self.left_solver.solve()
            right_joints = self.right_solver.solve()

            l_gripper = action[12:13] if type(action) == list else action[12:13].tolist()
            r_gripper = action[13:14] if type(action) == list else action[13:14].tolist()
            joint_actions.append(left_joints.tolist() + right_joints.tolist() + l_gripper + r_gripper)

        return joint_actions

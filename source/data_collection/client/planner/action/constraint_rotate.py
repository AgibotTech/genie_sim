# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import numpy as np

from client.layout.object import OmniObject
from client.planner.action.stage import Action, ActionSequence, Stage
from client.planner.func.sort_pose.sort_pose import sorted_by_position_humanlike
from client.robot.omni_robot import IsaacSimRpcRobot
from common.base_utils.logger import logger
from common.base_utils.transform_utils import calculate_rotation_matrix2, mat2quat_wxyz, rotate_along_axis, quat_multiplication_wxyz

class ConstraintRotateStage(Stage):
    def __init__(self, stage_config, objects):
        super().__init__(stage_config, objects)

    def select_pose(self, objects: list[OmniObject], robot: IsaacSimRpcRobot):
        arm = self.extra_params.get("arm", "right")
        path_constraint = np.array([0, 0, 0, 0.1, 0.1, 0.1])
        constraint_axises = self.extra_params.get("constraint_axises", [])
        for constraint_axis in constraint_axises:
            if constraint_axis == "x":
                path_constraint[0] = 0.1
            elif constraint_axis == "y":
                path_constraint[1] = 0.1
            elif constraint_axis == "z":
                path_constraint[2] = 0.1
            else:
                logger.error(f"Invalid constraint_axis: {constraint_axis}")
                return []
        if all(path_constraint != 0):
            logger.error("All path constraint are 0, no constraint rotate")
            return []
        
        gripper_pose = robot.get_ee_pose(ee_type="gripper", id=arm)
        pre_pose = objects[self.active_obj_id].obj_pose
        target_pose = self.extra_params.get("target_pose", None)
        if target_pose is None:
            logger.error("target_pose is None")
            return []
        if len(target_pose.shape) == 2:
            target_pose = target_pose[np.newaxis, ...]
        aligned_target_poses = []
        for single_target_pose in target_pose:
            gripper2obj = np.linalg.inv(pre_pose) @ gripper_pose
            target_gripper_pose = single_target_pose @ gripper2obj
            aligned_target_pose = target_gripper_pose.copy()
            aligned_target_pose[:3, 3] = gripper_pose[:3, 3]
            aligned_target_poses.append(aligned_target_pose)
        aligned_target_poses = np.array(aligned_target_poses)
        ik_success, ik_info = robot.solve_ik(
            aligned_target_poses, ee_type="gripper", type="Simple", arm=arm
        )
        aligned_target_poses = aligned_target_poses[ik_success]
        logger.info(
            f"Constraint Rotate, {self.passive_obj_id}, Filtered target pose with isaac-sim IK: {aligned_target_poses.shape[0]}/{ik_success.shape[0]}"
        )
        if aligned_target_poses.shape[0] == 0:
            logger.info(f"Constraint Rotate, {self.passive_obj_id}, No target pose can pass isaac-sim IK")
            return []
        ik_success, ik_info = robot.solve_ik(
            aligned_target_poses,
            ee_type="gripper",
            type="AvoidObs",
            arm=arm,
            output_link_pose=True,
        )
        aligned_target_poses = aligned_target_poses[ik_success]
        logger.info(
            f"Constraint Rotate, {self.passive_obj_id}, Filtered target pose with isaac-sim IK: {aligned_target_poses.shape[0]}/{ik_success.shape[0]}"
        )
        if aligned_target_poses.shape[0] == 0:
            logger.info(f"Constraint Rotate, {self.passive_obj_id}, No target pose can pass isaac-sim IK")
            return []
        result = []
        for aligned_target_pose in aligned_target_poses:
            result.append(
                {
                    "target_pose": aligned_target_pose,
                    "path_constraint": path_constraint,
                }
            )
        return result

    def generate_action_sequence(self, target_pose, path_constraint=None):
        action_sequence = ActionSequence()
        action_sequence.add_action(
            Action(
                target_pose,
                None,
                np.eye(4),
                "AvoidObs",
                extra_params={
                    "path_constraint": path_constraint,
                    "use_world_pose": True,
                    "offset_and_constraint_in_goal_frame":self.extra_params.get("offset_and_constraint_in_goal_frame", False)
                },
            )
        )

        return action_sequence

    def check_completion(self, objects, robot=None):
        assert (
            self.active_action_sequence is not None
        ), f"Active action for stage {self.action_type} is None"
        return True

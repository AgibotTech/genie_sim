# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import copy

import numpy as np

from client.planner.action.stage import Action, ActionSequence, Stage
from client.planner.common import get_aligned_fix_pose, get_aligned_pose
from client.planner.func.common import random_downsample, sorted_by_joint_pos_dist
from client.planner.func.sort_pose.sort_pose import sorted_by_position_humanlike
from common.base_utils.logger import logger
from common.base_utils.transform_utils import add_random_noise_to_pose


# When the grasp point of the current target cannot be reached, find points near the target point
def find_near_point_grasp_pose(
    robot,
    arm,
    target_gripper_poses,
    offset_range=np.linspace(0, 0.2, 20),
    offset_axis="x",
):
    new_target_gripper_poses = copy.deepcopy(target_gripper_poses)
    find_near_point_taget_pose = False
    for direction in [-1, 1]:
        for offset in offset_range:
            if offset_axis == "x":
                new_target_gripper_poses[:, 0, 3] += offset * direction
            elif offset_axis == "y":
                new_target_gripper_poses[:, 1, 3] += offset * direction
            elif offset_axis == "z":
                new_target_gripper_poses[:, 2, 3] += offset * direction
            ik_success, ik_info = robot.solve_ik(
                new_target_gripper_poses,
                ee_type="gripper",
                type="AvoidObs",
                arm=arm,
                output_link_pose=True,
            )
            target_gripper_poses_pass_ik = new_target_gripper_poses[ik_success]
            ik_joint_positions = ik_info["joint_positions"][ik_success]
            ik_joint_names = ik_info["joint_names"][ik_success]
            ik_jacobian_score = ik_info["jacobian_score"][ik_success]
            if len(target_gripper_poses_pass_ik) != 0:
                find_near_point_taget_pose = True
                break
        if find_near_point_taget_pose:
            break
    if find_near_point_taget_pose:
        return (
            target_gripper_poses_pass_ik,
            ik_joint_positions,
            ik_joint_names,
            ik_jacobian_score,
        )
    else:
        return [], [], [], []


class PlaceStage(Stage):
    def __init__(self, stage_config, objects):
        super().__init__(stage_config, objects)
        self.place_transform_up = np.array([0, 0, 0.01])
        self.use_pre_place = self.extra_params.get("use_pre_place", False)
        self.pre_place_offset = self.extra_params.get("pre_place_offset", 0.12)

    def select_pose(self, objects, robot):
        object_pose = objects[self.active_obj_id].obj_pose
        arm = self.extra_params.get("arm", "right")
        ee_pose = robot.get_ee_pose(ee_type="gripper", id=arm)
        gripper2obj = np.linalg.inv(object_pose) @ ee_pose
        passive_elements = self.passive_element
        active_elements = self.active_element
        if not isinstance(self.active_element, list):
            active_elements = [self.active_element]
        active_obj = objects[self.active_obj_id]
        passive_obj = objects[self.passive_obj_id]
        target_obj_pose_canonical_sorted = []
        for active_element in active_elements:
            target_gripper_poses = np.zeros((0, 4, 4))
            active_obj.obj_pose
            anchor_pose = passive_obj.obj_pose
            active_obj.update_aligned_info(active_element)
            for passive_element in passive_elements:
                # interaction between two rigid objects
                passive_obj.update_aligned_info(passive_element)
                # N_align equals the least common multiple of passive_obj.angle_sample_num and active_obj.angle_sample_num
                N_align = np.lcm(passive_obj.angle_sample_num, active_obj.angle_sample_num)
                if active_obj.name == "gripper":
                    gripper2obj = np.eye(4)

                if "fix_pose" in self.passive_obj_id:
                    target_obj_poses = get_aligned_fix_pose(active_obj, passive_obj, N=N_align)
                else:
                    target_obj_poses = get_aligned_pose(active_obj, passive_obj, N=N_align)

                target_gripper_poses = np.concatenate(
                    (
                        target_gripper_poses,
                        target_obj_poses @ gripper2obj[np.newaxis, ...],
                    ),
                    axis=0,
                )
            disable_upside_down = self.extra_params.get("disable_upside_down", False)
            if disable_upside_down:
                if disable_upside_down:
                    if "omnipicker" in robot.robot_cfg:
                        if arm == "left":
                            upright_mask = target_gripper_poses[:, 2, 1] < 0.0
                        else:
                            upright_mask = target_gripper_poses[:, 2, 1] > 0.0
                    else:
                        upright_mask = target_gripper_poses[:, 2, 0] > 0.0
                target_gripper_poses = target_gripper_poses[upright_mask]
            if not target_gripper_poses.shape[0]:
                logger.warning(
                    f"{self.action_type}: No target_gripper_poses can pass upright filter"
                )
                continue

            # downsample target_gripper_poses
            target_gripper_poses, _ = random_downsample(
                transforms=target_gripper_poses, downsample_num=100, replace=False
            )
            ik_success, _ = robot.solve_ik(
                target_gripper_poses,
                ee_type="gripper",
                type="Simple",
                arm=arm,
            )
            target_gripper_poses = target_gripper_poses[ik_success]

            if len(target_gripper_poses) == 0:
                logger.warning(f"{self.action_type}: No target_obj_pose can pass isaac-sim IK")
                continue
            ik_success, ik_info = robot.solve_ik(
                target_gripper_poses,
                ee_type="gripper",
                type="AvoidObs",
                arm=arm,
                output_link_pose=True,
            )

            target_gripper_poses_pass_ik = target_gripper_poses[ik_success]
            ik_joint_positions = ik_info["joint_positions"][ik_success]
            ik_joint_names = ik_info["joint_names"][ik_success]
            ik_jacobian_score = ik_info["jacobian_score"][ik_success]
            ik_link_poses = ik_info["link_poses"][ik_success]
            if len(target_gripper_poses_pass_ik) == 0:
                if self.extra_params.get("use_near_point", False):
                    logger.warning(
                        f"{self.action_type}: No target_obj_pose can pass isaac curobo IK"
                    )
                    logger.info("-" * 20 + " use near point " + "-" * 20)
                    if "fix_pose" in self.passive_obj_id:
                        target_obj_poses = get_aligned_fix_pose(active_obj, passive_obj, N=12)
                    else:
                        target_obj_poses = get_aligned_pose(active_obj, passive_obj, N=12)
                    target_gripper_poses = np.concatenate(
                        (
                            target_gripper_poses,
                            target_obj_poses @ gripper2obj[np.newaxis, ...],
                        ),
                        axis=0,
                    )

                    find_near_point_grasp_pose_success = False
                    for offset_axis in ["x", "y", "z"]:
                        logger.info(f"search {offset_axis}-axis near point")
                        (
                            target_gripper_poses_pass_ik,
                            ik_joint_positions,
                            ik_joint_names,
                            ik_jacobian_score,
                        ) = find_near_point_grasp_pose(
                            robot,
                            arm,
                            target_gripper_poses,
                            offset_range=np.linspace(0, 0.2, 5),
                            offset_axis=offset_axis,
                        )
                        if len(target_gripper_poses_pass_ik) != 0:
                            find_near_point_grasp_pose_success = True
                            break
                        else:
                            logger.warning(
                                f"{self.action_type}: No {offset_axis}-axis near target_obj_pose can pass isaac curobo IK"
                            )
                    if not find_near_point_grasp_pose_success:
                        logger.warning(
                            f"{self.action_type}: No any near target_obj_pose from any axis can pass isaac curobo IK"
                        )
                        continue
                    else:
                        logger.info("find near point grasp pose success")
                else:
                    logger.info(
                        f"Unable to find valid target_obj_pose for {self.action_type} action, try next active/passive element combination."
                    )
                    continue

            # filter by pre place pose
            pre_insert_pose_canonical = None
            if self.use_pre_place:
                target_obj_pose = (
                    target_gripper_poses_pass_ik @ np.linalg.inv(gripper2obj)[np.newaxis, ...]
                )
                target_obj_pose_canonical = (
                    np.linalg.inv(anchor_pose)[np.newaxis, ...] @ target_obj_pose
                )
                self.pre_insert_offset = self.extra_params.get("pre_insert_offset", 0.1)
                # Calculate direction perpendicular to placement point
                # We assume that the passive object has only one placement direction, and all passive_elements have the same direction
                normal_direction = np.array(self.passive_element[0]["direction"])
                # Create transformation matrix for pre-placement point
                # Pre-placement point is in the direction perpendicular to the placement point, at distance pre_insert_offset
                pre_insert_pose_canonical = target_obj_pose_canonical.copy()
                pre_insert_pose_canonical[:, :3, 3] += -normal_direction * self.pre_insert_offset
                MAX_TRY_TIMES = 5
                MIN_REMAIN_POSE_NUM = 5

                def add_noise(
                    indices,
                    pose_canonical,
                    origin_pose_canonical,
                    position_noise,
                    rotation_noise,
                ):
                    for i in indices:
                        pose_canonical[i] = add_random_noise_to_pose(
                            origin_pose_canonical[i],
                            rot_noise=rotation_noise,
                            pos_noise=position_noise,
                        )
                    return pose_canonical

                origin_pre_insert_pose_canonical = pre_insert_pose_canonical.copy()
                if "pre_pose_noise" in self.extra_params:
                    position_noise = self.extra_params["pre_pose_noise"].get("position_noise", 0)
                    rotation_noise = self.extra_params["pre_pose_noise"].get("rotation_noise", 0)
                    pre_insert_pose_canonical = add_noise(
                        range(len(pre_insert_pose_canonical)),
                        pre_insert_pose_canonical,
                        origin_pre_insert_pose_canonical,
                        position_noise,
                        rotation_noise,
                    )
                pre_insert_obj_pose = anchor_pose[np.newaxis, ...] @ pre_insert_pose_canonical
                pre_insert_gripper_pose = pre_insert_obj_pose @ gripper2obj[np.newaxis, ...]
                ik_success_pre, ik_info_pre = robot.solve_ik(
                    pre_insert_gripper_pose,
                    ee_type="gripper",
                    type="AvoidObs",
                    arm=arm,
                    output_link_pose=False,
                )
                # if noise makes the pre insert pose not reachable, try another noise or finally discard noise
                if (
                    not ik_success_pre.all()
                    and "pre_pose_noise" in self.extra_params
                    and ik_success_pre.sum() < MIN_REMAIN_POSE_NUM
                ):
                    for try_time in range(MAX_TRY_TIMES):
                        logger.info(f"try {try_time} times to add noise to pre insert pose")
                        if try_time == MAX_TRY_TIMES - 1:
                            logger.error(
                                "try to add noise to pre insert pose failed, use original pose"
                            )
                            position_noise = 0
                            rotation_noise = 0
                        pre_insert_pose_canonical = add_noise(
                            np.where(~ik_success_pre)[0],
                            pre_insert_pose_canonical,
                            origin_pre_insert_pose_canonical,
                            position_noise,
                            rotation_noise,
                        )
                        pre_insert_obj_pose = (
                            anchor_pose[np.newaxis, ...] @ pre_insert_pose_canonical
                        )
                        pre_insert_gripper_pose = pre_insert_obj_pose @ gripper2obj[np.newaxis, ...]
                        ik_success_pre, ik_info_pre = robot.solve_ik(
                            pre_insert_gripper_pose,
                            ee_type="gripper",
                            type="AvoidObs",
                            arm=arm,
                            output_link_pose=False,
                        )
                        if ik_success_pre.sum() >= MIN_REMAIN_POSE_NUM or ik_success_pre.all():
                            break
                target_gripper_poses_pass_ik = target_gripper_poses_pass_ik[ik_success_pre]
                ik_joint_positions = ik_joint_positions[ik_success_pre]
                ik_joint_names = ik_joint_names[ik_success_pre]
                ik_jacobian_score = ik_jacobian_score[ik_success_pre]
                ik_link_poses = ik_link_poses[ik_success_pre]
                pre_insert_gripper_pose = pre_insert_gripper_pose[ik_success_pre]
                pre_insert_pose_canonical = pre_insert_pose_canonical[ik_success_pre]
            if len(target_gripper_poses_pass_ik) == 0:
                logger.warning(f"{self.action_type}: No target_obj_pose can pass curobo IK")
                continue
            if "G2" in robot.robot_cfg:
                elbow_name = "arm_r_link4" if arm == "right" else "arm_l_link4"
                hand_name = "gripper_r_center_link" if arm == "right" else "gripper_l_center_link"
                idx_sorted = sorted_by_position_humanlike(
                    joint_positions=ik_joint_positions,
                    joint_names=ik_joint_names,
                    link_poses=ik_link_poses,
                    is_right=arm == "right",
                    elbow_name=elbow_name,
                    hand_name=hand_name,
                )
            else:
                idx_sorted = sorted_by_joint_pos_dist(
                    robot, arm, ik_joint_positions, ik_joint_names, ik_jacobian_score
                )

            target_obj_pose_sorted = (
                target_gripper_poses_pass_ik[idx_sorted]
                @ np.linalg.inv(gripper2obj)[np.newaxis, ...]
            )
            target_obj_pose_canonical_sorted = (
                np.linalg.inv(anchor_pose)[np.newaxis, ...] @ target_obj_pose_sorted
            )
            if pre_insert_pose_canonical is not None:
                pre_insert_pose_canonical = pre_insert_pose_canonical[idx_sorted]
        result = []
        for i in range(len(target_obj_pose_canonical_sorted)):
            tmp_result = {}
            tmp_result["grasp_pose"] = target_obj_pose_canonical_sorted[i]
            if pre_insert_pose_canonical is not None:
                tmp_result["pre_insert_pose"] = pre_insert_pose_canonical[i]
            result.append(tmp_result)
        return result

    def generate_action_sequence(self, grasp_pose):
        action_sequence = ActionSequence()
        target_pose_canonical = grasp_pose
        gripper_cmd = self.extra_params.get("gripper_state", "open")
        error_type = ""
        if "error_data" in self.extra_params:
            error_data = self.extra_params["error_data"]
            error_data.get("params", {})
            error_type = error_data.get("type", "")
            if error_type == "KeepClose":
                gripper_cmd = None
        self.extra_params.get("pre_place_direction", "z")
        post_place_action = self.extra_params.get(
            "post_place_action", None
        )  # place object without collision check, e.g., place on a surface
        palce_transform_up = np.eye(4)
        palce_transform_up[:3, 3] = self.place_transform_up
        action_sequence.add_action(
            Action(target_pose_canonical, None, palce_transform_up, "AvoidObs")
        )
        if post_place_action is not None:
            for post_action in post_place_action:
                post_place_gripper_cmd = post_action.get("gripper_cmd", None)
                if post_place_gripper_cmd is not None:
                    action_sequence.add_action(
                        Action(
                            target_pose_canonical,
                            post_place_gripper_cmd,
                            np.eye(4),
                            "Simple",
                        )
                    )
                post_place_distance = post_action.get("distance", 0.02)
                post_place_direction = np.array(
                    post_action.get("direction", [0, 0, 1])
                )  # in passive object local frame
                target_pose_canonical = target_pose_canonical.copy()
                target_pose_canonical[:3, 3] += (
                    post_place_direction
                    * post_place_distance
                    / np.linalg.norm(post_place_direction)
                )
                action_sequence.add_action(
                    Action(target_pose_canonical, gripper_cmd, np.eye(4), "Simple")
                )
        else:
            action_sequence.add_action(Action(None, gripper_cmd, np.eye(4), "Simple"))
        return action_sequence

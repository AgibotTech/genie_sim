# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import time

import numpy as np

from client.planner.action.stage import (
    PLACE_LIKE_ACTIONS,
    Action,
    ActionSequence,
    Stage,
    simple_check_completion,
)
from client.planner.common import (
    generate_random_pose,
    get_aligned_fix_pose,
    get_aligned_pose,
    overweite_grasp_data,
)
from client.planner.func.common import (
    filter_grasp_pose_by_gripper_up_direction,
    filter_grasp_poses_with_humanlike_posture,
    random_downsample,
    sorted_by_joint_pos_dist_and_grasp_pose,
)
from client.planner.func.sort_pose.sort_pose import sorted_by_position_humanlike
from common.base_utils.logger import logger
from common.base_utils.transform_utils import (
    calculate_rotation_matrix2,
    pose_from_position_quaternion,
    rotate_along_axis,
)


class PickStage(Stage):
    def __init__(self, stage_config, objects):
        super().__init__(stage_config, objects)
        self.use_pre_grasp = True
        self.pick_up_step = 999

    def select_pose(self, objects, robot):
        gripper2obj = []
        arm = self.extra_params.get("arm", "right")
        grasp_offset = self.extra_params.get("grasp_offset", 0.03)
        pre_grasp_offset = self.extra_params.get("pre_grasp_offset", 0.0)
        grasp_lower_percentile = self.extra_params.get("grasp_lower_percentile", 0)
        grasp_upper_percentile = self.extra_params.get("grasp_upper_percentile", 100)
        disable_upside_down = self.extra_params.get("disable_upside_down", False)
        flip_grasp = self.extra_params.get("flip_grasp", False)
        error_data = self.extra_params.get("error_data", {})
        error_type = error_data.get("type", None)
        set_grasp_pose_xy = self.extra_params.get("set_grasp_pose_xy", False)
        set_grasp_vertical = self.extra_params.get("set_grasp_vertical", False)

        """Select grasp points from Stage class"""
        """Filter out grasp poses without IK solutions"""
        grasp_poses_canonical = self.passive_element["grasp_pose"].copy()
        grasp_widths = self.passive_element["width"]
        if set_grasp_pose_xy:
            grasp_poses_canonical[:, 0, 3] = 0
            grasp_poses_canonical[:, 2, 3] = 0

        z_values = grasp_poses_canonical[:, 1, 3]
        z_lower_threshold = np.percentile(z_values, grasp_lower_percentile)
        z_upper_threshold = np.percentile(z_values, grasp_upper_percentile)
        grasp_poses_canonical_bak = grasp_poses_canonical.copy()
        # filter grasp pose with z_min and z_max value
        mask = (z_values <= z_upper_threshold) & (z_values >= z_lower_threshold)
        grasp_poses_canonical = grasp_poses_canonical_bak[mask]
        grasp_widths = grasp_widths[mask]
        grasp_poses_canonical[:, :3, :3] = (
            grasp_poses_canonical[:, :3, :3] @ robot.robot_gripper_2_grasp_gripper[np.newaxis, ...]
        )
        if set_grasp_vertical:
            # Directly align local y-axis positive direction with coordinate system y-axis (right hand), left hand aligns with negative direction
            for i in range(grasp_poses_canonical.shape[0]):
                local_y = grasp_poses_canonical[i][:3, 1]
                target_y = np.array([0, 1, 0]) if arm == "right" else np.array([0, -1, 0])
                calc_rotation_matrix = calculate_rotation_matrix2(local_y, target_y)
                grasp_poses_canonical[i][:3, :3] = calc_rotation_matrix @ grasp_poses_canonical[i][:3, :3]

        if flip_grasp:
            grasp_poses_canonical_flip = []
            for _i in range(grasp_poses_canonical.shape[0]):
                grasp_poses_canonical_flip.append(
                    rotate_along_axis(grasp_poses_canonical[_i], 180, "z", use_local=True)
                )
            grasp_poses_canonical_flip = np.stack(grasp_poses_canonical_flip)
            grasp_poses_canonical = np.concatenate([grasp_poses_canonical, grasp_poses_canonical_flip], axis=0)
            grasp_widths = np.concatenate([grasp_widths, grasp_widths], axis=0)

        random_grasp_pose = self.extra_params.get("random_grasp_pose", False)
        if random_grasp_pose:
            grasp_poses_canonical, grasp_widths = overweite_grasp_data(100)

        grasp_poses = objects[self.passive_obj_id].obj_pose[np.newaxis, ...] @ grasp_poses_canonical

        filter_grasp_pose_info = self.extra_params.get("filter_grasp_pose", {})
        if filter_grasp_pose_info:
            grasp_poses, grasp_widths, _ = filter_grasp_pose_by_gripper_up_direction(
                filter_grasp_pose_info, grasp_poses, grasp_widths
            )

        if self.extra_params.get("humanlike_filter", False):
            grasp_poses, grasp_widths, _ = filter_grasp_poses_with_humanlike_posture(grasp_poses, grasp_widths)

        if disable_upside_down:
            if "omnipicker" in robot.robot_cfg:
                if arm == "left":
                    upright_mask = grasp_poses[:, 2, 1] < 0.0
                else:
                    upright_mask = grasp_poses[:, 2, 1] > 0.0
            else:
                upright_mask = grasp_poses[:, 2, 0] > 0.0
            grasp_poses = grasp_poses[upright_mask]
            grasp_widths = grasp_widths[upright_mask]
            logger.info(
                f"{self.action_type}, {self.passive_obj_id}, Filtered upside-down grasp_poses: {grasp_poses.shape[0]}"
            )

        if len(grasp_poses) == 0:
            logger.warning(f"{self.action_type}: No grasp_gripper_pose can pass gripper y direction filter")
            return []

        # Downsample if there are too many grasp points
        grasp_poses, random_indices = random_downsample(grasp_poses, 300, False)
        if random_indices is not None:
            grasp_widths = grasp_widths[random_indices]

        # grasp offset
        grasp_rotate = grasp_poses.copy()
        grasp_rotate[:, :3, 3] = np.array([0, 0, 0])
        transport_vector = grasp_rotate @ np.array([0, 0, 1, 0])[:, np.newaxis]
        transport_vector = transport_vector[:, :3, 0]
        transport_vector = transport_vector / np.linalg.norm(transport_vector, axis=1, keepdims=True)

        grasp_poses[:, :3, 3] = grasp_poses[:, :3, 3] + transport_vector * grasp_offset
        object_pose_inverse = np.linalg.inv(objects[self.passive_obj_id].obj_pose)
        grasp_poses_canonical = object_pose_inverse[np.newaxis, ...] @ grasp_poses

        # filter with IK-checking
        ik_success, _ = robot.solve_ik(grasp_poses, ee_type="gripper", arm=arm, type="Simple")
        grasp_poses_canonical, grasp_poses = (
            grasp_poses_canonical[ik_success],
            grasp_poses[ik_success],
        )
        grasp_widths = grasp_widths[ik_success]

        if error_type == "RandomPerturbations":
            error_params = error_data.get("params", {})
            pos_std = error_params.get("pos_std", [0.2, 0.2, 0.2])
            rot_std = error_params.get("rot_std", [0.5, 0.5, 0.5])
            gripper2obj = []
            ik_joint_positions = []
            ik_joint_names = []
            ik_jacobian_score = []
            max_try = 10
            while len(gripper2obj) < 5 and max_try > 0:
                random_poses = generate_random_pose(
                    grasp_poses[max_try % grasp_poses.shape[0]],
                    position_std=pos_std,
                    rotation_std=rot_std,
                    num=40,
                )
                if disable_upside_down:
                    if "omnipicker" in robot.robot_cfg:
                        upright_mask = random_poses[:, 2, 1] > 0.0
                    else:
                        upright_mask = random_poses[:, 2, 0] > 0.0
                    random_poses = random_poses[upright_mask]
                ik_success, ik_info = robot.solve_ik(
                    random_poses,
                    ee_type="gripper",
                    arm=arm,
                    type="AvoidObs",
                    output_link_pose=False,
                )
                success_poses = random_poses[ik_success]
                if success_poses.shape[0] > 0:
                    gripper2obj += (
                        np.linalg.inv(objects[self.passive_obj_id].obj_pose)[np.newaxis, ...] @ success_poses
                    ).tolist()
                    ik_joint_positions += ik_info["joint_positions"][ik_success].tolist()
                    ik_joint_names += ik_info["joint_names"][ik_success].tolist()
                    ik_jacobian_score += ik_info["jacobian_score"][ik_success].tolist()
                max_try -= 1
            if len(gripper2obj):
                joint_names = robot.joint_names[arm]
                target_joint_positions = []
                for ik_joint_position, ik_joint_name in zip(ik_joint_positions, ik_joint_names):
                    temp_target_joint_positions = []
                    for joint_name in joint_names:
                        temp_target_joint_positions.append(ik_joint_position[list(ik_joint_name).index(joint_name)])
                    target_joint_positions.append(np.array(temp_target_joint_positions))
                target_joint_positions = np.array(target_joint_positions)
                cur_joint_states = robot.client.get_joint_positions().states
                cur_joint_positions = []
                for key in cur_joint_states:
                    if key.name in joint_names:
                        cur_joint_positions.append(key.position)
                cur_joint_positions = np.array(cur_joint_positions)
                joint_pos_dist = np.linalg.norm(target_joint_positions - cur_joint_positions[np.newaxis, :], axis=1)
                dist_mean = np.mean(joint_pos_dist)
                dist_std = np.std(joint_pos_dist)
                joint_pos_dist = (joint_pos_dist - dist_mean) / dist_std
                ik_jacobian_score = np.array(ik_jacobian_score)
                ik_jacobian_score = (ik_jacobian_score - np.min(ik_jacobian_score)) / (
                    np.max(ik_jacobian_score) - np.min(ik_jacobian_score)
                )
                cost = joint_pos_dist - ik_jacobian_score
                idx_sorted = np.argsort(cost)
                gripper2obj = np.array(gripper2obj)[idx_sorted]
            return gripper2obj

        if len(grasp_poses) == 0:
            logger.warning(f"{self.action_type}: No grasp_gripper_pose can pass Isaac IK")
            return []

        """Select optimal passive primitive element based on grasp pose scores with IK solutions, and select the optimal grasp pose"""
        next_active_stage = self.next_stage
        while next_active_stage:
            if next_active_stage.action_type not in ["turn", "reset"]:
                break
            next_active_stage = next_active_stage.next_stage
        if (
            next_active_stage
            and not error_type
            and next_active_stage.action_type not in ["pick", "grasp", "hook", "rotate"]
        ):
            place_with_origin_orientation = next_active_stage.extra_params.get("place_with_origin_orientation", True)
            single_obj = next_active_stage.active_obj_id == next_active_stage.passive_obj_id

            next_active_obj = objects[next_active_stage.active_obj_id]
            next_passive_obj = objects[next_active_stage.passive_obj_id]
            next_passive_element = next_active_stage.passive_element[
                np.random.choice(len(next_active_stage.passive_element))
            ]
            active_elements = next_active_stage.active_element

            if next_active_stage.action_type in PLACE_LIKE_ACTIONS and place_with_origin_orientation:
                obj_pose = next_active_obj.obj_pose
                direction = np.array([0, 0, -1])
                xyz_canonical = np.array([0, 0, 0])
                direction_canonical = (np.linalg.inv(obj_pose) @ np.array([*direction, 0]))[:3]
                active_elements = [{"xyz": xyz_canonical, "direction": direction_canonical}]

            time.time()
            element_ik_score = []
            grasp_pose_ik_score = []

            if type(active_elements) is not list:
                active_elements = [active_elements]
            for active_element in active_elements:
                # active_element contains the target pose of the active object in the current action
                # interaction between two rigid objects
                obj_pose = next_active_obj.obj_pose

                if not single_obj:
                    # Here obj.xyz only represents the final target position relative to the object's xyz_start position,
                    # direction represents the direction relative to the object's xyz_start,
                    # the object position is still stored in obj.pose
                    next_active_obj.update_aligned_info(active_element)
                    if "fix_pose" in next_active_stage.passive_obj_id:
                        next_passive_obj.update_aligned_info(next_passive_element)
                        # N_align equals the least common multiple of next_passive_obj.angle_sample_num and next_active_obj.angle_sample_num
                        N_align = np.lcm(
                            next_passive_obj.angle_sample_num,
                            next_active_obj.angle_sample_num,
                        )
                        N_align = min(N_align, 12)
                        target_obj_poses = get_aligned_fix_pose(next_active_obj, next_passive_obj, N=N_align)
                    else:
                        next_passive_obj.update_aligned_info(next_passive_element)
                        # N_align equals the least common multiple of next_passive_obj.angle_sample_num and next_active_obj.angle_sample_num
                        N_align = np.lcm(
                            next_passive_obj.angle_sample_num,
                            next_active_obj.angle_sample_num,
                        )
                        N_align = min(N_align, 12)
                        target_obj_poses = get_aligned_pose(next_active_obj, next_passive_obj, N=N_align)
                else:  # Object moves by itself
                    transform = np.eye(4)
                    transform[:3, 3] = active_element["xyz"]
                    target_obj_poses = (obj_pose @ transform)[np.newaxis, ...]
                    N_align = 1

                N_obj_pose = target_obj_poses.shape[0]
                N_grasp_pose = grasp_poses_canonical.shape[0]
                target_gripper_poses = (
                    target_obj_poses[:, np.newaxis, ...] @ grasp_poses_canonical[np.newaxis, ...]
                ).reshape(-1, 4, 4)

                ik_success, _ = robot.solve_ik(target_gripper_poses, ee_type="gripper", type="Simple", arm=arm)
                if next_active_stage.extra_params.get("disable_upside_down", False):
                    if "omnipicker" in robot.robot_cfg:
                        if arm == "left":
                            upright_mask = target_gripper_poses[:, 2, 1] < 0.0
                        else:
                            upright_mask = target_gripper_poses[:, 2, 1] > 0.0
                    else:
                        upright_mask = target_gripper_poses[:, 2, 0] > 0.0
                    ik_success = ik_success & upright_mask
                element_ik_score.append(np.max(ik_success.reshape(N_obj_pose, N_grasp_pose).sum(axis=1)))

                grasp_pose_ik = ik_success.reshape(N_obj_pose, N_grasp_pose)
                grasp_pose_ik_score.append(np.sum(grasp_pose_ik, axis=0))

            best_element_id = np.argmax(element_ik_score)
            best_active_element = active_elements[best_element_id]

            next_active_stage.active_element = best_active_element

            grasp_ik_score = grasp_pose_ik_score[best_element_id]

            _mask = grasp_ik_score >= max(np.median(grasp_ik_score) / 2, 1)
            best_grasp_poses = grasp_poses[_mask]
            if best_grasp_poses.shape[0] == 0 and next_active_stage.extra_params.get("use_near_point", False):
                logger.info("! search near point")
                x_offset_range = np.arange(-0.06, 0.06, 0.02)
                y_offset_range = np.arange(-0.06, 0.06, 0.02)
                ok = False
                for x_offset in x_offset_range:
                    for y_offset in y_offset_range:
                        time.time()
                        element_ik_score = []
                        grasp_pose_ik_score = []
                        for active_element in active_elements:
                            # active_element contains the target pose of the active object in the current action
                            # interaction between two rigid objects
                            obj_pose = next_active_obj.obj_pose

                            if not single_obj:
                                # Here obj.xyz only represents the final target position relative to the object's xyz_start position,
                                # direction represents the direction relative to the object's xyz_start,
                                # the object position is still stored in obj.pose
                                next_active_obj.update_aligned_info(active_element)
                                if "fix_pose" in next_active_stage.passive_obj_id:
                                    next_passive_obj.update_aligned_info(next_passive_element)
                                    # N_align equals the greatest common divisor of next_passive_obj.angle_sample_num and next_active_obj.angle_sample_num
                                    N_align = np.gcd(
                                        next_passive_obj.angle_sample_num,
                                        next_active_obj.angle_sample_num,
                                    )
                                    target_obj_poses = get_aligned_fix_pose(
                                        next_active_obj, next_passive_obj, N=N_align
                                    )
                                else:
                                    next_passive_obj.update_aligned_info(next_passive_element)
                                    next_passive_obj.xyz += [x_offset, y_offset, 0.0]
                                    # N_align equals the greatest common divisor of next_passive_obj.angle_sample_num and next_active_obj.angle_sample_num
                                    N_align = np.gcd(
                                        next_passive_obj.angle_sample_num,
                                        next_active_obj.angle_sample_num,
                                    )
                                    target_obj_poses = get_aligned_pose(next_active_obj, next_passive_obj, N=N_align)
                            else:  # Object moves by itself
                                transform = np.eye(4)
                                transform[:3, 3] = active_element["xyz"]
                                target_obj_poses = (obj_pose @ transform)[np.newaxis, ...]
                                N_align = 1

                            N_obj_pose = target_obj_poses.shape[0]
                            N_grasp_pose = grasp_poses_canonical.shape[0]
                            target_gripper_poses = (
                                target_obj_poses[:, np.newaxis, ...] @ grasp_poses_canonical[np.newaxis, ...]
                            ).reshape(-1, 4, 4)

                            ik_success, _ = robot.solve_ik(
                                target_gripper_poses,
                                ee_type="gripper",
                                type="Simple",
                                arm=arm,
                            )
                            element_ik_score.append(np.max(ik_success.reshape(N_obj_pose, N_grasp_pose).sum(axis=1)))

                            grasp_pose_ik = ik_success.reshape(N_obj_pose, N_grasp_pose)
                            grasp_pose_ik_score.append(np.sum(grasp_pose_ik, axis=0))

                        best_element_id = np.argmax(element_ik_score)
                        best_active_element = active_elements[best_element_id]

                        if not single_obj:
                            next_active_stage.active_element = best_active_element

                        grasp_ik_score = grasp_pose_ik_score[best_element_id]

                        _mask = grasp_ik_score >= max(np.median(grasp_ik_score) / 2, 1)
                        best_grasp_poses = grasp_poses[_mask]

                        if best_grasp_poses.shape[0] > 0:
                            logger.info(
                                f"get grasp pose {best_grasp_poses.shape[0]} with offset: [{x_offset}, {y_offset}]"
                            )
                            ok = True
                        else:
                            logger.info(f"can not get grasp pose with offset: [{x_offset}, {y_offset}]")
                        if ok:
                            break
                    if ok:
                        break
        else:
            best_grasp_poses = grasp_poses
        if best_grasp_poses.shape[0] == 0:
            logger.info("No grasp pose can pass next action IK")
            return []
        # downsample grasp pose
        best_grasp_poses, _ = random_downsample(best_grasp_poses, 100, False)
        if best_grasp_poses.shape[0] >= 1:
            if error_type == "KeepClose":
                robot.client.remove_objs_from_obstacle([objects[self.passive_obj_id].prim_path])
            joint_names = robot.joint_names[arm]

            ik_success, ik_info = robot.solve_ik(
                best_grasp_poses,
                ee_type="gripper",
                type="AvoidObs",
                arm=arm,
                output_link_pose=True,
            )
            best_grasp_poses = best_grasp_poses[ik_success]
            ik_joint_positions = ik_info["joint_positions"][ik_success]
            ik_joint_names = ik_info["joint_names"][ik_success]
            ik_jacobian_score = ik_info["jacobian_score"][ik_success]
            ik_link_poses = ik_info["link_poses"][ik_success]
            if len(best_grasp_poses) == 0:
                logger.warning(f"{self.action_type}: No best_grasp_poses can pass curobo IK")
                return []
            pre_grasp_poses = None
            if len(best_grasp_poses) == 0:
                logger.warning(f"{self.action_type}: No best_grasp_poses can pass curobo IK")
                return []
            if self.use_pre_grasp:
                # calculate pre grasp pose
                pre_grasp_distance = self.extra_params.get("pre_grasp_distance", 0.05)
                pre_grasp_offset = [0, 0, -pre_grasp_distance, 1, 0, 0, 0]
                pre_grasp_offset_matrix = pose_from_position_quaternion(pre_grasp_offset[:3], pre_grasp_offset[3:])
                pre_grasp_poses = best_grasp_poses @ pre_grasp_offset_matrix
                ik_success_pre, ik_info_pre = robot.solve_ik(
                    pre_grasp_poses,
                    ee_type="gripper",
                    type="AvoidObs",
                    arm=arm,
                    output_link_pose=True,
                )
                best_grasp_poses = best_grasp_poses[ik_success_pre]
                pre_grasp_poses = pre_grasp_poses[ik_success_pre]
                ik_joint_positions = ik_joint_positions[ik_success_pre]
                ik_joint_names = ik_joint_names[ik_success_pre]
                ik_jacobian_score = ik_jacobian_score[ik_success_pre]
                ik_link_poses = ik_link_poses[ik_success_pre]
            if len(best_grasp_poses) == 0:
                logger.warning(f"{self.action_type}: No best_grasp_poses can pass curobo IK")
                return []
            if "G2" in robot.robot_cfg:
                is_right = arm == "right"
                elbow_name = "arm_r_link4" if is_right else "arm_l_link4"
                hand_name = "gripper_r_center_link" if is_right else "gripper_l_center_link"
                idx_sorted = sorted_by_position_humanlike(
                    joint_positions=ik_joint_positions,
                    joint_names=ik_joint_names,
                    link_poses=ik_link_poses,
                    is_right=arm == "right",
                    elbow_name=elbow_name,
                    hand_name=hand_name,
                    is_from_up_side=self.extra_params.get("is_from_up_side", False),
                )
            else:
                # choose target pose based on the ik jacobian score and ik joint positions
                idx_sorted = sorted_by_joint_pos_dist_and_grasp_pose(
                    robot=robot,
                    arm=arm,
                    ik_joint_positions=ik_joint_positions,
                    ik_joint_names=ik_joint_names,
                    ik_jacobian_score=ik_jacobian_score,
                    grasp_poses=best_grasp_poses,
                    pre_grasp_offset=pre_grasp_offset,
                )
            grasp_poses_sorted = best_grasp_poses[idx_sorted]
            pre_grasp_poses_sorted = pre_grasp_poses[idx_sorted]

        else:
            logger.info("No grasp pose found")
            return []

        grasp_pose_canonical_sorted = (
            np.linalg.inv(objects[self.passive_obj_id].obj_pose)[np.newaxis, ...] @ grasp_poses_sorted
        )
        gripper2obj = grasp_pose_canonical_sorted
        if pre_grasp_poses is not None and len(pre_grasp_poses_sorted) == len(grasp_poses_sorted):
            pre_grasp_pose_canonical_sorted = (
                np.linalg.inv(objects[self.passive_obj_id].obj_pose)[np.newaxis, ...] @ pre_grasp_poses_sorted
            )
        result = []
        for i in range(len(gripper2obj)):
            tmp_result = {}
            tmp_result["grasp_pose"] = gripper2obj[i]
            tmp_result["pre_grasp_pose"] = pre_grasp_pose_canonical_sorted[i] if pre_grasp_poses is not None else None
            result.append(tmp_result)
        return result

    def generate_action_sequence(self, grasp_pose, pre_grasp_pose=None):
        action_sequence = ActionSequence()
        pick_up_distance = self.extra_params.get("pick_up_distance", 0.12)
        pick_up_type = self.extra_params.get("pick_up_type", "Simple")
        pick_up_direction = self.extra_params.get("pick_up_direction", "z")
        pre_grasp_vector = self.extra_params.get("pre_grasp_vector", [])

        if self.use_pre_grasp:
            if pre_grasp_pose is None:
                logger.info("No pre_grasp_pose found")
                return None
            # sub-stage-0   moveTo pregrasp pose
            path_constraint = [0.1, 0.1, 0.1, 0.1, 0.1, 0.0]
            offset_and_constraint_in_goal_frame = True
            # first to pregrasp pose
            action_sequence.add_action(
                Action(
                    grasp_pose=pre_grasp_pose,
                    gripper_action=None,
                    transform_world=np.eye(4),
                    motion_type="AvoidObs",
                )
            )
            # then to grasp pose
            action_sequence.add_action(
                Action(
                    grasp_pose=grasp_pose,
                    gripper_action="close",
                    transform_world=np.eye(4),
                    motion_type="AvoidObs",
                    extra_params={
                        "path_constraint": path_constraint,
                        "offset_and_constraint_in_goal_frame": offset_and_constraint_in_goal_frame,
                    },
                )
            )
            # pick up
            if pick_up_distance != 0.0:
                gripper_action = "close"
                goal_offset = [0, 0, pick_up_distance, 1, 0, 0, 0]
                path_constraint = [0.1, 0.1, 0.1, 0.1, 0.1, 0.0]
                if pick_up_direction == "x":
                    goal_offset = [pick_up_distance, 0, 0, 1, 0, 0, 0]
                    path_constraint = [0.1, 0.1, 0.1, 0, 0.1, 0.1]
                elif pick_up_direction == "y":
                    goal_offset = [0, pick_up_distance, 0, 1, 0, 0, 0]
                    path_constraint = [0.1, 0.1, 0.1, 0.1, 0, 0.1]
                offset_and_constraint_in_goal_frame = False
                from_current_pose = True
                action_sequence.add_action(
                    Action(
                        grasp_pose,
                        gripper_action,
                        np.eye(4),
                        "AvoidObs",
                        extra_params={
                            "goal_offset": goal_offset,
                            "path_constraint": path_constraint,
                            "offset_and_constraint_in_goal_frame": offset_and_constraint_in_goal_frame,
                            "from_current_pose": from_current_pose,
                        },
                    )
                )
            self.pick_up_step = 2
        else:
            if len(pre_grasp_vector) > 0:
                self.pick_up_step = 2
                # pre-grasp
                transform = np.eye(4)
                transform[:3, 3] = np.array(pre_grasp_vector)
                # sub-stage-0 moveTo pre-grasp pose
                action_sequence.add_action(Action(grasp_pose, "open", transform, "AvoidObs"))
            else:
                self.pick_up_step = 1
            if self.error_type == "MissGrasp":
                # sub-stage-0 moveTo grasp pose
                action_sequence.add_action(Action(grasp_pose, None, np.eye(4), "AvoidObs"))
            else:
                # grasp
                action_sequence.add_action(Action(grasp_pose, "close", np.eye(4), "AvoidObs"))
                if self.error_type != "RandomPerturbations":
                    # pick-up
                    gripper_action = None
                    motion_type = pick_up_type
                    transform_up = np.eye(4)
                    if pick_up_direction == "x":
                        transform_up[0, 3] = pick_up_distance
                        transform_up[2, 3] = 0.02
                    elif pick_up_direction == "y":
                        transform_up[1, 3] = pick_up_distance
                    else:
                        transform_up[2, 3] = pick_up_distance
                    action_sequence.add_action(Action(grasp_pose, gripper_action, transform_up, motion_type))
                else:
                    action_sequence.add_action(Action(grasp_pose, "open", np.eye(4), "AvoidObs"))
        if self.error_type == "WrongTarget":
            action = action_sequence[0]
            action.gripper_action = "open"
            action_sequence = ActionSequence([action])
        elif self.error_type == "KeepClose":
            action = action_sequence[0]
            action.gripper_action = None
            action_sequence = ActionSequence([action])
        return action_sequence

    def check_completion(self, objects, robot=None):
        assert self.active_action_sequence is not None, f"Active action for stage {self.action_type} is None"

        goal_datapack = [
            self.active_obj_id,
            self.passive_obj_id,
        ] + self.active_action_sequence.get_current_action().unpack_as_list()
        pre_action = self.active_action_sequence.get_previous_action()
        is_grasped = False
        if pre_action and pre_action.gripper_action == "close":
            is_grasped = True

        succ = True
        if not self.error_type:
            succ = simple_check_completion(goal_datapack, objects, is_grasped=is_grasped)
        return succ

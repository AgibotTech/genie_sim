# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import numpy as np

from client.layout.object import OmniObject
from client.planner.action.stage import Action, ActionSequence, Stage
from client.planner.func.sort_pose.sort_pose import sorted_by_position_humanlike
from client.robot.omni_robot import IsaacSimRpcRobot
from common.base_utils.logger import logger
from common.base_utils.transform_utils import calculate_rotation_matrix2, rotate_along_axis


class RotateStage(Stage):
    def __init__(self, stage_config, objects):
        super().__init__(stage_config, objects)

    def select_pose(self, objects: list[OmniObject], robot: IsaacSimRpcRobot):
        arm = self.extra_params.get("arm", "right")
        anchor_pose = objects[self.passive_obj_id].obj_pose
        place_up_axis = self.extra_params.get("place_up_axis", "y")
        gripper_pose = robot.get_ee_pose(ee_type="gripper", id=arm)
        pick_up_distance = self.extra_params.get("pick_up_distance", 0.0)
        pick_up_direction = self.extra_params.get("pick_up_direction", "z")
        place_origin_position = self.extra_params.get("place_origin_position", True)
        pick_up_pose = None
        # Calculate gripper transformation relative to object (in object coordinate system)
        gripper2obj = np.linalg.inv(anchor_pose) @ gripper_pose
        object2gripper = np.linalg.inv(gripper_pose) @ anchor_pose
        if pick_up_distance > 0.0:
            pick_up_vector = np.array([0, 0, pick_up_distance])
            if pick_up_direction == "x":
                pick_up_vector = np.array([pick_up_distance, 0, 0])
            elif pick_up_direction == "y":
                pick_up_vector = np.array([0, pick_up_distance, 0])
            else:
                pick_up_vector = np.array([0, 0, pick_up_distance])
            gripper_pose[:3, 3] += pick_up_vector
            anchor_pose[:3, 3] += pick_up_vector
            pick_up_pose = gripper_pose.copy()

        # Get the specified axis direction of the object (in world coordinate system)
        world_up_vector = np.array([0, 0, 1])
        if place_up_axis == "y":
            object_up_vector = anchor_pose[:3, 1]
        elif place_up_axis == "z":
            object_up_vector = anchor_pose[:3, 2]
        elif place_up_axis == "x":
            object_up_vector = anchor_pose[:3, 0]
        else:
            logger.error(f"Invalid place_up_axis: {place_up_axis}")
            return []

        R_align = calculate_rotation_matrix2(object_up_vector, world_up_vector)

        new_anchor_pose = anchor_pose.copy()
        new_anchor_pose[:3, :3] = R_align @ anchor_pose[:3, :3]

        new_gripper_pose = new_anchor_pose @ gripper2obj
        target_pose = gripper_pose.copy()
        target_pose[:3, :3] = new_gripper_pose[:3, :3]

        target_poses = [target_pose]
        rotate_angle = 0
        rotate_delta = 5
        while rotate_angle < 360:
            target_pose = rotate_along_axis(target_pose, rotate_delta, "z", False)
            target_poses.append(target_pose)
            rotate_angle += rotate_delta

        target_poses = np.array(target_poses)

        ik_success, ik_info = robot.solve_ik(
            target_poses, ee_type="gripper", type="Simple", arm=arm
        )
        target_poses = target_poses[ik_success]
        logger.info(
            f"Rotate, {self.passive_obj_id}, Filtered target pose with isaac-sim IK: {target_poses.shape[0]}/{ik_success.shape[0]}"
        )
        if target_poses.shape[0] == 0:
            logger.info(f"Rotate, {self.passive_obj_id}, No target pose can pass isaac-sim IK")
            return []

        ik_success, ik_info = robot.solve_ik(
            target_poses,
            ee_type="gripper",
            type="AvoidObs",
            arm=arm,
            output_link_pose=True,
        )
        target_poses = target_poses[ik_success]
        logger.info(
            f"Rotate, {self.passive_obj_id}, Filtered target pose with curobo IK: {target_poses.shape[0]}/{ik_success.shape[0]}"
        )
        if target_poses.shape[0] == 0:
            logger.info(f"Rotate, {self.passive_obj_id}, No target pose can pass curobo IK")
            return []
        if "G2" in robot.robot_cfg:
            is_right = arm == "right"
            elbow_name = "arm_r_link4" if is_right else "arm_l_link4"
            hand_name = "gripper_r_center_link" if is_right else "gripper_l_center_link"
            idx_sorted = sorted_by_position_humanlike(
                joint_positions=ik_info["joint_positions"][ik_success],
                joint_names=ik_info["joint_names"][ik_success],
                link_poses=ik_info["link_poses"][ik_success],
                is_right=is_right,
                elbow_name=elbow_name,
                hand_name=hand_name,
                is_from_up_side=False,
            )
            target_poses = target_poses[idx_sorted]
        target_place_poses = None
        if place_origin_position:
            # get place pose
            obj_info = objects[self.passive_obj_id].info
            origin_position = obj_info["position"]
            origin_up_axis = obj_info.get("upAxis", "y")
            size = objects[self.passive_obj_id].obj_length

            def get_axis_index(upaxis):
                if upaxis == "x":
                    return 0
                elif upaxis == "y":
                    return 1
                elif upaxis == "z":
                    return 2
                else:
                    return 1

            origin_up_axis_index = get_axis_index(origin_up_axis[0])
            place_up_axis_index = get_axis_index(place_up_axis)
            origin_length = size[origin_up_axis_index]
            place_length = size[place_up_axis_index]
            delta_world_z = (place_length - origin_length) / 2.0
            object_target_place_position = origin_position + np.array([0, 0, delta_world_z])
            after_rotation_object_positions = target_poses @ object2gripper
            delta_translation = (
                object_target_place_position - after_rotation_object_positions[:, :3, 3]
            )
            # gripper pose at object place position
            target_place_poses = target_poses.copy()
            target_place_poses[:, :3, 3] += delta_translation
            num_samples = 10
            if target_place_poses.shape[0] < num_samples:
                sample_indices = np.arange(target_place_poses.shape[0])
            else:
                sample_indices = np.linspace(
                    0, target_place_poses.shape[0] - 1, num_samples
                ).astype(int)
            sampled_poses = target_place_poses[sample_indices]
            num_samples = sampled_poses.shape[0]
            z_shifts = np.arange(-0.08, 0.02, 0.01)
            num_z = z_shifts.shape[0]
            sampled_poses_batch = []
            for z_shift in z_shifts:
                new_poses = sampled_poses.copy()
                new_poses[:, 2, 3] += z_shift
                sampled_poses_batch.append(new_poses)
            sampled_poses_batch = np.vstack(sampled_poses_batch)  # (num_samples * num_z, 4, 4)

            ik_success_array, _ = robot.solve_ik(
                sampled_poses_batch, ee_type="gripper", type="AvoidObs", arm=arm
            )
            z_success_counts = []
            for i in range(num_z):
                ik_success_this_z = ik_success_array[i * num_samples : (i + 1) * num_samples]
                success_count = ik_success_this_z.sum()
                z_success_counts.append(success_count)

            min_valid_idx = None
            threshold = max(min(1, num_samples // 2), 1)
            for idx, count in enumerate(z_success_counts):
                if count > threshold:
                    min_valid_idx = idx
                    break
            if min_valid_idx is None:
                logger.info(f"Rotate, {self.passive_obj_id}, No valid z_shift can pass ik")
                return []
            best_z = z_shifts[min_valid_idx]
            logger.info(f"best_z{best_z}")
            margin = 0.0
            target_place_poses[:, :3, 3] += np.array([0, 0, best_z + margin])

            ik_success, ik_info = robot.solve_ik(
                target_place_poses, ee_type="gripper", type="Simple", arm=arm
            )
            target_place_poses = target_place_poses[ik_success]
            target_poses = target_poses[ik_success]
            logger.info(
                f"Rotate, {self.passive_obj_id}, Filtered target place pose with isaac-sim IK: {target_place_poses.shape[0]}/{ik_success.shape[0]}"
            )
            if target_place_poses.shape[0] == 0:
                logger.info(
                    f"Rotate, {self.passive_obj_id}, No target place pose can pass isaac-sim IK"
                )
                return []

            ik_success, ik_info = robot.solve_ik(
                target_place_poses, ee_type="gripper", type="AvoidObs", arm=arm
            )
            target_poses = target_poses[ik_success]
            target_place_poses = target_place_poses[ik_success]
            logger.info(
                f"Rotate, {self.passive_obj_id}, Filtered target place pose with curobo IK: {target_place_poses.shape[0]}/{ik_success.shape[0]}"
            )
            if target_place_poses.shape[0] == 0:
                logger.info(
                    f"Rotate, {self.passive_obj_id}, No target place pose can pass curobo IK"
                )
                return []
        result = []

        for idx, target_pose_canonical in enumerate(target_poses):
            tmp_result = {
                "target_pose": target_pose_canonical,
            }
            if target_place_poses is not None:
                tmp_result["target_place_pose"] = target_place_poses[idx]
            if pick_up_pose is not None:
                tmp_result["pick_up_pose"] = pick_up_pose
            result.append(tmp_result)
        return result

    def generate_action_sequence(self, target_pose, target_place_pose=None, pick_up_pose=None):
        action_sequence = ActionSequence()
        action_sequence.add_action(
            Action(
                target_pose,
                None,
                np.eye(4),
                "AvoidObs",
                extra_params={
                    #  "path_constraint": path_constraint,
                    "use_world_pose": True,
                },
            )
        )
        # place to origin
        if target_place_pose is not None:
            action_sequence.add_action(
                Action(
                    target_place_pose,
                    "open",
                    np.eye(4),
                    "AvoidObs",
                    extra_params={
                        "use_world_pose": True,
                    },
                )
            )

        return action_sequence

    def check_completion(self, objects, robot=None):
        assert (
            self.active_action_sequence is not None
        ), f"Active action for stage {self.action_type} is None"
        return True

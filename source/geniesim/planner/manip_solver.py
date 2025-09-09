# Copyright (c) 2023-2025, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import os, sys
import copy
import json
import time
import trimesh

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]

import numpy as np

from geniesim.utils.transform_utils import calculate_rotation_matrix, rotate_around_axis
from geniesim.utils.object import OmniObject, transform_coordinates_3d
from geniesim.utils.data_utils import pose_difference_batch, vector_difference_batch
from geniesim.utils.fix_rotation import (
    rotate_180_along_axis,
    translate_along_axis,
    rotate_along_axis,
)


from .action import build_stage

from scipy.interpolate import interp1d


from geniesim.utils.logger import Logger

logger = Logger()  # Create singleton instance


def format_object(obj, distance, type="active"):
    if obj is None:
        return None
    xyz, direction = obj.xyz, obj.direction

    direction = direction / np.linalg.norm(direction) * distance
    type = type.lower()
    if type == "active":
        xyz_start = xyz
        xyz_end = xyz_start + direction
    elif type == "passive" or type == "plane":
        xyz_end = xyz
        xyz_start = xyz_end - direction

    part2obj = np.eye(4)
    part2obj[:3, 3] = xyz_start
    obj.obj2part = np.linalg.inv(part2obj)

    obj_info = {
        "pose": obj.obj_pose,
        "length": obj.obj_length,
        "xyz_start": xyz_start,
        "xyz_end": xyz_end,
        "obj2part": obj.obj2part,
    }
    return obj_info


def obj2world(obj_info):
    obj_pose = obj_info["pose"]
    obj_length = obj_info["length"]
    obj2part = obj_info["obj2part"]
    xyz_start = obj_info["xyz_start"]
    xyz_end = obj_info["xyz_end"]

    arrow_in_obj = np.array([xyz_start, xyz_end]).transpose(1, 0)
    arrow_in_world = transform_coordinates_3d(arrow_in_obj, obj_pose).transpose(1, 0)

    xyz_start_world, xyz_end_world = arrow_in_world
    direction_world = xyz_end_world - xyz_start_world
    direction_world = direction_world / np.linalg.norm(direction_world)

    obj_info_world = {
        "pose": obj_pose,
        "length": obj_length,
        "obj2part": obj2part,
        "xyz_start": xyz_start_world,
        "xyz_end": xyz_end_world,
        "direction": direction_world,
    }
    return obj_info_world


def get_aligned_fix_pose(active_obj, passive_obj, distance=0.01, N=1):
    try:
        active_object = format_object(active_obj, type="active", distance=distance)
        passive_object = format_object(passive_obj, type="passive", distance=distance)
    except:
        logger.error("error")

    active_obj_world = obj2world(active_object)
    current_obj_pose = active_obj_world["pose"]
    if passive_object is None:
        return current_obj_pose[np.newaxis, ...]

    passive_obj_world = obj2world(passive_object)
    passive_obj_world["direction"] = passive_obj.direction

    R = calculate_rotation_matrix(
        active_obj_world["direction"], passive_obj_world["direction"]
    )
    T = passive_obj_world["xyz_end"] - R @ active_obj_world["xyz_start"]
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = R
    transform_matrix[:3, 3] = T
    target_obj_pose = transform_matrix @ current_obj_pose
    target_obj_pose[:3, 3] = passive_obj.obj_pose[:3, 3]
    poses = []
    for angle in [i * 360 / N for i in range(N)]:
        pose_rotated = rotate_around_axis(
            target_obj_pose,
            passive_obj_world["xyz_start"],
            passive_obj_world["direction"],
            angle,
        )
        poses.append(pose_rotated)
    return np.stack(poses)


def get_aligned_pose(active_obj, passive_obj, distance=0.01, N=1):
    try:
        active_object = format_object(active_obj, type="active", distance=distance)
        passive_object = format_object(passive_obj, type="passive", distance=distance)
    except:
        logger.error("error")

    active_obj_world = obj2world(active_object)
    current_obj_pose = active_obj_world["pose"]
    if passive_object is None:
        return current_obj_pose[np.newaxis, ...]
    passive_obj_world = obj2world(passive_object)
    R = calculate_rotation_matrix(
        active_obj_world["direction"], passive_obj_world["direction"]
    )
    T = passive_obj_world["xyz_end"] - R @ active_obj_world["xyz_start"]
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = R
    transform_matrix[:3, 3] = T
    target_obj_pose = transform_matrix @ current_obj_pose

    poses = []
    for angle in [i * 360 / N for i in range(N)]:
        pose_rotated = rotate_around_axis(
            target_obj_pose,
            passive_obj_world["xyz_start"],
            passive_obj_world["direction"],
            angle,
        )
        poses.append(pose_rotated)
    return np.stack(poses)


def load_task_solution(task_info):
    stages = task_info["stages"]

    objects = {"gripper": OmniObject("gripper")}

    for obj_info in task_info["objects"]:
        obj_id = obj_info["object_id"]
        if obj_id == "fix_pose":
            obj = OmniObject("fix_pose")
            if "position" not in obj_info or "direction" not in obj_info:
                logger.error(f"Error: Missing position/direction in object {obj_id}")
                continue
            obj.set_pose(
                np.array(obj_info["position"]), np.array([0.001, 0.001, 0.001])
            )
            obj.elements = {
                "active": {},
                "passive": {
                    "place": {
                        "default": [
                            {
                                "xyz": np.array([0, 0, 0]),
                                "direction": np.array(obj_info["direction"]),
                            }
                        ]
                    }
                },
            }
            objects[obj_id] = obj
        else:
            obj_dir = obj_info["data_info_dir"]
            obj = OmniObject.from_obj_dir(obj_dir, obj_info=obj_info)
            objects[obj_id] = obj

            if hasattr(obj, "part_ids"):
                if (
                    hasattr(obj, "part_joint_limits")
                    and obj.part_joint_limits is not None
                ):
                    obj_parts_joint_limits = obj.part_joint_limits
                for part_id in obj.part_ids:
                    id = obj_id + "/%s" % part_id
                    objects[id] = copy.deepcopy(obj)
                    objects[id].name = id
                    objects[id].part_joint_limit = obj_parts_joint_limits[part_id]
                if len(obj.part_ids):
                    del objects[obj_id]
    return stages, objects


def parse_stage(stage, objects):
    action = stage["action"]
    if action in ["reset"]:
        return action, "gripper", "gripper", None, None, None, None

    active_obj_id = stage["active"]["object_id"]
    if "part_id" in stage["active"]:
        active_obj_id += "/%s" % stage["active"]["part_id"]

    passive_obj_id = stage["passive"]["object_id"]
    if "part_id" in stage["passive"]:
        passive_obj_id += "/%s" % stage["passive"]["part_id"]

    active_obj = objects[active_obj_id]
    passive_obj = objects[passive_obj_id]

    def _load_element(obj, type):
        if action in ["pick", "hook"]:
            action_mapped = "grasp"
        else:
            action_mapped = action
        if action_mapped == "grasp" and type == "active":
            return None, None
        elif obj.name == "gripper":
            element = obj.elements[type][action_mapped]
            return element, "default"
        primitive = (
            stage[type]["primitive"]
            if stage[type]["primitive"] is not None
            else "default"
        )
        if primitive != "default" or (action_mapped == "grasp" and type == "passive"):
            if action_mapped not in obj.elements[type]:
                logger.warning("No %s element for %s" % (action_mapped, obj.name))
                return None, None
            element = obj.elements[type][action_mapped][primitive]
        else:
            element = []
            for primitive in obj.elements[type][action_mapped]:
                _element = obj.elements[type][action_mapped][primitive]
                if isinstance(_element, list):
                    element += _element
                else:
                    element.append(_element)
        return element, primitive

    passive_element, passive_primitive = _load_element(passive_obj, "passive")
    active_element, active_primitive = _load_element(active_obj, "active")
    return (
        action,
        active_obj_id,
        passive_obj_id,
        active_element,
        passive_element,
        active_primitive,
        passive_primitive,
    )


def select_obj(objects, stages, robot):
    gripper2obj = None
    extra_params = stages[0].get("extra_params", {})
    arm = extra_params.get("arm", "right")
    grasp_offset = extra_params.get("grasp_offset", 0.0)
    grasp_lower_percentile = extra_params.get("grasp_lower_percentile", 0)
    grasp_upper_percentile = extra_params.get("grasp_upper_percentile", 100)
    disable_upside_down = extra_params.get("disable_upside_down", True)
    horizontal_pick = extra_params.get("horizontal_pick", False)

    """ Initial screening to grab poses, get grasp_poses_canonical, grasp_poses """
    grasp_stage_id = None

    if stages[0]["action"] in ["pick", "grasp", "hook"]:
        action = stages[0]["action"]

        """ Solve out grasp pose without IK solution """
        grasp_stage_id = 0
        grasp_stage = parse_stage(stages[0], objects)
        _, _, passive_obj_id, _, passive_element, _, _ = grasp_stage
        grasp_obj_id = passive_obj_id
        grasp_poses_canonical = passive_element["grasp_pose"].copy()
        grasp_widths = passive_element["width"]

        z_values = grasp_poses_canonical[:, 1, 3]
        # z_lower_threshold = np.percentile(z_values, 20)
        # z_upper_threshold = np.percentile(z_values, 40)

        z_lower_threshold = np.percentile(z_values, grasp_lower_percentile)
        z_upper_threshold = np.percentile(z_values, grasp_upper_percentile)
        # filter grasp pose with z_min and z_max value
        mask = (z_values <= z_upper_threshold) & (z_values >= z_lower_threshold)
        grasp_poses_canonical = grasp_poses_canonical[mask]
        grasp_widths = grasp_widths[mask]

        grasp_poses_canonical[:, :3, :3] = (
            grasp_poses_canonical[:, :3, :3]
            @ robot.robot_gripper_2_grasp_gripper[np.newaxis, ...]
        )
        grasp_poses_canonical_flip = []
        for _i in range(grasp_poses_canonical.shape[0]):
            grasp_poses_canonical_flip.append(
                rotate_along_axis(grasp_poses_canonical[_i], 180, "z", use_local=True)
            )

        grasp_poses_canonical = np.concatenate(
            [grasp_poses_canonical, grasp_poses_canonical_flip], axis=0
        )
        grasp_widths = np.concatenate([grasp_widths, grasp_widths], axis=0)
        grasp_poses = (
            objects[passive_obj_id].obj_pose[np.newaxis, ...] @ grasp_poses_canonical
        )

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
            print(
                "%s, %s, Filtered upside-down grasp_poses: %d"
                % (action, passive_obj_id, grasp_poses.shape[0])
            )

        if horizontal_pick:
            horizontal_mask = np.abs(grasp_poses[:, 2, 2]) < 0.5
            grasp_poses = grasp_poses[horizontal_mask]
            grasp_widths = grasp_widths[horizontal_mask]
            print(
                "%s, %s, Filtered horizontal grasp_poses: %d"
                % (action, passive_obj_id, grasp_poses.shape[0])
            )
        # grasp offset
        grasp_rotate = grasp_poses.copy()
        grasp_rotate[:, :3, 3] = np.array([0, 0, 0])
        transport_vector = grasp_rotate @ np.array([0, 0, 1, 0])[:, np.newaxis]
        transport_vector = transport_vector[:, :3, 0]
        transport_vector = transport_vector / np.linalg.norm(
            transport_vector, axis=1, keepdims=True
        )
        grasp_poses[:, :3, 3] = grasp_poses[:, :3, 3] + transport_vector * grasp_offset
        object_pose_inverse = np.linalg.inv(objects[passive_obj_id].obj_pose)
        grasp_poses_canonical = object_pose_inverse[np.newaxis, ...] @ grasp_poses

        # filter with IK-checking

        ik_success, jacobian_score = robot.solve_ik(
            grasp_poses, ee_type="gripper", arm=arm, type="Simple"
        )
        grasp_poses_canonical, grasp_poses = (
            grasp_poses_canonical[ik_success],
            grasp_poses[ik_success],
        )
        grasp_widths = grasp_widths[ik_success]
        logger.info(
            "%s, %s, Filtered grasp pose with isaac-sim IK: %d/%d"
            % (action, passive_obj_id, grasp_poses.shape[0], ik_success.shape[0])
        )
        if len(grasp_poses) == 0:
            logger.error(action, "No grasp_gripper_pose can pass IK")
            return []

    """ Based on the Grasp pose score with IK solution, select the optimal passive primitive element and select the optimal Grasp pose at the same time."""
    if grasp_stage_id is not None:
        next_stage_id = grasp_stage_id + 1
        if next_stage_id < len(stages):
            (
                action,
                active_obj_id,
                passive_obj_id,
                active_elements,
                passive_elements,
                active_primitive,
                passive_primitive,
            ) = parse_stage(stages[next_stage_id], objects)
            next_stage_extra_params = stages[next_stage_id].get("extra_params", {})
            place_with_origin_orientation = next_stage_extra_params.get(
                "place_with_origin_orientation", True
            )
            single_obj = active_obj_id == passive_obj_id

            active_obj = objects[active_obj_id]
            passive_obj = objects[passive_obj_id]
            passive_element = passive_elements[np.random.choice(len(passive_elements))]

            if action == "place" and place_with_origin_orientation:
                obj_pose = active_obj.obj_pose
                mesh = trimesh.load(active_obj.info["mesh_file"], force="mesh")
                mesh.apply_scale(0.001)
                mesh.apply_transform(obj_pose)
                pts, _ = trimesh.sample.sample_surface(mesh, 200)  # Surface sampling
                xyz = np.array(
                    [
                        np.mean(pts[:, 0]),
                        np.mean(pts[:, 1]),
                        np.percentile(pts[:, 2], 1),
                    ]
                )

                direction = np.array([0, 0, -1])
                xyz_canonical = (np.linalg.inv(obj_pose) @ np.array([*xyz, 1]))[:3]
                direction_canonical = (
                    np.linalg.inv(obj_pose) @ np.array([*direction, 0])
                )[:3]
                active_elements = [
                    {"xyz": xyz_canonical, "direction": direction_canonical}
                ]

            t0 = time.time()
            element_ik_score = []
            grasp_pose_ik_score = []
            for active_element in active_elements:
                # interaction between two rigid objects
                obj_pose = active_obj.obj_pose

                N_align = 12
                if not single_obj:
                    active_obj.xyz, active_obj.direction = (
                        active_element["xyz"],
                        active_element["direction"],
                    )
                    if passive_obj_id == "fix_pose":
                        passive_obj.xyz, passive_obj.direction = (
                            passive_element["xyz"],
                            passive_element["direction"],
                        )
                        target_obj_poses = get_aligned_fix_pose(
                            active_obj, passive_obj, N=N_align
                        )
                    else:
                        passive_obj.xyz, passive_obj.direction = (
                            passive_element["xyz"],
                            passive_element["direction"],
                        )
                        target_obj_poses = get_aligned_pose(
                            active_obj, passive_obj, N=N_align
                        )
                else:
                    transform = np.eye(4)
                    transform[:3, 3] = active_element["xyz"]
                    target_obj_poses = (obj_pose @ transform)[np.newaxis, ...]
                    N_align = 1

                N_obj_pose = target_obj_poses.shape[0]
                N_grasp_pose = grasp_poses_canonical.shape[0]
                target_gripper_poses = (
                    target_obj_poses[:, np.newaxis, ...]
                    @ grasp_poses_canonical[np.newaxis, ...]
                ).reshape(-1, 4, 4)
                print("target_gripper_poses：", target_gripper_poses)
                ik_success, _ = robot.solve_ik(
                    target_gripper_poses, ee_type="gripper", type="Simple", arm=arm
                )
                print("ik_success:", ik_success)
                element_ik_score.append(
                    np.max(ik_success.reshape(N_obj_pose, N_grasp_pose).sum(axis=1))
                )

                grasp_pose_ik = ik_success.reshape(N_obj_pose, N_grasp_pose)
                grasp_pose_ik_score.append(np.sum(grasp_pose_ik, axis=0))

            logger.info(time.time() - t0)
            best_element_id = np.argmax(element_ik_score)
            best_active_element = active_elements[best_element_id]

            if not single_obj:

                active_obj.elements["active"][action] = {
                    active_primitive: best_active_element
                }

            grasp_ik_score = grasp_pose_ik_score[best_element_id]

            _mask = grasp_ik_score >= max(np.median(grasp_ik_score) / 2, 1)
            best_grasp_poses = grasp_poses[_mask]
            best_grasp_widths = grasp_widths[_mask]
            print(
                "%s, %s, Filtered grasp pose with next action IK: %d/%d"
                % (
                    action,
                    passive_obj_id,
                    best_grasp_poses.shape[0],
                    grasp_ik_score.shape[0],
                )
            )
        else:
            best_grasp_poses = grasp_poses
        if best_grasp_poses.shape[0] == 0:
            print("No grasp pose can pass next action IK")
            return []

        downsample_num = 100
        if best_grasp_poses.shape[0] > downsample_num:
            best_grasp_poses = best_grasp_poses[:downsample_num]
        if (
            best_grasp_poses.shape[0] > 1
        ):  # further select the best grasp pose with the smallest pose difference

            joint_names = robot.joint_names[arm]

            ik_success, ik_info = robot.solve_ik(
                best_grasp_poses, ee_type="gripper", type="AvoidObs", arm=arm
            )
            # mask = best_grasp_poses[:, 2, 0] < 0.0
            # ik_success[mask] = False

            best_grasp_poses = best_grasp_poses[ik_success]

            print(
                "%s, %s, Filtered grasp pose with curobo IK: %d/%d"
                % (
                    action,
                    passive_obj_id,
                    best_grasp_poses.shape[0],
                    ik_success.shape[0],
                )
            )

            ik_joint_positions = ik_info["joint_positions"][ik_success]
            ik_joint_names = ik_info["joint_names"][ik_success]
            if len(best_grasp_poses) == 0:
                return []
            target_joint_positions = []
            for ik_joint_position, ik_joint_name in zip(
                ik_joint_positions, ik_joint_names
            ):
                temp_target_joint_positions = []
                for joint_name in joint_names:
                    temp_target_joint_positions.append(
                        ik_joint_position[list(ik_joint_name).index(joint_name)]
                    )
                target_joint_positions.append(np.array(temp_target_joint_positions))
            target_joint_positions = np.array(target_joint_positions)
            cur_joint_states = robot.client.get_joint_positions().states
            cur_joint_positions = []
            for key in cur_joint_states:
                if key.name in joint_names:
                    cur_joint_positions.append(key.position)
            cur_joint_positions = np.array(cur_joint_positions)
            joint_pos_dist = np.linalg.norm(
                target_joint_positions - cur_joint_positions[np.newaxis, :], axis=1
            )
            dist_mean = np.mean(joint_pos_dist)
            dist_std = np.std(joint_pos_dist)
            joint_pos_dist = (joint_pos_dist - dist_mean) / dist_std
            cost = joint_pos_dist
            idx_sorted = np.argsort(cost)
            best_grasp_pose = best_grasp_poses[idx_sorted][0]
        else:
            best_grasp_pose = best_grasp_poses[0]
            best_grasp_widths = best_grasp_widths[0]
        best_grasp_pose_canonical = (
            np.linalg.inv(objects[grasp_obj_id].obj_pose) @ best_grasp_pose
        )
        gripper2obj = best_grasp_pose_canonical
    return gripper2obj


def split_grasp_stages(stages):
    # split_stages = []
    # i = 0
    # while i < len(stages):
    #     if stages[i]["action"] in ["pick", "grasp", "hook"]:
    #         if (i + 1) < len(stages) and stages[i + 1]["action"] not in [
    #             "pick",
    #             "grasp",
    #             "hook",
    #         ]:
    #             split_stages.append([stages[i], stages[i + 1]])
    #             i += 2
    #         else:
    #             split_stages.append([stages[i]])
    #             i += 1
    #     else:
    #         split_stages.append([stages[i]])
    #         i += 1
    # return split_stages
    split_stages = []
    i = 0
    while i < len(stages):
        if stages[i]["action"] in ["pick", "grasp", "hook", "move", "shift"]:
            split_stages.append([stages[i]])
            i += 1
            while i < len(stages) and stages[i]["action"] not in [
                "pick",
                "grasp",
                "hook",
                "move",
                "shift",
            ]:
                split_stages[-1].append(stages[i])
                i += 1
        else:
            split_stages.append([stages[i]])
            i += 1
    return split_stages


def generate_action_stages(objects, all_stages, robot):
    split_stages = split_grasp_stages(all_stages)
    action_stages = []
    for stages in split_stages:
        gripper2obj = select_obj(objects, stages, robot)
        if gripper2obj is None or len(gripper2obj) == 0:
            logger.error("No gripper2obj pose can pass IK")
            return []
        for stage in stages:
            extra_params = stage.get("extra_params", {})
            arm = extra_params.get("arm", "right")
            (
                action,
                active_obj_id,
                passive_obj_id,
                active_elements,
                passive_elements,
                active_primitive,
                passive_primitive,
            ) = parse_stage(stage, objects)
            active_obj = objects[active_obj_id]
            passive_obj = objects[passive_obj_id]

            single_obj = active_obj_id == passive_obj_id

            substages = None
            if action in ["reset"]:
                substages = True
            elif action in ["pick", "grasp", "hook"]:
                substages = build_stage(action)(
                    active_obj_id,
                    passive_obj_id,
                    active_elements,
                    passive_elements,
                    gripper2obj,
                    extra_params=stage.get("extra_params", None),
                )
            else:
                passive_element = passive_elements[
                    np.random.choice(len(passive_elements))
                ]
                if not isinstance(active_elements, list):
                    active_elements = [active_elements]

                for active_element in active_elements:
                    target_gripper_poses = np.zeros((0, 4, 4))
                    joint_names = robot.joint_names[arm]
                    # interaction between two rigid objects
                    obj_pose = active_obj.obj_pose
                    anchor_pose = passive_obj.obj_pose

                    current_obj_pose_canonical = np.linalg.inv(anchor_pose) @ obj_pose
                    active_obj.xyz, active_obj.direction = (
                        active_element["xyz"],
                        active_element["direction"],
                    )
                    for passive_element in passive_elements:
                        passive_obj.xyz, passive_obj.direction = (
                            passive_element["xyz"],
                            passive_element["direction"],
                        )
                        if active_obj.name == "gripper":
                            gripper2obj = np.eye(4)

                        if "fix_pose" == passive_obj_id:
                            target_obj_poses = get_aligned_fix_pose(
                                active_obj, passive_obj, N=36
                            )
                        else:
                            target_obj_poses = get_aligned_pose(
                                active_obj, passive_obj, N=36
                            )
                        #
                        target_gripper_poses = np.concatenate(
                            (
                                target_gripper_poses,
                                target_obj_poses @ gripper2obj[np.newaxis, ...],
                            ),
                            axis=0,
                        )

                    downsample_num = 200
                    if target_gripper_poses.shape[0] > downsample_num:
                        target_gripper_poses = target_gripper_poses[:downsample_num]
                    ik_success, ik_info = robot.solve_ik(
                        target_gripper_poses,
                        ee_type="gripper",
                        type="AvoidObs",
                        arm=arm,
                    )
                    target_gripper_poses_pass_ik = target_gripper_poses[ik_success]
                    ik_joint_positions = ik_info["joint_positions"][ik_success]
                    ik_joint_names = ik_info["joint_names"][ik_success]
                    if len(target_gripper_poses_pass_ik) == 0:
                        logger.error(action, ": No target_obj_pose can pass isaac IK")
                        continue

                    target_joint_positions = []
                    for ik_joint_position, ik_joint_name in zip(
                        ik_joint_positions, ik_joint_names
                    ):
                        temp_target_joint_positions = []
                        for joint_name in joint_names:
                            temp_target_joint_positions.append(
                                ik_joint_position[list(ik_joint_name).index(joint_name)]
                            )
                        target_joint_positions.append(
                            np.array(temp_target_joint_positions)
                        )
                    target_joint_positions = np.array(target_joint_positions)
                    cur_joint_states = robot.client.get_joint_positions().states
                    cur_joint_positions = []
                    for key in cur_joint_states:
                        if key.name in joint_names:
                            cur_joint_positions.append(key.position)
                    cur_joint_positions = np.array(cur_joint_positions)
                    joint_pos_dist = np.linalg.norm(
                        target_joint_positions - cur_joint_positions[np.newaxis, :],
                        axis=1,
                    )
                    cost = joint_pos_dist
                    idx_sorted = np.argsort(cost)
                    best_target_gripper_pose = target_gripper_poses_pass_ik[idx_sorted][
                        idx_sorted
                    ][0]
                    best_target_obj_pose = best_target_gripper_pose @ np.linalg.inv(
                        gripper2obj
                    )
                    target_obj_pose_canonical = (
                        np.linalg.inv(anchor_pose) @ best_target_obj_pose
                    )
                    part2obj = np.eye(4)
                    part2obj[:3, 3] = active_obj.xyz
                    obj2part = np.linalg.inv(part2obj)

                    substages = build_stage(action)(
                        active_obj_id=active_obj_id,
                        passive_obj_id=passive_obj_id,
                        target_pose=target_obj_pose_canonical,
                        current_pose=current_obj_pose_canonical,
                        obj2part=obj2part,
                        vector_direction=passive_element["direction"],
                        passive_element=passive_element,
                        extra_params=stage.get("extra_params", None),
                    )
                    break

            if substages is None:
                logger.error(action, ": No target_obj_pose can pass IK")
                return []
            action_stages.append((action, substages))

    return action_stages
